import itertools
import typing
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
import triton
import triton.language as tl
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from chill_attention.utils import cached_static_property


class ChillMask(ABC):
    """
    Abstract base class for attention mask implementations.

    ChillMask provides the interface and common functionality for defining
    various attention masking patterns. Each mask type implements its own
    specific masking logic while sharing common operations.

    Attributes:
        name (str): The name of the mask (derived from class name).
        constargs (tuple): Constants used to parameterize the mask.
    """

    name: str
    constargs: typing.Tuple[typing.Union[int, float, bool]]

    def __init__(self, constargs=None):
        """
        Initialize a ChillMask.

        Args:
            constargs (tuple, optional): Constants used to parameterize the mask.
                Only int, float, or bool values are supported due to PyTorch/Triton
                interface limitations.
        """
        self.constargs = constargs or tuple()
        self.name = type(self).__name__

        assert isinstance(
            self.constargs, tuple
        ), "As of now, only tuple can be passed as args"
        for i in self.constargs:
            assert isinstance(
                i, typing.Union[int, float, bool]
            ), "Pytorch interface only supports typing.Union[int, float, bool] parametrization"

    def __str__(self):
        return f"{self.name}({', '.join(map(str,self.constargs))})"

    @staticmethod
    @abstractmethod
    def mask(
        q: tl.tensor,
        k: tl.tensor,
        seq_len: tl.tensor,
        args,
    ) -> tl.tensor:
        """
        The mask function that determines which query-key pairs are allowed to attend.
        Mask for seq_len limit is applied automatically

        Args:
            q (tl.tensor): Query token positions (m x 1)
            k (tl.tensor): Key token positions (n x 1)
            seq_len: (tl.tensor): Len of the current sequence
            args: Constants used to parameterize the mask (tuple from __init__)

        Returns:
            tl.tensor: Boolean tensor with True for positions that can attend (m x n shape)
        """
        ...

    @staticmethod
    @abstractmethod
    def q_range_for_k(k: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """
        Determine the range of query indices that can attend to a given key position.

        Args:
            k (int): Key position
            seq_len: (tl.tensor): Len of the current sequence
            args: Constants used to parameterize the mask (tuple from __init__)

        Returns:
            tuple[tl.tensor, tl.tensor]: (start_query_idx, end_query_idx) inclusive range
        """
        ...

    @staticmethod
    @abstractmethod
    def k_range_for_q(q: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """
        Determine the range of key indices that a given query position can attend to.

        Args:
            q (int): Query position
            seq_len: (tl.tensor): Len of the current sequence
            args: Constants used to parameterize the mask (tuple from __init__)

        Returns:
            tuple[tl.tensor, tl.tensor]: (start_key_idx, end_key_idx) inclusive range
        """
        ...

    @staticmethod
    def q_lims_continious():
        """
        Whether query limits for keys form a continuous range.
        Used for optimization.

        If True, we can identify the limits as (q_range_for_k(leftmost)[0], q_range_for_k(rightmost)[1])

        Returns:
            bool: True if limits are continuous, False otherwise
        """
        return True

    @staticmethod
    def k_lims_continious():
        """
        Whether key limits for queries form a continuous range.
        Used for optimization.

        If True, we can identify the limits as (k_range_for_q(leftmost)[0], k_range_for_q(rightmost)[1])

        Returns:
            bool: True if limits are continuous, False otherwise
        """
        return True

    @staticmethod
    def has_full_blocks(q_tile_size, k_tile_size, seq_len, args):
        """

        Whether mask has full blocks for the provided tile size.
        If True, is_full_block will be used to determine if masking is needed.

        Called before kernel execution and not inside, therefore can use any
        python functions.

        seq_len represents size of time dimension

        Returns:
            bool: True full blocks are available

        """
        return False

    @staticmethod
    def is_full_block(
        q,
        k,
        q_tile_size: tl.constexpr,
        k_tile_size: tl.constexpr,
        seq_len: tl.tensor,
        args,
    ):
        """

        Whether current block is fully unmasked.
        If true, will not calculate mask for the block.

        seq_len represents length of the current sequence.

        Returns:
            bool: True if the block is fully unmasked

        """
        return False

    @cached_static_property
    def mask_jit(self):
        """Cached JIT-compiled version of the mask function."""
        return triton.jit()(self.mask)

    @cached_static_property
    def q_range_for_k_jit(self):
        """Cached JIT-compiled version of q_range_for_k function."""
        return triton.jit()(self.q_range_for_k)

    @cached_static_property
    def k_range_for_q_jit(self):
        """Cached JIT-compiled version of k_range_for_q function."""
        return triton.jit()(self.k_range_for_q)

    @cached_static_property
    def is_full_block_jit(self):
        """Cached JIT-compiled version of is_full_block function."""
        return triton.jit()(self.is_full_block)

    @staticmethod
    @triton.jit
    def _mask_infer(
        CHUNK_SIZE: tl.constexpr,
        seq_len: int,
        result: torch.Tensor,
        result_stride_q: int,
        result_stride_k: int,
        args,
        mask,
    ):
        """
        JIT-compiled function to infer the mask for a grid of query-key pairs.

        Args:
            CHUNK_SIZE: Size of chunks to process
            seq_len: (tl.tensor): Len of the current sequence
            result: Output tensor for the mask
            result_stride_q: Stride for query dimension
            result_stride_k: Stride for key dimension
            args: Constants for mask parameterization
            mask: The mask function to apply
        """
        q_start, k_start = (
            tl.program_id(axis=0) * CHUNK_SIZE,
            tl.program_id(axis=1) * CHUNK_SIZE,
        )

        q_pos = q_start + tl.arange(0, CHUNK_SIZE)
        k_pos = k_start + tl.arange(0, CHUNK_SIZE)

        mask_result = mask(q_pos, k_pos, seq_len=seq_len, args=args)
        q_lens_mask = q_pos[:, None] < seq_len
        mask_result &= q_lens_mask & (k_pos[None, :] < seq_len)

        assert mask_result.dtype == tl.int1
        assert mask_result.shape[0] == mask_result.shape[1]
        assert mask_result.shape[0] == CHUNK_SIZE

        chunks_count = tl.num_programs(0)
        result_ptr = tl.make_block_ptr(
            base=result,
            shape=(chunks_count * CHUNK_SIZE, chunks_count * CHUNK_SIZE),
            strides=(result_stride_q, result_stride_k),
            offsets=(q_start, k_start),
            block_shape=(CHUNK_SIZE, CHUNK_SIZE),
            order=(1, 0),
        )
        tl.store(result_ptr, mask_result.to(tl.int8))

    @staticmethod
    @triton.jit
    def _limits_infer(
        CHUNK_SIZE: tl.constexpr,
        seq_len: int,
        result: torch.Tensor,
        result_stride_i: int,
        result_stride_pos: int,
        args,
        rule,
    ):
        """
        JIT-compiled function to infer the limits for each position.

        Args:
            CHUNK_SIZE: Size of chunks to process
            result: Output tensor for limits
            result_stride_i: Stride for position dimension
            result_stride_pos: Stride for limit dimension
            args: Constants for mask parameterization
            rule: The limit function to apply
        """
        i_start = tl.program_id(axis=0) * CHUNK_SIZE

        lower_limits = tl.zeros((CHUNK_SIZE,), dtype=tl.int32)
        upper_limits = tl.zeros((CHUNK_SIZE,), dtype=tl.int32)

        for i in tl.range(0, CHUNK_SIZE):
            mask = tl.arange(0, CHUNK_SIZE) == i
            start, end = rule(i_start + i, seq_len=seq_len, args=args)

            lower_limits = tl.where(mask, start, lower_limits)
            upper_limits = tl.where(mask, end, upper_limits)

        limits = tl.join(lower_limits, upper_limits)

        chunks_count = tl.num_programs(0)
        result_ptr = tl.make_block_ptr(
            base=result,
            shape=(chunks_count * CHUNK_SIZE, chunks_count * CHUNK_SIZE),
            strides=(result_stride_i, result_stride_pos),
            offsets=(i_start, 0),
            block_shape=(CHUNK_SIZE, 2),
            order=(1, 0),
        )
        tl.store(result_ptr, limits.to(tl.int32))

    def make_mask(self, max_pos):
        """
        Create a boolean attention mask matrix.

        Args:
            max_pos (int): Number of positions

        Returns:
            torch.Tensor: Boolean mask of shape (q_pos, k_pos)
        """
        chunk_size = 32
        chunks = triton.cdiv(max_pos, chunk_size)
        total_pos = chunk_size * chunks

        mask = torch.empty(
            (total_pos, total_pos),
            device="cuda",
            dtype=torch.bool,
        )
        self._mask_infer[(chunks, chunks)](
            CHUNK_SIZE=chunk_size,
            seq_len=max_pos,
            result=mask,
            result_stride_q=mask.stride(0),
            result_stride_k=mask.stride(1),
            args=self.constargs,
            mask=self.mask_jit,
        )
        return mask[:max_pos, :max_pos]

    def _calc_limits(self, max_pos, lim_func):
        """
        Calculate the limits for all positions.

        Args:
            max_pos (int): Maximum position
            lim_func: Function to calculate limits

        Returns:
            torch.Tensor: Tensor of limits with shape (max_pos, 2)
        """
        chunk_size = 32
        chunks = triton.cdiv(max_pos, chunk_size)
        total_pos = chunk_size * chunks

        lims = torch.zeros(
            (total_pos, 2),
            device="cuda",
            dtype=torch.int32,
        )
        self._limits_infer[(chunks, 1)](
            CHUNK_SIZE=chunk_size,
            seq_len=total_pos,
            result=lims,
            result_stride_i=lims.stride(0),
            result_stride_pos=lims.stride(1),
            args=self.constargs,
            rule=lim_func,
        )
        lims = torch.where(
            lims >= max_pos,
            max_pos - 1,
            lims,
        )
        return lims[:max_pos]

    def _calc_full_block_indices(self, max_pos, q_tile_size, k_tile_size):
        result = []

        for q in range(0, max_pos, q_tile_size):
            for k in range(0, max_pos, k_tile_size):
                if self.is_full_block(
                    q=q,
                    k=k,
                    q_tile_size=q_tile_size,
                    k_tile_size=k_tile_size,
                    seq_len=max_pos,
                    args=self.constargs,
                ):
                    result.append((q, k))

        return result

    def make_flex_mask(self, max_pos) -> BlockMask | None:
        """
        Create a PyTorch FlexAttention compatible mask if supported.

        Args:
            max_pos (int): Number of positions

        Returns:
            BlockMask | None: FlexAttention compatible mask or None
        """
        return None

    def plot(self, max_pos, draw_full_blocks: bool | int | tuple[int, int] = False):
        """
        Create a visualization of the mask pattern.

        Args:
            max_pos (int): Maximum position to visualize

        Returns:
            matplotlib.figure.Figure: Figure with mask visualization
        """
        mask = self.make_mask(max_pos).cpu()
        q_lims_for_k = self._calc_limits(max_pos, self.q_range_for_k_jit).cpu()
        k_lims_for_q = self._calc_limits(max_pos, self.k_range_for_q_jit).cpu()

        import matplotlib.patches
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(mask, origin="lower", vmin=0, vmax=1)
        axes[0].plot(
            np.array(list(range(len(q_lims_for_k)))),
            np.clip(q_lims_for_k[:, 0].numpy() - 0.5, -0.5, len(q_lims_for_k) - 1),
            color="red",
            label="min q_lims_for_k",
        )
        axes[0].plot(
            np.array(list(range(len(q_lims_for_k)))),
            np.clip(q_lims_for_k[:, 1].numpy() + 0.5, -0.5, len(q_lims_for_k) - 1),
            color="blue",
            label="max q_lims_for_k",
        )
        axes[0].set_xlabel("k index")
        axes[0].set_ylabel("q index")
        axes[0].legend()

        axes[1].imshow(mask.T, origin="lower", vmin=0, vmax=1)
        axes[1].plot(
            np.array(list(range(len(k_lims_for_q)))),
            np.clip(k_lims_for_q[:, 0].numpy() - 0.5, -0.5, len(k_lims_for_q) - 1),
            color="red",
            label="min k_lims_for_q",
        )
        axes[1].plot(
            np.array(list(range(len(k_lims_for_q)))),
            np.clip(k_lims_for_q[:, 1].numpy() + 0.5, -0.5, len(k_lims_for_q) - 1),
            color="blue",
            label="max k_lims_for_q",
        )
        axes[1].set_xlabel("q index")
        axes[1].set_ylabel("k index")

        colors = itertools.cycle(
            itertools.product(
                [
                    "black",
                    "green",
                    "violet",
                    "purple",
                    "darkorange",
                    "white",
                    "lightblue",
                ],
                ["-", "--", ":"],
            )
        )
        if draw_full_blocks:
            if isinstance(draw_full_blocks, bool):
                tile_sizes_q = [16, 32, 64, 128, 256]
                tile_sizes_k = tile_sizes_q
            elif isinstance(draw_full_blocks, int):
                tile_sizes_q = [draw_full_blocks]
                tile_sizes_k = [draw_full_blocks]
            elif isinstance(draw_full_blocks, tuple):
                tile_sizes_q = [draw_full_blocks[0]]
                tile_sizes_k = [draw_full_blocks[1]]

            for q_tile_size in tile_sizes_q:
                for k_tile_size in tile_sizes_k:
                    if not self.has_full_blocks(
                        q_tile_size, k_tile_size, seq_len=max_pos, args=self.constargs
                    ):
                        continue
                    color, style = next(colors)
                    full_blocks = self._calc_full_block_indices(
                        max_pos, q_tile_size, k_tile_size
                    )
                    for q, k in full_blocks:
                        if not (
                            q + q_tile_size <= max_pos and k + k_tile_size <= max_pos
                        ):
                            continue
                        if (~mask[q : q + q_tile_size, k : k + k_tile_size]).all():
                            continue
                        for (x, y, width, height), ax in zip(
                            [
                                (k, q, k_tile_size, q_tile_size),
                                (q, k, q_tile_size, k_tile_size),
                            ],
                            axes,
                        ):
                            rectangle = matplotlib.patches.Rectangle(
                                (x - 0.5, y - 0.5),
                                width,
                                height,
                                ls=style,
                                ec=color,
                                zorder=2,
                                alpha=0.3,
                                fill=False,
                            )
                            ax.add_patch(rectangle)

        axes[1].legend()
        return fig

    def verify(self, max_pos):
        """
        Verify that the mask implementation is correct by comparing
        analytical calculations with actual mask values.

        Args:
            max_pos (int): Maximum position to verify

        Raises:
            AssertionError: If mask implementation is inconsistent
        """
        mask = self.make_mask(max_pos).cpu()
        assert mask.dtype == torch.bool, "Mask must be boolean"

        q_lims_for_k = self._calc_limits(max_pos, self.q_range_for_k_jit).cpu()
        assert q_lims_for_k.dtype in {
            torch.int,
            torch.int64,
        }, "q_lims_for_k must be int"

        k_lims_for_q = self._calc_limits(max_pos, self.k_range_for_q_jit).cpu()
        assert k_lims_for_q.dtype in {
            torch.int,
            torch.int64,
        }, "k_lims_for_q must be int"

        total_pos = len(mask)

        k_lims_for_q_real = defaultdict(lambda: (None, None))
        q_lims_for_k_real = defaultdict(lambda: (None, None))

        def update_lims(prev_lims, new_value):
            return (
                new_value if prev_lims[0] is None else min(prev_lims[0], new_value),
                new_value if prev_lims[1] is None else max(prev_lims[1], new_value),
            )

        for q in range(total_pos):
            for k in range(total_pos):
                if not mask[q, k]:
                    continue

                k_lims_for_q_real[q] = update_lims(k_lims_for_q_real[q], k)
                q_lims_for_k_real[k] = update_lims(q_lims_for_k_real[k], q)

        for q in k_lims_for_q_real:
            real = torch.tensor(k_lims_for_q_real[q])
            analytical = k_lims_for_q[q]
            match = (real == analytical).all().item()
            assert match, (
                "Mismatch real vs analytical k lims for q position. "
                f"{real = }, {analytical = }"
            )

        for k in q_lims_for_k_real:
            real = torch.tensor(q_lims_for_k_real[k])
            analytical = q_lims_for_k[k]
            match = (real == analytical).all().item()
            assert match, (
                "Mismatch real vs analytical q lims for k position. "
                f"{real = }, {analytical = }"
            )

        tile_sizes = [16, 32, 64, 128, 256]
        for q_tile_size in tile_sizes:
            for k_tile_size in tile_sizes:
                if not self.has_full_blocks(
                    q_tile_size, k_tile_size, seq_len=max_pos, args=self.constargs
                ):
                    continue

                full_blocks = self._calc_full_block_indices(
                    max_pos, q_tile_size, k_tile_size
                )
                for q, k in full_blocks:
                    tile = mask[q : q + q_tile_size, k : k + k_tile_size]
                    assert tile.all() or (~tile).all(), (
                        "Encountered unmasked block, while the mask says otherwise. "
                        f"{q = }, {k = }, {q_tile_size = }, {k_tile_size = },\n{tile = }"
                    )


class FullChillMask(ChillMask):
    """
    Full attention mask where each query can attend to all keys.

    This is the standard unrestricted attention pattern used in the original
    Transformer architecture, allowing global context.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def mask(q: tl.tensor, k: tl.tensor, seq_len: tl.tensor, args) -> tl.tensor:
        """
        Full attention - all positions can attend to all other positions.

        Returns:
            tl.tensor: A tensor of all True values
        """
        return tl.full((q.shape[0], k.shape[0]), True, dtype=tl.int1)

    @staticmethod
    def q_range_for_k(k: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """All queries can attend to any key."""
        return 0, seq_len - 1

    @staticmethod
    def k_range_for_q(q: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """Any query can attend to all keys."""
        return 0, seq_len - 1

    def make_flex_mask(self, max_pos) -> BlockMask | None:
        """Create a BlockMask for FlexAttention."""

        def full(b, h, q_idx, kv_idx):
            return q_idx >= 0

        return create_block_mask(full, B=None, H=None, Q_LEN=max_pos, KV_LEN=max_pos)

    @staticmethod
    def has_full_blocks(q_tile_size, k_tile_size, seq_len, args):
        return True

    @staticmethod
    def is_full_block(
        q,
        k,
        q_tile_size: tl.constexpr,
        k_tile_size: tl.constexpr,
        seq_len: tl.tensor,
        args,
    ):
        """All blocks are unmasked"""
        return True


class CausalChillMask(ChillMask):
    """
    Causal attention mask where each query can only attend to itself and preceding keys.

    This is the standard masking pattern used in autoregressive models like GPT,
    preventing information leakage from future tokens.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def mask(q: tl.tensor, k: tl.tensor, seq_len: tl.tensor, args) -> tl.tensor:
        """
        Causal attention - each query can only attend to itself and preceding positions.

        Returns:
            tl.tensor: Boolean tensor with True where q_idx >= k_idx
        """
        return q[:, None] >= k[None, :]

    @staticmethod
    def q_range_for_k(k: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """A key at position k can be attended to by queries at position k or later."""
        return k, seq_len - 1

    @staticmethod
    def k_range_for_q(q: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """A query at position q can attend to keys at position q or earlier."""
        return 0, q

    def make_flex_mask(self, max_pos) -> BlockMask | None:
        """Create a BlockMask for FlexAttention."""

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return create_block_mask(causal, B=None, H=None, Q_LEN=max_pos, KV_LEN=max_pos)

    @staticmethod
    def has_full_blocks(q_tile_size, k_tile_size, seq_len, args):
        return (k_tile_size - 1) <= seq_len - q_tile_size

    @staticmethod
    def is_full_block(
        q,
        k,
        q_tile_size: tl.constexpr,
        k_tile_size: tl.constexpr,
        seq_len: tl.tensor,
        args,
    ):
        """If top left token in the tile is unmasked, all other unmasked too"""
        return (k + k_tile_size - 1) <= q


class SlidingWindowChillMask(ChillMask):
    """
    Sliding window attention mask with configurable left and right context.

    This mask allows each query to attend to a specific number of positions
    before and after it, creating a local attention window.

    Args:
        left_context (int): Number of preceding positions to attend to
        right_context (int): Number of following positions to attend to
    """

    def __init__(self, left_context, right_context):
        super().__init__((left_context, right_context))

    @staticmethod
    def mask(q: tl.tensor, k: tl.tensor, seq_len: tl.tensor, args) -> tl.tensor:
        """
        Sliding window attention - each query attends to positions within a window.

        Args:
            args[0]: left_context - number of preceding positions to attend to
            args[1]: right_context - number of following positions to attend to

        Returns:
            tl.tensor: Boolean tensor with True for positions within the window
        """
        left_context, right_context = args[0], args[1]

        diff = q[:, None] - k[None, :]
        return ((diff <= left_context) & (diff >= 0)) | (
            (diff >= -right_context) & (diff <= 0)
        )

    @staticmethod
    def q_range_for_k(k: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """
        For a key at position k, determine which queries can attend to it.

        Returns:
            tuple: max(0, k - right_context), min(k + left_context, seq_len - 1)
        """
        left_context, right_context = args
        return max(0, k - right_context), min(k + left_context, seq_len - 1)

    @staticmethod
    def k_range_for_q(q: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """
        For a query at position q, determine which keys it can attend to.

        Returns:
            tuple: (max(0, q-left_context), q+right_context)
        """
        left_context, right_context = args
        return max(0, q - left_context), min(q + right_context, seq_len - 1)

    def make_flex_mask(self, max_pos) -> BlockMask | None:
        """Create a BlockMask for FlexAttention."""
        left_context, right_context = self.constargs

        def sliding_window(b, h, q_idx, kv_idx):
            diff = q_idx - kv_idx
            return ((diff <= left_context) & (diff >= 0)) | (
                (diff >= -right_context) & (diff <= 0)
            )

        return create_block_mask(
            sliding_window, B=None, H=None, Q_LEN=max_pos, KV_LEN=max_pos
        )

    @staticmethod
    def has_full_blocks(q_tile_size, k_tile_size, seq_len, args):
        left_context, right_context = args
        return (
            ((q_tile_size <= left_context) and (k_tile_size <= right_context))
            or (q_tile_size * 2 <= left_context)
            or (k_tile_size * 2 <= right_context)
        )

    @staticmethod
    def is_full_block(
        q,
        k,
        q_tile_size: tl.constexpr,
        k_tile_size: tl.constexpr,
        seq_len: tl.tensor,
        args,
    ):
        """If top left token in the tile is unmasked, all other unmasked too"""
        left_context, right_context = args[0], args[1]

        diff_left = (q + q_tile_size - 1) - k
        diff_right = q - (k + k_tile_size - 1)
        return (
            ((diff_left <= left_context) & (diff_left >= 0))
            | ((diff_left >= -right_context) & (diff_left <= 0))
        ) & (
            ((diff_right <= left_context) & (diff_right >= 0))
            | ((diff_right >= -right_context) & (diff_right <= 0))
        )


class ChunkwiseChillMask(ChillMask):
    """
    Chunkwise attention mask where tokens attend to blocks of tokens.

    This mask divides the sequence into fixed-size chunks and allows each query
    to attend to its chunk and a specific number of preceding chunks.

    Args:
        context_size (int): Size of each chunk
        back_contexts (int): Number of preceding chunks to attend to
    """

    def __init__(self, context_size, back_contexts):
        super().__init__(
            (context_size, back_contexts),
        )

    @staticmethod
    def mask(
        q: tl.tensor,
        k: tl.tensor,
        seq_len: tl.tensor,
        args,
    ) -> tl.tensor:
        """
        Chunkwise attention - each query attends to its chunk and preceding chunks.

        Args:
            args[0]: context_size - size of each chunk
            args[1]: back_contexts - number of preceding chunks to attend to

        Returns:
            tl.tensor: Boolean tensor with True for positions in allowed chunks
        """
        context_size, back_contexts = args
        q, k = q // context_size, k // context_size
        blocks_diff = q[:, None] - k[None, :]
        return (blocks_diff >= 0) & (blocks_diff <= back_contexts)

    @staticmethod
    def q_range_for_k(
        k: int,
        seq_len: tl.tensor,
        args,
    ) -> tuple[tl.tensor, tl.tensor]:
        """
        For a key at position k, determine which queries can attend to it.

        Returns:
            tuple: Range of query indices that can attend to key position k
        """
        context_size, back_contexts = args
        kv_tile_context = k // context_size
        q_start_idx = kv_tile_context * context_size
        q_end_tile_idx = (kv_tile_context + back_contexts + 1) * context_size - 1
        return q_start_idx, min(q_end_tile_idx, seq_len - 1)

    @staticmethod
    def k_range_for_q(
        q: int,
        seq_len: tl.tensor,
        args,
    ) -> tuple[tl.tensor, tl.tensor]:
        """
        For a query at position q, determine which keys it can attend to.

        Returns:
            tuple: Range of key indices that query position q can attend to
        """
        context_size, back_contexts = args
        q_tile_context = q // context_size
        kv_start_idx = max(0, ((q_tile_context - back_contexts) * context_size))
        kv_end_tile_idx = (q_tile_context + 1) * context_size - 1

        return kv_start_idx, min(kv_end_tile_idx, seq_len - 1)

    def make_flex_mask(self, max_pos) -> BlockMask | None:
        """Create a BlockMask for FlexAttention."""
        context_size, back_contexts = self.constargs

        def chunkwise_mask(b, h, q_idx, kv_idx):
            q_idx, kv_idx = q_idx // context_size, kv_idx // context_size
            blocks_diff = q_idx - kv_idx
            return (blocks_diff >= 0) & (blocks_diff <= back_contexts)

        return create_block_mask(
            chunkwise_mask, B=None, H=None, Q_LEN=max_pos, KV_LEN=max_pos
        )

    @staticmethod
    def has_full_blocks(q_tile_size, k_tile_size, seq_len, args):
        context_size, back_contexts = args
        first_q_max = context_size * (back_contexts + 1)

        # worst scenario
        return ChunkwiseChillMask.is_full_block(
            q=first_q_max - q_tile_size,
            k=0,
            q_tile_size=q_tile_size,
            k_tile_size=k_tile_size,
            seq_len=seq_len,
            args=args,
        )

    @staticmethod
    def is_full_block(
        q,
        k,
        q_tile_size: tl.constexpr,
        k_tile_size: tl.constexpr,
        seq_len: tl.tensor,
        args,
    ):
        """If first token in the tile is unmasked, all other unmasked too"""
        context_size, back_contexts = args

        q_block, k_block = q // context_size, (k + k_tile_size - 1) // context_size
        blocks_diff = q_block - k_block
        result = (blocks_diff >= 0) & (blocks_diff <= back_contexts)

        q_block, k_block = (q + q_tile_size - 1) // context_size, k // context_size
        blocks_diff = q_block - k_block
        return result & (blocks_diff >= 0) & (blocks_diff <= back_contexts)


class PrefixLMChillMask(ChillMask):
    """
    Prefix Language Model attention mask.

    This mask allows bidirectional attention within a prefix region (like BERT)
    and causal attention for the rest of the sequence (like GPT).

    Args:
        prefix_size (int): Size of the bidirectional prefix region
    """

    def __init__(self, prefix_size):
        super().__init__(
            (prefix_size,),
        )

    @staticmethod
    def mask(
        q: tl.tensor,
        k: tl.tensor,
        seq_len: tl.tensor,
        args,
    ) -> tl.tensor:
        """
        Prefix LM attention - bidirectional in prefix, causal elsewhere.

        Args:
            args[0]: prefix_size - size of the bidirectional prefix region

        Returns:
            tl.tensor: Boolean tensor combining prefix and causal masks
        """
        prefix_size = args[0]
        prefix_mask = k < prefix_size
        causal_mask = q[:, None] >= k[None, :]
        return prefix_mask[None, :] | causal_mask

    @staticmethod
    def q_range_for_k(
        k: int,
        seq_len: tl.tensor,
        args,
    ) -> tuple[tl.tensor, tl.tensor]:
        """
        For a key at position k, determine which queries can attend to it.

        Returns:
            tuple: Range of query indices that can attend to key position k
        """
        prefix_size = args[0]
        left_border = k
        if k < prefix_size:
            left_border = 0
        return left_border, seq_len - 1

    @staticmethod
    def k_range_for_q(
        q: int,
        seq_len: tl.tensor,
        args,
    ) -> tuple[tl.tensor, tl.tensor]:
        """
        For a query at position q, determine which keys it can attend to.

        Returns:
            tuple: Range of key indices that query position q can attend to
        """
        prefix_size = args[0]
        right_window = max(q, prefix_size - 1)
        return 0, min(right_window, seq_len - 1)

    def make_flex_mask(self, max_pos) -> BlockMask | None:
        """Create a BlockMask for FlexAttention."""
        prefix_size = self.constargs[0]

        def prefix_lm(b, h, q, k):
            prefix_mask = k < prefix_size
            causal_mask = q >= k
            return prefix_mask | causal_mask

        return create_block_mask(
            prefix_lm, B=None, H=None, Q_LEN=max_pos, KV_LEN=max_pos
        )

    @staticmethod
    def has_full_blocks(q_tile_size, k_tile_size, seq_len, args):
        return True

    @staticmethod
    def is_full_block(
        q,
        k,
        q_tile_size: tl.constexpr,
        k_tile_size: tl.constexpr,
        seq_len: tl.tensor,
        args,
    ):
        """If first token in the tile is unmasked, all other unmasked too"""
        prefix_size = args[0]
        causal_full = (k + k_tile_size - 1) <= q
        in_prefix = (q + q_tile_size <= prefix_size) & (k + k_tile_size <= prefix_size)
        return causal_full | in_prefix


if __name__ == "__main__":
    # full_mask = ChunkwiseChillMask(
    #     32, 1
    # )
    full_mask = SlidingWindowChillMask(10, 4)
    # full_mask = PrefixLMChillMask(64)
    # full_mask = FullChillMask()
    # full_mask = CausalChillMask()
    fig = full_mask.plot(200, True)
    fig.savefig("mask.png")
    full_mask.verify(1024)
