import math

import torch
import triton
import triton.language as tl


def strides(t: torch.Tensor, expected_size=None):
    """
    Extract strides from a PyTorch tensor with optional size validation.

    Args:
        t (torch.Tensor): Input tensor
        expected_size (int, optional): Expected number of dimensions for validation

    Returns:
        list: List of strides for each dimension of the tensor

    Raises:
        AssertionError: If the tensor is None or doesn't match the expected size
    """
    assert t is not None
    if expected_size is not None:
        assert t.ndim == expected_size
    return [t.stride(i) for i in range(t.ndim)]


class cached_static_property(object):
    """
    Decorator for creating cached static properties.

    This property is computed once per class and then cached,
    avoiding redundant calculations for the same class.

    Example:
        class Example:
            @cached_static_property
            def value(cls):
                # Computed once and cached
                return expensive_calculation()
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, _, typ):
        value = self.func(typ)
        setattr(typ, self.func.__name__, value)
        return value


@triton.autotune(
    configs=[
        triton.Config(
            dict(
                TILE_SIZE=tile,
            ),
            num_warps=num_warps,
            num_stages=stages,
        )
        for num_warps in [2, 4, 8]
        for tile in [32, 64, 128]
        for stages in [1, 2, 3]
    ],
    key=["HEAD_DIM", "DTYPE", "TIME_BUCKET"],
)
@triton.heuristics(
    dict(
        BLOCK_DIVISIBLE=lambda args: args["T"] % args["TILE_SIZE"] == 0,
    )
)
@triton.jit
def _chill_attn_bwd_precompute(
    O: tl.tensor,
    DO: tl.tensor,
    RES: tl.tensor,
    stride_ob: int,
    stride_oh: int,
    stride_ot: int,
    stride_ok: int,  #
    stride_dob: int,
    stride_doh: int,
    stride_dot: int,
    stride_dok: int,  #
    stride_rb: int,
    stride_rh: int,
    stride_rt: int,
    T: int,
    TIME_BUCKET: int,  #
    HEAD_DIM: tl.constexpr,
    DTYPE: tl.constexpr,  #
    TILE_SIZE: tl.constexpr,
    BLOCK_DIVISIBLE: tl.constexpr,  #
):
    """
    Precompute operation for attention backward pass.

    This kernel computes the dot product of the output tensor O and
    its gradient DO for efficient backward pass computation.

    Args:
        O: Output tensor
        DO: Output gradient tensor
        RES: Result tensor for storing the dot product
        stride_*: Strides for various tensor dimensions
        T: Sequence length
        TIME_BUCKET: Time bucket size
        HEAD_DIM: Head dimension (compile-time constant)
        DTYPE: Data type (compile-time constant)
        TILE_SIZE: Tile size for computation (compile-time constant)
        BLOCK_DIVISIBLE: Whether T is divisible by TILE_SIZE (compile-time constant)
    """
    batch = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)

    token_idx = tile * TILE_SIZE

    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(token_idx, 0),
        block_shape=(TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dot, stride_dok),
        offsets=(token_idx, 0),
        block_shape=(TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    if BLOCK_DIVISIBLE:
        o_tile = tl.load(
            o_tile_ptr,
        )
        do_tile = tl.load(
            do_tile_ptr,
        )
    else:
        o_tile = tl.load(o_tile_ptr, boundary_check=(0,))
        do_tile = tl.load(do_tile_ptr, boundary_check=(0,))

    res = tl.sum(o_tile.to(tl.float32) * do_tile.to(tl.float32), 1)

    rbatch_head_offset = batch * stride_rb + head * stride_rh
    res_ptr = tl.make_block_ptr(
        base=RES + rbatch_head_offset,
        shape=(T,),
        strides=(stride_rt,),
        offsets=(token_idx,),
        block_shape=(TILE_SIZE,),
        order=(0,),
    )

    if BLOCK_DIVISIBLE:
        tl.store(res_ptr, res)
    else:
        tl.store(res_ptr, res, boundary_check=(0,))


@triton.jit
def _get_min_max_tiles(
    i,
    seq_len,
    fn_lims: tl.constexpr,
    mask_args: tl.constexpr,
    continious: tl.constexpr,
    TILE_SIZE_IN: tl.constexpr,
    TILE_SIZE_OUT: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    """
    Calculate minimum and maximum tile indices for efficient attention computation.

    This utility function helps determine the range of tiles that need to be processed
    for a given input position, optimizing the computation by skipping unnecessary tiles.

    Args:
        i: Starting position
        seq_len: Sequence length
        fn_lims: Limit function (compile-time constant)
        mask_args: Mask arguments (compile-time constant)
        continious: Whether limits are continuous (compile-time constant)
        TILE_SIZE_IN: Input tile size (compile-time constant)
        TILE_SIZE_OUT: Output tile size (compile-time constant)

    Returns:
        tuple: (start_tile, end_tile) indicating the range of tiles to process
    """
    if continious:
        i_max = min(i + TILE_SIZE_IN, seq_len) - 1
        left_start, _ = fn_lims(i, seq_len=seq_len, args=mask_args)
        _, right_end = fn_lims(i_max, seq_len=seq_len, args=mask_args)

        start = left_start
        end = right_end
    else:
        start = seq_len - 1
        end = 0
        for i in i + tl.arange(0, TILE_SIZE_IN):
            new_start, new_end = fn_lims(i, seq_len=seq_len, args=mask_args)
            start = min(start, new_start)
            end = max(end, new_end)
    start_tile = start // TILE_SIZE_OUT
    end_tile = tl.cdiv(end + 1, TILE_SIZE_OUT)
    return start_tile, end_tile
