import logging
import typing

import torch
import torch.nn.functional as F
import triton

from chill_attention.autotune import (
    _get_backward_autotune_configs,
    _get_default_config_bwd,
    _get_default_config_fwd,
    _get_forward_autotune_configs,
    _prune_notfitting_configs,
)

from .chill_bwd import _chill_attn_bwd, attention_backward_adapter_op_setup_context
from .chill_fwd import _chill_attn_fwd
from .mask import ChillMask
from .utils import (
    _chill_attn_bwd_precompute,
    _triton_set_alloc,
    strides,
)

logger = logging.getLogger(__name__)


def register_chill_mask(mask: ChillMask):
    """
    Register a ChillMask with PyTorch's dispatcher system.

    This function creates and registers custom PyTorch operations for the forward and backward
    passes of attention with the provided mask pattern. The registration happens only once
    per mask type.

    Args:
        mask (ChillMask): The mask pattern to register
    """
    mask_name = mask.name
    if hasattr(torch.ops.chill_attention, f"forward{mask_name}"):
        return

    mask_fns = (
        mask.mask_jit,
        mask.k_range_for_q_jit,
        mask.q_range_for_k_jit,
        mask.is_full_block_jit,
        mask.k_full_range_for_q_jit,
        mask.q_full_range_for_k_jit,
    )

    q_lims_continious = mask.q_lims_continious()
    k_lims_continious = mask.k_lims_continious()

    autotunes = dict()
    autotunes_bwd = dict()

    @torch.library.custom_op(
        f"chill_attention::forward{mask_name}", mutates_args=(), device_types=("cuda",)
    )
    def attention_forward_adapter(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lens: torch.Tensor,
        mask_args: typing.List[typing.Union[int, float, bool]],
        sm_scale: float,
        return_lse: bool,
        prescale_qk: bool,
        precision: str,
        autotune: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, heads, T, HEAD_DIM = q.shape
        num_kv_heads = k.shape[1]
        assert heads % num_kv_heads == 0
        group_size = heads // num_kv_heads

        assert HEAD_DIM in {16, 32, 64, 128, 256}
        assert HEAD_DIM == k.shape[-1] and HEAD_DIM == v.shape[-1]
        assert T == k.shape[-2] and T == v.shape[-2]
        assert sm_scale is not None
        assert lens is None or (
            lens.dtype == torch.int32 and batch == len(lens) and lens.ndim == 1
        )

        _triton_set_alloc()

        O = torch.zeros_like(q, memory_format=torch.contiguous_format)
        LSE = None
        if return_lse:
            LSE = torch.zeros(q.shape[:3], dtype=torch.float32, device=q.device)

        use_tma = (
            q.stride(-1) == 1  # q is contiguous
            and k.stride(-1) == 1  # k is contiguous
            and v.stride(-1) == 1  # v is contiguous
            and O.stride(-1) == 1  # O is contiguous
            and (LSE is None or LSE.stride(-1) == 1)  # LSE is contiguous
            and q.stride(0) % 16 == 0  # stride_qb is multiple of 16
            and k.stride(0) % 16 == 0  # stride_kb is multiple of 16
            and v.stride(0) % 16 == 0  # stride_vb is multiple of 16
            and O.stride(0) % 16 == 0  # stride_0b is multiple of 16
            and q.stride(1) % 16 == 0  # stride_qh is multiple of 16
            and k.stride(1) % 16 == 0  # stride_kh is multiple of 16
            and v.stride(1) % 16 == 0  # stride_vh is multiple of 16
            and O.stride(1) % 16 == 0  # stride_ob is multiple of 16
            and (
                LSE is None
                or LSE.stride(0) % 16 == 0  # stride_mb is multiple of 16
                and LSE.stride(1) % 16 == 0  # stride_mh is multiple of 16
            )
        )

        def grid(args):
            return batch, heads, triton.cdiv(T, args["TILE_Q_SIZE"])

        MASK_SAFE = mask.is_guaranteed_safe() and lens is None

        args = [
            q,
            k,
            v,
            lens,
            LSE,
            O,
            *strides(q, 4),
            *strides(k, 4),
            *strides(v, 4),
            *(strides(LSE, 3) if LSE is not None else [0] * 3),
            *strides(O, 4),
            *(strides(lens, 1) if lens is not None else [0]),
        ]
        kwargs = dict(
            T=T,
            HEAD_DIM=HEAD_DIM,
            INPUT_PRECISION=precision,
            PRESCALE_QK=prescale_qk,
            DTYPE=q.dtype,
            TIME_BUCKET=triton.next_power_of_2(T),
            OUTPUT_LOGSUMEXP=return_lse,
            SM_SCALE=sm_scale,
            mask_fns=tuple(mask_fns),
            mask_args=tuple(mask_args),
            k_lims_continious=k_lims_continious,
            USE_TMA=use_tma,
            GROUP_SIZE=group_size,
        )

        kernel = triton.heuristics(
            dict(
                HAS_FULL_BLOCKS=lambda args: mask.has_full_blocks(
                    args["TILE_Q_SIZE"],
                    args["TILE_K_SIZE"],
                    seq_len=args["T"],
                    args=mask_args,
                ),
                ROWS_GUARANTEED_SAFE=lambda args: (
                    MASK_SAFE and lens is None and (args["T"] >= args["TILE_Q_SIZE"])
                ),
            )
        )(_chill_attn_fwd)

        if autotune:
            if (HEAD_DIM, q.dtype) not in autotunes:
                configs = _get_forward_autotune_configs(
                    HEAD_DIM, q.dtype, has_k_full_range=mask.has_k_full_range()
                )
                configs = _prune_notfitting_configs(configs, kernel, args, kwargs, grid)
                logger.info(
                    f"{mask_name} {(HEAD_DIM, q.dtype)}: using {len(configs)} configs for forward autotune"
                )
                autotunes[(HEAD_DIM, q.dtype)] = triton.autotune(
                    configs=configs,
                    key=[
                        "TIME_BUCKET",
                        "INPUT_PRECISION",
                        "PRESCALE_QK",
                        "OUTPUT_LOGSUMEXP",
                    ],
                    use_cuda_graph=False,
                )(kernel)
            kernel = autotunes[(HEAD_DIM, q.dtype)]
        else:
            TILE_Q_SIZE, TILE_K_SIZE, N_WARPS, PIPELINING, TENSORS_PRELOAD = (
                _get_default_config_fwd(HEAD_DIM, dtype=q.dtype)
            )
            kernel = triton.heuristics(
                dict(
                    TILE_Q_SIZE=lambda _: TILE_Q_SIZE,
                    TILE_K_SIZE=lambda _: TILE_K_SIZE,
                    PIPELINING=lambda _: PIPELINING,
                    TENSORS_PRELOAD=lambda _: TENSORS_PRELOAD,
                    SPLIT_LOOPS=lambda _: mask.has_k_full_range(),
                    num_warps=lambda _: N_WARPS,
                    num_stages=lambda _: PIPELINING,
                )
            )(kernel)

        kernel[grid](*args, **kwargs)

        if LSE is None:
            LSE = torch.empty(0)
        return O, LSE

    @torch.library.register_fake(f"chill_attention::forward{mask_name}")
    def attention_forward_adapter_abstract(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lens: torch.Tensor,
        mask_args: typing.List[typing.Union[int, float, bool]],
        sm_scale: float,
        return_lse: bool,
        prescale_qk: bool,
        precision: str,
        autotune: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty_like(q, memory_format=torch.contiguous_format),
            (
                torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
                if return_lse
                else torch.empty(0)
            ),
        )

    @torch.library.custom_op(
        f"chill_attention::backward{mask_name}", mutates_args=(), device_types=("cuda",)
    )
    def attention_backward_adapter(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lens: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        do: torch.Tensor,
        mask_args: typing.List[typing.Union[int, float, bool]],
        sm_scale: float,
        prescale_qk: bool,
        precision: str,
        autotune: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, heads, T, HEAD_DIM = q.shape
        num_kv_heads = k.shape[1]
        assert heads % num_kv_heads == 0
        group_size = heads // num_kv_heads

        _triton_set_alloc()

        delta = torch.empty(o.shape[:-1], dtype=torch.float32, device=o.device)

        def grid(args):
            return batch, heads, triton.cdiv(T, args["TILE_SIZE"])

        use_precompute_tma = (
            o.stride(-1) == 1
            and o.stride(0) % 16 == 0
            and o.stride(1) % 16 == 0
            and do.stride(-1) == 1
            and do.stride(0) % 16 == 0
            and do.stride(1) % 16 == 0
            and delta.stride(-1) == 1
            and delta.stride(0) % 16 == 0
            and delta.stride(1) % 16 == 0
        )

        _chill_attn_bwd_precompute[grid](
            o,
            do,
            delta,
            *strides(o, 4),
            *strides(do, 4),
            *strides(delta, 3),
            T=T,
            HEAD_DIM=HEAD_DIM,
            DTYPE=q.dtype,
            TIME_BUCKET=triton.next_power_of_2(T),
            USE_TMA=use_precompute_tma,
        )

        DQ = torch.zeros_like(q, memory_format=torch.contiguous_format)
        DK = torch.zeros_like(k, memory_format=torch.contiguous_format)
        DV = torch.zeros_like(v, memory_format=torch.contiguous_format)

        use_tma = (
            q.stride(-1) == 1  # q is contiguous
            and k.stride(-1) == 1  # k is contiguous
            and v.stride(-1) == 1  # v is contiguous
            and o.stride(-1) == 1  # o is contiguous
            and do.stride(-1) == 1  # do is contiguous
            and (lens is None or lens.stride(-1) == 1)  # lens is contiguous
            and delta.stride(-1) == 1  # delta is contiguous
            and lse.stride(-1) == 1  # lse is contiguous
            and q.stride(0) % 16 == 0  # stride_qb is multiple of 16
            and k.stride(0) % 16 == 0  # stride_kb is multiple of 16
            and v.stride(0) % 16 == 0  # stride_vb is multiple of 16
            and lse.stride(0) % 16 == 0  # stride_mb is multiple of 16
            and delta.stride(0) % 16 == 0  # stride_deltab is multiple of 16
            and do.stride(0) % 16 == 0  # stride_dob is multiple of 16
            and DQ.stride(0) % 16 == 0  # stride_dqb is multiple of 16
            and DK.stride(0) % 16 == 0  # stride_dkb is multiple of 16
            and DV.stride(0) % 16 == 0  # stride_dvb is multiple of 16
            and q.stride(1) % 16 == 0  # stride_qh is multiple of 16
            and k.stride(1) % 16 == 0  # stride_kh is multiple of 16
            and v.stride(1) % 16 == 0  # stride_vh is multiple of 16
            and lse.stride(1) % 16 == 0  # stride_mh is multiple of 16
            and delta.stride(1) % 16 == 0  # stride_deltah is multiple of 16
            and do.stride(1) % 16 == 0  # stride_doh is multiple of 16
            and DQ.stride(1) % 16 == 0  # stride_dqh is multiple of 16
            and DK.stride(1) % 16 == 0  # stride_dkh is multiple of 16
            and DV.stride(1) % 16 == 0  # stride_dvh is multiple of 16
        )

        def grid(args):
            return (
                batch,
                heads * triton.cdiv(T, args["TILE_DQ_Q_SIZE"])
                + num_kv_heads * triton.cdiv(T, args["TILE_DK_K_SIZE"]),
            )

        MASK_SAFE = mask.is_guaranteed_safe()

        args = [
            q,
            k,
            v,
            lens,
            delta,
            lse,
            do,
            DQ,
            DK,
            DV,
            *strides(q, 4),
            *strides(k, 4),
            *strides(v, 4),
            *strides(delta, 3),
            *strides(lse, 3),
            *strides(do, 4),
            *strides(DQ, 4),
            *strides(DK, 4),
            *strides(DV, 4),
            *(strides(lens, 1) if lens is not None else [0]),
        ]
        kwargs = dict(
            T=T,
            HEAD_DIM=HEAD_DIM,
            TIME_BUCKET=triton.next_power_of_2(T),
            INPUT_PRECISION=precision,
            DTYPE=q.dtype,
            SM_SCALE=sm_scale,
            PRESCALE_QK=prescale_qk,
            mask_fns=tuple(mask_fns),
            mask_args=tuple(mask_args),
            k_lims_continious=k_lims_continious,
            q_lims_continious=q_lims_continious,
            USE_TMA=use_tma,
            GROUP_SIZE=group_size,
            NUM_KV_HEADS=num_kv_heads,
        )

        kernel = triton.heuristics(
            dict(
                HAS_FULL_BLOCKS_DQ=lambda args: mask.has_full_blocks(
                    args["TILE_DQ_Q_SIZE"],
                    args["TILE_DQ_K_SIZE"],
                    seq_len=args["T"],
                    args=mask_args,
                ),
                HAS_FULL_BLOCKS_DK=lambda args: mask.has_full_blocks(
                    args["TILE_DK_Q_SIZE"],
                    args["TILE_DK_K_SIZE"],
                    seq_len=args["T"],
                    args=mask_args,
                ),
                ROWS_GUARANTEED_SAFE=lambda args: (
                    MASK_SAFE
                    and lens is None
                    and (
                        args["T"] >= args["TILE_DQ_Q_SIZE"]
                        and args["T"] >= args["TILE_DK_Q_SIZE"]
                    )
                ),
            )
        )(_chill_attn_bwd)

        if autotune:
            if (HEAD_DIM, q.dtype) not in autotunes_bwd:
                configs = _get_backward_autotune_configs(
                    HEAD_DIM,
                    q.dtype,
                    has_k_full_range=mask.has_k_full_range(),
                    has_q_full_range=mask.has_q_full_range(),
                )
                configs = _prune_notfitting_configs(configs, kernel, args, kwargs, grid)
                logger.info(
                    f"{mask_name} {(HEAD_DIM, q.dtype)}: using {len(configs)} configs for backward autotune"
                )
                autotunes_bwd[(HEAD_DIM, q.dtype)] = triton.autotune(
                    configs=configs,
                    key=[
                        "TIME_BUCKET",
                        "INPUT_PRECISION",
                        "PRESCALE_QK",
                        "OUTPUT_LOGSUMEXP",
                    ],
                    use_cuda_graph=False,
                )(kernel)
            kernel = autotunes_bwd[(HEAD_DIM, q.dtype)]
        else:
            TILE_DQ_Q_SIZE, TILE_DQ_K_SIZE, N_WARPS, PIPELINING, TENSORS_PRELOAD = (
                _get_default_config_bwd(HEAD_DIM, dtype=q.dtype)
            )
            TILE_DK_Q_SIZE = TILE_DQ_Q_SIZE
            TILE_DK_K_SIZE = TILE_DQ_K_SIZE
            kernel = triton.heuristics(
                dict(
                    TILE_DQ_Q_SIZE=lambda _: TILE_DQ_Q_SIZE,
                    TILE_DQ_K_SIZE=lambda _: TILE_DQ_K_SIZE,
                    TILE_DK_Q_SIZE=lambda _: TILE_DK_Q_SIZE,
                    TILE_DK_K_SIZE=lambda _: TILE_DK_K_SIZE,
                    TENSORS_PRELOAD=lambda _: TENSORS_PRELOAD,
                    PIPELINING=lambda _: PIPELINING,
                    SPLIT_LOOPS=lambda _: mask.has_k_full_range(),
                    SPLIT_LOOPS_KV=lambda _: mask.has_q_full_range(),
                    num_warps=lambda _: N_WARPS,
                    num_stages=lambda _: PIPELINING,
                )
            )(kernel)

        kernel[grid](*args, **kwargs)

        return DQ, DK, DV

    @torch.library.register_fake(f"chill_attention::backward{mask_name}")
    def attention_backward_adapter_abstract(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lens: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        do: torch.Tensor,
        mask_args: typing.List[typing.Union[int, float, bool]],
        sm_scale: float,
        prescale_qk: bool,
        precision: str,
        autotune: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        DQ = torch.empty_like(q, memory_format=torch.contiguous_format)
        DK = torch.empty_like(k, memory_format=torch.contiguous_format)
        DV = torch.empty_like(v, memory_format=torch.contiguous_format)
        return DQ, DK, DV

    def attention_backward_adapter_op(ctx, do, dlse):
        q, k, v, o, lse, lens = ctx.saved_tensors
        mask_args = ctx.mask_args
        sm_scale = ctx.sm_scale
        prescale_qk = ctx.prescale_qk
        precision = ctx.precision
        autotune = ctx.autotune

        _triton_set_alloc()
        DQ, DK, DV = attention_backward_adapter(
            q=q,
            k=k,
            v=v,
            lens=lens,
            o=o,
            lse=lse,
            do=do,
            sm_scale=sm_scale,
            prescale_qk=prescale_qk,
            precision=precision,
            mask_args=mask_args,
            autotune=autotune,
        )

        return DQ, DK, DV, None, None, None, None, None, None, None, None, None

    attention_forward_adapter.register_autograd(
        attention_backward_adapter_op,
        setup_context=attention_backward_adapter_op_setup_context,
    )


def _chill_reference_naive(
    mask: ChillMask,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
):
    """
    A reference implementation of attention for testing/verification purposes.

    This function implements attention using PyTorch's native operations, which is
    slower but useful for testing and validating the correctness of the optimized version.

    Args:
        mask (ChillMask): Attention mask pattern
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        v (torch.Tensor): Value tensor
        lens (torch.Tensor | None): Optional sequence lengths

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (output tensor, mask tensor)
    """
    assert q.size(2) == k.size(2)
    if q.shape[1] != k.shape[1]:
        group_size = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

    attn_mask = mask.make_mask(q.size(2))
    T = q.size(2)

    if lens is not None:
        key_padding_mask = (
            torch.arange(T, device="cuda").unsqueeze(0) < lens.unsqueeze(-1)
        ).unsqueeze(-1)
        key_padding_mask_ref = key_padding_mask
        key_padding_mask = key_padding_mask & key_padding_mask.transpose(-1, -2)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) & key_padding_mask.unsqueeze(1)
        res_mask = key_padding_mask_ref.unsqueeze(1)
    else:
        res_mask = 1
    res_mask = torch.as_tensor(res_mask, device=q.device)
    return (
        F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask) * res_mask,
        res_mask,
    )


def _chill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lens: torch.Tensor | None,
    mask_name: str,
    mask_args: typing.List[typing.Union[int, float, bool]],
    sm_scale: float | None,
    return_lse: bool,
    prescale_qk: bool,
    precision: str,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    requires_grad = any(i.requires_grad for i in (q, k, v))
    attention_forward = getattr(torch.ops.chill_attention, f"forward{mask_name}")
    O, LSE = attention_forward(
        q=q,
        k=k,
        v=v,
        lens=lens,
        mask_args=mask_args,
        sm_scale=sm_scale,
        prescale_qk=prescale_qk,
        return_lse=return_lse or requires_grad,
        precision=precision,
        autotune=autotune,
    )
    if return_lse:
        return O, LSE
    return O


def chill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: ChillMask,
    lens: torch.Tensor | None = None,
    sm_scale: float | None = None,
    return_lse=False,
    prescale_qk=False,
    precision="ieee",
    autotune=False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """
    Efficient attention implementation using Triton kernels with flexible masking.

    This function performs the multi-head attention operation efficiently on CUDA GPUs
    using Triton kernels. It supports various masking patterns through the ChillMask interface.
    GQA is activated if num_kv_heads != num_heads.

    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        k (torch.Tensor): Key tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
        v (torch.Tensor): Value tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
        mask (ChillMask): Attention mask pattern that determines which query-key pairs can attend
        lens (torch.Tensor | None): Optional tensor of sequence lengths for each batch element
                                   Shape [batch_size], dtype torch.int32
        sm_scale (float | None): Scaling factor for attention scores. If None, uses 1/sqrt(head_dim)
        return_lse (bool): Whether to return the log-sum-exp values alongside the output. Default: False
        prescale_qk (bool): Whether to apply the scale before the softmax. Default: False
        precision (str): Precision to use for computations. Options: "ieee" (default), "tf32"
        autotune (bool): Whether to use autotuning for kernel parameters. Default: False

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
        torch.Tensor: (Optional, only if return_lse=True) LogSumExp values

    Examples:
        >>> q = torch.randn(2, 8, 512, 64, device="cuda")
        >>> k = torch.randn(2, 8, 512, 64, device="cuda")
        >>> v = torch.randn(2, 8, 512, 64, device="cuda")
        >>> mask = CausalChillMask()
        >>> output = chill_attention(q, k, v, lens=None, mask=mask)
    """

    # Compile and register the mask if not already done
    register_chill_mask(mask)
    if not torch.compiler.is_compiling():
        for i in (q, k, v):
            torch._dynamo.mark_static(i, 1)
            torch._dynamo.mark_static(i, 3)
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Call through to the registered PyTorch operation
    return _chill_attention(
        q=q,
        k=k,
        v=v,
        lens=lens,
        mask_name=mask.name,
        mask_args=mask.constargs,
        sm_scale=sm_scale,
        return_lse=return_lse,
        prescale_qk=prescale_qk,
        precision=precision,
        autotune=autotune,
    )
