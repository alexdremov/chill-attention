import logging
import math

import triton
import triton.language as tl

from .chill_bwd_dkdv import _chill_attn_bwd_dkdv_inner
from .chill_bwd_dq import _chill_attn_bwd_dq_inner

logger = logging.getLogger(__name__)


# fmt: off
@triton.heuristics(
    dict(
        RCP_LN2=lambda _: math.log2(math.e),
        DQ_TILES_NUM=lambda args: triton.cdiv(args['T'], args["TILE_DQ_Q_SIZE"]),
        DK_TILES_NUM=lambda args: triton.cdiv(args['T'], args["TILE_DK_K_SIZE"]),
        DQ_Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DQ_Q_SIZE'] == 0,
        DQ_K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DQ_K_SIZE'] == 0,
        DK_Q_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DK_Q_SIZE'] == 0,
        DK_K_BLOCK_DIVISIBLE=lambda args : args['T'] % args['TILE_DK_K_SIZE'] == 0,
    )
)
@triton.jit
def _chill_attn_bwd(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, L: tl.tensor, #
    DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor, DK: tl.tensor, DV: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: tl.constexpr,  #
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: tl.constexpr,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: tl.constexpr,  #
    stride_deltab: int, stride_deltah: int, stride_deltat: tl.constexpr,  #
    stride_mb: int, stride_mh: int, stride_mt: tl.constexpr,  #
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: tl.constexpr,  #
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: tl.constexpr,  #
    stride_dkb: int, stride_dkh: int, stride_dkt: int, stride_dkk: tl.constexpr,  #
    stride_dvb: int, stride_dvh: int, stride_dvt: int, stride_dvk: tl.constexpr,  #
    lens_stride: int,
    T: int,  #
    TIME_BUCKET: tl.constexpr,  #
    DQ_TILES_NUM: int,  #
    DK_TILES_NUM: int,  #
    HEAD_DIM: tl.constexpr,  #
    DTYPE: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    DQ_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DQ_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    HAS_FULL_BLOCKS_DQ: tl.constexpr,  #
    HAS_FULL_BLOCKS_DK: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    GROUP_SIZE: tl.constexpr, #
    NUM_KV_HEADS: tl.constexpr, #
    TILE_DQ_Q_SIZE: tl.constexpr, TILE_DQ_K_SIZE: tl.constexpr,  #
    TILE_DK_Q_SIZE: tl.constexpr, TILE_DK_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    SPLIT_LOOPS: tl.constexpr,  #
    SPLIT_LOOPS_KV: tl.constexpr,  #
    TENSORS_PRELOAD: tl.constexpr,
    USE_TMA: tl.constexpr,
    mask_fns,
    mask_args,
    q_lims_continious: tl.constexpr,
    k_lims_continious: tl.constexpr,
    ROWS_GUARANTEED_SAFE: tl.constexpr,
):
    batch = tl.program_id(0)
    idx = tl.program_id(1)

    num_heads = NUM_KV_HEADS * GROUP_SIZE
    dq_total_tiles = num_heads * DQ_TILES_NUM

    if idx < dq_total_tiles:
        head = idx // DQ_TILES_NUM
        tile_id = idx % DQ_TILES_NUM
        dq_worker = True
    else:
        idx -= dq_total_tiles
        head = idx // DK_TILES_NUM
        tile_id = idx % DK_TILES_NUM
        dq_worker = False

    if L is not None:
        seq_len = tl.load(L + batch * lens_stride)
        seq_len = min(seq_len, T)
    else:
        seq_len = T

    if dq_worker:
        _chill_attn_bwd_dq_inner(
            Q, K, V, DELTA, LSE,
            DO, DQ,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dqb, stride_dqh, stride_dqt, stride_dqk,
            batch=batch,
            head=head,
            tile_id=tile_id,
            seq_len=seq_len,
            T=T,
            HEAD_DIM=HEAD_DIM,
            DTYPE=DTYPE,
            USE_TMA=USE_TMA,
            INPUT_PRECISION=INPUT_PRECISION,
            SM_SCALE=SM_SCALE,
            PRESCALE_QK=PRESCALE_QK,
            DQ_Q_BLOCK_DIVISIBLE=DQ_Q_BLOCK_DIVISIBLE,
            DQ_K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE,
            HAS_FULL_BLOCKS=HAS_FULL_BLOCKS_DQ,
            RCP_LN2=RCP_LN2,
            GROUP_SIZE=GROUP_SIZE,
            TILE_DQ_Q_SIZE=TILE_DQ_Q_SIZE,
            TILE_DQ_K_SIZE=TILE_DQ_K_SIZE,
            PIPELINING=PIPELINING,
            SPLIT_LOOPS=SPLIT_LOOPS,
            TENSORS_PRELOAD=TENSORS_PRELOAD,
            ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
            mask_fns=mask_fns,
            mask_args=mask_args,
            k_lims_continious=k_lims_continious,
        )
    else:
        _chill_attn_bwd_dkdv_inner(
            Q, K, V, DELTA, LSE, DO, DK, DV,
            stride_qb, stride_qh, stride_qt, stride_qk,
            stride_kb, stride_kh, stride_kt, stride_kk,
            stride_vb, stride_vh, stride_vt, stride_vk,
            stride_deltab, stride_deltah, stride_deltat,
            stride_mb, stride_mh, stride_mt,
            stride_dob, stride_doh, stride_dot, stride_dok,
            stride_dkb, stride_dkh, stride_dkt, stride_dkk,
            stride_dvb, stride_dvh, stride_dvt, stride_dvk,
            batch=batch,
            head=head,
            tile_id=tile_id,
            seq_len=seq_len,
            T=T,
            HEAD_DIM=HEAD_DIM,
            DTYPE=DTYPE,
            USE_TMA=USE_TMA,
            INPUT_PRECISION=INPUT_PRECISION,
            SM_SCALE=SM_SCALE,
            PRESCALE_QK=PRESCALE_QK,
            DK_Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
            DK_K_BLOCK_DIVISIBLE=DK_K_BLOCK_DIVISIBLE,
            HAS_FULL_BLOCKS=HAS_FULL_BLOCKS_DK,
            RCP_LN2=RCP_LN2,
            GROUP_SIZE=GROUP_SIZE,
            TILE_DK_Q_SIZE=TILE_DK_Q_SIZE,
            TILE_DK_K_SIZE=TILE_DK_K_SIZE,
            PIPELINING=PIPELINING,
            SPLIT_LOOPS_KV=SPLIT_LOOPS_KV,
            TENSORS_PRELOAD=TENSORS_PRELOAD,
            ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
            NUM_KV_HEADS=NUM_KV_HEADS,
            mask_fns=mask_fns,
            mask_args=mask_args,
            q_lims_continious=q_lims_continious,
        )
# fmt: on


def attention_backward_adapter_op_setup_context(ctx, inputs, output):
    O, LSE = output
    (
        q,
        k,
        v,
        lens,
        mask_args,
        sm_scale,
        return_lse,
        prescale_qk,
        precision,
        autotune,
    ) = inputs
    ctx.save_for_backward(
        q,
        k,
        v,
        O,
        LSE,
        lens,
    )
    ctx.mask_args = mask_args
    ctx.sm_scale = sm_scale
    ctx.prescale_qk = prescale_qk
    ctx.precision = precision
    ctx.autotune = autotune
