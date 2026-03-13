import logging
import math

import triton
import triton.language as tl

from .utils import (
    _get_min_max_tiles,
)

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

@triton.jit()
def _chill_attn_bwd_dq_inner(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: tl.constexpr,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: tl.constexpr,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: tl.constexpr,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: tl.constexpr,
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: tl.constexpr,
    batch: int,
    head: int,
    tile_id: int,
    seq_len: tl.tensor,
    T: int,  #
    HEAD_DIM: tl.constexpr,  #
    DTYPE: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    DQ_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DQ_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    HAS_FULL_BLOCKS: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    GROUP_SIZE: tl.constexpr, #
    TILE_DQ_Q_SIZE: tl.constexpr,  #
    TILE_DQ_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    SPLIT_LOOPS: tl.constexpr,  #
    TENSORS_PRELOAD: tl.constexpr,
    USE_TMA: tl.constexpr,
    ROWS_GUARANTEED_SAFE: tl.constexpr,
    mask_fns,
    mask_args,
    k_lims_continious: tl.constexpr,  #
):
    q_tile_idx = tile_id
    q_token_idx = q_tile_idx * TILE_DQ_Q_SIZE
    kv_head = head // GROUP_SIZE

    if q_token_idx >= seq_len:
        return

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    if USE_TMA:
        q_desc = tl.make_tensor_descriptor(
            base=Q + qbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_qt, stride_qk),
            block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        )
    else:
        q_desc = tl.make_block_ptr(
            base=Q + qbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_qt, stride_qk),
            offsets=(q_token_idx, 0),
            block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
            order=(1, 0),
        )

    lsebatch_head_offset = batch * stride_mb + head * stride_mh
    if USE_TMA:
        lse_desc = tl.make_tensor_descriptor(
            base=LSE + lsebatch_head_offset,
            shape=(T,),
            strides=(stride_mt,),
            block_shape=(TILE_DQ_Q_SIZE,),
        )
    else:
        lse_desc = tl.make_block_ptr(
            base=LSE + lsebatch_head_offset,
            shape=(T,),
            strides=(stride_mt,),
            offsets=(q_token_idx,),
            block_shape=(TILE_DQ_Q_SIZE,),
            order=(0,),
        )

    delta_offset = batch * stride_deltab + head * stride_deltah
    if USE_TMA:
        delta_desc = tl.make_tensor_descriptor(
            base=DELTA + delta_offset,
            shape=(T,),
            strides=(stride_deltat,),
            block_shape=(TILE_DQ_Q_SIZE,),
        )
    else:
        delta_desc = tl.make_block_ptr(
            base=DELTA + delta_offset,
            shape=(T,),
            strides=(stride_deltat,),
            offsets=(q_token_idx,),
            block_shape=(TILE_DQ_Q_SIZE,),
            order=(0,),
        )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    if USE_TMA:
        do_desc = tl.make_tensor_descriptor(
            base=DO + dobatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dot, stride_dok),
            block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        )
    else:
        do_desc = tl.make_block_ptr(
            base=DO + dobatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dot, stride_dok),
            offsets=(q_token_idx, 0),
            block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
            order=(1, 0),
        )

    if USE_TMA:
        q = q_desc.load([q_token_idx, 0])
        m = lse_desc.load([q_token_idx])[:, None]
        di = delta_desc.load([q_token_idx])
        do = do_desc.load([q_token_idx, 0])
    else:
        if DQ_Q_BLOCK_DIVISIBLE:
            q = tl.load(q_desc)
            m = tl.load(lse_desc)[:, None]
            di = tl.load(delta_desc)
            do = tl.load(do_desc)
        else:
            q = tl.load(q_desc, boundary_check=(0,))
            m = tl.load(lse_desc, boundary_check=(0,))[:, None]
            di = tl.load(delta_desc, boundary_check=(0,))
            do = tl.load(do_desc, boundary_check=(0,))

    kbatch_head_offset = batch * stride_kb + kv_head * stride_kh
    if USE_TMA:
        k_desc = tl.make_tensor_descriptor(
            base=K + kbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_kt, stride_kk),
            block_shape=(TILE_DQ_K_SIZE, HEAD_DIM),
        )
    else:
        k_desc = tl.make_block_ptr(
            base=K + kbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_kt, stride_kk),
            block_shape=(TILE_DQ_K_SIZE, HEAD_DIM),
            offsets=(0, 0),
            order=(1, 0),
        )

    vbatch_head_offset = batch * stride_vb + kv_head * stride_vh
    if USE_TMA:
        v_desc = tl.make_tensor_descriptor(
            base=V + vbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_vt, stride_vk),
            block_shape=(TILE_DQ_K_SIZE, HEAD_DIM),
        )
    else:
        v_desc = tl.make_block_ptr(
            base=V + vbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_vt, stride_vk),
            block_shape=(TILE_DQ_K_SIZE, HEAD_DIM),
            offsets=(0, 0),
            order=(1, 0),
        )

    dq = tl.zeros([TILE_DQ_Q_SIZE, HEAD_DIM], dtype=tl.float32)
    dq = _chill_attn_bwd_dq(
        dq, q, m, di, do,
        k_desc, v_desc,
        seq_len=seq_len,
        q_token_idx=q_token_idx,
        TILE_Q_SIZE=TILE_DQ_Q_SIZE,
        TILE_K_SIZE=TILE_DQ_K_SIZE,
        INPUT_PRECISION=INPUT_PRECISION,
        PIPELINING=PIPELINING,
        K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE,
        HAS_FULL_BLOCKS=HAS_FULL_BLOCKS,
        SPLIT_LOOPS=SPLIT_LOOPS,
        RCP_LN2=RCP_LN2,
        SM_SCALE=SM_SCALE,
        USE_TMA=USE_TMA,
        PRESCALE_QK=PRESCALE_QK,
        TENSORS_PRELOAD=TENSORS_PRELOAD,
        ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
        mask_fns=mask_fns,
        mask_args=mask_args,
        k_lims_continious=k_lims_continious,
    )

    dqbatch_head_offset = batch * stride_dqb + head * stride_dqh
    if USE_TMA:
        dq_desc = tl.make_tensor_descriptor(
            base=DQ + dqbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dqt, stride_dqk),
            block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        )

        dq_desc.store([q_token_idx, 0], dq.to(q.dtype))
    else:
        dq_tile_ptr = tl.make_block_ptr(
            base=DQ + dqbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dqt, stride_dqk),
            offsets=(q_token_idx, 0),
            block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
            order=(1, 0),
        )
        if DQ_Q_BLOCK_DIVISIBLE:
            tl.store(dq_tile_ptr, dq.to(dq_tile_ptr.type.element_ty))
        else:
            tl.store(dq_tile_ptr, dq.to(dq_tile_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def _chill_attn_bwd_dkdv_inner(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor,
    DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DK: tl.tensor, DV: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: tl.constexpr,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: tl.constexpr,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: tl.constexpr,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: tl.constexpr,
    stride_dkb: int, stride_dkh: int, stride_dkt: int, stride_dkk: tl.constexpr,
    stride_dvb: int, stride_dvh: int, stride_dvt: int, stride_dvk: tl.constexpr,
    batch: int,
    head: int,
    tile_id: int,
    seq_len: tl.tensor,
    T: int,  #
    HEAD_DIM: tl.constexpr,  #
    DTYPE: tl.constexpr,  #
    USE_TMA: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    HAS_FULL_BLOCKS: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    GROUP_SIZE: tl.constexpr, #
    TILE_DK_Q_SIZE: tl.constexpr,  #
    TILE_DK_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    SPLIT_LOOPS_KV: tl.constexpr,  #
    TENSORS_PRELOAD: tl.constexpr,
    ROWS_GUARANTEED_SAFE: tl.constexpr,  #
    NUM_KV_HEADS: tl.constexpr,  #
    mask_fns,
    mask_args,
    q_lims_continious: tl.constexpr,  #
):
    kv_tile_idx = tile_id
    kv_token_idx = kv_tile_idx * TILE_DK_K_SIZE
    kv_head = head

    if kv_token_idx >= seq_len:
        return

    kbatch_head_offset = batch * stride_kb + kv_head * stride_kh
    if USE_TMA:
        k_desc = tl.make_tensor_descriptor(
            base=K + kbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_kt, stride_kk),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        )
    else:
        k_desc = tl.make_block_ptr(
            base=K + kbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_kt, stride_kk),
            offsets=(kv_token_idx, 0),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
            order=(1, 0),
        )

    vbatch_head_offset = batch * stride_vb + kv_head * stride_vh
    if USE_TMA:
        v_desc = tl.make_tensor_descriptor(
            base=V + vbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_vt, stride_vk),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        )
    else:
        v_desc = tl.make_block_ptr(
            base=V + vbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_vt, stride_vk),
            offsets=(kv_token_idx, 0),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
            order=(1, 0),
        )

    if USE_TMA:
        k = k_desc.load([kv_token_idx, 0])
        v = v_desc.load([kv_token_idx, 0])
    else:
        if DK_K_BLOCK_DIVISIBLE:
            k = tl.load(
                    k_desc,
                )
            v = tl.load(
                    v_desc,
                )
        else:
            k = tl.load(
                    k_desc,
                    boundary_check=(0,),
                )
            v = tl.load(
                    v_desc,
                    boundary_check=(0,),
                )

    if PRESCALE_QK:
        k = (k * (SM_SCALE * RCP_LN2)).to(k.dtype)

    dv = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)

    num_heads = NUM_KV_HEADS * GROUP_SIZE
    q_batch_offset = batch * stride_qb
    dobatch_offset = batch * stride_dob
    lsebatch_offset = batch * stride_mb
    deltabatch_offset = batch * stride_deltab

    if USE_TMA:
        q_desc_3d = tl.make_tensor_descriptor(
            base=Q + q_batch_offset,
            shape=(num_heads, T, HEAD_DIM),
            strides=(stride_qh, stride_qt, stride_qk),
            block_shape=(1, TILE_DK_Q_SIZE, HEAD_DIM),
        )
        do_desc_3d = tl.make_tensor_descriptor(
            base=DO + dobatch_offset,
            shape=(num_heads, T, HEAD_DIM),
            strides=(stride_doh, stride_dot, stride_dok),
            block_shape=(1, TILE_DK_Q_SIZE, HEAD_DIM),
        )
        lse_desc_2d = tl.make_tensor_descriptor(
            base=LSE + lsebatch_offset,
            shape=(num_heads, T),
            strides=(stride_mh, stride_mt),
            block_shape=(1, TILE_DK_Q_SIZE),
        )
        delta_desc_2d = tl.make_tensor_descriptor(
            base=DELTA + deltabatch_offset,
            shape=(num_heads, T),
            strides=(stride_deltah, stride_deltat),
            block_shape=(1, TILE_DK_Q_SIZE),
        )

    for g in tl.range(0, GROUP_SIZE):
        q_head = kv_head * GROUP_SIZE + g
        qbatch_head_offset = q_batch_offset + q_head * stride_qh
        if USE_TMA:
            q_desc = q_desc_3d
            do_desc = do_desc_3d
            lse_desc = lse_desc_2d
            delta_desc = delta_desc_2d
        else:
            q_desc = tl.make_block_ptr(
                base=Q + qbatch_head_offset,
                shape=(T, HEAD_DIM),
                strides=(stride_qt, stride_qk),
                offsets=(0, 0),
                block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
                order=(1, 0),
            )

            dobatch_head_offset = dobatch_offset + q_head * stride_doh
            do_desc = tl.make_block_ptr(
                base=DO + dobatch_head_offset,
                shape=(T, HEAD_DIM),
                strides=(stride_dot, stride_dok),
                offsets=(0, 0),
                block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
                order=(1, 0),
            )

            lsebatch_head_offset = lsebatch_offset + q_head * stride_mh
            lse_desc = tl.make_block_ptr(
                base=LSE + lsebatch_head_offset,
                shape=(T,),
                strides=(stride_mt,),
                offsets=(0,),
                block_shape=(TILE_DK_Q_SIZE,),
                order=(0,),
            )

            deltabatch_head_offset = deltabatch_offset + q_head * stride_deltah
            delta_desc = tl.make_block_ptr(
                base=DELTA + deltabatch_head_offset,
                shape=(T,),
                strides=(stride_deltat,),
                offsets=(0,),
                block_shape=(TILE_DK_Q_SIZE,),
                order=(0,),
            )

        dk, dv = _chill_attn_bwd_dkdv(
            dk, dv,
            q_desc, do_desc, lse_desc, delta_desc,
            k, v,
            seq_len=seq_len,
            kv_token_idx=kv_token_idx,
            q_head=q_head,
            TILE_Q_SIZE=TILE_DK_Q_SIZE,
            TILE_K_SIZE=TILE_DK_K_SIZE,
            INPUT_PRECISION=INPUT_PRECISION,
            PIPELINING=PIPELINING,
            Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
            HAS_FULL_BLOCKS=HAS_FULL_BLOCKS,
            RCP_LN2=RCP_LN2,
            SM_SCALE=SM_SCALE,
            PRESCALE_QK=PRESCALE_QK,
            SPLIT_LOOPS_KV=SPLIT_LOOPS_KV,
            TENSORS_PRELOAD=TENSORS_PRELOAD,
            ROWS_GUARANTEED_SAFE=ROWS_GUARANTEED_SAFE,
            USE_TMA=USE_TMA,
            mask_fns=mask_fns,
            mask_args=mask_args,
            q_lims_continious=q_lims_continious,
        )

    dk *= SM_SCALE

    dkbatch_head_offset = batch * stride_dkb + kv_head * stride_dkh
    if USE_TMA:
        dk_desc = tl.make_tensor_descriptor(
            base=DK + dkbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dkt, stride_dkk),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        )
        dk_desc.store([kv_token_idx, 0], dk.to(k.dtype))
    else:
        dk_desc = tl.make_block_ptr(
            base=DK + dkbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dkt, stride_dkk),
            offsets=(kv_token_idx, 0),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
            order=(1, 0),
        )
        if DK_K_BLOCK_DIVISIBLE:
            tl.store(dk_desc, dk.to(dk_desc.type.element_ty))
        else:
            tl.store(dk_desc, dk.to(dk_desc.type.element_ty), boundary_check=(0,))


    dvbatch_head_offset = batch * stride_dvb + kv_head * stride_dvh
    if USE_TMA:
        dv_desc = tl.make_tensor_descriptor(
            base=DV + dvbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dvt, stride_dvk),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        )
        dv_desc.store([kv_token_idx, 0], dv.to(v.dtype))
    else:
        dv_desc = tl.make_block_ptr(
            base=DV + dvbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_dvt, stride_dvk),
            offsets=(kv_token_idx, 0),
            block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
            order=(1, 0),
        )
        if DK_K_BLOCK_DIVISIBLE:
            tl.store(dv_desc, dv.to(dv_desc.type.element_ty))
        else:
            tl.store(dv_desc, dv.to(dv_desc.type.element_ty), boundary_check=(0,))


@triton.jit
def _chill_attn_bwd_dq(
    dq: tl.tensor, q: tl.tensor, m: tl.tensor,
    di: tl.tensor, do: tl.tensor,
    k_desc: tl.tensor, v_desc: tl.tensor,
    seq_len: tl.tensor,
    q_token_idx: int,
    TILE_Q_SIZE: tl.constexpr,
    TILE_K_SIZE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    PIPELINING: tl.constexpr,
    K_BLOCK_DIVISIBLE: tl.constexpr,
    HAS_FULL_BLOCKS: tl.constexpr,  #
    RCP_LN2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    PRESCALE_QK: tl.constexpr,
    SPLIT_LOOPS: tl.constexpr,
    TENSORS_PRELOAD: tl.constexpr,
    USE_TMA: tl.constexpr,
    ROWS_GUARANTEED_SAFE: tl.constexpr,
    mask_fns,
    mask_args,
    k_lims_continious: tl.constexpr,
):
    fn_mask: tl.constexpr = mask_fns[0]
    fn_k_lims_for_q: tl.constexpr = mask_fns[1]
    is_full_block: tl.constexpr = mask_fns[3]

    kv_start_tile_idx, kv_end_tile_idx = _get_min_max_tiles(
        q_token_idx,
        seq_len=seq_len,
        fn_lims=fn_k_lims_for_q,
        mask_args=mask_args,
        continious=k_lims_continious,
        TILE_SIZE_IN=TILE_Q_SIZE,
        TILE_SIZE_OUT=TILE_K_SIZE,
    )

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)

    q_len_mask = q_tile_indices[:, None] < seq_len
    tile_k_arange = tl.arange(0, TILE_K_SIZE)

    softmax_scale_ln2: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q.dtype)
    if PRESCALE_QK:
        q = (q * softmax_scale_ln2).to(q.dtype)

    # Determine full block range for loop splitting
    full_kv_start_tile = kv_end_tile_idx
    full_kv_end_tile = kv_start_tile_idx

    if SPLIT_LOOPS and HAS_FULL_BLOCKS and (q_token_idx + TILE_Q_SIZE <= seq_len):
        fn_k_full_range: tl.constexpr = mask_fns[4]
        full_k_start, full_k_end = fn_k_full_range(q_token_idx, TILE_Q_SIZE, seq_len, mask_args)
        full_kv_start_tile = tl.cdiv(full_k_start, TILE_K_SIZE)
        full_kv_end_tile = full_k_end // TILE_K_SIZE

    full_kv_start_tile = tl.maximum(kv_start_tile_idx, full_kv_start_tile)
    full_kv_end_tile = tl.minimum(kv_end_tile_idx, full_kv_end_tile)
    if full_kv_start_tile >= full_kv_end_tile:
        full_kv_start_tile = kv_end_tile_idx
        full_kv_end_tile = kv_end_tile_idx

    m_safe = m
    if not ROWS_GUARANTEED_SAFE:
        m_safe = tl.where(m == float("-inf"), 0.0, m)

    # --- LOOP 1: Prefix Partial Blocks ---
    for kv_tile_idx in tl.range(
        kv_start_tile_idx, full_kv_start_tile, num_stages=PIPELINING
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        if USE_TMA:
            k = k_desc.load([kv_token_idx, 0])
            if TENSORS_PRELOAD: v = v_desc.load([kv_token_idx, 0])
        else:
            k = tl.load(tl.advance(k_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())
            if TENSORS_PRELOAD: v = tl.load(tl.advance(v_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())

        qk = tl.dot(q, k.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qk = qk * softmax_scale_ln2
        p = tl.math.exp2(qk - m_safe)

        kv_indices = kv_token_idx + tile_k_arange
        mask = q_len_mask & (kv_indices[None, :] < seq_len)
        mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args)

        if not TENSORS_PRELOAD:
            v = v_desc.load([kv_token_idx, 0]) if USE_TMA else tl.load(tl.advance(v_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())

        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do.to(v.dtype), v.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        ds = p * (dp - di[:, None])
        dq = tl.dot(ds.to(k.dtype), k, dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    # --- LOOP 2: Full Blocks (Hot Loop) ---
    for kv_tile_idx in tl.range(
        full_kv_start_tile, full_kv_end_tile, num_stages=PIPELINING
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        if USE_TMA:
            k = k_desc.load([kv_token_idx, 0])
            if TENSORS_PRELOAD: v = v_desc.load([kv_token_idx, 0])
        else:
            k = tl.load(tl.advance(k_desc, (kv_token_idx, 0)))
            if TENSORS_PRELOAD: v = tl.load(tl.advance(v_desc, (kv_token_idx, 0)))

        qk = tl.dot(q, k.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qk = qk * softmax_scale_ln2
        p = tl.math.exp2(qk - m_safe)
        # NO MASKING HERE
        if not TENSORS_PRELOAD:
            v = v_desc.load([kv_token_idx, 0]) if USE_TMA else tl.load(tl.advance(v_desc, (kv_token_idx, 0)))

        dp = tl.dot(do.to(v.dtype), v.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        ds = p * (dp - di[:, None])
        dq = tl.dot(ds.to(k.dtype), k, dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    # --- LOOP 3: Suffix Partial Blocks ---
    for kv_tile_idx in tl.range(
        full_kv_end_tile, kv_end_tile_idx, num_stages=PIPELINING
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        if USE_TMA:
            k = k_desc.load([kv_token_idx, 0])
            if TENSORS_PRELOAD: v = v_desc.load([kv_token_idx, 0])
        else:
            k = tl.load(tl.advance(k_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())
            if TENSORS_PRELOAD: v = tl.load(tl.advance(v_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())

        qk = tl.dot(q, k.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qk = qk * softmax_scale_ln2
        p = tl.math.exp2(qk - m_safe)

        kv_indices = kv_token_idx + tile_k_arange
        mask = q_len_mask & (kv_indices[None, :] < seq_len)
        mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args)

        if not TENSORS_PRELOAD:
            v = v_desc.load([kv_token_idx, 0]) if USE_TMA else tl.load(tl.advance(v_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())

        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do.to(v.dtype), v.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        ds = p * (dp - di[:, None])
        dq = tl.dot(ds.to(k.dtype), k, dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    dq *= SM_SCALE
    return dq


@triton.jit
def _chill_attn_bwd_dkdv(
    dk: tl.tensor, dv: tl.tensor,
    q_desc: tl.tensor, do_desc: tl.tensor,
    lse_desc: tl.tensor, delta_desc: tl.tensor,
    k: tl.tensor, v: tl.tensor,
    seq_len: tl.tensor,
    kv_token_idx: int,
    q_head: int,
    TILE_Q_SIZE: tl.constexpr,
    TILE_K_SIZE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    PIPELINING: tl.constexpr,
    Q_BLOCK_DIVISIBLE: tl.constexpr,
    HAS_FULL_BLOCKS: tl.constexpr,  #
    RCP_LN2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    PRESCALE_QK: tl.constexpr,
    SPLIT_LOOPS_KV: tl.constexpr,
    TENSORS_PRELOAD: tl.constexpr,
    USE_TMA: tl.constexpr,
    mask_fns,
    mask_args,
    q_lims_continious: tl.constexpr,
    ROWS_GUARANTEED_SAFE: tl.constexpr,
):
    fn_mask: tl.constexpr = mask_fns[0]
    fn_q_range_for_k: tl.constexpr = mask_fns[2]
    is_full_block: tl.constexpr = mask_fns[3]

    q_start_tile_idx, q_end_tile_idx = _get_min_max_tiles(
        kv_token_idx,
        seq_len=seq_len,
        fn_lims=fn_q_range_for_k,
        mask_args=mask_args,
        continious=q_lims_continious,
        TILE_SIZE_IN=TILE_K_SIZE,
        TILE_SIZE_OUT=TILE_Q_SIZE,
    )

    kv_indices = kv_token_idx + tl.arange(0, TILE_K_SIZE)
    tile_q_arange = tl.arange(0, TILE_Q_SIZE)
    kv_lens_mask = (
        kv_indices[:, None] < seq_len
    )

    softmax_scale_ln2: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, k.dtype)

    # Determine full block range for loop splitting
    full_q_start_tile = q_end_tile_idx
    full_q_end_tile = q_start_tile_idx

    if SPLIT_LOOPS_KV and HAS_FULL_BLOCKS and (kv_token_idx + TILE_K_SIZE <= seq_len):
        fn_q_full_range: tl.constexpr = mask_fns[5]
        full_q_start, full_q_end = fn_q_full_range(kv_token_idx, TILE_K_SIZE, seq_len, mask_args)

        full_q_start_tile = tl.cdiv(full_q_start, TILE_Q_SIZE)
        full_q_end_tile = full_q_end // TILE_Q_SIZE

    full_q_start_tile = tl.maximum(q_start_tile_idx, full_q_start_tile)
    full_q_end_tile = tl.minimum(q_end_tile_idx, full_q_end_tile)
    if full_q_start_tile >= full_q_end_tile:
        full_q_start_tile = q_end_tile_idx
        full_q_end_tile = q_end_tile_idx

    # --- LOOP 1: Prefix Partial Blocks ---
    for q_tile_idx in tl.range(q_start_tile_idx, full_q_start_tile, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        if USE_TMA:
            q = q_desc.load([q_head, q_token_idx, 0])[0, :, :]
            if TENSORS_PRELOAD:
                m = lse_desc.load([q_head, q_token_idx])[0, :]
                do = do_desc.load([q_head, q_token_idx, 0])[0, :, :]
                Di = delta_desc.load([q_head, q_token_idx])[0, :]
        else:
            q = tl.load(tl.advance(q_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
            if TENSORS_PRELOAD:
                m = tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                do = tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                Di = tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

        qkT = tl.dot(k, q.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qkT *= softmax_scale_ln2

        if not TENSORS_PRELOAD:
            m = lse_desc.load([q_head, q_token_idx])[0, :] if USE_TMA else tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

        m_safe = m
        if not ROWS_GUARANTEED_SAFE:
            m_safe = tl.where(m == float("-inf"), 0.0, m)

        pT = tl.math.exp2(qkT - m_safe[None, :])
        q_tile_indices = q_token_idx + tile_q_arange
        mask = kv_lens_mask & (q_tile_indices[None, :] < seq_len)
        mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args).T
        pT = tl.where(mask, pT, 0.0)

        if not TENSORS_PRELOAD:
            do = do_desc.load([q_head, q_token_idx, 0])[0, :, :] if USE_TMA else tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dpT = tl.dot(v.to(do.dtype), do.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not TENSORS_PRELOAD:
            Di = delta_desc.load([q_head, q_token_idx])[0, :] if USE_TMA else tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    # --- LOOP 2: Full Blocks (Hot Loop) ---
    for q_tile_idx in tl.range(full_q_start_tile, full_q_end_tile, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        if USE_TMA:
            q = q_desc.load([q_head, q_token_idx, 0])[0, :, :]
            if TENSORS_PRELOAD:
                m = lse_desc.load([q_head, q_token_idx])[0, :]
                do = do_desc.load([q_head, q_token_idx, 0])[0, :, :]
                Di = delta_desc.load([q_head, q_token_idx])[0, :]
        else:
            q = tl.load(tl.advance(q_desc, (q_token_idx, 0)))
            if TENSORS_PRELOAD:
                m = tl.load(tl.advance(lse_desc, (q_token_idx,)))
                do = tl.load(tl.advance(do_desc, (q_token_idx, 0)))
                Di = tl.load(tl.advance(delta_desc, (q_token_idx,)))

        qkT = tl.dot(k, q.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qkT *= softmax_scale_ln2

        if not TENSORS_PRELOAD:
            m = lse_desc.load([q_head, q_token_idx])[0, :] if USE_TMA else tl.load(tl.advance(lse_desc, (q_token_idx,)))

        m_safe = m
        if not ROWS_GUARANTEED_SAFE:
            m_safe = tl.where(m == float("-inf"), 0.0, m)

        pT = tl.math.exp2(qkT - m_safe[None, :])
        # NO MASKING HERE
        if not TENSORS_PRELOAD:
            do = do_desc.load([q_head, q_token_idx, 0])[0, :, :] if USE_TMA else tl.load(tl.advance(do_desc, (q_token_idx, 0)))
        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dpT = tl.dot(v.to(do.dtype), do.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not TENSORS_PRELOAD:
            Di = delta_desc.load([q_head, q_token_idx])[0, :] if USE_TMA else tl.load(tl.advance(delta_desc, (q_token_idx,)))
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    # --- LOOP 3: Suffix Partial Blocks ---
    for q_tile_idx in tl.range(full_q_end_tile, q_end_tile_idx, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        if USE_TMA:
            q = q_desc.load([q_head, q_token_idx, 0])[0, :, :]
            if TENSORS_PRELOAD:
                m = lse_desc.load([q_head, q_token_idx])[0, :]
                do = do_desc.load([q_head, q_token_idx, 0])[0, :, :]
                Di = delta_desc.load([q_head, q_token_idx])[0, :]
        else:
            q = tl.load(tl.advance(q_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
            if TENSORS_PRELOAD:
                m = tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                do = tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                Di = tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

        qkT = tl.dot(k, q.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qkT *= softmax_scale_ln2

        if not TENSORS_PRELOAD:
            m = lse_desc.load([q_head, q_token_idx])[0, :] if USE_TMA else tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

        m_safe = m
        if not ROWS_GUARANTEED_SAFE:
            m_safe = tl.where(m == float("-inf"), 0.0, m)

        pT = tl.math.exp2(qkT - m_safe[None, :])
        q_tile_indices = q_token_idx + tile_q_arange
        mask = kv_lens_mask & (q_tile_indices[None, :] < seq_len)
        mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args).T
        pT = tl.where(mask, pT, 0.0)

        if not TENSORS_PRELOAD:
            do = do_desc.load([q_head, q_token_idx, 0])[0, :, :] if USE_TMA else tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dpT = tl.dot(v.to(do.dtype), do.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not TENSORS_PRELOAD:
            Di = delta_desc.load([q_head, q_token_idx])[0, :] if USE_TMA else tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
    return dk, dv
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
