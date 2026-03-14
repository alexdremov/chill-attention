import logging
import math

import triton
import triton.language as tl

from .utils import (
    _get_min_max_tiles,
)

logger = logging.getLogger(__name__)


# fmt: off
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
            HEAD_DIM=HEAD_DIM,
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
            tl.store(dk_desc, dk.to(k.dtype))
        else:
            tl.store(dk_desc, dk.to(k.dtype), boundary_check=(0,))


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
            tl.store(dv_desc, dv.to(v.dtype))
        else:
            tl.store(dv_desc, dv.to(v.dtype), boundary_check=(0,))


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
    HEAD_DIM: tl.constexpr,
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

    DO_SPLIT: tl.constexpr = SPLIT_LOOPS_KV and HAS_FULL_BLOCKS
    if DO_SPLIT and (kv_token_idx + TILE_K_SIZE <= seq_len):
        fn_q_full_range: tl.constexpr = mask_fns[5]
        full_q_start, full_q_end = fn_q_full_range(kv_token_idx, TILE_K_SIZE, seq_len, mask_args)

        full_q_start_tile = tl.cdiv(full_q_start, TILE_Q_SIZE)
        full_q_end_tile = full_q_end // TILE_Q_SIZE

    if DO_SPLIT:
        full_q_start_tile = tl.maximum(q_start_tile_idx, full_q_start_tile)
        full_q_end_tile = tl.minimum(q_end_tile_idx, full_q_end_tile)
        if full_q_start_tile >= full_q_end_tile:
            full_q_start_tile = q_end_tile_idx
            full_q_end_tile = q_end_tile_idx

    # --- LOOP 1: Prefix Partial Blocks ---
    for q_tile_idx in tl.range(q_start_tile_idx, full_q_start_tile, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        if USE_TMA:
            q = q_desc.load([q_head, q_token_idx, 0]).reshape((TILE_Q_SIZE, HEAD_DIM))
            if TENSORS_PRELOAD:
                m = lse_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
                do = do_desc.load([q_head, q_token_idx, 0]).reshape((TILE_Q_SIZE, HEAD_DIM))
                Di = delta_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
        else:
            q = tl.load(tl.advance(q_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
            if TENSORS_PRELOAD:
                m = tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                do = tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                Di = tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

        qkT = tl.dot(k, q.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qkT *= softmax_scale_ln2

        if not TENSORS_PRELOAD:
            if USE_TMA:
                m = lse_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
            else:
                m = tl.load(
                    tl.advance(lse_desc, (q_token_idx,)),
                    boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ()
                )

        m_safe = m
        if not ROWS_GUARANTEED_SAFE:
            m_safe = tl.where(m == float("-inf"), 0.0, m)

        pT = tl.math.exp2(qkT - m_safe[None, :])
        q_tile_indices = q_token_idx + tile_q_arange
        mask = kv_lens_mask & (q_tile_indices[None, :] < seq_len)
        if DO_SPLIT or not HAS_FULL_BLOCKS or not is_full_block(q_token_idx, kv_token_idx, TILE_Q_SIZE, TILE_K_SIZE, seq_len, mask_args):
            mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args).T
        pT = tl.where(mask, pT, 0.0)

        if not TENSORS_PRELOAD:
            if USE_TMA:
                do = do_desc.load([q_head, q_token_idx, 0]).reshape((TILE_Q_SIZE, HEAD_DIM))
            else:
                do = tl.load(
                    tl.advance(do_desc, (q_token_idx, 0)),
                    boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ()
                )
        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dpT = tl.dot(v.to(do.dtype), do.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not TENSORS_PRELOAD:
            if USE_TMA:
                Di = delta_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE, ))
            else:
                Di = tl.load(
                    tl.advance(delta_desc, (q_token_idx,)),
                    boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ()
                )
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    if DO_SPLIT:
        # --- LOOP 2: Full Blocks (Hot Loop) ---
        for q_tile_idx in tl.range(full_q_start_tile, full_q_end_tile, num_stages=PIPELINING):
            q_token_idx = q_tile_idx * TILE_Q_SIZE
            if USE_TMA:
                q = q_desc.load([q_head, q_token_idx, 0]).reshape((TILE_Q_SIZE, HEAD_DIM))
                if TENSORS_PRELOAD:
                    m = lse_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
                    do = do_desc.load([q_head, q_token_idx, 0]).reshape((TILE_Q_SIZE, HEAD_DIM))
                    Di = delta_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
            else:
                q = tl.load(tl.advance(q_desc, (q_token_idx, 0)))
                if TENSORS_PRELOAD:
                    m = tl.load(tl.advance(lse_desc, (q_token_idx,)))
                    do = tl.load(tl.advance(do_desc, (q_token_idx, 0)))
                    Di = tl.load(tl.advance(delta_desc, (q_token_idx,)))

            qkT = tl.dot(k, q.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            if not PRESCALE_QK: qkT *= softmax_scale_ln2

            if not TENSORS_PRELOAD:
                if USE_TMA:
                    m = lse_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
                else:
                    m = tl.load(tl.advance(lse_desc, (q_token_idx,)))

            m_safe = m
            if not ROWS_GUARANTEED_SAFE:
                m_safe = tl.where(m == float("-inf"), 0.0, m)

            pT = tl.math.exp2(qkT - m_safe[None, :])
            # NO MASKING HERE
            if not TENSORS_PRELOAD:
                if USE_TMA:
                    do = do_desc.load([q_head, q_token_idx, 0]).reshape((TILE_Q_SIZE, HEAD_DIM))
                else:
                    do = tl.load(tl.advance(do_desc, (q_token_idx, 0)))
            dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            dpT = tl.dot(v.to(do.dtype), do.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            if not TENSORS_PRELOAD:
                if USE_TMA:
                    Di = delta_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
                else:
                    Di = tl.load(tl.advance(delta_desc, (q_token_idx,)))
            dsT = pT * (dpT - Di[None, :])
            dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

        # --- LOOP 3: Suffix Partial Blocks ---
        for q_tile_idx in tl.range(full_q_end_tile, q_end_tile_idx, num_stages=PIPELINING):
            q_token_idx = q_tile_idx * TILE_Q_SIZE
            if USE_TMA:
                q = q_desc.load([q_head, q_token_idx, 0]).reshape((TILE_Q_SIZE, HEAD_DIM))
                if TENSORS_PRELOAD:
                    m = lse_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
                    do = do_desc.load([q_head, q_token_idx, 0]).reshape((TILE_Q_SIZE, HEAD_DIM))
                    Di = delta_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
            else:
                q = tl.load(tl.advance(q_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                if TENSORS_PRELOAD:
                    m = tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                    do = tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                    Di = tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

            qkT = tl.dot(k, q.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            if not PRESCALE_QK: qkT *= softmax_scale_ln2

            if not TENSORS_PRELOAD:
                if USE_TMA:
                    m = lse_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
                else:
                    m = tl.load(
                        tl.advance(lse_desc, (q_token_idx,)),
                        boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ()
                    )

            m_safe = m
            if not ROWS_GUARANTEED_SAFE:
                m_safe = tl.where(m == float("-inf"), 0.0, m)

            pT = tl.math.exp2(qkT - m_safe[None, :])
            q_tile_indices = q_token_idx + tile_q_arange
            mask = kv_lens_mask & (q_tile_indices[None, :] < seq_len)
            if DO_SPLIT or not HAS_FULL_BLOCKS or not is_full_block(q_token_idx, kv_token_idx, TILE_Q_SIZE, TILE_K_SIZE, seq_len, mask_args):
                mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args).T
            pT = tl.where(mask, pT, 0.0)

            if not TENSORS_PRELOAD:
                if USE_TMA:
                    do = do_desc.load([q_head, q_token_idx, 0]).reshape((TILE_Q_SIZE, HEAD_DIM))
                else:
                    do = tl.load(
                        tl.advance(do_desc, (q_token_idx, 0)),
                        boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ()
                    )
            dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            dpT = tl.dot(v.to(do.dtype), do.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            if not TENSORS_PRELOAD:
                if USE_TMA:
                    Di = delta_desc.load([q_head, q_token_idx]).reshape((TILE_Q_SIZE,))
                else:
                    Di = tl.load(
                        tl.advance(delta_desc, (q_token_idx,)),
                        boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ()
                    )
            dsT = pT * (dpT - Di[None, :])
            dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
    return dk, dv
