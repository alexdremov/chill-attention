import logging

import triton
import triton.language as tl

from .utils import (
    _get_min_max_tiles,
)

logger = logging.getLogger(__name__)


# fmt: off
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
            tl.store(dq_tile_ptr, dq.to(q.dtype))
        else:
            tl.store(dq_tile_ptr, dq.to(q.dtype), boundary_check=(0,))

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

    DO_SPLIT: tl.constexpr = SPLIT_LOOPS and HAS_FULL_BLOCKS
    if DO_SPLIT and (q_token_idx + TILE_Q_SIZE <= seq_len):
        fn_k_full_range: tl.constexpr = mask_fns[4]
        full_k_start, full_k_end = fn_k_full_range(q_token_idx, TILE_Q_SIZE, seq_len, mask_args)
        full_kv_start_tile = tl.cdiv(full_k_start, TILE_K_SIZE)
        full_kv_end_tile = full_k_end // TILE_K_SIZE

    if DO_SPLIT:
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
        if DO_SPLIT or not HAS_FULL_BLOCKS or not is_full_block(q_token_idx, kv_token_idx, TILE_Q_SIZE, TILE_K_SIZE, seq_len, mask_args):
            mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args)

        if not TENSORS_PRELOAD:
            if USE_TMA:
                v = v_desc.load([kv_token_idx, 0])
            else:
                v = tl.load(
                    tl.advance(v_desc, (kv_token_idx, 0)),
                    boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ()
                )

        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do.to(v.dtype), v.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        ds = p * (dp - di[:, None])
        dq = tl.dot(ds.to(k.dtype), k, dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    if DO_SPLIT:
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
                if USE_TMA:
                    v = v_desc.load([kv_token_idx, 0])
                else:
                    v = tl.load(
                        tl.advance(v_desc, (kv_token_idx, 0))
                    )

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
                if USE_TMA:
                    v = v_desc.load([kv_token_idx, 0])
                else:
                    v = tl.load(
                        tl.advance(v_desc, (kv_token_idx, 0)),
                        boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ()
                    )

            p = tl.where(mask, p, 0.0)
            dp = tl.dot(do.to(v.dtype), v.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            ds = p * (dp - di[:, None])
            dq = tl.dot(ds.to(k.dtype), k, dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    dq *= SM_SCALE
    return dq
