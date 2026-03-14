import logging
import math

import torch.nn.functional as F
import triton
import triton.language as tl

from .utils import _get_min_max_tiles

logger = logging.getLogger(__name__)

# fmt: off
@triton.heuristics(
    dict(
        Q_BLOCK_DIVISIBLE=lambda args: args['T'] % args['TILE_Q_SIZE'] == 0,
        K_BLOCK_DIVISIBLE=lambda args: args['T'] % args['TILE_K_SIZE'] == 0,
        RCP_LN2=lambda _: math.log2(math.e),
    )
)
@triton.jit
def _chill_attn_fwd(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, L: tl.tensor, #
    LSE: tl.tensor, O: tl.tensor,  #
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: tl.constexpr,  #
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: tl.constexpr,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: tl.constexpr,  #
    stride_mb: int, stride_mh: int, stride_mt: tl.constexpr,  #
    stride_ob: int, stride_oh: int, stride_ot: int, stride_ok: tl.constexpr, #
    lens_stride: int,
    T: int,  #
    TIME_BUCKET: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    DTYPE:  tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    USE_TMA: tl.constexpr,  #
    OUTPUT_LOGSUMEXP: tl.constexpr,  #
    TILE_Q_SIZE: tl.constexpr,  #
    TILE_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    SPLIT_LOOPS: tl.constexpr,  #
    GROUP_SIZE: tl.constexpr, #
    Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    K_BLOCK_DIVISIBLE: tl.constexpr,  #
    HAS_FULL_BLOCKS: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    TENSORS_PRELOAD:  tl.constexpr,  #
    k_lims_continious: tl.constexpr,  #
    ROWS_GUARANTEED_SAFE: tl.constexpr,  #
    mask_fns,  #
    mask_args,  #
):
    fn_mask: tl.constexpr = mask_fns[0]
    fn_k_lims_for_q: tl.constexpr = mask_fns[1]
    is_full_block: tl.constexpr = mask_fns[3]

    batch = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP_SIZE
    q_tile_idx = tl.program_id(2)
    q_token_idx = q_tile_idx * TILE_Q_SIZE

    if L is not None:
        seq_len = tl.load(L + batch * lens_stride)
        seq_len = min(seq_len, T)
    else:
        seq_len = T

    if seq_len <= q_token_idx:
        return

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    if USE_TMA:
        q_desc = tl.make_tensor_descriptor(
            base=Q + qbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_qt, stride_qk),
            block_shape=(TILE_Q_SIZE, HEAD_DIM),
        )
    else:
        q_desc = tl.make_block_ptr(
            base=Q + qbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_qt, stride_qk),
            offsets=(q_token_idx, 0),
            block_shape=(TILE_Q_SIZE, HEAD_DIM),
            order=(1, 0),
        )

    kbatch_head_offset = batch * stride_kb + kv_head * stride_kh
    if USE_TMA:
        k_desc = tl.make_tensor_descriptor(
            base=K + kbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_kt, stride_kk),
            block_shape=(TILE_K_SIZE, HEAD_DIM),
        )
    else:
        k_desc = tl.make_block_ptr(
            base=K + kbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_kt, stride_kk),
            offsets=(0, 0),
            block_shape=(TILE_K_SIZE, HEAD_DIM),
            order=(1, 0),
        )

    vbatch_head_offset = batch * stride_vb + kv_head * stride_vh
    if USE_TMA:
        v_desc = tl.make_tensor_descriptor(
            base=V + vbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_vt, stride_vk),
            block_shape=(TILE_K_SIZE, HEAD_DIM),
        )
    else:
        v_desc = tl.make_block_ptr(
            base=V + vbatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_vt, stride_vk),
            offsets=(0, 0),
            block_shape=(TILE_K_SIZE, HEAD_DIM),
            order=(1, 0),
        )

    m_i = tl.full([TILE_Q_SIZE], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32)
    acc = tl.zeros([TILE_Q_SIZE, HEAD_DIM], dtype=tl.float32)

    kv_start_tile_idx, kv_end_tile_idx = _get_min_max_tiles(
        q_token_idx,
        seq_len=seq_len,
        fn_lims=fn_k_lims_for_q,
        mask_args=mask_args,
        continious=k_lims_continious,
        TILE_SIZE_IN=TILE_Q_SIZE,
        TILE_SIZE_OUT=TILE_K_SIZE
    )

    q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
    k_tile_arange = tl.arange(0, TILE_K_SIZE)

    if USE_TMA:
        # TMA inherently zero-pads out-of-bounds loads
        q_tile = q_desc.load([q_token_idx, 0])
    else:
        if Q_BLOCK_DIVISIBLE:
            q_tile = tl.load(q_desc)
        else:
            q_tile = tl.load(
                q_desc,
                boundary_check=(0,),
            )

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)

    if PRESCALE_QK:
        q_tile = q_tile * softmax_scale

    # Determine full block range for loop splitting
    full_kv_start_tile = kv_end_tile_idx
    full_kv_end_tile = kv_start_tile_idx

    Q_NOT_BOUNDARY = q_token_idx + TILE_Q_SIZE <= seq_len
    GUARANTEED_SAFE = ROWS_GUARANTEED_SAFE

    DO_SPLIT: tl.constexpr = SPLIT_LOOPS and HAS_FULL_BLOCKS
    if DO_SPLIT and Q_NOT_BOUNDARY:
        fn_k_full_range: tl.constexpr = mask_fns[4]

        # Explicitly calculate the range of keys that are unmasked for ALL queries in the tile
        full_k_start, full_k_end = fn_k_full_range(q_token_idx, TILE_Q_SIZE, seq_len, mask_args)

        # Convert token indices to tile indices
        # A tile is FULL only if its entire range [kv_token_idx, kv_token_idx + TILE_K_SIZE)
        # is within the unmasked range [full_k_start, full_k_end).
        full_kv_start_tile = tl.cdiv(full_k_start, TILE_K_SIZE)
        full_kv_end_tile = full_k_end // TILE_K_SIZE

    # Ensure boundaries are valid
    if DO_SPLIT:
        full_kv_start_tile = tl.maximum(kv_start_tile_idx, full_kv_start_tile)
        full_kv_end_tile = tl.minimum(kv_end_tile_idx, full_kv_end_tile)
        if full_kv_start_tile >= full_kv_end_tile:
            full_kv_start_tile = kv_end_tile_idx
            full_kv_end_tile = kv_end_tile_idx

    # --- LOOP 1: Prefix Partial Blocks ---
    for kv_tile_idx in tl.range(
        kv_start_tile_idx, full_kv_start_tile, num_stages=PIPELINING
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        # [Load and Dot logic...]
        if USE_TMA:
            k_tile = k_desc.load([kv_token_idx, 0])
            if TENSORS_PRELOAD: v_tile = v_desc.load([kv_token_idx, 0])
        else:
            k_tile = tl.load(
                tl.advance(k_desc, (kv_token_idx, 0)),
                boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ()
            )
            if TENSORS_PRELOAD:
                v_tile = tl.load(
                    tl.advance(v_desc, (kv_token_idx, 0)),
                    boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ()
                )

        qk = tl.dot(q_tile, k_tile.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        kv_indices = kv_token_idx + k_tile_arange
        if DO_SPLIT or not HAS_FULL_BLOCKS or not is_full_block(q_token_idx, kv_token_idx, TILE_Q_SIZE, TILE_K_SIZE, seq_len, mask_args):
            mask = kv_indices[None, :] < seq_len
            mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args)
        else:
            mask = tl.broadcast_to(kv_indices[None, :] < seq_len, TILE_Q_SIZE, TILE_K_SIZE)

        if not PRESCALE_QK: qk = qk * softmax_scale
        qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        if GUARANTEED_SAFE:
            m_ij_safe = m_ij
        else:
            m_ij_safe = tl.where(m_ij == float("-inf"), tl.cast(0, m_ij.dtype), m_ij)
        alpha = tl.math.exp2(m_i - m_ij_safe)
        p = tl.math.exp2(qk - m_ij_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if not TENSORS_PRELOAD:
            v_tile = v_desc.load([kv_token_idx, 0]) if USE_TMA else tl.load(tl.advance(v_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())
        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        m_i = m_ij

    if DO_SPLIT:
        # --- LOOP 2: Full Blocks (Hot Loop) ---
        for kv_tile_idx in tl.range(
            full_kv_start_tile, full_kv_end_tile, num_stages=PIPELINING
        ):
            kv_token_idx = kv_tile_idx * TILE_K_SIZE
            if USE_TMA:
                k_tile = k_desc.load([kv_token_idx, 0])
                if TENSORS_PRELOAD: v_tile = v_desc.load([kv_token_idx, 0])
            else:
                # We know these are full blocks, but they might still be near seq_len if not divisible
                k_tile = tl.load(tl.advance(k_desc, (kv_token_idx, 0)))
                if TENSORS_PRELOAD: v_tile = tl.load(tl.advance(v_desc, (kv_token_idx, 0)))

            qk = tl.dot(q_tile, k_tile.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            if not PRESCALE_QK: qk = qk * softmax_scale
            # NO MASKING HERE
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            m_ij_safe = tl.where(m_ij == float("-inf"), tl.cast(0, m_ij.dtype), m_ij)
            alpha = tl.math.exp2(m_i - m_ij_safe)
            p = tl.math.exp2(qk - m_ij_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            if not TENSORS_PRELOAD:
                v_tile = v_desc.load([kv_token_idx, 0]) if USE_TMA else tl.load(tl.advance(v_desc, (kv_token_idx, 0)))
            acc = tl.dot(p.to(v_tile.dtype), v_tile, acc, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            m_i = m_ij

        # --- LOOP 3: Suffix Partial Blocks ---
        for kv_tile_idx in tl.range(
            full_kv_end_tile, kv_end_tile_idx, num_stages=PIPELINING
        ):
            kv_token_idx = kv_tile_idx * TILE_K_SIZE
            if USE_TMA:
                k_tile = k_desc.load([kv_token_idx, 0])
                if TENSORS_PRELOAD: v_tile = v_desc.load([kv_token_idx, 0])
            else:
                k_tile = tl.load(tl.advance(k_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())
                if TENSORS_PRELOAD: v_tile = tl.load(tl.advance(v_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())

            qk = tl.dot(q_tile, k_tile.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            kv_indices = kv_token_idx + k_tile_arange
            mask = kv_indices[None, :] < seq_len
            mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args)

            if not PRESCALE_QK: qk = qk * softmax_scale
            qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            if GUARANTEED_SAFE:
                m_ij_safe = m_ij
            else:
                m_ij_safe = tl.where(m_ij == float("-inf"), tl.cast(0, m_ij.dtype), m_ij)
            alpha = tl.math.exp2(m_i - m_ij_safe)
            p = tl.math.exp2(qk - m_ij_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            if not TENSORS_PRELOAD:
                v_tile = v_desc.load([kv_token_idx, 0]) if USE_TMA else tl.load(tl.advance(v_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())
            acc = tl.dot(p.to(v_tile.dtype), v_tile, acc, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
            m_i = m_ij

    l_i = tl.where(l_i == 0.0, 1, l_i)
    acc = acc / l_i[:, None]

    # reinit to lower register pressure
    batch = tl.program_id(0)
    head = tl.program_id(1)
    q_tile_idx = tl.program_id(2)
    q_token_idx = q_tile_idx * TILE_Q_SIZE

    obatch_head_offset = batch * stride_ob + head * stride_oh
    if USE_TMA:
        o_desc = tl.make_tensor_descriptor(
            base=O + obatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_ot, stride_ok),
            block_shape=(TILE_Q_SIZE, HEAD_DIM),
        )
        o_desc.store([q_token_idx, 0], acc.to(q_tile.dtype))
    else:
        o_desc = tl.make_block_ptr(
            base=O + obatch_head_offset,
            shape=(T, HEAD_DIM),
            strides=(stride_ot, stride_ok),
            offsets=(q_token_idx, 0),
            block_shape=(TILE_Q_SIZE, HEAD_DIM),
            order=(1, 0),
        )
        if Q_BLOCK_DIVISIBLE:
            tl.store(
                o_desc,
                acc.to(q_tile.dtype),
            )
        else:
            tl.store(
                o_desc,
                acc.to(q_tile.dtype),
                boundary_check=(0,),
            )

    if OUTPUT_LOGSUMEXP and LSE is not None:
        m_i += tl.math.log2(l_i)

        mbatch_head_offset = batch * stride_mb + head * stride_mh
        if USE_TMA:
            m_desc = tl.make_tensor_descriptor(
                base=LSE + mbatch_head_offset,
                shape=(T,),
                strides=(stride_mt,),
                block_shape=(TILE_Q_SIZE,),
            )
            m_desc.store([q_token_idx], m_i)
        else:
            m_tile_ptr = tl.make_block_ptr(
                base=LSE + mbatch_head_offset,
                shape=(T,),
                strides=(stride_mt,),
                offsets=(q_token_idx,),
                block_shape=(TILE_Q_SIZE,),
                order=(0,),
            )

            if Q_BLOCK_DIVISIBLE:
                tl.store(
                    m_tile_ptr,
                    m_i,
                )
            else:
                tl.store(
                    m_tile_ptr,
                    m_i,
                    boundary_check=(0,),
                )
