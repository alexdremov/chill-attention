import logging
import math
import typing

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from chill_attention.autotune import (
    _get_backward_autotune_configs,
    _get_default_config_bwd,
    _get_default_config_fwd,
    _get_forward_autotune_configs,
    _prune_notfitting_configs,
)

from .mask import ChillMask
from .utils import (
    _chill_attn_bwd_precompute,
    _get_min_max_tiles,
    _triton_set_alloc,
    strides,
)

MAX_TILE_SIZE = 128
MIN_TILE_SIZE = 32

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
    TIME_BUCKET:  int,  #
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

    m_i = tl.zeros([TILE_Q_SIZE], dtype=tl.float32) - float("inf")
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
    q_lens_mask = (
        q_tile_indices[:, None] < seq_len
    )

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

    if SPLIT_LOOPS and HAS_FULL_BLOCKS and (q_token_idx + TILE_Q_SIZE <= seq_len):
        fn_k_full_range: tl.constexpr = mask_fns[4]

        # Explicitly calculate the range of keys that are unmasked for ALL queries in the tile
        full_k_start, full_k_end = fn_k_full_range(q_token_idx, TILE_Q_SIZE, seq_len, mask_args)

        # Convert token indices to tile indices
        # A tile is FULL only if its entire range [kv_token_idx, kv_token_idx + TILE_K_SIZE)
        # is within the unmasked range [full_k_start, full_k_end).
        full_kv_start_tile = tl.cdiv(full_k_start, TILE_K_SIZE)
        full_kv_end_tile = full_k_end // TILE_K_SIZE

    # Ensure boundaries are valid
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
            k_tile = tl.load(tl.advance(k_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())
            if TENSORS_PRELOAD: v_tile = tl.load(tl.advance(v_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())

        qk = tl.dot(q_tile, k_tile.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        kv_indices = kv_token_idx + k_tile_arange
        mask = q_lens_mask & (kv_indices[None, :] < seq_len)
        mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args)

        if not PRESCALE_QK: qk = qk * softmax_scale
        qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_safe = tl.where(m_ij == float("-inf"), tl.cast(0, m_ij.dtype), m_ij)
        alpha = tl.math.exp2(m_i - m_ij_safe)
        p = tl.math.exp2(qk - m_ij_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if not TENSORS_PRELOAD:
            v_tile = v_desc.load([kv_token_idx, 0]) if USE_TMA else tl.load(tl.advance(v_desc, (kv_token_idx, 0)), boundary_check=(0,) if not K_BLOCK_DIVISIBLE else ())
        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        m_i = m_ij

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
        mask = q_lens_mask & (kv_indices[None, :] < seq_len)
        mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args)

        if not PRESCALE_QK: qk = qk * softmax_scale
        qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
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

    # reinit to lower registers pressure
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
                acc.to(o_desc.type.element_ty),
            )
        else:
            tl.store(
                o_desc,
                acc.to(o_desc.type.element_ty),
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
    TIME_BUCKET: int,  #
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
    TENSORS_PRELOAD: tl.constexpr,
    USE_TMA: tl.constexpr,
    mask_fns,
    mask_args,
    q_lims_continious: tl.constexpr,
    k_lims_continious: tl.constexpr,
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
            SPLIT_LOOPS=SPLIT_LOOPS,
            TENSORS_PRELOAD=TENSORS_PRELOAD,
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
    SPLIT_LOOPS: tl.constexpr,  #
    TENSORS_PRELOAD: tl.constexpr,
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

    for g in tl.range(0, GROUP_SIZE):
        q_head = kv_head * GROUP_SIZE + g
        qbatch_head_offset = batch * stride_qb + q_head * stride_qh
        if USE_TMA:
            q_desc = tl.make_tensor_descriptor(
                base=Q + qbatch_head_offset,
                shape=(T, HEAD_DIM),
                strides=(stride_qt, stride_qk),
                block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
            )
        else:
            q_desc = tl.make_block_ptr(
                base=Q + qbatch_head_offset,
                shape=(T, HEAD_DIM),
                strides=(stride_qt, stride_qk),
                offsets=(0, 0),
                block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
                order=(1, 0),
            )

        dobatch_head_offset = batch * stride_dob + q_head * stride_doh
        if USE_TMA:
            do_desc = tl.make_tensor_descriptor(
                base=DO + dobatch_head_offset,
                shape=(T, HEAD_DIM),
                strides=(stride_dot, stride_dok),
                block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
            )
        else:
            do_desc = tl.make_block_ptr(
                base=DO + dobatch_head_offset,
                shape=(T, HEAD_DIM),
                strides=(stride_dot, stride_dok),
                offsets=(0, 0),
                block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
                order=(1, 0),
            )

        lsebatch_head_offset = batch * stride_mb + q_head * stride_mh
        if USE_TMA:
            lse_desc = tl.make_tensor_descriptor(
                base=LSE + lsebatch_head_offset,
                shape=(T,),
                strides=(stride_mt,),
                block_shape=(TILE_DK_Q_SIZE,),
            )
        else:
            lse_desc = tl.make_block_ptr(
                base=LSE + lsebatch_head_offset,
                shape=(T,),
                strides=(stride_mt,),
                offsets=(0,),
                block_shape=(TILE_DK_Q_SIZE,),
                order=(0,),
            )

        deltabatch_head_offset = batch * stride_deltab + q_head * stride_deltah
        if USE_TMA:
            delta_desc = tl.make_tensor_descriptor(
                base=DELTA + deltabatch_head_offset,
                shape=(T,),
                strides=(stride_deltat,),
                block_shape=(TILE_DK_Q_SIZE,),
            )
        else:
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
            TILE_Q_SIZE=TILE_DK_Q_SIZE,
            TILE_K_SIZE=TILE_DK_K_SIZE,
            INPUT_PRECISION=INPUT_PRECISION,
            PIPELINING=PIPELINING,
            Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
            HAS_FULL_BLOCKS=HAS_FULL_BLOCKS,
            RCP_LN2=RCP_LN2,
            SM_SCALE=SM_SCALE,
            PRESCALE_QK=PRESCALE_QK,
            SPLIT_LOOPS=SPLIT_LOOPS,
            TENSORS_PRELOAD=TENSORS_PRELOAD,
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
        p = tl.math.exp2(qk - m)

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
        p = tl.math.exp2(qk - m)
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
        p = tl.math.exp2(qk - m)

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
    TILE_Q_SIZE: tl.constexpr,
    TILE_K_SIZE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    PIPELINING: tl.constexpr,
    Q_BLOCK_DIVISIBLE: tl.constexpr,
    HAS_FULL_BLOCKS: tl.constexpr,  #
    RCP_LN2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    PRESCALE_QK: tl.constexpr,
    SPLIT_LOOPS: tl.constexpr,
    TENSORS_PRELOAD: tl.constexpr,
    USE_TMA: tl.constexpr,
    mask_fns,
    mask_args,
    q_lims_continious: tl.constexpr,
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

    if SPLIT_LOOPS and HAS_FULL_BLOCKS and (kv_token_idx + TILE_K_SIZE <= seq_len):
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
            q = q_desc.load([q_token_idx, 0])
            if TENSORS_PRELOAD:
                m = lse_desc.load([q_token_idx])
                do = do_desc.load([q_token_idx, 0])
                Di = delta_desc.load([q_token_idx])
        else:
            q = tl.load(tl.advance(q_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
            if TENSORS_PRELOAD:
                m = tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                do = tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                Di = tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

        qkT = tl.dot(k, q.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qkT *= softmax_scale_ln2

        if not TENSORS_PRELOAD:
            m = lse_desc.load([q_token_idx]) if USE_TMA else tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

        pT = tl.math.exp2(qkT - m[None, :])
        q_tile_indices = q_token_idx + tile_q_arange
        mask = kv_lens_mask & (q_tile_indices[None, :] < seq_len)
        mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args).T
        pT = tl.where(mask, pT, 0.0)

        if not TENSORS_PRELOAD:
            do = do_desc.load([q_token_idx, 0]) if USE_TMA else tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dpT = tl.dot(v.to(do.dtype), do.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not TENSORS_PRELOAD:
            Di = delta_desc.load([q_token_idx]) if USE_TMA else tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    # --- LOOP 2: Full Blocks (Hot Loop) ---
    for q_tile_idx in tl.range(full_q_start_tile, full_q_end_tile, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        if USE_TMA:
            q = q_desc.load([q_token_idx, 0])
            if TENSORS_PRELOAD:
                m = lse_desc.load([q_token_idx])
                do = do_desc.load([q_token_idx, 0])
                Di = delta_desc.load([q_token_idx])
        else:
            q = tl.load(tl.advance(q_desc, (q_token_idx, 0)))
            if TENSORS_PRELOAD:
                m = tl.load(tl.advance(lse_desc, (q_token_idx,)))
                do = tl.load(tl.advance(do_desc, (q_token_idx, 0)))
                Di = tl.load(tl.advance(delta_desc, (q_token_idx,)))

        qkT = tl.dot(k, q.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qkT *= softmax_scale_ln2

        if not TENSORS_PRELOAD:
            m = lse_desc.load([q_token_idx]) if USE_TMA else tl.load(tl.advance(lse_desc, (q_token_idx,)))

        pT = tl.math.exp2(qkT - m[None, :])
        # NO MASKING HERE
        if not TENSORS_PRELOAD:
            do = do_desc.load([q_token_idx, 0]) if USE_TMA else tl.load(tl.advance(do_desc, (q_token_idx, 0)))
        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dpT = tl.dot(v.to(do.dtype), do.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not TENSORS_PRELOAD:
            Di = delta_desc.load([q_token_idx]) if USE_TMA else tl.load(tl.advance(delta_desc, (q_token_idx,)))
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    # --- LOOP 3: Suffix Partial Blocks ---
    for q_tile_idx in tl.range(full_q_end_tile, q_end_tile_idx, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        if USE_TMA:
            q = q_desc.load([q_token_idx, 0])
            if TENSORS_PRELOAD:
                m = lse_desc.load([q_token_idx])
                do = do_desc.load([q_token_idx, 0])
                Di = delta_desc.load([q_token_idx])
        else:
            q = tl.load(tl.advance(q_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
            if TENSORS_PRELOAD:
                m = tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                do = tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
                Di = tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

        qkT = tl.dot(k, q.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK: qkT *= softmax_scale_ln2

        if not TENSORS_PRELOAD:
            m = lse_desc.load([q_token_idx]) if USE_TMA else tl.load(tl.advance(lse_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())

        pT = tl.math.exp2(qkT - m[None, :])
        q_tile_indices = q_token_idx + tile_q_arange
        mask = kv_lens_mask & (q_tile_indices[None, :] < seq_len)
        mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args).T
        pT = tl.where(mask, pT, 0.0)

        if not TENSORS_PRELOAD:
            do = do_desc.load([q_token_idx, 0]) if USE_TMA else tl.load(tl.advance(do_desc, (q_token_idx, 0)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
        dv = tl.dot(pT.to(do.dtype), do, dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        dpT = tl.dot(v.to(do.dtype), do.trans(), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not TENSORS_PRELOAD:
            Di = delta_desc.load([q_token_idx]) if USE_TMA else tl.load(tl.advance(delta_desc, (q_token_idx,)), boundary_check=(0,) if not Q_BLOCK_DIVISIBLE else ())
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
            )
        )(_chill_attn_bwd)

        if autotune:
            if (HEAD_DIM, q.dtype) not in autotunes_bwd:
                configs = _get_backward_autotune_configs(
                    HEAD_DIM, q.dtype, has_k_full_range=mask.has_k_full_range()
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
