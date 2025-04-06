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
from .utils import _chill_attn_bwd_precompute, _get_min_max_tiles, strides

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
    Q: tl.tensor, Kt: tl.tensor, V: tl.tensor, L: tl.tensor, #
    LSE: tl.tensor, O: tl.tensor,  #
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: tl.constexpr,  #
    stride_kb: int, stride_kh: int, stride_kk: tl.constexpr, stride_kt: int,  #
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
    OUTPUT_LOGSUMEXP: tl.constexpr,  #
    TILE_Q_SIZE: tl.constexpr,  #
    TILE_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
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
    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr = tl.make_block_ptr(
        base=Kt + kbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_K_SIZE),
        order=(0, 1),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
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
    q_lens_mask = (
        q_tile_indices[:, None] < seq_len
    )

    if Q_BLOCK_DIVISIBLE:
        q_tile = tl.load(q_tile_ptr)
    else:
        q_tile = tl.load(
            q_tile_ptr,
            boundary_check=(0,),
        )

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE * RCP_LN2, q_tile.dtype)

    if PRESCALE_QK:
        q_tile = q_tile * softmax_scale

    for kv_tile_idx in tl.range(
        kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE

        if K_BLOCK_DIVISIBLE:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
            )
            if TENSORS_PRELOAD:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                )
        else:
            kt_tile = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                boundary_check=(1,),
            )
            if TENSORS_PRELOAD:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_token_idx, 0)),
                    boundary_check=(0,),
                )

        qk = tl.dot(
            q_tile, kt_tile, input_precision=INPUT_PRECISION, out_dtype=tl.float32
        )

        kv_indices = kv_token_idx + tl.arange(0, TILE_K_SIZE)
        mask = q_lens_mask & (
            kv_indices[None, :] < seq_len
        )
        if not HAS_FULL_BLOCKS or not is_full_block(q_token_idx, kv_token_idx, TILE_Q_SIZE, TILE_K_SIZE, seq_len=seq_len, args=mask_args):
            q_tile_indices = q_token_idx + tl.arange(0, TILE_Q_SIZE)
            mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args)

        if not PRESCALE_QK:
            qk = qk * softmax_scale
        qk = tl.where(mask, qk, tl.cast(-float("inf"), qk.dtype))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_safe = tl.where(m_ij == float("-inf"), tl.cast(0, m_ij.dtype), m_ij)
        alpha = tl.math.exp2(m_i - m_ij_safe)

        p = tl.math.exp2(qk - m_ij_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        if not TENSORS_PRELOAD:
            if K_BLOCK_DIVISIBLE:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_tile_idx * TILE_K_SIZE, 0)),
                )
            else:
                v_tile = tl.load(
                    tl.advance(v_tile_ptr, (kv_tile_idx * TILE_K_SIZE, 0)),
                    boundary_check=(0,),
                )

        acc = tl.dot(
            p.to(v_tile.dtype),
            v_tile,
            acc,
            input_precision=INPUT_PRECISION,
            out_dtype=tl.float32,
        )
        m_i = m_ij

    l_i = tl.where(l_i == 0.0, 1, l_i)
    acc = acc / l_i[:, None]

    batch = tl.program_id(0)
    head = tl.program_id(1)
    q_tile_idx = tl.program_id(2)
    q_token_idx = q_tile_idx * TILE_Q_SIZE

    obatch_head_offset = batch * stride_ob + head * stride_oh
    o_tile_ptr = tl.make_block_ptr(
        base=O + obatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_ot, stride_ok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if Q_BLOCK_DIVISIBLE:
        tl.store(
            o_tile_ptr,
            acc.to(o_tile_ptr.type.element_ty),
        )
    else:
        tl.store(
            o_tile_ptr,
            acc.to(o_tile_ptr.type.element_ty),
            boundary_check=(0,),
        )

    if OUTPUT_LOGSUMEXP and LSE is not None:
        m_i += tl.math.log2(l_i)

        mbatch_head_offset = batch * stride_mb + head * stride_mh
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
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,  #
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,  #
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,  #
    stride_deltab: int, stride_deltah: int, stride_deltat: int,  #
    stride_mb: int, stride_mh: int, stride_mt: int,  #
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,  #
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: int,  #
    stride_dkb: int, stride_dkh: int, stride_dkt: int, stride_dkk: int,  #
    stride_dvb: int, stride_dvh: int, stride_dvt: int, stride_dvk: int,  #
    lens_stride: int,
    T: int,  #
    TIME_BUCKET: int,  #
    DQ_TILES_NUM: int,  #
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
    TILE_DQ_Q_SIZE: tl.constexpr, TILE_DQ_K_SIZE: tl.constexpr,  #
    TILE_DK_Q_SIZE: tl.constexpr, TILE_DK_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    TENSORS_PRELOAD: tl.constexpr,
    mask_fns,
    mask_args,
    q_lims_continious: tl.constexpr,
    k_lims_continious: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    dkv_worker = tl.program_id(2) >= DQ_TILES_NUM
    tile_id = tl.program_id(2) - (DQ_TILES_NUM * dkv_worker)

    if L is not None:
        seq_len = tl.load(L + batch * lens_stride)
        seq_len = min(seq_len, T)
    else:
        seq_len = T

    if dkv_worker:
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
            INPUT_PRECISION=INPUT_PRECISION,
            SM_SCALE=SM_SCALE,
            PRESCALE_QK=PRESCALE_QK,
            DK_Q_BLOCK_DIVISIBLE=DK_Q_BLOCK_DIVISIBLE,
            DK_K_BLOCK_DIVISIBLE=DK_K_BLOCK_DIVISIBLE,
            HAS_FULL_BLOCKS=HAS_FULL_BLOCKS_DK,
            RCP_LN2=RCP_LN2,
            TILE_DK_Q_SIZE=TILE_DK_Q_SIZE,
            TILE_DK_K_SIZE=TILE_DK_K_SIZE,
            PIPELINING=PIPELINING,
            TENSORS_PRELOAD=TENSORS_PRELOAD,
            mask_fns=mask_fns,
            mask_args=mask_args,
            q_lims_continious=q_lims_continious,
        )
    else:
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
            INPUT_PRECISION=INPUT_PRECISION,
            SM_SCALE=SM_SCALE,
            PRESCALE_QK=PRESCALE_QK,
            DQ_Q_BLOCK_DIVISIBLE=DQ_Q_BLOCK_DIVISIBLE,
            DQ_K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE,
            HAS_FULL_BLOCKS=HAS_FULL_BLOCKS_DQ,
            RCP_LN2=RCP_LN2,
            TILE_DQ_Q_SIZE=TILE_DQ_Q_SIZE,
            TILE_DQ_K_SIZE=TILE_DQ_K_SIZE,
            PIPELINING=PIPELINING,
            TENSORS_PRELOAD=TENSORS_PRELOAD,
            mask_fns=mask_fns,
            mask_args=mask_args,
            k_lims_continious=k_lims_continious,
        )

@triton.jit()
def _chill_attn_bwd_dq_inner(
    Q: tl.tensor, K: tl.tensor, V: tl.tensor, DELTA: tl.tensor, LSE: tl.tensor,
    DO: tl.tensor, DQ: tl.tensor,
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int, stride_dok: int,
    stride_dqb: int, stride_dqh: int, stride_dqt: int, stride_dqk: int,
    batch: int,
    head: int,
    tile_id: int,
    seq_len: tl.tensor,
    T: int,  #
    HEAD_DIM: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    DQ_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DQ_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    HAS_FULL_BLOCKS: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    TILE_DQ_Q_SIZE: tl.constexpr,  #
    TILE_DQ_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    TENSORS_PRELOAD: tl.constexpr,
    mask_fns,
    mask_args,
    k_lims_continious: tl.constexpr,  #
):
    q_tile_idx = tile_id
    q_token_idx = q_tile_idx * TILE_DQ_Q_SIZE

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    q_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_qt, stride_qk),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    lsebatch_head_offset = batch * stride_mb + head * stride_mh
    lse_tile_ptr = tl.make_block_ptr(
        base=LSE + lsebatch_head_offset,
        shape=(T,),
        strides=(stride_mt,),
        offsets=(q_token_idx,),
        block_shape=(TILE_DQ_Q_SIZE,),
        order=(0,),
    )

    delta_tile_ptr = batch * stride_deltab + head * stride_deltah
    delta_tile_ptr = tl.make_block_ptr(
        base=DELTA + delta_tile_ptr,
        shape=(T,),
        strides=(stride_deltat,),
        offsets=(q_token_idx,),
        block_shape=(TILE_DQ_Q_SIZE,),
        order=(0,),
    )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dot, stride_dok),
        offsets=(q_token_idx, 0),
        block_shape=(TILE_DQ_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    if DQ_Q_BLOCK_DIVISIBLE:
        q = tl.load(q_tile_ptr)
        m = tl.load(lse_tile_ptr)[:, None]
        di = tl.load(delta_tile_ptr)
        do = tl.load(do_tile_ptr)
    else:
        q = tl.load(q_tile_ptr, boundary_check=(0,))
        m = tl.load(lse_tile_ptr, boundary_check=(0,))[:, None]
        di = tl.load(delta_tile_ptr, boundary_check=(0,))
        do = tl.load(do_tile_ptr, boundary_check=(0,))

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    kt_tile_ptr = tl.make_block_ptr(
        base=K + kbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_kk, stride_kt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_DQ_K_SIZE),
        order=(0, 1),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    vt_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_vk, stride_vt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_DQ_K_SIZE),
        order=(1, 0),
    )

    dq = tl.zeros([TILE_DQ_Q_SIZE, HEAD_DIM], dtype=tl.float32)
    dq = _chill_attn_bwd_dq(
        dq, q, m, di, do,
        kt_tile_ptr, vt_tile_ptr,
        seq_len=seq_len,
        q_token_idx=q_token_idx,
        TILE_Q_SIZE=TILE_DQ_Q_SIZE,
        TILE_K_SIZE=TILE_DQ_K_SIZE,
        INPUT_PRECISION=INPUT_PRECISION,
        PIPELINING=PIPELINING,
        K_BLOCK_DIVISIBLE=DQ_K_BLOCK_DIVISIBLE,
        HAS_FULL_BLOCKS=HAS_FULL_BLOCKS,
        RCP_LN2=RCP_LN2,
        SM_SCALE=SM_SCALE,
        PRESCALE_QK=PRESCALE_QK,
        TENSORS_PRELOAD=TENSORS_PRELOAD,
        mask_fns=mask_fns,
        mask_args=mask_args,
        k_lims_continious=k_lims_continious,
    )

    dqbatch_head_offset = batch * stride_dqb + head * stride_dqh
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
    stride_qb: int, stride_qh: int, stride_qt: int, stride_qk: int,
    stride_kb: int, stride_kh: int, stride_kt: int, stride_kk: int,
    stride_vb: int, stride_vh: int, stride_vt: int, stride_vk: int,
    stride_deltab: int, stride_deltah: int, stride_deltat: int,
    stride_mb: int, stride_mh: int, stride_mt: int,
    stride_dob: int, stride_doh: int, stride_dot: int,
    stride_dok: int, stride_dkb: int, stride_dkh: int,
    stride_dkt: int, stride_dkk: int, stride_dvb: int,
    stride_dvh: int, stride_dvt: int, stride_dvk: int,
    batch: int,
    head: int,
    tile_id: int,
    seq_len: tl.tensor,
    T: int,  #
    HEAD_DIM: tl.constexpr,  #
    INPUT_PRECISION: tl.constexpr,  #
    SM_SCALE: tl.constexpr,  #
    PRESCALE_QK: tl.constexpr,  #
    DK_Q_BLOCK_DIVISIBLE: tl.constexpr,  #
    DK_K_BLOCK_DIVISIBLE: tl.constexpr,  #
    HAS_FULL_BLOCKS: tl.constexpr,  #
    RCP_LN2: tl.constexpr,  #
    TILE_DK_Q_SIZE: tl.constexpr,  #
    TILE_DK_K_SIZE: tl.constexpr,  #
    PIPELINING: tl.constexpr,  #
    TENSORS_PRELOAD: tl.constexpr,
    mask_fns,
    mask_args,
    q_lims_continious: tl.constexpr,  #
):
    kv_tile_idx = tile_id
    kv_token_idx = kv_tile_idx * TILE_DK_K_SIZE

    qbatch_head_offset = batch * stride_qb + head * stride_qh
    qt_tile_ptr = tl.make_block_ptr(
        base=Q + qbatch_head_offset,
        shape=(HEAD_DIM, T),
        strides=(stride_qk, stride_qt),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, TILE_DK_Q_SIZE),
        order=(0, 1),
    )

    kbatch_head_offset = batch * stride_kb + head * stride_kh
    k_tile_ptr = tl.make_block_ptr(
        base=K + kbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_kt, stride_kk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    vbatch_head_offset = batch * stride_vb + head * stride_vh
    v_tile_ptr = tl.make_block_ptr(
        base=V + vbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_vt, stride_vk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    dobatch_head_offset = batch * stride_dob + head * stride_doh
    do_tile_ptr = tl.make_block_ptr(
        base=DO + dobatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dot, stride_dok),
        offsets=(0, 0),
        block_shape=(TILE_DK_Q_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    lsebatch_head_offset = batch * stride_mb + head * stride_mh
    lse_tile_ptr = tl.make_block_ptr(
        base=LSE + lsebatch_head_offset,
        shape=(T,),
        strides=(stride_mt,),
        offsets=(0,),
        block_shape=(TILE_DK_Q_SIZE,),
        order=(0,),
    )

    deltabatch_head_offset = batch * stride_deltab + head * stride_deltah
    delta_tile_ptr = tl.make_block_ptr(
        base=DELTA + deltabatch_head_offset,
        shape=(T,),
        strides=(stride_deltat,),
        offsets=(0,),
        block_shape=(TILE_DK_Q_SIZE,),
        order=(0,),
    )

    dv = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([TILE_DK_K_SIZE, HEAD_DIM], dtype=tl.float32)

    if DK_K_BLOCK_DIVISIBLE:
        k = tl.load(
                k_tile_ptr,
            )
        v = tl.load(
                v_tile_ptr,
            )
    else:
        k = tl.load(
                k_tile_ptr,
                boundary_check=(0,),
            )
        v = tl.load(
                v_tile_ptr,
                boundary_check=(0,),
            )

    dk, dv = _chill_attn_bwd_dkdv(
        dk, dv,
        qt_tile_ptr, do_tile_ptr, lse_tile_ptr, delta_tile_ptr,
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
        TENSORS_PRELOAD=TENSORS_PRELOAD,
        mask_fns=mask_fns,
        mask_args=mask_args,
        q_lims_continious=q_lims_continious,
    )

    dkbatch_head_offset = batch * stride_dkb + head * stride_dkh
    dk_tile_ptr = tl.make_block_ptr(
        base=DK + dkbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dkt, stride_dkk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if DK_K_BLOCK_DIVISIBLE:
        tl.store(dk_tile_ptr, dk.to(dk_tile_ptr.type.element_ty))
    else:
        tl.store(dk_tile_ptr, dk.to(dk_tile_ptr.type.element_ty), boundary_check=(0,))

    dvbatch_head_offset = batch * stride_dvb + head * stride_dvh
    dv_tile_ptr = tl.make_block_ptr(
        base=DV + dvbatch_head_offset,
        shape=(T, HEAD_DIM),
        strides=(stride_dvt, stride_dvk),
        offsets=(kv_token_idx, 0),
        block_shape=(TILE_DK_K_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    if DK_K_BLOCK_DIVISIBLE:
        tl.store(dv_tile_ptr, dv.to(dv_tile_ptr.type.element_ty))
    else:
        tl.store(dv_tile_ptr, dv.to(dv_tile_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def _chill_attn_bwd_dq(
    dq: tl.tensor, q: tl.tensor, m: tl.tensor,
    di: tl.tensor, do: tl.tensor,
    kt_tile_ptr: tl.tensor, vt_tile_ptr: tl.tensor,
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
    TENSORS_PRELOAD: tl.constexpr,
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

    softmax_scale: tl.constexpr = tl.cast(SM_SCALE, q.dtype)
    if PRESCALE_QK:
        q = q * softmax_scale * RCP_LN2

    for kv_tile_idx in tl.range(
        kv_start_tile_idx, kv_end_tile_idx, num_stages=PIPELINING
    ):
        kv_token_idx = kv_tile_idx * TILE_K_SIZE
        if K_BLOCK_DIVISIBLE:
            kT = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
            )
            if TENSORS_PRELOAD:
                vT = tl.load(
                    tl.advance(vt_tile_ptr, (0, kv_token_idx)),
                )
        else:
            kT = tl.load(
                tl.advance(kt_tile_ptr, (0, kv_token_idx)),
                boundary_check=(1,),
            )
            if TENSORS_PRELOAD:
                vT = tl.load(
                    tl.advance(vt_tile_ptr, (0, kv_token_idx,)),
                    boundary_check=(1,),
                )

        qk = tl.dot(q, kT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK:
            qk = qk * softmax_scale * RCP_LN2
        p = tl.math.exp2(qk - m)

        kv_indices = kv_token_idx + tile_k_arange
        mask = q_len_mask & (
            kv_indices[None, :] < seq_len
        )
        if not HAS_FULL_BLOCKS or not is_full_block(q_token_idx, kv_token_idx, TILE_Q_SIZE, TILE_K_SIZE, seq_len=seq_len, args=mask_args):
            mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args)

        if not TENSORS_PRELOAD:
            if K_BLOCK_DIVISIBLE:
                vT = tl.load(
                    tl.advance(vt_tile_ptr, (0, kv_token_idx)),
                )
            else:
                vT = tl.load(
                    tl.advance(vt_tile_ptr, (0, kv_token_idx,)),
                    boundary_check=(1,),
                )

        p = tl.where(mask, p, 0.0)
        dp = tl.dot(do.to(vT.dtype), vT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        ds = p * (dp - di[:, None])
        dq = tl.dot(ds.to(kT.dtype), tl.trans(kT), dq, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

    dq *= softmax_scale
    return dq


@triton.jit
def _chill_attn_bwd_dkdv(
    dk: tl.tensor, dv: tl.tensor,
    qt_tile_ptr: tl.tensor, do_tile_ptr: tl.tensor,
    lse_tile_ptr: tl.tensor, delta_tile_ptr: tl.tensor,
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
    TENSORS_PRELOAD: tl.constexpr,
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

    if PRESCALE_QK:
        k *= RCP_LN2 * SM_SCALE

    for q_tile_idx in tl.range(q_start_tile_idx, q_end_tile_idx, num_stages=PIPELINING):
        q_token_idx = q_tile_idx * TILE_Q_SIZE
        # NOTE: triton will not reorder loads
        # if there are problems with shared memory, do and Di loads can be moved just before usage
        # (via constexpr flag)
        if Q_BLOCK_DIVISIBLE:
            qT = tl.load(
                tl.advance(qt_tile_ptr, (0, q_token_idx)),
            )
            if TENSORS_PRELOAD:
                m = tl.load(
                    tl.advance(lse_tile_ptr, (q_token_idx,)),
                )
                do = tl.load(
                    tl.advance(do_tile_ptr, (q_token_idx, 0)),
                )
                Di = tl.load(
                    tl.advance(delta_tile_ptr, (q_token_idx,)),
                )
        else:
            qT = tl.load(
                tl.advance(qt_tile_ptr, (0, q_token_idx)),
                boundary_check=(1,),
            )
            if TENSORS_PRELOAD:
                m = tl.load(
                    tl.advance(lse_tile_ptr, (q_token_idx,)),
                    boundary_check=(0,),
                )
                do = tl.load(
                    tl.advance(do_tile_ptr, (q_token_idx, 0)),
                    boundary_check=(0,),
                )
                Di = tl.load(
                    tl.advance(delta_tile_ptr, (q_token_idx,)),
                    boundary_check=(0,),
                )

        qkT = tl.dot(k, qT, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not PRESCALE_QK:
            qkT *= RCP_LN2 * SM_SCALE

        if not TENSORS_PRELOAD:
            if Q_BLOCK_DIVISIBLE:
                m = tl.load(
                    tl.advance(lse_tile_ptr, (q_token_idx,)),
                )
            else:
                m = tl.load(
                    tl.advance(lse_tile_ptr, (q_token_idx,)),
                    boundary_check=(0,),
                )

        tl.static_assert(m.dtype == tl.float32)
        pT = tl.math.exp2(qkT - m[None, :])

        q_tile_indices = q_token_idx + tile_q_arange
        mask = kv_lens_mask & (
            q_tile_indices[None, :] < seq_len
        )
        if not HAS_FULL_BLOCKS or not is_full_block(q_token_idx, kv_token_idx, TILE_Q_SIZE, TILE_K_SIZE, seq_len=seq_len, args=mask_args):
            mask &= fn_mask(q_tile_indices, kv_indices, seq_len=seq_len, args=mask_args).T
        pT = tl.where(mask, pT, 0.0)

        if not TENSORS_PRELOAD:
            if Q_BLOCK_DIVISIBLE:
                do = tl.load(
                    tl.advance(do_tile_ptr, (q_token_idx, 0)),
                )
            else:
                do = tl.load(
                    tl.advance(do_tile_ptr, (q_token_idx, 0)),
                    boundary_check=(0,),
                )

        dv = tl.dot(pT, do.to(pT.dtype), dv, input_precision=INPUT_PRECISION, out_dtype=tl.float32)

        dpT = tl.dot(v.to(do.dtype), tl.trans(do), input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        if not TENSORS_PRELOAD:
            if Q_BLOCK_DIVISIBLE:
                Di = tl.load(
                    tl.advance(delta_tile_ptr, (q_token_idx,)),
                )
            else:
                Di = tl.load(
                    tl.advance(delta_tile_ptr, (q_token_idx,)),
                    boundary_check=(0,),
                )

        tl.static_assert(Di.dtype == tl.float32)
        # Compute dP and dS.
        dsT = pT * (dpT - Di[None, :])
        dk = tl.dot(dsT.to(qT.dtype), tl.trans(qT), dk, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
    dk *= SM_SCALE
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

        assert HEAD_DIM in {16, 32, 64, 128, 256}
        assert HEAD_DIM == k.shape[-1] and HEAD_DIM == v.shape[-1]
        assert T == k.shape[-2] and T == v.shape[-2]
        assert sm_scale is not None
        assert lens is None or (
            lens.dtype == torch.int32 and batch == len(lens) and lens.ndim == 1
        )

        O = torch.zeros_like(q, memory_format=torch.contiguous_format)
        LSE = None
        if return_lse:
            LSE = torch.zeros(q.shape[:3], dtype=torch.float32, device=q.device)

        grid = lambda args: (
            batch,
            heads,
            triton.cdiv(T, args["TILE_Q_SIZE"]),
        )

        kt = k.transpose(-1, -2)  # just stride tricks, same data

        args = [
            q,
            kt,
            v,
            lens,
            LSE,
            O,
            *strides(q, 4),
            *strides(kt, 4),
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
                configs = _get_forward_autotune_configs(HEAD_DIM, q.dtype)
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

        delta = torch.empty(o.shape[:-1], dtype=torch.float32, device=o.device)
        grid = lambda args: (
            batch,
            heads,
            triton.cdiv(T, args["TILE_SIZE"]),
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
        )

        # assert do.isfinite().all()

        DQ = torch.zeros_like(q, memory_format=torch.contiguous_format)
        DK = torch.zeros_like(k, memory_format=torch.contiguous_format)
        DV = torch.zeros_like(v, memory_format=torch.contiguous_format)

        grid = lambda args: (
            batch,
            heads,
            triton.cdiv(T, args["TILE_DQ_Q_SIZE"])
            + triton.cdiv(T, args["TILE_DK_K_SIZE"]),
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
                configs = _get_backward_autotune_configs(HEAD_DIM, q.dtype)
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

    Args:
        q (torch.Tensor): Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        k (torch.Tensor): Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        v (torch.Tensor): Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
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
