import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.stats as sps
import statsmodels.stats.multitest as multitest
import torch
import triton
from tqdm.auto import tqdm, trange

sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/..")


from chill_attention import PrefixLMChillMask
from chill_attention.autotune import _get_default_config_bwd, _get_default_config_fwd
from chill_attention.kernel import _chill_attn_bwd, _chill_attn_fwd
from chill_attention.utils import strides

mask = PrefixLMChillMask(256)


def make_dummies_bwd(head_dim, dtype, B=1, T=8000, HEADS=1):
    q, k, v = torch.randn(3, B, HEADS, T, head_dim, dtype=dtype, device="cuda")
    lens = torch.full((B,), T, dtype=torch.int32, device="cuda")
    do = torch.zeros_like(q, memory_format=torch.contiguous_format)
    delta = torch.empty(do.shape[:-1], dtype=torch.float32, device=q.device)
    DQ = torch.zeros_like(q, memory_format=torch.contiguous_format)
    DK = torch.zeros_like(k, memory_format=torch.contiguous_format)
    DV = torch.zeros_like(v, memory_format=torch.contiguous_format)
    lse = torch.zeros(q.shape[:3], dtype=torch.float32, device=q.device)
    return dict(
        q=q,
        k=k,
        v=v,
        lens=lens,
        delta=delta,
        DQ=DQ,
        DK=DK,
        DV=DV,
        lse=lse,
        do=do,
    )


def make_dummies(head_dim, dtype, B=1, T=8000, HEADS=1):
    q, k, v = torch.randn(3, B, HEADS, T, head_dim, dtype=dtype, device="cuda")
    lens = torch.full((B,), T, dtype=torch.int32, device="cuda")
    return dict(
        q=q,
        k=k,
        v=v,
        lens=lens,
    )


def _make_fwd_args(q, k, v, lens, config):
    batch, heads, T, HEAD_DIM = q.shape
    TILE_Q_SIZE, TILE_K_SIZE, N_WARPS, PIPELINING, TENSORS_PRELOAD = config
    kt = k.transpose(-1, -2)
    Q_BLOCK_DIVISIBLE = T % TILE_Q_SIZE == 0
    K_BLOCK_DIVISIBLE = T % TILE_K_SIZE == 0
    grid = lambda args: (
        batch,
        heads,
        triton.cdiv(T, args["TILE_Q_SIZE"]),
    )
    O = torch.zeros_like(q, memory_format=torch.contiguous_format)
    LSE = torch.zeros(q.shape[:3], dtype=torch.float32, device=q.device)

    mask_fns = (
        mask.mask_jit,
        mask.k_range_for_q_jit,
        mask.q_range_for_k_jit,
        mask.is_full_block_jit,
    )

    HAS_FULL_BLOCKS = (
        mask.has_full_blocks(TILE_Q_SIZE, TILE_K_SIZE, T, args=mask.constargs),
    )

    q_lims_continious = mask.q_lims_continious()
    k_lims_continious = mask.k_lims_continious()

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
        INPUT_PRECISION="ieee",
        PRESCALE_QK=False,
        DTYPE=q.dtype,
        TIME_BUCKET=triton.next_power_of_2(T),
        OUTPUT_LOGSUMEXP=True,
        SM_SCALE=1,
        RCP_LN2=math.log2(math.e),
        PIPELINING=PIPELINING,
        Q_BLOCK_DIVISIBLE=Q_BLOCK_DIVISIBLE,
        K_BLOCK_DIVISIBLE=K_BLOCK_DIVISIBLE,
        TILE_Q_SIZE=TILE_Q_SIZE,
        TILE_K_SIZE=TILE_K_SIZE,
        HAS_FULL_BLOCKS=HAS_FULL_BLOCKS,
        TENSORS_PRELOAD=TENSORS_PRELOAD,
        mask_fns=mask_fns,
        mask_args=tuple(mask.constargs),
        k_lims_continious=k_lims_continious,
        num_warps=N_WARPS,
        num_stages=PIPELINING,
    )

    return grid, args, kwargs


def _make_bwd_args(q, k, v, lens, delta, DQ, DK, DV, lse, do, config):
    batch, heads, T, HEAD_DIM = q.shape
    grid = lambda args: (
        batch,
        heads,
        triton.cdiv(T, args["TILE_DQ_Q_SIZE"]) + triton.cdiv(T, args["TILE_DK_K_SIZE"]),
    )
    TILE_DQ_Q_SIZE, TILE_DQ_K_SIZE, N_WARPS, PIPELINING, TENSORS_PRELOAD = config
    TILE_DK_Q_SIZE = TILE_DQ_Q_SIZE
    TILE_DK_K_SIZE = TILE_DQ_K_SIZE

    HAS_FULL_BLOCKS_DQ = (
        mask.has_full_blocks(TILE_DQ_Q_SIZE, TILE_DQ_K_SIZE, T, args=mask.constargs),
    )
    HAS_FULL_BLOCKS_DK = mask.has_full_blocks(
        TILE_DK_Q_SIZE, TILE_DQ_K_SIZE, T, args=mask.constargs
    )

    mask_fns = (
        mask.mask_jit,
        mask.k_range_for_q_jit,
        mask.q_range_for_k_jit,
        mask.is_full_block_jit,
    )

    q_lims_continious = mask.q_lims_continious()
    k_lims_continious = mask.k_lims_continious()

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
        INPUT_PRECISION="ieee",
        DTYPE=q.dtype,
        SM_SCALE=1,
        PRESCALE_QK=False,
        mask_fns=mask_fns,
        mask_args=tuple(mask.constargs),
        k_lims_continious=k_lims_continious,
        q_lims_continious=q_lims_continious,
        RCP_LN2=math.log2(math.e),
        DQ_TILES_NUM=triton.cdiv(T, TILE_DQ_Q_SIZE),
        DQ_Q_BLOCK_DIVISIBLE=T % TILE_DQ_Q_SIZE == 0,
        DQ_K_BLOCK_DIVISIBLE=T % TILE_DQ_K_SIZE == 0,
        DK_Q_BLOCK_DIVISIBLE=T % TILE_DK_Q_SIZE == 0,
        DK_K_BLOCK_DIVISIBLE=T % TILE_DK_K_SIZE == 0,
        PIPELINING=PIPELINING,
        TILE_DQ_Q_SIZE=TILE_DQ_Q_SIZE,
        TILE_DQ_K_SIZE=TILE_DQ_K_SIZE,
        TILE_DK_Q_SIZE=TILE_DK_Q_SIZE,
        TILE_DK_K_SIZE=TILE_DK_K_SIZE,
        TENSORS_PRELOAD=TENSORS_PRELOAD,
        HAS_FULL_BLOCKS_DQ=HAS_FULL_BLOCKS_DQ,
        HAS_FULL_BLOCKS_DK=HAS_FULL_BLOCKS_DK,
        num_stages=PIPELINING,
        num_warps=N_WARPS,
    )

    return grid, args, kwargs


def check_forward(head_dim, dtype):
    dummies = make_dummies(head_dim, dtype)
    q = dummies["q"]
    batch, heads, T, HEAD_DIM = q.shape
    config = _get_default_config_fwd(HEAD_DIM, q.dtype)
    grid, args, kwargs = _make_fwd_args(**dummies, config=config)
    kernel = _chill_attn_fwd.warmup(
        *args,
        **kwargs,
        grid=grid,
    )
    kernel._init_handles()

    return kernel.n_regs, kernel.n_spills


def check_backward(head_dim, dtype):
    dummies = make_dummies_bwd(head_dim, dtype)
    q = dummies["q"]
    batch, heads, T, HEAD_DIM = q.shape
    config = _get_default_config_bwd(HEAD_DIM, q.dtype)
    grid, args, kwargs = _make_bwd_args(**dummies, config=config)
    kernel = _chill_attn_bwd.warmup(
        *args,
        **kwargs,
        grid=grid,
    )
    kernel._init_handles()

    return kernel.n_regs, kernel.n_spills


def bechmark_kernel(grid, args, kwargs, kernel_fn, nospill=True, n_repeat=20):
    if nospill:
        try:
            kernel = kernel_fn.warmup(
                *args,
                **kwargs,
                grid=grid,
            )
            kernel._init_handles()
            if kernel.n_spills > 0:
                return None
        except Exception as e:
            if "out of resource" in str(e):
                return None
            raise

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                kernel_fn[grid](*args, **kwargs)
        torch.cuda.synchronize()

        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
    except Exception as e:
        if "out of resource" in str(e):
            return None
        raise
    torch.cuda.synchronize()
    return [start.elapsed_time(end) / n_repeat]


def benchmark_forward(q, k, v, lens, config, nospill=True):
    grid, args, kwargs = _make_fwd_args(q, k, v, lens, config)
    return bechmark_kernel(
        grid,
        args,
        kwargs,
        _chill_attn_fwd,
        nospill=False,
    )


def benchmark_backward(q, k, v, lens, delta, DQ, DK, DV, lse, do, config, nospill=True):
    grid, args, kwargs = _make_bwd_args(
        q, k, v, lens, delta, DQ, DK, DV, lse, do, config
    )
    return bechmark_kernel(
        grid,
        args,
        kwargs,
        _chill_attn_bwd,
        nospill=nospill,
    )


def make_forward_configs(dtype, head_dim):
    configs = [
        (q, k, warp, pipe, preload)
        for q in [16, 32, 64, 128]
        for k in [16, 32, 64]
        for warp in [1, 2, 4, 8]
        for pipe in [1, 2, 3, 4]
        for preload in [True, False]
    ]
    if dtype == torch.float32 or head_dim > 128:
        configs = [
            (q, k, warp, pipe, preload)
            for q in [16, 32, 64, 128]
            for k in [16, 32, 64]
            for warp in [1, 2, 4, 8]
            for pipe in [1, 2]
            for preload in [True, False]
        ]
    return configs


def make_backward_configs(dtype, head_dim):
    configs = [
        (q, k, warp, pipe, preload)
        for q in [16, 32, 64]
        for k in [16, 32, 64]
        for warp in [1, 2, 4]
        for pipe in [1, 2, 3]
        for preload in [True, False]
    ]
    if dtype == torch.float32 or head_dim >= 128:
        configs = [
            (q, k, warp, pipe, preload)
            for q in [16, 32]
            for k in ([16, 32] if head_dim <= 128 else [16])
            for warp in [1, 2, 4]
            for pipe in ([1, 2] if head_dim <= 128 else [1])
            for preload in [True, False]
        ]
    return configs


def significantly_worse(references, new):
    pvals = [
        sps.ttest_ind(
            np.array(reference),
            np.array(new),
            equal_var=False,
            alternative="less",
        ).pvalue
        for reference in references
        if np.isfinite(np.array(reference)).all()
    ]

    result = multitest.multipletests(
        pvals=pvals,
    )
    return result[0].any(), np.min(result[1])


def get_results_dict(result_path, configs):
    if os.path.exists(result_path):
        with open(result_path) as file:
            bench = json.load(file)
            bench["configs"] = list(map(tuple, bench["configs"]))

        for config in configs:
            if config in bench["configs"]:
                continue
            bench["configs"].append(config)
            bench["result"].append([])
    else:
        bench = dict(configs=configs, result=[[] for _ in configs])
    return bench


def infitite():
    i = 0
    while True:
        yield i
        i += 1


def find_best(
    num_repetitions,
    configs_factory,
    dummies_factory,
    benchmark_fn,
    path_prefix,
    nospill=False,
):
    for dtype, head_dim in itertools.product(DTYPES, HEAD_DIMS):
        result_path = f"{os.path.dirname(os.path.realpath(__file__))}/timings/{path_prefix}-{dtype}-{head_dim}.json"
        print(result_path)

        configs = configs_factory(dtype=dtype, head_dim=head_dim)
        bench = get_results_dict(result_path, configs)

        def save():
            with open(result_path, "w") as file:
                json.dump(
                    bench,
                    file,
                    indent=2,
                )

        configs = bench["configs"]
        results = bench["result"]
        assert len(configs) == len(results)
        dummies = dummies_factory(head_dim, dtype)
        for config, times in zip(tqdm(configs), results):
            dummies["config"] = config

            if len(times) >= num_repetitions:
                continue

            benchmark_fn(**dummies, nospill=nospill)

            exps = trange(num_repetitions, leave=False)
            for _ in exps:
                time_ns = benchmark_fn(**dummies, nospill=nospill)
                if time_ns is None:
                    times.append(float("inf"))
                    break
                times += time_ns
                save()
        save()


def significantly_better_than_all(references, new):
    pvals = [
        (
            (
                sps.ttest_ind(
                    np.array(reference),
                    np.array(new),
                    equal_var=False,
                    alternative="greater",
                ).pvalue
                if len(reference) > 1
                else 1.0
            )
            if np.isfinite(np.array(reference)).all()
            else 0.0
        )
        for reference in references
    ]

    result = multitest.multipletests(
        pvals=pvals,
    )
    return result[1]


def purify(maxiter, dummies_factory, configs_factory, benchmark_fn, path_prefix, alpha):
    for dtype, head_dim in itertools.product(DTYPES, HEAD_DIMS):
        gpu_capability = torch.cuda.get_device_capability()
        gpu_capability = f"nv-{gpu_capability[0]}_{gpu_capability[1]}"

        result_path = f"{os.path.dirname(os.path.realpath(__file__))}/timings/{path_prefix}-{gpu_capability}-{dtype}-{head_dim}.json"
        print(result_path)

        configs = configs_factory(dtype=dtype, head_dim=head_dim)
        dummies = dummies_factory(head_dim, dtype)
        bench = get_results_dict(result_path, configs)

        def save():
            with open(result_path, "w") as file:
                json.dump(
                    bench,
                    file,
                    indent=2,
                )

        def best_config_i():
            configs_sorted = sorted(
                enumerate(bench["result"]), key=lambda x: np.mean(x[1])
            )
            return configs_sorted[0][0]

        def best_vs_all_pvalues():
            i = best_config_i()
            return significantly_better_than_all(bench["result"], bench["result"][i])

        def expansion_index():
            i = best_config_i()
            pvalues = best_vs_all_pvalues()
            pvalues[i] = -1
            return np.random.choice([np.argmax(pvalues), i]), sorted(
                pvalues, reverse=True
            )

        def termination_condition():
            i = best_config_i()
            pvalues = best_vs_all_pvalues()
            pvalues[i] = -1
            return max(pvalues) < alpha

        warmed_up = defaultdict(lambda: False)
        pbar = tqdm(range(maxiter))
        for _ in pbar:
            if termination_condition():
                break
            best_i = best_config_i()
            i, max_pvalue = expansion_index()
            dummies["config"] = tuple(bench["configs"][i])
            pbar.set_description(
                f"p_max={np.max(max_pvalue):.3f}, "
                f"p_mean={np.mean(max_pvalue):.3f}, "
                f"1_count={(abs(np.array(max_pvalue) - 1.0) < 1e-4).sum()}, "
                f"config={dummies['config']}, "
                f"best_config={bench['configs'][best_i]}"
            )
            pbar.refresh()

            if not warmed_up[dummies["config"]]:
                benchmark_fn(**dummies, nospill=False)
                warmed_up[dummies["config"]] = True

            time_ns = benchmark_fn(**dummies, nospill=False)
            if time_ns is None:
                time_ns = [float("inf")]
            bench["result"][i] += time_ns
            save()
        save()


def benchmark_all_forward(num_repetitions):
    find_best(
        num_repetitions=num_repetitions,
        configs_factory=make_forward_configs,
        dummies_factory=make_dummies,
        benchmark_fn=benchmark_forward,
        path_prefix="fwd",
    )


def benchmark_all_backward(num_repetitions):
    find_best(
        num_repetitions=num_repetitions,
        configs_factory=make_backward_configs,
        dummies_factory=make_dummies_bwd,
        benchmark_fn=benchmark_backward,
        path_prefix="bwd",
    )


def show_forward_configs():
    for dtype, head_dim in itertools.product(DTYPES, HEAD_DIMS):
        n_regs, n_spills = check_forward(head_dim, dtype)
        print(f"({dtype}, {head_dim}): ({n_regs = } {n_spills = })")


def show_backward_configs():
    for dtype, head_dim in itertools.product(DTYPES, HEAD_DIMS):
        n_regs, n_spills = check_backward(head_dim, dtype)
        print(f"({dtype}, {head_dim}): ({n_regs = } {n_spills = })")


def purify_all_forward(alpha):
    purify(
        dummies_factory=make_dummies,
        benchmark_fn=benchmark_forward,
        configs_factory=make_forward_configs,
        path_prefix="fwd",
        alpha=alpha,
        maxiter=2000,
    )


def purify_all_backward(alpha):
    purify(
        dummies_factory=make_dummies_bwd,
        benchmark_fn=benchmark_backward,
        configs_factory=make_backward_configs,
        path_prefix="bwd",
        alpha=alpha,
        maxiter=2000,
    )


HEAD_DIMS = [16, 32, 64, 128, 256]
DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

if __name__ == "__main__":
    target = sys.argv[1]
    match target:
        case "benchmark_all_forward":
            benchmark_all_forward(0)
        case "benchmark_all_backward":
            benchmark_all_backward(0)
        case "purify_all_forward":
            purify_all_forward(alpha=0.2)
        case "purify_all_backward":
            purify_all_backward(alpha=0.05)
        case "show_forward_configs":
            show_forward_configs()
        case "show_backward_configs":
            show_backward_configs()
