import contextlib
import functools
import logging
import os
import sys

import numpy as np
import torch
import triton
from torch.nn.attention.flex_attention import flex_attention

sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/..")

from chill_attention import (
    CausalChillMask,
    ChillMask,
    ChunkwiseChillMask,
    FullChillMask,
    PrefixLMChillMask,
    SlidingWindowChillMask,
    _chill_reference_naive,
    chill_attention,
)

masks_to_bench = [
    SlidingWindowChillMask(10, 20),
    SlidingWindowChillMask(16, 16),
    CausalChillMask(),
    ChunkwiseChillMask(16, 8),
    PrefixLMChillMask(128),
    FullChillMask(),
]

num_points = 5
max_time = 8000
batches = (64,)


def make_configs_for_mask(mask):
    configs = []
    params = [
        dict(
            batch=batch,
            dim=64,
            heads=12,
            name="small",
        )
        for batch in batches
    ]

    gpu_capability = torch.cuda.get_device_capability()
    gpu_capability = f"nv-{gpu_capability[0]}_{gpu_capability[1]}"

    for param in params:
        for mode in ("bwd", "fwd"):
            line_vals = [
                # f"chill-{mode}-compile",
                f"chill-{mode}-compile-autotune",
                # f"flex-{mode}-compile",
                f"flex-{mode}-compile-autotune",
            ]
            dim = param["dim"]
            heads = param["heads"]
            batch = param["batch"]

            x_vals = np.linspace(128, max_time, num_points).astype(int).tolist()
            x_vals = np.unique(x_vals)
            x_vals = sorted(x_vals)
            x_vals = [int(i) for i in x_vals]

            config = triton.testing.Benchmark(
                x_names=["time"],
                x_vals=x_vals,
                line_arg="provider",
                line_vals=line_vals,
                line_names=line_vals,
                styles=[
                    ("red", "-"),
                    ("blue", "-"),
                    ("green", "-"),
                    ("yellow", "-"),
                    ("black", "-"),
                    ("darkorange", "-"),
                    ("purple", "-"),
                ],
                ylabel="ms",
                plot_name=f"result-{gpu_capability}-{mode}-{param['name']}-{mask}-dim-{dim}-heads-{heads}-batch-{batch}",
                args=dict(
                    batch=batch,
                    heads=heads,
                    dim=dim,
                    dtype=torch.float16,
                    mask=mask,
                ),
            )

            configs.append(config)

    return configs


logging.basicConfig(level=logging.INFO)


for mask in masks_to_bench:
    configs = make_configs_for_mask(mask)

    @triton.testing.perf_report(configs)
    def bench_attention(provider, time, batch, heads, dim, dtype, mask: ChillMask):
        torch._dynamo.reset()
        device = "cuda"
        torch.set_float32_matmul_precision("highest")

        q = (
            torch.randn((batch, heads, time, dim), dtype=dtype, device=device)
            .normal_(mean=0.0, std=0.01)
            .requires_grad_()
        )
        k = (
            torch.randn((batch, heads, time, dim), dtype=dtype, device=device)
            .normal_(mean=0.0, std=0.01)
            .requires_grad_()
        )
        v = (
            torch.randn((batch, heads, time, dim), dtype=dtype, device=device)
            .normal_(mean=0.0, std=0.01)
            .requires_grad_()
        )

        lens = None
        autotune = "autotune" in provider
        cudagraph_available = True

        operation = None
        args = dict()

        if "chill" in provider:
            operation = chill_attention
            args = dict(
                q=q,
                k=k,
                v=v,
                lens=lens,
                mask=mask,
                autotune=autotune,
            )
        elif "torch" in provider:
            operation = lambda **kwargs: _chill_reference_naive(**kwargs)[0]
            args = dict(mask=mask, q=q, k=k, v=v, lens=lens)
        elif "flex" in provider:
            flex_mask = mask.make_flex_mask(time)
            operation = flex_attention
            args = dict(query=q, key=k, value=v, block_mask=flex_mask)

        assert operation is not None
        fn = lambda: operation(**args)
        if "compile" in provider:
            fn = lambda: torch.compile(
                operation, mode="max-autotune-no-cudagraphs" if autotune else None
            )(**args)

        ref, res_mask = _chill_reference_naive(mask, q, k, v, lens)
        ref, res_mask = ref.cuda(), res_mask.cuda()

        if "bwd" in provider:
            cudagraph_available = False
            do = (
                torch.randn(
                    (batch, heads, time, dim), dtype=torch.float32, device=device
                ).normal_(mean=0.0, std=0.1)
                * res_mask
            )

            def fn_back(fn):
                res = fn()
                res.backward(do)
                return res, q.grad, k.grad, v.grad

            fn = functools.partial(fn_back, fn=fn)
        else:

            def fn_forward(fn):
                with torch.no_grad(), torch.inference_mode():
                    return fn()

            fn = functools.partial(fn_forward, fn=fn)

        print(f"Starting {provider}")
        if "bwd" in provider:
            ref.backward(do)
            dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
            q.grad, k.grad, v.grad = [None] * 3

            actual, dq, dk, dv = fn()
            dq, dk, dv = dq.clone(), dk.clone(), dv.clone()
        else:
            actual = fn()

        actual = actual * res_mask.broadcast_to(actual.shape)

        atol = 3e-3
        torch.testing.assert_close(
            actual,
            ref,
            atol=atol,
            rtol=0,
            msg=lambda x: f"error in {provider}\n{x}",
        )

        if "bwd" in provider and not "flex" in provider:  # flex is failing
            for i, (d_ref, d_tri) in enumerate(
                [(dq_ref, dq), (dk_ref, dk), (dv_ref, dv)]
            ):
                atol = 1e-2
                torch.testing.assert_close(
                    d_tri,
                    d_ref,
                    atol=atol,
                    rtol=0,
                )

        with torch.inference_mode() if "fwd" in provider else contextlib.nullcontext():
            bench = triton.testing.do_bench
            if cudagraph_available:
                bench = triton.testing.do_bench_cudagraph
            else:
                print("bench with disabled cudagraph")
            ms = bench(fn, rep=1000, return_mode="mean", grad_to_none=(q, k, v))

        mask_tensor = mask.make_mask(time)
        time_sq = time * time * (mask_tensor.sum().item() / mask_tensor.numel())
        ATTN_FLOPS = (4 * batch * heads * time_sq * dim) / 1e12

        if "bwd" in provider:
            ATTN_FLOPS *= 2.5

        return ms

    bench_attention.run(
        save_path=f"{os.path.dirname(os.path.realpath(__file__))}/results",
        print_data=True,
    )
