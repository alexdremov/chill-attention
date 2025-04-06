import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

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

masks_to_test = [
    FullChillMask(),
    CausalChillMask(),
    SlidingWindowChillMask(10, 4),
    ChunkwiseChillMask(10, 4),
    PrefixLMChillMask(10),
]


def make_lens(lens, B, T):
    if lens == "none":
        lens = None
    elif lens == "tricky":
        tricky_lens = [
            1,
            2,
            5,
            T + 1,
            T,
            max(T // 2, 1),
            max(T // 4, 1),
        ]
        lens = torch.tensor(
            np.random.choice(tricky_lens, B), dtype=torch.int32, device="cuda"
        )
    else:
        lens = torch.randint(1, T + 1, (B,), dtype=torch.int32, device="cuda")
    return lens


@pytest.fixture(autouse=True)
def run_around_tests(request):
    devices = list(range(torch.cuda.device_count()))
    torch.cuda.set_device(devices[int(hash(request.node.callspec.id)) % len(devices)])
    torch._dynamo.reset()

    torch.manual_seed(20)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()
    yield


@pytest.mark.parametrize("mask", masks_to_test, ids=lambda x: str(x))
def test_masks_verify(mask):
    mask.verify(1024)


@pytest.mark.parametrize("mask", masks_to_test, ids=lambda x: str(x))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=lambda x: str(x))
@pytest.mark.parametrize(
    "lens", ["none", "tricky", "random"], ids=lambda x: f"lens-{x}"
)
@pytest.mark.parametrize(
    "noncontiguous", [False, True], ids=lambda x: f"noncontiguous-{x}"
)
@pytest.mark.parametrize("HEAD_DIM", [16, 128], ids=lambda x: f"dim-{x}")
@pytest.mark.parametrize("B", [1, 40, 64], ids=lambda x: f"batch-{x}")
@pytest.mark.parametrize("H", [1, 6, 8], ids=lambda x: f"heads-{x}")
@pytest.mark.parametrize("T", [1, 10, 16, 800, 1025], ids=lambda x: f"time-{x}")
@pytest.mark.parametrize("autotune", [False], ids=lambda x: f"autotune-{x}")
def test_simple_chill_forward(
    mask,
    dtype,
    lens,
    noncontiguous,
    HEAD_DIM,
    B,
    H,
    T,
    autotune,
):
    if os.environ.get("TRITON_INTERPRET") == "1" and dtype == torch.bfloat16:
        pytest.skip("skipping bf16 in interpreter mode")
        return

    if autotune and not (
        B
        in {
            1,
        }
        and H
        in {
            1,
        }
        and T in {10, 1025}
    ):
        pytest.skip("reduced set for autotune")
        return

    q, k, v = [
        torch.testing.make_tensor(
            (B, H, T, HEAD_DIM),
            dtype=dtype,
            device="cuda",
            noncontiguous=noncontiguous,
            low=-0.1,
            high=0.1,
        )
        for _ in range(3)
    ]
    for i in (q, k, v):
        i.normal_()

    lens = make_lens(lens, B, T)
    reference, res_mask = _chill_reference_naive(
        mask, q.float(), k.float(), v.float(), lens=lens
    )
    reference = reference.to(q.dtype)
    chill = chill_attention(q, k, v, mask=mask, lens=lens, autotune=autotune) * res_mask

    atol = 3e-3
    if dtype == torch.float32:
        atol = 7e-6

    torch.testing.assert_close(
        actual=chill,
        expected=reference,
        atol=atol,
        rtol=0,
    )


@pytest.mark.parametrize("mask", masks_to_test, ids=lambda x: str(x))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=lambda x: str(x))
@pytest.mark.parametrize(
    "lens", ["none", "tricky", "random"], ids=lambda x: f"lens-{x}"
)
@pytest.mark.parametrize(
    "noncontiguous", [False, True], ids=lambda x: f"noncontiguous-{x}"
)
@pytest.mark.parametrize("HEAD_DIM", [16, 128], ids=lambda x: f"dim-{x}")
@pytest.mark.parametrize("B", [1, 40, 64], ids=lambda x: f"batch-{x}")
@pytest.mark.parametrize("H", [1, 6, 8], ids=lambda x: f"heads-{x}")
@pytest.mark.parametrize("T", [1, 10, 16, 800, 1025], ids=lambda x: f"time-{x}")
@pytest.mark.parametrize("autotune", [False], ids=lambda x: f"autotune-{x}")
def test_simple_chill_backward(
    mask,
    dtype,
    lens,
    noncontiguous,
    HEAD_DIM,
    B,
    H,
    T,
    autotune,
):
    torch._dynamo.reset()

    torch.manual_seed(20)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()

    if os.environ.get("TRITON_INTERPRET") == "1" and dtype == torch.bfloat16:
        pytest.skip("skipping bf16 in interpreter mode")
        return

    if autotune and not (
        B
        in {
            1,
        }
        and H
        in {
            1,
        }
        and T in {10, 16}
    ):
        pytest.skip("reduced set for autotune")
        return

    q, k, v = [
        torch.testing.make_tensor(
            (B, H, T, HEAD_DIM),
            dtype=dtype,
            device="cuda",
            noncontiguous=noncontiguous,
            low=-0.1,
            high=0.1,
        )
        for _ in range(3)
    ]
    for i in (q, k, v):
        i.normal_().requires_grad_()
    lens = make_lens(lens, B, T)

    reference, res_mask = _chill_reference_naive(
        mask, q.float(), k.float(), v.float(), lens=lens
    )

    dout = torch.testing.make_tensor(
        (B, H, T, HEAD_DIM),
        dtype=torch.float32,
        device="cuda",
        noncontiguous=noncontiguous,
        low=-0.1,
        high=0.1,
    )
    dout.normal_().requires_grad_()
    dout = dout * res_mask.broadcast_to(dout.shape)

    reference.backward(dout.float())
    reference = reference.to(q.dtype)

    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    for i in (q, k, v):
        assert i.isfinite().all()

    chill = chill_attention(q, k, v, mask=mask, lens=lens, autotune=autotune)
    chill.backward(dout.float())

    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    chill = chill * res_mask.broadcast_to(chill.shape)
    atol = 3e-3
    if dtype == torch.float32:
        atol = 7e-6

    torch.testing.assert_close(
        actual=chill,
        expected=reference,
        atol=atol,
        rtol=0,
    )

    for i, (d_ref, d_tri) in enumerate(
        [(ref_dv, tri_dv), (ref_dk, tri_dk), (ref_dq, tri_dq)]
    ):
        atol = 1e-2
        if dtype == torch.float32:
            atol = 5e-5

        torch.testing.assert_close(
            d_tri,
            d_ref,
            atol=atol,
            rtol=0,
            msg=lambda x: f"error in d{'vkq'[i]}\n{(~torch.isfinite(d_tri)).sum().item() = }, {(~torch.isfinite(d_ref)).sum().item() = }\n{x}",
        )


@pytest.mark.parametrize("mask", masks_to_test, ids=lambda x: str(x))
def test_flex_mask_consistent(mask: ChillMask):
    max_pos = 1024
    flex_mask = mask.make_flex_mask(max_pos)
    if flex_mask is None:
        pytest.skip("None flex mask")
        return

    real_mask = mask.make_mask(max_pos)
    flex_mask_tensor = create_mask(flex_mask.mask_mod, None, None, max_pos, max_pos)[
        0, 0
    ]

    assert (real_mask == flex_mask_tensor).all().item()
