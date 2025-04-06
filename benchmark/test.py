import os
import sys

import torch

sys.path.insert(0, f"{os.path.dirname(os.path.realpath(__file__))}/..")

from chill_attention import (
    CausalChillMask,
    ChunkwiseChillMask,
    PrefixLMChillMask,
    chill_attention,
)

q, k, v = torch.randn(
    3,
    64,
    16,
    3000,
    64,
    dtype=torch.float16,
    device="cuda",
    requires_grad=True,
)
do = torch.randn_like(q)
mask = CausalChillMask()

for i in range(5):
    if i > 1:
        torch.cuda.cudart().cudaProfilerStart()
    res = chill_attention(
        q=q,
        k=k,
        v=v,
        lens=None,
        mask=mask,
        autotune=True,
    )
torch.cuda.cudart().cudaProfilerStop()
torch.cuda.synchronize()


# # mask = PrefixLMChillMask(128)
# # for i in range(5):
# #     if i > 1:
# #         torch.cuda.cudart().cudaProfilerStart()
# #     res = chill_attention(
# #         q=q,
# #         k=k,
# #         v=v,
# #         lens=None,
# #         mask=mask,
# #         autotune=True,
# #     )
# # torch.cuda.cudart().cudaProfilerStop()
# # torch.cuda.synchronize()
