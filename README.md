# Chill Attention
> A fast, flexible, and chill sparse flash attention kernel

Chill Attention provides an efficient sparse flash attention operator with optimized attention masking for exact attention calculation.


## Features

- ⚡ **High-Performance Kernels**

  A Triton-based sparse flash attention implementation with custom masks that outperforms **PyTorch SDPA attention** and is comparable to or better than **FlexAttention**, depending on the use case.

- 🎭 **Flexible Masking Patterns**

  Supports custom-defined attention masks. Examples include `FullChillMask`, `CausalChillMask`, `SlidingWindowChillMask`, `ChunkwiseChillMask`, and `PrefixLMChillMask`. **Define your own mask with just three simple methods.**

- 🏎️ **Kernel Tuning**

  Optimized default configurations for different hardware (A100, H100). Autotuning is also available to optimize performance for custom masks.

- 🎯 **Multiple Precision Types**

  Supports FP32, FP16, and BF16.

- 🚀 **PyTorch 2 Integration**

  Supports `torch.compile` since the kernels are defined as custom PyTorch operators.

## Why So Chill?

Masking and sparsity patterns are calculated in-place, requiring no additional memory loads compared to a simple attention kernel. This is the main difference from PyTorch's **FlexAttention**.

For simple parametric masking, **FlexAttention** may be overkill. That's where **ChillAttention** comes into play.

## Installation

```bash
pip install chill-attention
```

Or install from source:

```bash
git clone https://github.com/alexdremov/chill-attention.git
cd chill-attention
uv sync --all-extras
uv pip install -e .
```

## Requirements

- Python ≥ 3.11, < 3.13

- PyTorch >= 2.7.0

- CUDA-compatible GPU

## Usage

### Basic Example

```python
import torch
from chill_attention import chill_attention, CausalChillMask

# Create input tensors (batch_size, num_heads, seq_len, head_dim)
batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")

# Create a mask
mask = CausalChillMask()

# Compute attention
output = chill_attention(q, k, v, mask=mask)
```

### Using Different Masks

```python
from chill_attention import (
    FullChillMask,
    CausalChillMask,
    SlidingWindowChillMask,
    ChunkwiseChillMask,
    PrefixLMChillMask,
)

# Full attention (standard transformer)
full_mask = FullChillMask()

# Causal attention
causal_mask = CausalChillMask()

# Sliding window attention (local attention within a window)
# Args: left_context, right_context
sliding_mask = SlidingWindowChillMask(
  left_context=64,
  right_context=32,
)

# Chunkwise attention (block-wise attention)
# Args: chunk_size, number_of_preceding_chunks_to_attend
chunk_mask = ChunkwiseChillMask(
  context_size=128,
  back_contexts=2,
)

# Prefix LM attention (bidirectional attention in prefix, causal elsewhere)
# Args: prefix_size
prefix_mask = PrefixLMChillMask(
  prefix_size=128,
)
```

## Creating Your Own Mask

Creating a custom mask can be as simple as implementing three methods. To do this, you need to define:

- **mask**: The mask for the provided query (q) and key (k) indices.
- **q_range_for_k**: The range of q positions for a specified k position.
- **k_range_for_q**: The range of k positions for a specified q position.

In essence, the last two methods define the sparsity of your mask.

A simple example would be:

```python
class SlidingWindowChillMask(ChillMask):
    """
    Sliding window attention mask with configurable left and right context.
    """

    def __init__(self, left_context, right_context):
        super().__init__((left_context, right_context))

    # These are Triton methods that control the mask's behavior
    @staticmethod
    def mask(q: tl.tensor, k: tl.tensor, args) -> tl.tensor:
        """
        Sliding window attention - each query attends to positions within a window.
        """
        left_context, right_context = args

        diff = q[:, None] - k[None, :]
        return ((diff <= left_context) & (diff >= 0)) | (
            (diff >= -right_context) & (diff <= 0)
        )

    @staticmethod
    def q_range_for_k(k: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """
        For a key at position k, determine which queries can attend to it.
        """
        left_context, right_context = args
        return max(0, k - right_context), min(k + left_context, seq_len - 1)

    @staticmethod
    def k_range_for_q(q: int, seq_len: tl.tensor, args) -> tuple[tl.tensor, tl.tensor]:
        """
        For a query at position q, determine which keys it can attend to.
        """
        left_context, right_context = args
        return max(0, q - left_context), min(q + right_context, seq_len - 1)
```

Additional methods to optimize performance are also available:
- `q_lims_continuous`, `k_lims_continuous` — Optimize the computation of tiling ranges (True by default).
- `has_full_blocks`, `is_full_block` — Optimize performance for fully unmasked blocks.

### Verification

Creating `q_range_for_k` and `k_range_for_q` can be complex. However, you can verify their correctness manually for a fixed number of positions.

To do this, call the mask's `verify` method:

```python
from chill_attention import SlidingWindowChillMask

# Create a mask
mask = SlidingWindowChillMask(10, 20)

# Verifying the first 512 positions
# Finishes successfully or raises an assertion error
mask.verify(512)
```

If verification fails, you can visualize the mask using the `plot` method. The plot will also display your analytical `q_range_for_k` and `k_range_for_q` predictions, so you can easily identify where the mistakes are.

## Visualizing Masks

You can visualize mask patterns to better understand their behavior (requires matplotlib):

```python
from chill_attention import ChunkwiseChillMask

# Create a mask
mask = ChunkwiseChillMask(
  context_size=16,
  back_contexts=3
)

# Create a visualization for the first 128 positions
fig = mask.plot(128)
fig.savefig("chunkwise_mask.png")
```

## Limitations

Since only `int`, `float`, and `bool` can be used as parameters, no additional tensors can be passed in the `args` tuple. However, this is a potential area for future improvement.

If your mask structure is parameterized purely by query (q) and key (k) indices, along with some additional constants, then this kernel is for you.

## Benchmarking

The following plots show a comparison with **FlexAttention** for several attention masks. **PyTorch SDPA** does not take advantage of mask sparsity and therefore performs poorly (not shown). The code for these benchmarks is available in `benchmark/benchmark.py`. There are cases where the kernel performs worse than FlexAttention, but I believe this can be improved through kernel optimizations, primarily for the backward pass.

Some notable results are:

`bwd SlidingWindowChillMask(16, 16)`

<img src="https://github.com/alexdremov/chill-attention/blob/main/benchmark/results/result-nv-9_0-bwd-small-SlidingWindowChillMask(16, 16)-dim-64-heads-12-batch-64.png">

`fwd ChunkwiseChillMask(16, 8)`

<img src="https://github.com/alexdremov/chill-attention/blob/main/benchmark/results/result-nv-9_0-fwd-small-ChunkwiseChillMask(16, 8)-dim-64-heads-12-batch-64.png">

`fwd PrefixLMChillMask(128)`

<img src="https://github.com/alexdremov/chill-attention/blob/main/benchmark/results/result-nv-9_0-fwd-small-PrefixLMChillMask(128)-dim-64-heads-12-batch-64.png">


However, in some cases, further optimization is still needed (especially for the backward pass on H100 hardware).

`bwd CausalChillMask()`

<img src="https://github.com/alexdremov/chill-attention/blob/main/benchmark/results/result-nv-9_0-bwd-small-CausalChillMask()-dim-64-heads-12-batch-64.png">

`bwd ChunkwiseChillMask(16, 8)`

<img src="https://github.com/alexdremov/chill-attention/blob/main/benchmark/results/result-nv-9_0-bwd-small-ChunkwiseChillMask(16, 8)-dim-64-heads-12-batch-64.png">

`bwd PrefixLMChillMask(128)`

<img src="https://github.com/alexdremov/chill-attention/blob/main/benchmark/results/result-nv-9_0-bwd-small-PrefixLMChillMask(128)-dim-64-heads-12-batch-64.png">


## Future Improvements

- This code has not been profiled in-depth. Therefore, I believe there are simple Triton tricks (like load reordering) that could yield simple speed-ups.
- Adding support for custom tensors would make ChillAttention almost as powerful as FlexAttention, while still being able to achieve lower overheads when needed.
- The backward pass kernel is less optimized than the forward pass one, so some performance issues may be present.
- H100 performance needs to be optimized (e.g., by using Triton's new TMA instructions).
- Migrate to PyTorch's native Triton operators as soon as they become stable.

## License

GPL-3.0 License

## Citation

If you use this library in your research, please cite:

```
@software{dremov2025chillattention,
  author = {Aleksandr Dremov},
  title = {Chill Attention: A fast, flexible, and chill sparse flash attention kernel},
  year = {2025},
  url = {https://github.com/alexdremov/chill-attention}
}
```
