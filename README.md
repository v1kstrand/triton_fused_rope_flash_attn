
# Triton FlashAttention-Style Scaled Dot Product Attention (with fused RoPE)

Custom [Triton](https://github.com/triton-lang/triton) kernels for Scaled Dot Product Attention (SDPA), including both forward and backward passes, designed for Vision Transformer (ViT)–style workloads and small to medium sequence lengths.

---

## Overview

This repository contains a custom FlashAttention-style implementation of Scaled Dot Product Attention (SDPA) in Triton:

- Fully custom **forward** and **backward** passes (gradients for query (Q), key (K), and value (V) are computed in Triton).
- Online softmax with log-sum-exp for stability in half precision.
- Tiling and autotuning tuned for ViT-like shapes (for example, sequence length around 197).

The goal is to serve both as:

- A **practical kernel** you can plug into PyTorch, and  
- A **readable reference** for learning how to implement attention in Triton.

---

## Features

- **Full SDPA pipeline**
  - Forward: attention scores, online softmax, and output.
  - Backward: gradients for Q, K, V (no fallback to PyTorch autograd for core math).

- **Numerical stability**
  - Online softmax with running max and log-sum-exp in FP32.
  - Accumulation in FP32, cast back to original dtype (for example, `torch.bfloat16`).

- **Triton-specific optimizations**
  - Blocked tiling over queries and keys/values (for example, `BLOCK_Q` × `BLOCK_KV`).
  - Swizzled program IDs (for example, `GROUP_M` or `GROUP_N`) to improve load balancing.
  - Autotuning over:
    - Block sizes (for example, `BLOCK_Q`, `BLOCK_KV`)
    - Number of warps
    - Number of stages

## Requirements

* Python: `3.10+`
* PyTorch: `2.7+`
* Triton: `3.3+`
* CUDA: `cu118+` (tested on A100 80GB)

---

## Usage

### Getting the latest changes locally

If you cloned this repository before the RoPE support was added, make sure you
have the latest commit on the active branch:

```bash
git pull
```

You should see the RoPE-specific entry points and benchmark script listed at the
root of the repo:

- `flash_attn.py` — dispatcher exposing `flash_attention(..., impl="fa_rope")`
  alongside the baseline kernel.
- `rope_flash_attn_kernel.py` — fused Triton kernel with in-kernel RoPE.
- `bench_rope_flash_attn.py` — benchmark entry point for the RoPE kernel.

With those files present you are on the updated version that includes fused
RoPE support.

### Basic example (drop-in SDPA — Scaled Dot Product Attention)

```python
import torch
from triton_flash_attention.vit_fa_triton import sdpa_triton_fa

B, H, N, D = 2, 8, 197, 64
dtype = torch.bfloat16
device = "cuda"

q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

o = sdpa_triton_fa(q, k, v) 

loss = o.sum()
loss.backward()

print("Output shape:", o.shape)
print("Grad q mean:", q.grad.float().abs().mean().item())
```

### RoPE-enabled Triton FlashAttention

The fused RoPE kernel lives in `rope_flash_attn_kernel.py` and is dispatched
through `flash_attn.flash_attention`. The baseline (non-RoPE) kernel remains the
default; pass `impl="fa_rope"` to enable RoPE and supply the cosine/sine table
expected by the Triton kernel.

```python
import torch
from flash_attn import CosSinTable, flash_attention

B, H, N, D = 2, 8, 197, 64
dtype = torch.bfloat16
device = "cuda"

q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

cos_sin = CosSinTable(base=10000.0, H_img=14, D=D, device=device)
o = flash_attention(q, k, v, impl="fa_rope", cos_sin=cos_sin)
o.sum().backward()
```

### Benchmarks for fused RoPE

The RoPE-specific benchmark is available in `bench_rope_flash_attn.py` and
uses the same shapes as the baseline benchmarks by default. For example:

```bash
python bench_rope_flash_attn.py --batch 1024 --heads 6 --seq 197 --dim 64 --mode fwdbwd
```

This compares the fused RoPE Triton kernel against the PyTorch SDPA Flash and
memory-efficient backends, reporting median latency and elements/second.

---

## Correctness

We check both **forward outputs** and **backward gradients** against PyTorch SDPA for different backends (`math`, `mem`, `flash`).

Each block below shows the maximum and mean absolute error between `sdpa_triton_fa` and the corresponding PyTorch backend, using the same `(B, H, N, D)` and dtype as above.

### Versus Torch math backend

```text
[O ] max_abs_err = 9.194970e-03, mean_abs_err = 2.914664e-04
[dQ] max_abs_err = 1.565409e-02, mean_abs_err = 3.622163e-04
[dK] max_abs_err = 2.064800e-02, mean_abs_err = 3.542830e-04
[dV] max_abs_err = 1.052654e-02, mean_abs_err = 2.979040e-04
```

### Versus Torch memory-efficient backend

```text
[O ] max_abs_err = 3.939509e-03, mean_abs_err = 1.471445e-04
[dQ] max_abs_err = 1.608646e-02, mean_abs_err = 2.793679e-04
[dK] max_abs_err = 1.820827e-02, mean_abs_err = 2.731781e-04
[dV] max_abs_err = 3.893614e-03, mean_abs_err = 1.269000e-04
```

### Versus Torch Flash backend

```text
[O ] max_abs_err = 3.939509e-03, mean_abs_err = 1.974907e-04
[dQ] max_abs_err = 1.421165e-02, mean_abs_err = 1.595823e-04
[dK] max_abs_err = 1.820827e-02, mean_abs_err = 1.578689e-04
[dV] max_abs_err = 3.893614e-03, mean_abs_err = 1.269283e-04
```

These error levels are consistent with floating-point differences between implementations using different fusion patterns and accumulation orders.

---

## Reproducing these numbers

The benchmarks and comparisons above can be reproduced with helper functions in `triton_utils.py`:

```python
import torch
from triton_flash_attention.triton_utils import bench_sdpa_throughput, compare_sdpa_variants
from triton_flash_attention.vit_fa_triton import sdpa_triton_fa


dtype = torch.float32
device = "cuda"

B, H, D = 1024, 6, 64
N = 197

Q = torch.randn(B, H, N, D, device=device, dtype=dtype)
K = torch.randn(B, H, N, D, device=device, dtype=dtype)
V = torch.randn(B, H, N, D, device=device, dtype=dtype)

# 1) Forward + backward correctness against different PyTorch SDPA backends
compare_sdpa_variants(
    Q, K, V,
    sdpa_triton_fa,
    dO=None,  
    kernels=["math", "mem", "flash"],
)

# 2) Throughput / latency table for Triton vs PyTorch SDPA variants
bench_sdpa_throughput(
    Q, K, V,
    mode="fwdbwd",          # forward + backward
    print_table=True,
    variants=(sdpa_triton_fa, "math", "mem", "flash"),
)
```

* `compare_sdpa_variants` runs your kernel and the chosen PyTorch SDPA backends, then prints max and mean absolute errors for:

  * Output `O`
  * Gradients `dQ`, `dK`, `dV`
* `bench_sdpa_throughput` measures median latency and effective elements/second for each variant and prints a compact table like the one above.



### License
This project is licensed under the MIT License. 

