
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

## Benchmarks

All benchmarks in this section were run on:

* **Shape:** `(B, H, N, D) = (1024, 6, 197, 64)`
* **Dtype:** `torch.bfloat16`
* **Device:** `NVIDIA A100 80GB`
* **Mode:** forward + backward (`fwdbwd`)
* **Metric:** median latency over multiple runs, and effective elements/second (attention “elements” processed per second).

We compare:

* `sdpa_triton_fa` — this Triton FlashAttention-style kernel
* `Torch math` — PyTorch Scaled Dot Product Attention (SDPA — Scaled Dot Product Attention) math backend
* `Torch mem` — PyTorch SDPA memory-efficient backend
* `Torch flash` — PyTorch SDPA FlashAttention backend

### Throughput and latency

| Variant          | Latency (ms, median) | Elements / s   |
| ---------------- | -------------------- | -------------- |
| `sdpa_triton_fa` | **7.796**            | **1.987×10¹⁰** |
| Torch math       | 22.238               | 6.967×10⁹      |
| Torch mem        | 10.781               | 1.437×10¹⁰     |
| Torch flash      | 8.090                | 1.915×10¹⁰     |

For this ViT (Vision Transformer)-style configuration, the custom Triton kernel is:

* ~**2.9× faster** than the PyTorch math backend
* ~**1.4× faster** than the PyTorch memory-efficient backend
* Slightly **faster than PyTorch FlashAttention**, with comparable throughput

(Exact speedups will vary by GPU, driver, and PyTorch/Triton versions.)

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

You can change `dtype`, `(B, H, N, D)`, or `mode` (for example, `"fwd"` or `"bwd"`) to explore other regimes and verify that performance and accuracy behave as expected.

### RoPE benchmark

To benchmark the fused RoPE Triton kernel against PyTorch SDPA backends (with
RoPE applied on the Python side), run:

```bash
python bench_rope_flash_attn.py --batch 1024 --heads 6 --seq 197 --dim 64 --mode fwdbwd
```

The script prints median latency and effective elements/second for each variant,
mirroring the format of the existing FlashAttention benchmark helpers.


---

## Implementation details and intuition

This section gives a high-level picture of how the Triton kernels are structured.
It is meant to be readable first, then you can dive into the code.

### Intuition: forward pass (`_attn_fwd`)

The forward kernel `_attn_fwd` computes the standard attention operation

* “scores” = Q·Kᵀ (query times key transpose)
* apply softmax over keys
* multiply by V (values) to get the output

but does it in a memory- and cache-friendly way.

Roughly, each Triton program instance:

1. **Owns a small block of queries.**
   For example, a tile of shape `(BLOCK_Q, HEAD_DIM)` for a fixed batch and head.

2. **Streams over all keys and values in blocks.**
   Instead of materializing the full score matrix S (of shape `N × N`), it reads a `(BLOCK_KV, HEAD_DIM)` tile of K and V at a time, computes the partial scores between the local Q block and this K block, and immediately folds that into an **online softmax**.

3. **Uses online softmax with running statistics.**
   For each query row, the kernel keeps track of:

   * A running maximum of logits (for numerical stability).
   * A running normalized sum for the softmax denominator.
   * A running accumulator for the output row.

   When a new block of scores arrives, the kernel updates these running quantities, so it never needs to store the full scores S explicitly.

4. **Accumulates the output in FP32.**
   The partial contributions from each K/V block are accumulated into an output buffer in FP32. At the end, the result is cast back to the requested dtype (for example `bfloat16`).

5. **Writes out per-token statistics for backward.**
   Along the way, the kernel stores compact information (for example, per-query max/normalizer) into `M` (and optionally `D` or `L`, depending on your implementation).
   These saved tensors allow the backward pass to reconstruct the softmax probabilities without recomputing everything from scratch.

The result is a forward pass that:

* Never forms the full attention matrix in memory.
* Keeps most data in on-chip memory and registers.
* Is numerically stable even in half precision.

### Intuition: backward pass

`_attn_bwd_preprocess`, `_attn_bwd_dk_dv`, `_attn_bwd_dq`

The backward pass is split into three kernels for clarity and performance:

#### Preprocess: softmax “delta” (`_attn_bwd_preprocess`)

The kernel `_attn_bwd_preprocess` computes a per-token scalar that is needed for the softmax gradient. Intuitively, for each query position it:

* Looks at the upstream gradient `dO` (gradient of the loss with respect to the output) and the forward output `O`.
* Computes a summary term like “how much did this row’s softmax contribute overall”, which is used to form the softmax Jacobian efficiently in later kernels.

This is done once and stored in a compact tensor `D` that the next kernels can reuse.

#### Gradients with respect to K and V (`_attn_bwd_dk_dv`)

The kernel `_attn_bwd_dk_dv` computes gradients for keys and values:

* Each Triton program instance **fixes a block of K/V positions** (for example, a range of key indices).
* For that fixed K/V block, it **loops over all query blocks**.

Inside the loop:

1. It **rebuilds the local part of the score matrix** S (or equivalently the logits) for this Q×K tile by recomputing Q·Kᵀ. This is similar to the forward pass, but limited to local tiles.
2. Using the saved forward statistics (for example, `M`) and the precomputed `D`, it reconstructs the needed softmax gradients for that tile.
3. It uses those to accumulate:

   * `dV` by mixing `dO` with the softmax probabilities.
   * `dK` by mixing the softmax gradient with Q.

Because each program “owns” a unique K/V block and only writes to its own rows of `dK` and `dV`, there is **no need for atomic operations**. This keeps the kernel simple and fast, even though it recomputes the local scores S one more time.

#### Gradients with respect to Q (`_attn_bwd_dq`)

The kernel `_attn_bwd_dq` is symmetric to `_attn_bwd_dk_dv`, but with roles reversed:

* Each Triton program instance **fixes a block of Q positions**.
* For that block of queries, it **loops over all K/V blocks**.

Inside the loop:

1. It recomputes the same Q·Kᵀ tiles as in the other kernels.
2. Uses the saved forward statistics and `D` to reconstruct softmax gradients.
3. Accumulates contributions to `dQ` by mixing:

   * The softmax gradient.
   * The values V and upstream gradient `dO`.

Again, each program “owns” its unique slice of `dQ`, so there is no need for atomic additions.

Putting it all together:

* `_attn_bwd_preprocess` computes the softmax-related “delta” once.
* `_attn_bwd_dk_dv` and `_attn_bwd_dq` both **rebuild S locally from Q and K**, but in complementary directions (fixed K/V vs fixed Q).
* This design trades a small amount of extra compute (recomputing S) to avoid atomics and large intermediate storage, which tends to give **better throughput and simpler code** on modern GPUs.

---

## Limitations and future work

Current limitations:

* No support yet for:

  * Causal masking
  * Arbitrary attention masks
  * Dropout
  * Attention bias (for example, relative position bias or Continuous Position Bias)

Potential future extensions:
* Causal and windowed attention variants.
* Fused support for:
  * RoPE
  * Continuous position bias (CPB)

### License
This project is licensed under the MIT License. 

