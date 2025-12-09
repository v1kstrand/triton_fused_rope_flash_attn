# Triton FlashAttention-Style Scaled Dot Product Attention with Fused 2D RoPE

Custom [Triton](https://github.com/triton-lang/triton) kernels for Scaled Dot Product Attention (SDPA), including both forward and backward passes, with **fused 2D axial RoPE (Rotary Positional Embedding)** for Vision Transformer (ViT)–style workloads.

Designed and tuned for patch-grid ViTs (for example, 14×14 patches + optional CLS token) and small–medium sequence lengths.

---

## Overview

This repository contains a FlashAttention-style SDPA implementation in Triton with **RoPE fused directly inside the kernel**:

* Fully custom **forward** and **backward** passes (gradients for query (Q), key (K) and value (V) are computed in Triton).
* Online softmax with log-sum-exp in FP32 for stability in half precision.
* **2D axial RoPE** applied in-pair to Q and K inside the kernel, using a precomputed cosine/sine table over the patch grid.
* Layout and tiling tuned for ViT-like shapes, for example:

  * `SEQ_LEN = 1 + H_img * H_img` (CLS + 2D patch grid),
  * `HEAD_DIM % 4 == 0` (required for splitting into 2D RoPE pairs). 

The goal is to provide:

* A **practical fused RoPE attention kernel** you can plug into PyTorch, and
* A **readable reference** for implementing FlashAttention-style SDPA with 2D RoPE in Triton (forward + backward).

---

## Features

* **Full SDPA with fused 2D RoPE**

  * Forward: Q/K rotation via 2D axial RoPE, attention logits, online softmax and output.
  * Backward: gradients for Q, K, V entirely in Triton, including the RoPE rotation/unrotation in pair space.

* **2D axial RoPE (Rotary Positional Embedding)**

  * Positions are laid out on an `H_img × H_img` grid (for example, 14×14 patches → 196 positions).
  * Optional CLS token at index 0 that **bypasses RoPE** (identity rotation).
  * Even/odd channel pairs are mapped to x/y axes, so the first half of pairs encode x-rotations and the second half encode y-rotations.

* **Numerical stability**

  * Online softmax with running max + log-sum-exp in FP32.
  * Accumulation in FP32, then cast back to the original dtype (for example, `torch.bfloat16`).

* **Triton-specific optimizations**

  * Blocked tiling over queries (`BLOCK_Q`) and keys/values (`BLOCK_KV`).
  * Swizzled program IDs (`GROUP_M`, `GROUP_N`) to improve load balancing for both Q- and KV-tiles.
  * Autotuning over:

    * Block sizes (`BLOCK_Q`, `BLOCK_KV`)
    * Number of warps
    * Number of stages

---

## Requirements

* Python: `3.10+`
* PyTorch: `2.7+`
* Triton: `3.3+`
* CUDA: `cu118+` (tested on A100 80GB)

---

## Usage

### Cosine/sine table for 2D RoPE

The kernel expects a precomputed pairwise cosine/sine table over the 2D patch grid. This is wrapped in a small helper module:

```python
from fa_rope_full import CosSinTable  # adjust import to your repo layout

H_img = 14       # 14 x 14 patch grid
D = 64           # head dim (must be divisible by 4)
device = "cuda"

cos_sin = CosSinTable(base=100.0, H_img=H_img, D=D, device=device)
```

* `CosSinTable` builds **pairwise** RoPE tables:

  * `COSP`, `SINP`: `[N_pos, D2]` where `N_pos = H_img * H_img` and `D2 = HEAD_DIM // 2`.
* The Triton kernels consume these tables directly in their internal pair layout.

### Basic example (fused RoPE SDPA)

```python
import torch
from fa_rope_full import CosSinTable, sdpa_triton_fa_rope  # adjust import to your repo layout

B, H, H_img, D = 2, 8, 14, 64
N = 1 + H_img * H_img      # CLS + 2D grid
dtype = torch.float32
device = "cuda"

q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

cos_sin = CosSinTable(base=100.0, H_img=H_img, D=D, device=device)

# Fused FlashAttention-style SDPA + 2D RoPE
with torch.autocast("cuda", dtype=torch.bfloat16):
    o = sdpa_triton_fa_rope(q, k, v, cos_sin)

loss = o.sum()
loss.backward()

print("Output shape:", o.shape)
print("Grad q mean:", q.grad.float().abs().mean().item())
```

By default the kernel assumes:

* `HEAD_DIM % 4 == 0`
* `SEQ_LEN = 1 + H_img * H_img` (CLS at index 0) or `SEQ_LEN = H_img * H_img` when `has_cls=False` inside the autograd wrapper.

---

## Correctness

We check **forward outputs** and **backward gradients** against a PyTorch **FlashAttention (Flash SDPA)** reference implementation that applies the same 2D RoPE in Python.

Below are the maximum and mean absolute errors between the **Triton fused RoPE kernel** and the PyTorch Flash SDPA reference (same `(B, H, N, D)`, dtype and RoPE tables):

```text
[O ] max_abs_err = 2.343750e-02, mean_abs_err = 5.833786e-04
[dQ] max_abs_err = 2.619576e-02, mean_abs_err = 5.742905e-04
[dK] max_abs_err = 3.096879e-02, mean_abs_err = 5.423786e-04
[dV] max_abs_err = 1.741505e-02, mean_abs_err = 4.707939e-04
```

These error levels are consistent with expected floating-point differences between two independently fused implementations (different reduction orders and kernel fusion patterns).

---

## Benchmarks

All results below are from A100 80GB, ViT-like shapes, and a RoPE-enabled `vit_small_patch16_rope_224`-style configuration unless otherwise noted.

### PyTorch eager (no `torch.compile`)

In PyTorch **eager** mode (no `torch.compile`), the fused Triton RoPE kernel is competitive with the native PyTorch RoPE SDPA and is designed as a **drop-in, inspectable alternative**:

```text
dtype: bfloat16
backend: PyTorch eager
model: vit_small_patch16_rope_224
batch: 512
steps: 50
```

| Model                      | FA impl        | Batch | Steps | Time / step (ms) |
| -------------------------- | -------------- | ----- | ----- | ---------------- |
| vit_small_patch16_rope_224 | Triton RoPE FA | 512   | 50    | 179.5            |
| vit_small_patch16_rope_224 | Torch RoPE FA  | 512   | 50    | 145.2            |

So in eager mode the Triton kernel is on the same order of magnitude as PyTorch’s optimized SDPA with RoPE, despite being a fully custom implementation with fused 2D rotations and custom backward.

In end-to-end training setups, the Triton kernel can give a meaningful speedup over a purely Python-layer RoPE implementation (for example, on the order of **hundreds of milliseconds per step** when comparing unfused vs fused RoPE in eager mode), while remaining easy to read and modify.

### With `torch.compile` (Inductor)

For completeness, we also compare against `torch.compile(..., backend="inductor")` for the same attention shapes. Here we look at **kernel-level throughput** for several SDPA variants:

```python
results = {
    "flash":           {"ms": 10.75, "throughput": 1.44e10},
    "mem":             {"ms": 12.32, "throughput": 1.26e10},
    "math":            {"ms": 22.55, "throughput": 6.87e9},
    "sdpa_triton_fa_rope": {"ms": 17.62, "throughput": 8.79e9},
}
```

Rendered as a table:

| SDPA impl             | Time / step (ms) | Elements / s |
| --------------------- | ---------------- | ------------ |
| PyTorch Flash backend | 10.75            | 1.44 × 10¹⁰  |
| PyTorch Mem backend   | 12.32            | 1.26 × 10¹⁰  |
| PyTorch Math backend  | 22.55            | 6.87 × 10⁹   |
| Triton RoPE FA (this) | 17.62            | 8.79 × 10⁹   |

Key observations:

* In **compiled** mode, PyTorch’s Flash SDPA (without fused 2D RoPE) is extremely strong and reaches higher throughput than this first version of the fused Triton RoPE kernel.
* The Triton RoPE kernel lands between the PyTorch math and mem/flash implementations in terms of raw throughput.
* In actual compiled ViT training runs, this translates to PyTorch SDPA with RoPE reaching roughly the same speed as **non-RoPE FlashAttention**, while the Triton RoPE kernel improves less dramatically (for example, numbers around ~280 ms/step for PyTorch vs ~350 ms/step for Triton in one representative setup).

The takeaway:

> This project targets **PyTorch eager** and **kernel-level understanding** first. In that regime, the fused Triton RoPE kernel is competitive and useful. Under `torch.compile`, PyTorch’s SDPA stack is heavily optimized and remains the fastest option in this configuration.

---

## License

This project is licensed under the MIT License.
