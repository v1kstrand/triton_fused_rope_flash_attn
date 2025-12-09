from statistics import median
from itertools import combinations

import math
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from itertools import combinations

def compare_sdpa_variants(Q, K, V, triton_kernel, dO=None, kernels=("math", "mem", "flash")):
    def _torch_sdpa(q, k, v, *, backend: str):
        kernel = {
            "math" : SDPBackend.MATH,
            "mem" : SDPBackend.EFFICIENT_ATTENTION,
            "flash" : SDPBackend.FLASH_ATTENTION
        }[backend]

        dtype = torch.float32 if backend == "math" else torch.bfloat16
        with torch.autocast("cuda", dtype=dtype), sdpa_kernel(kernel):
            oh = F.scaled_dot_product_attention(q, k, v)
            oh.backward(dO.detach().clone())
        return oh

    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dtype == K.dtype == V.dtype
    B, H, M, D = Q.shape
    N = K.shape[2]
    if dO is None:
        dO = torch.randn(B, H, M, D, device=Q.device, dtype=Q.dtype)

    results = {}
    # --- Triton (baseline) ---
    q = Q.detach().clone().requires_grad_(True)
    k = K.detach().clone().requires_grad_(True)
    v = V.detach().clone().requires_grad_(True)
        
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = triton_kernel(q, k, v)
        o.backward(dO.detach().clone())
        results["triton"] = (o.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach())

    # --- PyTorch SDPA backends ---
    for name in kernels:
        q = Q.detach().clone().requires_grad_(True)
        k = K.detach().clone().requires_grad_(True)
        v = V.detach().clone().requires_grad_(True)
        o = _torch_sdpa(q, k, v, backend=name)
        results[name] = (o.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach())

    # --- Compare to Triton baseline ---
    def rep(tag, a, b):
        mx = (a - b).abs().max().item()
        mean = (a - b).abs().mean().item()
        print(f"[{tag}] max_abs_err={mx:.6e}, mean_abs_err={mean:.6e}")

    for k1, k2 in combinations(results, 2):
        o1, dQ1, dK1, dV1 = results[k1] 
        o2, dQ2, dK2, dV2 = results[k2] 
        
        print(k1, k2)
        rep("O ", o1.float(),   o2.float())
        rep("dQ", dQ1.float(), dQ2.float())
        rep("dK", dK1.float(), dK2.float())
        rep("dV", dV1.float(), dV2.float())
        print()


# -----------------------------
# Benchmark
# -----------------------------
def bench_sdpa_throughput(
    Q, K, V,
    dO=None,
    variants=("math", "mem", "flash"),
    mode="fwdbwd",              # "fwd" | "bwd" | "fwdbwd"
    warmup=10,
    repeats=100,
    print_table=True,
):
    """
    Returns: dict {variant: {"ms": median_ms, "throughput": elems_per_s}}
    - Throughput is a simple elements/s proxy:
        fwd:    elems = B*H*M*D
        bwd:    elems = 2*B*H*M*D  (roughly)
        fwdbwd: elems = 2*B*H*M*D  (uses same)
    """
    def _torch_sdpa(q, k, v, backend: str):
        kernel = {
            "math" : SDPBackend.MATH,
            "mem"  : SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
        }[backend]
        dtype = torch.float32 if backend == "math" else torch.bfloat16
        with torch.autocast("cuda", dtype=dtype), sdpa_kernel(kernel):
            out = F.scaled_dot_product_attention(q, k, v)
            out.backward(dO.detach())

    def _triton_sdpa(q, k, v, backend):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = backend(q, k, v)
            o.backward(dO.detach())

    assert Q.is_cuda and K.is_cuda and V.is_cuda
    B, H, M, D = Q.shape
    elems_base = B * H * M * D
    elems = elems_base if mode == "fwd" else 2 * elems_base

    if dO is None:
        dO = torch.randn_like(Q)
        
    results = {}
    for name in variants:
        def runner():
            q = Q.detach().clone().requires_grad_(True)
            k = K.detach().clone().requires_grad_(True)
            v = V.detach().clone().requires_grad_(True)
            fwd = _torch_sdpa if isinstance(name, str) else _triton_sdpa
            fwd(q, k, v, name)

        # Warmup
        for _ in range(warmup):
            _ = runner()
        torch.cuda.synchronize()

        # Timed runs
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        samples = []
        for _ in range(repeats):
            start.record()
            _ = runner()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))  # ms
        ms = median(samples)
        thr = (elems / (ms / 1e3))  # elems/s
        name = name if isinstance(name, str) else name.__name__
        results[name] = {"ms": ms, "throughput": thr}

    if print_table and results:
        print(f"{'variant':<12} {'ms (median)':>12} {'elems/s':>16}")
        for k, v in results.items():
            print(f"{k:<12} {v['ms']:>12.3f} {v['throughput']:>16.3e}")
    return results
