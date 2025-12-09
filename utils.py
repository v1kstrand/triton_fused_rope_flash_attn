from statistics import median
from itertools import combinations

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from itertools import combinations
from .rope import sdpa_triton_fa_rope, CosSinTable # TODO




def apply_rot_embed_cat(x, emb) -> torch.Tensor:
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    rot_x = torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape) 
    return x * cos_emb + rot_x * sin_emb
    
    
def rope_hook(q, k, v, rope, npt=1):
    q = torch.cat([q[:, :,  :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
    k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)
    return q, k


def compare_sdpa_variants(Q, K, V, triton_kernel, rope_embed, cos_sin, dO=None, dtype = torch.bfloat16, kernels=("flash",)):
    def _torch_sdpa(q, k, v, *, backend: str):
        kernel = {
            "math" : SDPBackend.MATH,
            "mem" : SDPBackend.EFFICIENT_ATTENTION,
            "flash" : SDPBackend.FLASH_ATTENTION
        }[backend]

        with torch.autocast("cuda", dtype=dtype), sdpa_kernel(kernel):
            q, k = rope_hook(q, k, v, rope_embed)
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
        o = triton_kernel(q, k, v, cos_sin)
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
    rope_embed,
    cos_sin,
    B=1024,
    H=6,
    S=197,
    D=64,
    device = "cuda",
    dO=None,
    variants=None,
    mode="fwdbwd",              # "fwd" | "bwd" | "fwdbwd"
    warmup=10,
    repeats=100,
    print_table=True,
    compile=False,
    dtype = torch.bfloat16,
    per_step = False
):
    """
    Returns: dict {variant: {"ms": median_ms, "throughput": elems_per_s}}
    - Throughput is a simple elements/s proxy:
        fwd:    elems = B*H*M*D
        bwd:    elems = 2*B*H*M*D  (roughly)
        fwdbwd: elems = 2*B*H*M*D  (uses same)
    """   
    variants = variants or ["flash", "mem", "math", "triton"]
    # inputs
    Q = torch.randn(B, H, S, D, device=device, dtype=torch.float32, requires_grad=True)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    dO = torch.randn_like(Q)

    # RoPE tables for Triton kernel
    cos_sin = CosSinTable(base=100., H_img=14, D=D, device=device)
    COSP, SINP = cos_sin.tables()
    sin_full = SINP.repeat_interleave(2, dim=-1)  # [H*W, D]
    cos_full = COSP.repeat_interleave(2, dim=-1)  # [H*W, D]
    rope_embed = torch.cat([sin_full, cos_full], dim=-1).contiguous()  # [H*W, 2*D]

    assert Q.is_cuda and K.is_cuda and V.is_cuda
    B, H, M, D = Q.shape
    elems_base = B * H * M * D
    elems = elems_base if mode == "fwd" else 2 * elems_base

    if dO is None:
        dO = torch.randn_like(Q)
        
    def _triton_sdpa(q, k, v):
        o = sdpa_triton_fa_rope(q, k, v, cos_sin)
        o.backward(dO.detach())
        
    results = {}
    for name in variants:
        kernel = {
                "mem"  : SDPBackend.EFFICIENT_ATTENTION,
                "flash": SDPBackend.FLASH_ATTENTION,
            }.get(name, SDPBackend.MATH)
        
        def _torch_sdpa(q, k, v):
            with sdpa_kernel(kernel):
                q, k = rope_hook(q, k, v, rope_embed)
                out = F.scaled_dot_product_attention(q, k, v)
                out.backward(dO.detach())
                
        fnc = _torch_sdpa if name  != "triton" else _triton_sdpa
        
        runner = torch.compile(
            fnc,
            backend="inductor",
            mode="max-autotune",
            fullgraph=True
        ) if compile else fnc
            
        # Warmup
        for _ in range(warmup):
            q = Q.detach().clone().requires_grad_(True)
            k = K.detach().clone().requires_grad_(True)
            v = V.detach().clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=dtype), sdpa_kernel(kernel):
                runner(q, k, v)
        torch.cuda.synchronize()

        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        samples = []
        if not per_step:
            start.record()
        for _ in range(repeats):
            q = Q.detach().clone().requires_grad_(True)
            k = K.detach().clone().requires_grad_(True)
            v = V.detach().clone().requires_grad_(True)
            if per_step:
                start.record()
            with torch.autocast("cuda", dtype=dtype):
                runner(q, k, v)
            if per_step:
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))  # ms
        if not per_step:
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
