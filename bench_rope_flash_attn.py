import argparse
from statistics import median

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attn import CosSinTable, flash_attention
from utils import rope_hook


def bench_rope_flash_attn(
    B: int,
    H: int,
    S: int,
    D: int,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10,
    repeats: int = 50,
    mode: str = "fwdbwd",
    base: float = 10000.0,
    h_img: int = 14,
):
    """Benchmark fused RoPE FlashAttention against PyTorch SDPA backends."""

    # inputs
    Q = torch.randn(B, H, S, D, device=device, dtype=torch.float32, requires_grad=True)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    dO = torch.randn_like(Q)

    cos_sin = CosSinTable(base=base, H_img=h_img, D=D, device=device)
    COSP, SINP = cos_sin.tables()
    sin_full = SINP.repeat_interleave(2, dim=-1)
    cos_full = COSP.repeat_interleave(2, dim=-1)
    rope_embed = torch.cat([sin_full, cos_full], dim=-1).contiguous()

    elems_base = B * H * S * D
    elems = elems_base if mode == "fwd" else 2 * elems_base

    backends = {
        "torch_flash_rope": SDPBackend.FLASH_ATTENTION,
        "torch_mem_rope": SDPBackend.EFFICIENT_ATTENTION,
    }

    results = {}

    def _torch_runner(name: str):
        backend = backends[name]
        q = Q.detach().clone().requires_grad_(True)
        k = K.detach().clone().requires_grad_(True)
        v = V.detach().clone().requires_grad_(True)
        q_r, k_r = rope_hook(q, k, v, rope_embed)
        dtype_local = torch.float32 if backend == SDPBackend.MATH else dtype
        with torch.autocast("cuda", dtype=dtype_local), sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(q_r, k_r, v)
            if mode != "fwd":
                out.backward(dO.detach())

    def _triton_runner():
        q = Q.detach().clone().requires_grad_(True)
        k = K.detach().clone().requires_grad_(True)
        v = V.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=dtype):
            out = flash_attention(q, k, v, impl="fa_rope", cos_sin=cos_sin)
            if mode != "fwd":
                out.backward(dO.detach())

    variants = list(backends.keys()) + ["triton_fa_rope"]

    def _run_variant(name: str):
        if name.startswith("torch"):
            _torch_runner(name)
        else:
            _triton_runner()

    for name in variants:
        for _ in range(warmup):
            _run_variant(name)
        torch.cuda.synchronize()

        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        samples = []
        for _ in range(repeats):
            start.record()
            _run_variant(name)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        ms = median(samples)
        thr = elems / (ms / 1e3)
        results[name] = {"ms": ms, "throughput": thr}

    if results:
        print(f"{'variant':<16} {'ms (median)':>12} {'elems/s':>16}")
        for k, v in results.items():
            print(f"{k:<16} {v['ms']:>12.3f} {v['throughput']:>16.3e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RoPE Triton FlashAttention")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size")
    parser.add_argument("--heads", type=int, default=6, help="Number of heads")
    parser.add_argument("--seq", type=int, default=197, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Autocast dtype")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=50, help="Measured iterations")
    parser.add_argument("--mode", type=str, default="fwdbwd", choices=["fwd", "fwdbwd"], help="Benchmark mode")
    parser.add_argument("--rope-base", type=float, default=10000.0, help="RoPE base")
    parser.add_argument("--rope-grid", type=int, default=14, help="Grid side length for axial RoPE")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    bench_rope_flash_attn(
        args.batch,
        args.heads,
        args.seq,
        args.dim,
        device=args.device,
        dtype=dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        mode=args.mode,
        base=args.rope_base,
        h_img=args.rope_grid,
    )


if __name__ == "__main__":
    main()
