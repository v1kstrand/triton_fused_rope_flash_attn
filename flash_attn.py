"""Public entry points for Triton FlashAttention kernels.

This module keeps the baseline FlashAttention implementation untouched while
adding a RoPE-aware variant that consumes the fused RoPE Triton kernel.
"""
from typing import Optional

from torch import Tensor

from vit_fa_triton import sdpa_triton_fa
from rope_flash_attn_kernel import CosSinTable, sdpa_triton_fa_rope

__all__ = [
    "CosSinTable",
    "flash_attention",
    "sdpa_triton_fa",
    "sdpa_triton_fa_rope",
]


_DEF_IMPL = "fa"
_ROPE_IMPLS = {"fa_rope", "rope"}


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    impl: str = _DEF_IMPL,
    cos_sin: Optional[CosSinTable] = None,
) -> Tensor:
    """Dispatch to the plain or RoPE-aware Triton FlashAttention kernels.

    Args:
        q, k, v: Input tensors shaped ``[batch, heads, seq, head_dim]``.
        impl: Either ``"fa"`` (baseline Triton FlashAttention) or ``"fa_rope"``/``"rope"``
            to select the fused RoPE kernel.
        cos_sin: Required when ``impl`` requests the RoPE kernel. Provide a
            :class:`CosSinTable` instance to supply the pairwise cosine/sine tables.

    Returns:
        Attention output tensor with the same shape and dtype as ``q``.
    """

    if impl == _DEF_IMPL:
        return sdpa_triton_fa(q, k, v)

    if impl in _ROPE_IMPLS:
        if cos_sin is None:
            raise ValueError("cos_sin must be provided when impl='fa_rope'")
        return sdpa_triton_fa_rope(q, k, v, cos_sin)

    raise ValueError(f"Unknown attention impl '{impl}'. Expected 'fa' or 'fa_rope'.")
