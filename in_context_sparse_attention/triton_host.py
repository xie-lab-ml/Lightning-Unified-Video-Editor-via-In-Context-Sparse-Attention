"""
Triton sparse-attention entrypoint.

Thin wrapper around the autograd Function defined in triton_kernels.
"""

import torch
from .triton_kernels import SparsePiecewiseAttn1stIntervalsFunction


@torch.no_grad()
def sparse_piecewise_attn_1st_intervals(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: float = 0.1,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Triton sparse piecewise attention (1st-interval approximation).

    Args:
        q, k, v:  [B, H, T, D]  (bfloat16)
        sparsity: fraction of KV blocks for exact TopK
        block_size: block size (64)

    Returns:
        out:  [B, H, T_Q, D]  (bfloat16)
    """
    return SparsePiecewiseAttn1stIntervalsFunction.apply(q, k, v, sparsity, block_size)
