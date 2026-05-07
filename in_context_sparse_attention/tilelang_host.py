import torch
import torch.nn.functional as F
from functools import lru_cache

from .tilelang_kernels import (
    reduce_q_kernel,
    reduce_kv_kernel,
    sparse_topk_kernel,
    sparse_interval_kernel,
    sparse_fused_vblocks1_kernel,
    sparse_fused_vblocks1_kernel_auto,
    ceil_div,
)


@lru_cache(maxsize=None)
def _get_reduce_q_kernel(BH: int, NQ: int, BT: int, Kdim: int, TQ: int):
    return reduce_q_kernel(BH, NQ, BT, Kdim, TQ)


@lru_cache(maxsize=None)
def _get_reduce_kv_kernel(BH: int, NKV: int, BT: int, Kdim: int, BV: int, VBlocks: int, TKV: int, Vdim: int):
    return reduce_kv_kernel(BH, NKV, BT, Kdim, BV, VBlocks, TKV, Vdim)


@lru_cache(maxsize=None)
def _get_sparse_topk_kernel(
    BH: int,
    NQ: int,
    NKV: int,
    TopK: int,
    BT: int,
    BK: int,
    BV: int,
    Kdim: int,
    Vdim: int,
    TQ: int,
    TKV: int,
    VBlocks: int,
    threads: int,
):
    return sparse_topk_kernel(BH, NQ, NKV, TopK, BT, BK, BV, Kdim, Vdim, TQ, TKV, VBlocks, threads=threads)


@lru_cache(maxsize=None)
def _get_sparse_interval_kernel(
    BH: int,
    NQ: int,
    NKV: int,
    TopK: int,
    BT: int,
    BK: int,
    BV: int,
    Kdim: int,
    Vdim: int,
    TQ: int,
    TKV: int,
    VBlocks: int,
    NKV_REAL: int,
    BN: int,
    threads: int,
):
    return sparse_interval_kernel(
        BH,
        NQ,
        NKV,
        TopK,
        BT,
        BK,
        BV,
        Kdim,
        Vdim,
        TQ,
        TKV,
        VBlocks,
        NKV_REAL=NKV_REAL,
        BN=BN,
        threads=threads,
    )


@lru_cache(maxsize=None)
def _get_sparse_fused_vblocks1_kernel(
    BH: int,
    NQ: int,
    NKV_PAD: int,
    TopK: int,
    BT: int,
    BK: int,
    BV: int,
    Kdim: int,
    Vdim: int,
    TQ: int,
    TKV: int,
    BN: int,
    NKV_REAL: int,
    threads: int,
    use_wgmma: bool = False,
):
    kernel_fn = sparse_fused_vblocks1_kernel
    return kernel_fn(
        BH,
        NQ,
        NKV_PAD,
        TopK,
        BT,
        BK,
        BV,
        Kdim,
        Vdim,
        TQ,
        TKV,
        BN=BN,
        NKV_REAL=NKV_REAL,
        threads=threads,
        use_swizzle_addr=False,
        use_wgmma=use_wgmma,
    )


def sparse_piecewise_attn_tilelang_hosttopk(
    q,
    k,
    v,
    sparsity=0.0625,
    BT=64,
    BV=32,
    BN=64,
    threads_topk: int = 128,
    threads_interval: int = 128,
    sorted_topk: bool = False,
    fused_use_wgmma: bool = False,
):
    """
    TileLang sparse attention pipeline with host-side TopK (PyTorch).

    Rough flow:
    - reduce Q / reduce KV on GPU (TileLang kernels)
    - score = QC @ KC^T on GPU (torch.matmul)
    - indices = torch.topk(score)
    - attention:
      - fast path: fused kernel if VBlocks==1 (BV==Vdim)
      - otherwise: two kernels (topk + interval)
    """
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    B, H, TQ, Kdim = q.shape
    TKV = k.shape[2]
    Vdim = v.shape[3]

    BH = B * H
    NQ = ceil_div(TQ, BT)
    NKV = ceil_div(TKV, BT)
    VBlocks = ceil_div(Vdim, BV)

    assert TQ % BT == 0 and TKV % BT == 0, "TQ/TKV must be divisible by BT"
    assert Vdim % BV == 0, "Vdim must be divisible by BV"

    q_flat = q.view(BH, TQ, Kdim).contiguous()
    k_flat = k.view(BH, TKV, Kdim).contiguous()
    v_flat = v.view(BH, TKV, Vdim).contiguous()

    # 1) Block reductions on GPU
    rq = _get_reduce_q_kernel(BH, NQ, BT, Kdim, TQ)
    QC = rq(q_flat)  # [BH, NQ, Kdim]

    rkv = _get_reduce_kv_kernel(BH, NKV, BT, Kdim, BV, VBlocks, TKV, Vdim)
    KC, VC = rkv(k_flat, v_flat)
    NKV_REAL = NKV

    # 2) Block relevance & TopK (PyTorch)
    top_k = max(1, int(sparsity * NKV))
    scores = torch.matmul(QC, KC.transpose(-1, -2))  # [BH, NQ, NKV]
    indices_flat = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=sorted_topk).indices.to(torch.int32)

    # Fast path: if VBlocks==1 (BV==Vdim), use the fused kernel
    if VBlocks == 1 and BV == Vdim:
        pad_blocks = (BN - (NKV_REAL % BN)) % BN
        if pad_blocks:
            KC_pad = F.pad(KC, (0, 0, 0, pad_blocks))
            VC_pad = F.pad(VC, (0, 0, 0, pad_blocks))
            NKV_PAD = NKV_REAL + pad_blocks
        else:
            KC_pad = KC
            VC_pad = VC
            NKV_PAD = NKV_REAL

        fused_kernel = _get_sparse_fused_vblocks1_kernel(
            BH,
            NQ,
            NKV_PAD,
            top_k,
            BT,
            Kdim,
            BV,
            Kdim,
            Vdim,
            TQ,
            TKV,
            BN,
            NKV_REAL,
            threads_topk,
            use_wgmma=fused_use_wgmma,
        )
        out_blocks, _lse = fused_kernel(q_flat, k_flat, v_flat, KC_pad, VC_pad, indices_flat)
        out = out_blocks.view(B, H, NQ, BT, Vdim).reshape(B, H, TQ, Vdim)
        return out

    # Slow path: 2 launches (topk kernel + interval kernel)
    topk_kernel = _get_sparse_topk_kernel(BH, NQ, NKV, top_k, BT, Kdim, BV, Kdim, Vdim, TQ, TKV, VBlocks, threads_topk)
    out_partial, l_partial, m_partial = topk_kernel(q_flat, k_flat, v_flat, KC, VC, indices_flat)

    pad_blocks = (BN - (NKV_REAL % BN)) % BN
    if pad_blocks:
        KC = F.pad(KC, (0, 0, 0, pad_blocks))
        VC = F.pad(VC, (0, 0, 0, pad_blocks))
        NKV_PAD = NKV_REAL + pad_blocks
    else:
        NKV_PAD = NKV_REAL

    interval_kernel = _get_sparse_interval_kernel(
        BH, NQ, NKV_PAD, top_k, BT, Kdim, BV, Kdim, Vdim, TQ, TKV, VBlocks, NKV_REAL, BN, threads_interval
    )
    out_blocks, _lse = interval_kernel(q_flat, KC, VC, indices_flat, out_partial, l_partial, m_partial)

    out = out_blocks.view(B, H, NQ, BT, Vdim).reshape(B, H, TQ, Vdim)
    return out


@torch.no_grad()
def sparse_piecewise_attn_1st_intervals_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparsity: float = 0.1,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Minimal entrypoint for sparse piecewise attention using TileLang.
    Equivalent to Triton's sparse_piecewise_attn_1st_intervals.

    Args:
        q: [B, H, TQ, D] in bfloat16
        k: [B, H, TKV, D] in bfloat16
        v: [B, H, TKV, D] in bfloat16 (Vdim must equal D for fast fused path)
        sparsity: fraction of KV blocks to attend exactly
        block_size: block size (must be 64)

    Returns:
        out: [B, H, TQ, D] in bfloat16
    """
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert block_size == 64, "Only block_size=64 supported."
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    assert q.shape[0] == k.shape[0] == v.shape[0]
    assert q.shape[1] == k.shape[1] == v.shape[1]
    assert q.shape[3] == k.shape[3], "Kdim must match between q and k"
    assert k.shape[2] == v.shape[2], "TKV must match between k and v"
    assert v.shape[3] == q.shape[3], "This entrypoint assumes Vdim == D (so VBlocks==1)."

    D = q.shape[3]
    return sparse_piecewise_attn_tilelang_hosttopk(
        q,
        k,
        v,
        sparsity=sparsity,
        BT=block_size,
        BV=D,
        BN=64,
        threads_topk=128,
        threads_interval=256,
        sorted_topk=False,
        fused_use_wgmma=False,
    )
