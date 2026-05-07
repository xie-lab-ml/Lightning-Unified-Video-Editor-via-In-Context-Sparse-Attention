"""
Editing sparse attention with pluggable sparse-kernel backend (tilelang / triton).

Usage:
    from editing_sparse_attention import editing_sparse_attention, Backend

    out = editing_sparse_attention(q, k, v, t_seq_len, s_seq_len)             # default: tilelang
    out = editing_sparse_attention(q, k, v, t_seq_len, s_seq_len, backend=Backend.TRITON)
    out = editing_sparse_attention(q, k, v, t_seq_len, s_seq_len, backend='triton')
"""

import torch
import torch.nn.functional as F
try:
    from flash_attn_interface import flash_attn_func  # FA3 (H100)
except ImportError:
    from flash_attn import flash_attn_func            # FA2 fallback

from .tilelang_host import sparse_piecewise_attn_1st_intervals_tilelang
from .triton_host import sparse_piecewise_attn_1st_intervals as _sparse_attn_triton

BLOCK_M = 64


class Backend:
    """Sparse-attention backend constants."""
    TILELANG = 'tilelang'
    TRITON  = 'triton'

    @classmethod
    def validate(cls, backend: str) -> str:
        if backend not in (cls.TILELANG, cls.TRITON):
            raise ValueError(f"Unknown backend '{backend}'.  Valid: {cls.TILELANG}, {cls.TRITON}")
        return backend


def compress_seq(x, block_size, sizes):
    """
    Compress sequence by averaging over blocks. Handles partial last block.
    """
    B, S, H, D = x.shape
    num_blocks = sizes.shape[0]

    if S % block_size == 0:
        return (x.view(B, num_blocks, block_size, H, D).sum(dim=2) / sizes.view(1, -1, 1, 1)).to(x.dtype)

    n_full = (num_blocks - 1) * block_size
    x_main = x[:, :n_full]
    x_last = x[:, n_full:]

    main_sum = x_main.view(B, num_blocks - 1, block_size, H, D).sum(dim=2)
    last_sum = x_last.sum(dim=1, keepdim=True)

    compressed = torch.cat([main_sum, last_sum], dim=1)
    return (compressed / sizes.view(1, -1, 1, 1)).to(x.dtype)


def torch_attention(q, k, v):
    """Vanilla scaled dot-product attention. q,k,v: [B, H, S, D]."""
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1) ** 0.5)
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK


def editing_sparse_attention(
    q, k, v, t_seq_len, s_seq_len,
    alpha_2=1.0, alpha_3=1.0, SPARSITY_RATIO=0.0625,
    backend=Backend.TILELANG,
):
    """
    Editing sparse attention with selectable sparse-kernel backend.

    Args:
        q, k, v: [B, S, H, D]
        t_seq_len: target-video token count
        s_seq_len: source-video token count (S = t_seq_len + s_seq_len)
        alpha_2: residual weight for compressed-attention shortcut
        alpha_3: unused (API compatibility)
        SPARSITY_RATIO: fraction of KV blocks for exact TopK
        backend: 'tilelang' (default) or 'triton'

    Returns:
        x: [B, S, H, D]
    """
    backend = Backend.validate(backend)

    B, S, H, D = q.shape
    NUM_BLOCKS = (S + BLOCK_M - 1) // BLOCK_M

    # variable block sizes (last block may be smaller)
    variable_block_sizes = torch.ones(NUM_BLOCKS, device=q.device, dtype=torch.int32) * BLOCK_M
    variable_block_sizes[-1] = S - (NUM_BLOCKS - 1) * BLOCK_M

    padding_len = NUM_BLOCKS * BLOCK_M - S

    if padding_len > 0:
        q_padded = F.pad(q, (0, 0, 0, 0, 0, padding_len))
        k_padded = F.pad(k, (0, 0, 0, 0, 0, padding_len))
        v_padded = F.pad(v, (0, 0, 0, 0, 0, padding_len))
    else:
        q_padded = q
        k_padded = k
        v_padded = v

    # ---- 1. block-wise compression ----
    q_compress = (q_padded.view(B, NUM_BLOCKS, BLOCK_M, H, D).sum(dim=2)
                  / variable_block_sizes.view(1, -1, 1, 1)).to(q.dtype)
    k_compress = compress_seq(k, BLOCK_M, variable_block_sizes)
    v_compress = compress_seq(v, BLOCK_M, variable_block_sizes)

    # ---- 2. compressed attention for block relevance + global shortcut ----
    out_compress, QK_compress = torch_attention(
        q_compress.permute(0, 2, 1, 3),
        k_compress.permute(0, 2, 1, 3),
        v_compress.permute(0, 2, 1, 3),
    )
    # out_compress: [B, H, NUM_BLOCKS, D] → broadcast back to [B, S, H, D]
    out_compress = out_compress.view(B, H, NUM_BLOCKS, 1, D)
    out_compress = out_compress.repeat(1, 1, 1, BLOCK_M, 1)
    out_compress = out_compress.view(B, H, NUM_BLOCKS * BLOCK_M, D)
    out_compress = out_compress.permute(0, 2, 1, 3)[:, :S, :, :]

    # ---- 3. select top-K source-video KV blocks ----
    n_t_BLOCK = t_seq_len // BLOCK_M
    n_s_BLOCK = NUM_BLOCKS - n_t_BLOCK

    topK = n_s_BLOCK // 8
    second_topk = n_s_BLOCK // 8

    QK_compress_subset = QK_compress[:, :, :n_t_BLOCK, n_t_BLOCK:]
    QK_compress_summed = QK_compress_subset.sum(dim=-2)  # [B, H, n_s_BLOCK]

    topK_indices = torch.topk(QK_compress_summed, topK + second_topk, dim=-1).indices
    first_topk_indices = topK_indices[:, :, :topK]

    # Build extended KV = [t-part || selected-source-blocks] in [B, H, Tkv, D]
    Tkv = t_seq_len + topK * BLOCK_M
    new_k = torch.empty(B, H, Tkv, D, device=k.device, dtype=k.dtype)
    new_v = torch.empty(B, H, Tkv, D, device=v.device, dtype=v.dtype)

    # t-part: permute only the slice
    new_k[:, :, :t_seq_len] = k[:, :t_seq_len].permute(0, 2, 1, 3).contiguous()
    new_v[:, :, :t_seq_len] = v[:, :t_seq_len].permute(0, 2, 1, 3).contiguous()

    # s-part: gather from [B, S_pad, H, D] → permute → write to [B, H, *, D]
    s_idx = (
        first_topk_indices[..., None] * BLOCK_M
        + torch.arange(BLOCK_M, device=k.device)
        + (t_seq_len // BLOCK_M * BLOCK_M)
    )  # [B, H, topK, BLOCK_M]
    s_idx = s_idx.reshape(B, H, -1).permute(0, 2, 1)  # [B, topK*BLOCK_M, H]
    s_idx_exp = s_idx.unsqueeze(-1).expand(-1, -1, -1, D)

    sk = torch.gather(k_padded, 1, s_idx_exp)  # [B, topK*BLOCK_M, H, D]
    sv = torch.gather(v_padded, 1, s_idx_exp)
    new_k[:, :, t_seq_len:] = sk.permute(0, 2, 1, 3).contiguous()
    new_v[:, :, t_seq_len:] = sv.permute(0, 2, 1, 3).contiguous()

    # ---- 4. split Q blocks into sharp vs flat by attention sharpness ----
    QK_sub = QK_compress[:, :, :, :n_t_BLOCK]
    sorted_QK_sub, _ = torch.sort(QK_sub, dim=-1, descending=True)
    k_sparsity = max(1, int(n_t_BLOCK * SPARSITY_RATIO))
    sharpness = sorted_QK_sub[:, :, :, :k_sparsity].sum(dim=-1)  # [B, H, NUM_BLOCKS]

    sorted_indices = torch.argsort(sharpness, dim=-1, descending=True)

    num_q_blocks = NUM_BLOCKS
    num_sharp = num_q_blocks // 2
    flat_indices = sorted_indices[:, :, :num_sharp]      # first half → full attention
    sharp_indices = sorted_indices[:, :, num_sharp:]     # second half → sparse attention

    q_padded_perm = q_padded.permute(0, 2, 1, 3)  # [B, H, S_pad, D]
    q_reshaped = q_padded_perm.view(B, H, NUM_BLOCKS, BLOCK_M, D)

    def gather_blocks(q_reshaped, indices):
        idx_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, BLOCK_M, D)
        return torch.gather(q_reshaped, 2, idx_expanded)

    q_sharp = gather_blocks(q_reshaped, sharp_indices)  # [B, H, N_sharp_blocks, BLOCK_M, D]
    q_flat = gather_blocks(q_reshaped, flat_indices)    # [B, H, N_flat_blocks, BLOCK_M, D]
    q_sharp = q_sharp.view(B, H, -1, D)
    q_flat = q_flat.view(B, H, -1, D)

    # ---- 5. sharp blocks → sparse attention (selected backend) ----
    q_sharp_bf16 = q_sharp.to(torch.bfloat16)

    if backend == Backend.TILELANG:
        # TileLang requires TKV to be a multiple of BLOCK_M.
        TKV = new_k.shape[2]
        tkv_pad = (BLOCK_M - (TKV % BLOCK_M)) % BLOCK_M
        if tkv_pad > 0:
            new_k_sp = F.pad(new_k, (0, 0, 0, tkv_pad))
            new_v_sp = F.pad(new_v, (0, 0, 0, tkv_pad))
        else:
            new_k_sp = new_k
            new_v_sp = new_v
        out_sharp = sparse_piecewise_attn_1st_intervals_tilelang(
            q_sharp_bf16, new_k_sp.to(torch.bfloat16), new_v_sp.to(torch.bfloat16),
            sparsity=SPARSITY_RATIO, block_size=64,
        )
    else:  # Backend.TRITON
        out_sharp = _sparse_attn_triton(
            q_sharp_bf16, new_k.to(torch.bfloat16), new_v.to(torch.bfloat16),
            sparsity=SPARSITY_RATIO, block_size=64,
        )

    # ---- 6. flat blocks → flash-attn (FlashAttention-2) ----
    # flash_attn_func expects [B, T, H, D] in fp16/bf16
    out_flat = flash_attn_func(
        q_flat.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16),
        new_k.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16),
        new_v.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16),
    ).to(q_flat.dtype).permute(0, 2, 1, 3).contiguous()  # back to [B, H, TQ, D]

    # ---- 7. scatter back to full output ----
    out_reshaped = torch.zeros_like(q_reshaped)  # [B, H, NUM_BLOCKS, BLOCK_M, D]

    out_sharp_reshaped = out_sharp.view(B, H, num_q_blocks - num_sharp, BLOCK_M, D).to(out_reshaped.dtype)
    idx = sharp_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, BLOCK_M, D)
    out_reshaped.scatter_(2, idx, out_sharp_reshaped)

    out_flat_reshaped = out_flat.view(B, H, num_sharp, BLOCK_M, D).to(out_reshaped.dtype)
    idx = flat_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, BLOCK_M, D)
    out_reshaped.scatter_(2, idx, out_flat_reshaped)

    # ---- 8. unpad + residual ----
    x = out_reshaped.view(B, H, NUM_BLOCKS * BLOCK_M, D).permute(0, 2, 1, 3)  # [B, S_pad, H, D]
    x = x[:, :S, :, :]
    x = x + out_compress * alpha_2
    return x


# Backward-compatible alias (always uses tilelang).
editing_sparse_attention_tilelang = editing_sparse_attention
