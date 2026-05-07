"""
WanModel adapted for LIVEditor inference.

Uses in_context_sparse_attention (tilelang/triton) for self-attention.
Stripped of all training-only code (DDP, gradient checkpointing, EMA, etc.).
"""

import math
import os
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from in_context_sparse_attention import editing_sparse_attention
from wanx.attention import flash_attention

BLOCK_M = 64

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def compress_seq(x, block_size, sizes):
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
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1) ** 0.5)
    QK = nn.functional.softmax(QK, dim=-1)
    return torch.matmul(QK, v), QK


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len),
                        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device)
    return torch.cat([original_tensor, padding_tensor], dim=0)


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, source_video_grid_sizes, TAG, freqs):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        s_f, s_h, s_w = source_video_grid_sizes.tolist()[i]
        s_seq_len = s_f * s_h * s_w
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        if TAG == 1:
            s_freqs_i = torch.cat([
                freqs[0][:s_f].view(s_f, 1, 1, -1).expand(s_f, s_h, s_w, -1),
                freqs[1][:s_h].view(1, s_h, 1, -1).expand(s_f, s_h, s_w, -1),
                freqs[2][:s_w].view(1, 1, s_w, -1).expand(s_f, s_h, s_w, -1),
            ], dim=-1).reshape(s_seq_len, 1, -1)
            freqs_i = torch.cat([freqs_i, s_freqs_i], dim=0)
        freqs_i = pad_freqs(freqs_i, s)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        output.append(x_i)
    return torch.stack(output).float()


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class WanSelfAttention(nn.Module):
    """Self-attention with in_context_sparse_attention (tilelang/triton backend)."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, backend='tilelang'):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.backend = backend

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.alpha_1 = nn.Linear(dim, 1, bias=False)
        self.alpha_2 = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.alpha_2.weight)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, source_video_grid_sizes, TAG, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        alpha_1 = (self.alpha_1(x).view(b, s, 1, 1).sigmoid() - 0.5).expand(-1, -1, 1, -1) * 0.5

        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            t_seq_len = f * h * w
            s_f, s_h, s_w = source_video_grid_sizes.tolist()[i]
            s_seq_len = s_f * s_h * s_w
            break

        q = rope_apply(q, grid_sizes, source_video_grid_sizes, TAG, freqs)
        k = rope_apply(k, grid_sizes, source_video_grid_sizes, TAG, freqs)

        x = editing_sparse_attention(q, k, v, t_seq_len, s_seq_len,
                                     alpha_2=alpha_1, backend=self.backend)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        return self.o(x)


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        return self.o(x)


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


# ---------------------------------------------------------------------------
# Block, Head, MLPProj
# ---------------------------------------------------------------------------

class WanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads,
                 window_size=(-1, -1), qk_norm=True, cross_attn_norm=False,
                 eps=1e-6, backend='tilelang'):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, backend=backend)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
                source_video_grid_sizes, TAG):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens, grid_sizes, source_video_grid_sizes, TAG, freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim),
            nn.GELU(), nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        return self.proj(image_embeds)


# ---------------------------------------------------------------------------
# WanModel
# ---------------------------------------------------------------------------

class WanModel(ModelMixin, ConfigMixin):
    """Wan diffusion backbone for video editing inference."""

    ignore_for_config = ['cross_attn_norm', 'qk_norm', 'text_dim', 'window_size']
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=5120,
                 ffn_dim=13824,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=40,
                 num_layers=40,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 use_siglip=False,
                 use_riflex=False,
                 backend='tilelang'):
        super().__init__()
        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        self.backend = backend

        self.patch_size = patch_size
        self.source_video_patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.source_video_embedding_more_memory = nn.Conv3d(
            in_dim, dim, kernel_size=self.source_video_patch_size, stride=self.source_video_patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              backend=backend)
            for _ in range(num_layers)
        ])

        # Head
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        if model_type == 'i2v':
            if use_siglip:
                self.img_emb_siglip = MLPProj(1152, dim)
            else:
                self.img_emb = MLPProj(1280, dim)

        self.gradient_checkpointing = False
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)

    def _forward(self, x, t, context, seq_len, source_video_latent):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        source_video_latent = [self.source_video_embedding_more_memory(u.unsqueeze(0))
                               for u in source_video_latent]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        source_video_grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in source_video_latent])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        source_video_latent = [u.flatten(2).transpose(1, 2) for u in source_video_latent]
        source_video_latent_len = [u.size(1) for u in source_video_latent]
        x = [torch.cat([u, v], dim=1) for u, v in zip(x, source_video_latent)]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        print(f"[DEBUG model] grid_sizes={grid_sizes.tolist()} src_grid_sizes={source_video_grid_sizes.tolist()}")
        print(f"[DEBUG model] patch_embed_out tokens: t={seq_lens[0].item() - source_video_latent_len[0]} src={source_video_latent_len[0]}")
        print(f"[DEBUG model] seq_lens={seq_lens.tolist()} seq_len={seq_len}")
        assert seq_lens.max() <= seq_len, f"seq_lens.max()={seq_lens.max().item()} > seq_len={seq_len}"
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

        TAG = 1
        for block in self.blocks:
            x = block(x, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                      freqs=self.freqs, context=context, context_lens=None,
                      source_video_grid_sizes=source_video_grid_sizes, TAG=TAG)

        x = torch.stack([u[:-source_video_latent_len[i]] for i, u in enumerate(x)], 0)
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def forward(self, x, t, context, seq_len, source_video_latent):
        return self._forward(x, t, context, seq_len, source_video_latent)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
