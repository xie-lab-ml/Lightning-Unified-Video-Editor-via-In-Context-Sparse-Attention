"""
LIVEditor — single-GPU video editing inference.

Usage:
    python inference.py \
        --config configs/inference.yaml \
        --checkpoint /path/to/checkpoint.bin \
        --input /path/to/input.mp4 \
        --prompt "your editing prompt" \
        --output /path/to/output.mp4

Or with defaults:
    python inference.py --checkpoint ckpt.bin --input video.mp4 --prompt "..."
"""

import argparse
import math
import random
import sys
import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from wanx.model import WanModel
from wanx.vae import WanVAE
from wanx.t5 import T5EncoderModel
from wanx.scheduler import FlowUniPCMultistepScheduler
from wanx.utils import save_video_ffmpeg, read_video_ffmpeg


class LIVEditor:
    """Single-GPU video editing inference engine."""

    def __init__(self, config: dict, checkpoint_path: str):
        self.cfg = config
        self.device = torch.device('cuda')

        # ---------- VAE ----------
        vae_path = os.path.join(config['base_model_path'], config['vae_checkpoint'])
        self.vae = WanVAE(vae_pth=vae_path, device=self.device)
        self.vae.model.to(self.device, dtype=eval(config['model']['vae_dtype']))
        self.vae.model.requires_grad_(False)
        self.vae.set_tiling(False)

        # ---------- T5 ----------
        t5_ckpt = os.path.join(config['base_model_path'], config['t5_checkpoint'])
        t5_tok = os.path.join(config['base_model_path'], config['t5_tokenizer'])
        self.text_encoder = T5EncoderModel(
            text_len=config['model']['text_len'],
            dtype=eval(config['model']['t5_dtype']),
            device=torch.device('cpu'),
            checkpoint_path=t5_ckpt,
            tokenizer_path=t5_tok,
        )
        self.neg_prompt_embed = torch.load(config['neg_prompt_embed_path']).to(
            'cpu', dtype=eval(config['model']['dtype']))

        # ---------- WanModel ----------
        self.model = WanModel(
            patch_size=tuple(config['model']['patch_size']),
            text_len=config['model']['text_len'],
            dim=config['model']['dim'],
            ffn_dim=config['model']['ffn_dim'],
            freq_dim=config['model']['freq_dim'],
            text_dim=config['model']['text_dim'],
            out_dim=config['model']['out_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            backend=config['attention']['backend'],
        )
        # Load checkpoint FIRST (on CPU, fp32), then move to GPU
        self._load_checkpoint(checkpoint_path)
        self.model.to(self.device, dtype=eval(config['model']['dtype']))
        self.model.gradient_checkpointing = False
        self.model.eval()

        # ---------- Scheduler ----------
        self.shift = config['scheduler']['shift']
        self.steps = config['inference']['num_inference_steps']
        self.guidance = config['inference']['guidance_scale']
        self.seed = config['inference']['seed']

    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights, handling various ckpt formats."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        for key in ['state_dict', 'module', 'weights']:
            if key in state_dict:
                state_dict = state_dict[key]
                break

        # Handle wanx_unimodel prefix (some checkpoints)
        has_wanxunimodel = any('wanx_unimodel.guidance_model.feedforward_model.' in k for k in state_dict.keys())
        if has_wanxunimodel:
            print('[ckpt] detected wanx_unimodel prefix, stripping...')
            state_dict = {k.replace('wanx_unimodel.guidance_model.feedforward_model.', ''): v
                          for k, v in state_dict.items()}

        # Filter and remap keys
        new_sd = {}
        for k, v in state_dict.items():
            # Skip non-model keys
            if k.startswith('vae.') or 'optimizer' in k or 'scheduler' in k:
                continue
            # Handle LoRA key naming: .weight.lora -> .lora.
            k = k.replace('.weight.lora', '.lora.')
            # Strip model/model_ema prefix
            if k.startswith('model.'):
                k = k[6:]
            elif k.startswith('model_ema.'):
                k = k[10:]
            new_sd[k] = v

        print(f'[ckpt] {len(state_dict)} raw keys -> {len(new_sd)} model keys')
        missing, unexpected = self.model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f'[ckpt] missing ({len(missing)}):')
            for mk in missing[:15]:
                print(f'  - {mk}')
            if len(missing) > 15:
                print(f'  ... and {len(missing)-15} more')
        if unexpected:
            print(f'[ckpt] unexpected ({len(unexpected)}):')
            for uk in unexpected[:10]:
                print(f'  - {uk}')
            if len(unexpected) > 10:
                print(f'  ... and {len(unexpected)-10} more')

    # Matches remote dataset RESIZE_CONFIGS_540P
    _ASPECT_RATIOS = {
        (9, 16):  (544, 960),
        (1, 1):   (720, 720),
        (16, 9):  (960, 544),
        (1, 2):   (480, 960),
        (2, 1):   (960, 480),
    }

    @classmethod
    def _adaptive_resize(cls, video: torch.Tensor) -> torch.Tensor:
        """Match remote dataset resize logic: find closest standard aspect ratio,
        resize preserving aspect, then center-crop to target dimensions."""
        _, _, H, W = video.shape  # [C, T, H, W]
        ratio = W / H

        # Find closest standard aspect ratio
        closest = min(cls._ASPECT_RATIOS, key=lambda k: abs(k[0]/k[1] - ratio))
        target_w, target_h = cls._ASPECT_RATIOS[closest]
        target_aspect = target_w / target_h

        # Compute intermediate size (preserving aspect, matching larger dim)
        if ratio > target_aspect:  # wider
            new_h, new_w = target_h, int(target_h * ratio)
        else:  # taller
            new_w, new_h = target_w, int(target_w / ratio)

        # Align to 16 for VAE/patch compatibility
        ALIGN = 16
        new_h = ((new_h + ALIGN - 1) // ALIGN) * ALIGN
        new_w = ((new_w + ALIGN - 1) // ALIGN) * ALIGN

        print(f'[resize] {W}x{H} (ratio={ratio:.3f}) -> closest {closest[0]}:{closest[1]} -> '
              f'intermediate {new_w}x{new_h} -> crop to {target_w}x{target_h}')

        # Resize
        video = torch.nn.functional.interpolate(
            video.float(), size=(new_h, new_w),
            mode='bicubic', antialias=True).to(video.dtype)

        # Center crop
        _, _, h, w = video.shape
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        video = video[:, :, top:top+target_h, left:left+target_w]

        return video.contiguous()

    @torch.no_grad()
    def encode_video(self, video_path: str) -> tuple:
        """Read video, adaptive-resize, encode to latent.
        Returns (latent, num_frames, H_used, W_used)."""
        video = read_video_ffmpeg(video_path).to(self.device)
        _, T_orig, _, _ = video.shape

        video = self._adaptive_resize(video)
        _, _, H_used, W_used = video.shape

        video = video.sub_(0.5).div_(0.5).unsqueeze(0)
        latent = self.vae.encode(video)[0].unsqueeze(0)
        return latent, T_orig, H_used, W_used

    @torch.no_grad()
    def encode_text(self, prompt: str) -> tuple:
        """Encode prompt and negative prompt."""
        self.text_encoder.model.to(self.device)
        ctx = self.text_encoder([prompt], self.device)
        ctx = [c.contiguous() for c in ctx]

        ctx_null = self.text_encoder([self.cfg['n_prompt']], self.device)
        ctx_null = [c.contiguous() for c in ctx_null]

        self.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        return ctx, ctx_null

    @staticmethod
    def _get_target_shape(z_dim: int, num_frames: int, height: int, width: int,
                          vae_stride: tuple, patch_size: tuple) -> tuple:
        """Compute noise target shape from ORIGINAL video dimensions (before VAE)."""
        target_t = (num_frames - 1) // vae_stride[0] + 1
        target_h = height // vae_stride[1]
        target_w = width // vae_stride[2]
        return (z_dim, target_t, target_h, target_w)

    @torch.no_grad()
    def run(self, input_video: str, prompt: str,
            guidance_scale: float = None, num_steps: int = None,
            seed: int = None) -> torch.Tensor:
        """Run video editing inference."""

        if guidance_scale is None:
            guidance_scale = self.guidance
        if num_steps is None:
            num_steps = self.steps
        if seed is None:
            seed = self.seed
        if seed < 0:
            seed = random.randint(0, sys.maxsize)

        use_cfg = guidance_scale != 1.0

        # Encode inputs (adaptive resize)
        source_latent, num_frames, orig_h, orig_w = self.encode_video(input_video)
        ctx, ctx_null = self.encode_text(prompt)
        self.model.to(self.device)

        patch_size = tuple(self.cfg['model']['patch_size'])
        vae_stride = tuple(self.cfg['model']['vae_stride'])
        z_dim = self.vae.model.z_dim
        target_shape = self._get_target_shape(z_dim, num_frames, orig_h, orig_w, vae_stride, patch_size)

        t_seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) / (patch_size[1] * patch_size[2]) * target_shape[1])
        s_seq_len = t_seq_len
        seq_len = t_seq_len + s_seq_len

        print(f"[DEBUG] orig video: frames={num_frames} H={orig_h} W={orig_w}")
        print(f"[DEBUG] vae_stride={vae_stride} patch_size={patch_size}")
        print(f"[DEBUG] target_shape={target_shape}")
        print(f"[DEBUG] t_seq_len={t_seq_len} s_seq_len={s_seq_len} seq_len={seq_len}")
        print(f"[DEBUG] source_latent.shape={source_latent.shape}")

        # Setup scheduler (matches remote: FlowUniPCMultistepScheduler)
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.cfg['scheduler']['num_train_timesteps'],
            shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(num_steps, device=self.device, shift=self.shift)
        timesteps = scheduler.timesteps

        seed_g = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(target_shape, device=self.device, dtype=torch.float32, generator=seed_g)
        latent = noise

        with torch.autocast(device_type='cuda', dtype=self.model.dtype, enabled=True):
            for t in tqdm(timesteps, desc='denoising'):
                timestep = torch.tensor([t], device=self.device)

                if use_cfg:
                    noise_uncond = self.model(
                        [latent], t=timestep, context=ctx_null,
                        seq_len=seq_len,
                        source_video_latent=[source_latent[0]]
                    )[0]
                    noise_cond = self.model(
                        [latent], t=timestep, context=ctx,
                        seq_len=seq_len,
                        source_video_latent=[source_latent[0]]
                    )[0]
                    noise_pred = guidance_scale * (noise_cond - noise_uncond) + noise_uncond
                else:
                    noise_pred = self.model(
                        [latent], t=timestep, context=ctx,
                        seq_len=seq_len,
                        source_video_latent=[source_latent[0]]
                    )[0]

                latent = scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                    return_dict=False)[0].squeeze(0)

        # Decode
        video = self.vae.decode([latent])
        self.model.cpu()
        torch.cuda.empty_cache()
        return video[0]


def main():
    parser = argparse.ArgumentParser(description='LIVEditor — video editing inference')
    parser.add_argument('--config', type=str, default='configs/inference.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--guidance', type=float, default=None)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--backend', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.backend:
        config['attention']['backend'] = args.backend

    print(f"[LIVEditor] Loading checkpoint: {args.checkpoint}")
    print(f"[LIVEditor] Backend: {config['attention']['backend']}")

    editor = LIVEditor(config, args.checkpoint)

    print(f"[LIVEditor] Input: {args.input}")
    print(f"[LIVEditor] Prompt: {args.prompt}")

    video = editor.run(
        input_video=args.input,
        prompt=args.prompt,
        guidance_scale=args.guidance,
        num_steps=args.steps,
        seed=args.seed,
    )

    save_video_ffmpeg(video, args.output, fps=config['inference']['output_fps'])
    print(f"[LIVEditor] Saved output to: {args.output}")


if __name__ == '__main__':
    main()
