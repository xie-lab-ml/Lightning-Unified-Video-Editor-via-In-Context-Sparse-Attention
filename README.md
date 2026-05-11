# LIVEditor

### Lightning Unified Video Editing via In-Context Sparse Attention

<div align="center">

**Shitong Shao** · **Zikai Zhou** · **Haopeng Li** · **Yingwei Song** · **Wenliang Zhong** · **Lichen Bai** · **Zeke Xie**

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://xie-lab-ml.github.io/liveditor-page/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2605.04569)
[![Hugging Face](https://img.shields.io/badge/🤗_Hub-Model-yellow)](https://huggingface.co/sst12345/liveditor)

<p align="center">
  <img src="./assets/live_visualization.png" alt="Teaser" width="90%">
</p>

</div>

---

## 📖 Introduction

Video editing with diffusion transformers suffers from the quadratic complexity of full self-attention — O(S²) in total token count — making it prohibitively expensive when both source and generated video tokens must be processed jointly.

**LIVEditor** addresses this with **In-Context Sparse Attention**: a lightweight, training-free block-retrieval mechanism that efficiently selects the most relevant source-video tokens for each query block, avoiding the need for dense attention over the full sequence.

> **Key Idea**: store compressed KV representations of the source video, retrieve only the top-*k* most relevant blocks via compressed attention scores, and apply sparse piecewise attention for the diffuse query blocks while using FlashAttention only for the most peaked ones.

**Key results**:
- A strong open-source video editing model leading in multiple aspects.
- The first sparse attention for video editing
- ⚡ **2.8× faster** than FlashAttention-2 at 65K tokens on RTX 4090
- 🎯 Lightweight fine-tuning — only **80 steps** on ~100K video pairs
- 🔌 Pluggable backend — supports **TileLang** and **Triton** sparse kernels

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/xie-lab-ml/LIVEditor.git
cd LIVEditor
pip install -r requirements.txt
```

### 2. Download Weights

| Component | Source | Path |
|-----------|--------|------|
| Wan 2.2-T2V-A14B | [Official](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | `pretrained_weights/Wan2.2-T2V-A14B/` |
| LIVEditor checkpoint | [🤗 Hugging Face](https://huggingface.co/sst12345/liveditor) | `liveditor_ckpt.bin` |

```bash
# Download fine-tuned checkpoint from Hugging Face
pip install huggingface_hub
huggingface-cli download sst12345/liveditor liveditor_ckpt.bin --local-dir .
```

Configure paths in `inference.yaml`:

```yaml
base_model_path: pretrained_weights/Wan2.2-T2V-A14B/
# resume_ckpt is passed via CLI:  --checkpoint liveditor_ckpt.bin
```

### 3. Run Demo

```bash
python inference.py \
    --config inference.yaml \
    --checkpoint liveditor_ckpt.bin \
    --input assets/input.mp4 \
    --prompt "Add a small golden crown with delicate jewels on top of the girl's head..." \
    --output result.mp4
```

<table>
<tr><th>Input</th><th>Output (TileLang)</th><th>Output (Triton)</th></tr>
<tr>
  <td><video src="assets/input.mp4" width="200"/></td>
  <td><video src="assets/output_tilelang.mp4" width="200"/></td>
  <td><video src="assets/output_triton.mp4" width="200"/></td>
</tr>
</table>

---

## 🔧 Usage

### Inference

```
python inference.py \
    --config inference.yaml \                     # config file
    --checkpoint <path-to-ckpt> \                 # fine-tuned checkpoint
    --input <input-video.mp4> \                   # source video
    --prompt "<editing-instruction>" \            # text prompt
    --output <output.mp4>                         # output path

# Optional flags
    --guidance 2.5 \                              # CFG scale (default: 2.5)
    --steps 32 \                                  # denoising steps (default: 32)
    --seed 42 \                                   # random seed
    --backend tilelang                            # sparse kernel: tilelang (default) | triton
```

### Switching the Sparse Backend

```python
# inference.yaml
attention:
  backend: tilelang      # or 'triton'
```

Both backends produce visually identical results (mean absolute error < 1e-4).

---

## 🧠 Method

<div align="center">
  <img src="./assets/in_context_sparse_attention.png" alt="Architecture" width="90%">
</div>

LIVEditor introduces three key components on top of the Wan 2.2 diffusion backbone:

### 1. Block-wise Compression

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BLOCK_M` | 64 | Block size for Q/K/V partitioning |
| Compressed dim | `[B, H, NUM_BLOCKS, D]` | Average-pooled per-block representation |

Raw Q, K, V tensors of shape `[B, H, S, D]` are divided into blocks of 64 tokens and averaged, yielding compact proxies for efficient relevance scoring.

### 2. In-Context Top-K Retrieval

For each query block, a compressed attention score matrix `Q_c @ K_c^T` is computed. The top-*k* source-video blocks (from the `s_part`) with the highest scores are selected and appended to the target-video KV cache:

```
new_KV = [K_target | K_selected_source_blocks]
```

This keeps the KV length to `t_seq + topK × 64` tokens, far smaller than the full `t_seq + s_seq`.

### 3. Sharpness-Aware Split

Query blocks are ranked by attention sharpness (the sum of top-*k* compressed attention weights). The most peaked blocks receive **full FlashAttention**, while the diffuse blocks use **sparse piecewise attention**:

| Split | Ratio | Attention | Cost |
|-------|-------|-----------|------|
| Flat (peaked) | 50% | FlashAttention-3 | O(T²) — accurate |
| Sharp (diffuse) | 50% | Sparse Top-K + Interval Approx. | O(T·(t+topK)) — fast |

### Inference Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SAR` (Sparsity) | 0.0625 | Fraction of KV blocks for exact Top-K |
| `SRR` (Select Ratio) | 0.125 | Block selection ratio |
| `Flat Ratio` | 0.5 | Fraction of Q blocks using full attention |
| `Block Size` | 64 | Token block granularity |
| Sampling Steps | 32 | Denoising steps (Flow UniPC) |
| Guidance Scale | 2.5 | Classifier-free guidance |
| Shift | 6.0 | Flow-matching timestep shift |

---

## 📊 Benchmark

### EditVerse Benchmark

| Method | CLIP-T ↑ | PickScore ↑ | FlowSim ↑ | TemCon ↑ |
|--------|----------|-------------|-----------|----------|
| TokenFlow | 0.261 | 0.193 | 0.883 | 0.972 |
| CoDeF | 0.252 | 0.187 | 0.854 | 0.965 |
| FateZero | 0.249 | 0.188 | 0.861 | 0.968 |
| Pix2Video | 0.262 | 0.192 | 0.887 | 0.971 |
| AnyV2V | 0.268 | 0.198 | 0.891 | 0.973 |
| InsV2V | 0.272 | 0.201 | 0.895 | 0.974 |
| UniEdit | 0.276 | 0.204 | 0.898 | 0.976 |
| I2VEdit | 0.280 | 0.208 | 0.902 | 0.977 |
| **LIVEditor** | **0.289** | **0.215** | **0.911** | **0.981** |

> Detailed benchmark results and comparisons on VBench, VIPSeg, and DAVIS are available in the [paper](https://arxiv.org/abs/2605.04569).

---

## 📁 Project Structure

```
LIVEditor/
├── inference.py                       # Main inference entrypoint
├── inference.yaml                     # Default config
├── model.py                           # WanModel with in-context sparse attention
├── scheduler.py                       # Flow UniPC scheduler
├── fm_solvers.py                      # Flow matching utilities
├── requirements.txt                   # Python dependencies
├── infer.sh                           # Inference launch script
├── in_context_sparse_attention/       # Pluggable sparse kernels
│   ├── editing_sparse_attention.py    # Main attention function (backend-agnostic)
│   ├── tilelang_kernels.py            # TileLang kernel definitions
│   ├── tilelang_host.py               # TileLang host wrapper
│   ├── triton_kernels.py              # Triton kernel definitions
│   └── triton_host.py                 # Triton host wrapper
├── wanx/                              # Wan model components
│   ├── model.py                       # WanModel class
│   ├── vae.py                         # WanVAE (encode/decode)
│   ├── t5.py                          # T5 text encoder
│   ├── scheduler.py                   # Scheduler (legacy path)
│   ├── attention.py                   # FlashAttention wrapper (FA3)
│   └── utils.py                       # Video I/O utilities
├── model_dit/                         # Distributed training stubs
├── configs/                           # Experiment configs
├── assets/                            # Demo assets
│   ├── input.mp4                      # Example input video
│   ├── output_tilelang.mp4            # TileLang backend output
│   ├── output_triton.mp4              # Triton backend output
│   └── prompt.txt                     # Example prompt
└── README.md
```

---

## 🛠 Backend Details

| | TileLang | Triton |
|---|---|---|
| Kernel framework | TVM-based TileLang | Triton 3.5 |
| K/V alignment | Manual pad to 64× | Auto boundary handling |
| Forward precision | bf16 (mean abs err < 1e-4 vs Triton) | bf16 (reference) |
| Recommended GPU | H100 / RTX 4090 | H100 / RTX 4090 |

---

## 📝 Citation

```bibtex
@inproceedings{shao2026liveditor,
  title   = {Lightning Unified Video Editing via In-Context Sparse Attention},
  author  = {Shitong Shao and Zikai Zhou and Haopeng Li and Yingwei Song and Wenliang Zhong and Lichen Bai and Zeke Xie},
  booktitle={The Forty-Third International Conference on Machine Learning},
  year    = {2026},
}
```

## 📄 License & Acknowledgements

This project is built upon [Wan 2.2](https://github.com/Wan-Video/Wan2.2) and [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) by Alibaba. The in-context sparse attention kernels are powered by [TileLang](https://github.com/tilelang/tilelang) and [Triton](https://github.com/triton-lang/triton). FlashAttention is provided by [flash-attn](https://github.com/Dao-AILab/flash-attention). We thank the authors for their open-source contributions.
