#!/bin/bash
# LIVEditor-14B single-GPU inference
# Usage: bash infer.sh <input_video> <prompt> <output_path>

INPUT=${1:?"Usage: $0 <input_video> <prompt> <output_path>"}
PROMPT=${2:?"Usage: $0 <input_video> <prompt> <output_path>"}
OUTPUT=${3:?"Usage: $0 <input_video> <prompt> <output_path>"}

CHECKPOINT="/path/to/checkpoint"
CONFIG="configs/inference.yaml"

export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID:-0}"
source activate v2v

cd /path/to/LIVEditor-14B

python inference.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --input "$INPUT" \
    --prompt "$PROMPT" \
    --output "$OUTPUT"
