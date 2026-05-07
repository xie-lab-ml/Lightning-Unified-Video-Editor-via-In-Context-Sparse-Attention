#!/bin/bash
# Submit LIVEditor single-GPU inference via slurm
# Usage: bash slurm_infer.sh <input_video> <prompt> <output_path>

INPUT=${1:?"Usage: $0 <input_video> <prompt> <output_path>"}
PROMPT=${2:?"Usage: $0 <input_video> <prompt> <output_path>"}
OUTPUT=${3:?"Usage: $0 <input_video> <prompt> <output_path>"}

sbatch \
    --chdir=/data/user/sshao213/zk_workspace/LIVEditor \
    --exclusive --nodes=1 --gres=gpu:1 --ntasks-per-node=1 \
    --cpus-per-task=8 --mem=128G \
    -p shaoshitong_rent \
    --nodelist=ACD1-[55] \
    -o slurm-%j.txt -e slurm-%j.txt \
    infer.sh "$INPUT" "$PROMPT" "$OUTPUT"
