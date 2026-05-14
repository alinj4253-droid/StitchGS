#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_pipeline_example.sh <config> [merge_seed]"
  exit 1
fi

CONFIG_PATH="$1"
MERGE_SEED="${2:-0}"

K_NEIGHBORS="${K_NEIGHBORS:-5}"
QUANT_THRESHOLD="${QUANT_THRESHOLD:-0.02}"
QAT_ITERS="${QAT_ITERS:-7000}"
TRAIN_EVAL_SPLIT="${TRAIN_EVAL_SPLIT:-1}"
EVAL_ONLY="${EVAL_ONLY:-1}"

python scene_partition.py -c "$CONFIG_PATH"
python train.py           -c "$CONFIG_PATH"
python merge.py           -c "$CONFIG_PATH" --k_neighbors "$K_NEIGHBORS" --seed "$MERGE_SEED"
python finetune_qat.py    -c "$CONFIG_PATH" --iterations "$QAT_ITERS"    --seed "$MERGE_SEED"
python compress_adaptive.py -c "$CONFIG_PATH" -t "$QUANT_THRESHOLD"

if [ "$TRAIN_EVAL_SPLIT" = "1" ]; then
  if [ "$EVAL_ONLY" = "1" ]; then
    python render.py  -c "$CONFIG_PATH" --train_eval_split --eval_only
    python metrics.py -c "$CONFIG_PATH" --train_eval_split --eval_only
  else
    python render.py  -c "$CONFIG_PATH" --train_eval_split
    python metrics.py -c "$CONFIG_PATH" --train_eval_split
  fi
else
  python render.py  -c "$CONFIG_PATH"
  python metrics.py -c "$CONFIG_PATH"
fi
