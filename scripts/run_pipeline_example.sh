#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash scripts/run_pipeline_example.sh <config> <scene_train_dir> <work_dir> [merge_seed]"
  exit 1
fi

CONFIG_PATH="$1"
SCENE_DIR="$2"
WORK_DIR="$3"
MERGE_SEED="${4:-0}"

K_NEIGHBORS="${K_NEIGHBORS:-5}"
QUANT_THRESHOLD="${QUANT_THRESHOLD:-0.02}"
QAT_ITERS="${QAT_ITERS:-7000}"
TRAIN_EVAL_SPLIT="${TRAIN_EVAL_SPLIT:-1}"
EVAL_ONLY="${EVAL_ONLY:-1}"

python scene_partition.py -c "$CONFIG_PATH" -s "$SCENE_DIR" -o "$WORK_DIR"
python train.py -c "$CONFIG_PATH" -s "$SCENE_DIR" -o "$WORK_DIR"
python merge.py -o "$WORK_DIR" --k_neighbors "$K_NEIGHBORS" --seed "$MERGE_SEED"
python finetune_qat.py -s "$SCENE_DIR" -m "$WORK_DIR/point_cloud_merged_seed${MERGE_SEED}.ply" -o "$WORK_DIR" --iterations "$QAT_ITERS"
python compress_adaptive.py -i "$WORK_DIR/point_cloud_finetuned_final.ply" -o "$WORK_DIR/point_cloud_quantized.ply" -t "$QUANT_THRESHOLD"

if [ "$TRAIN_EVAL_SPLIT" = "1" ]; then
  if [ "$EVAL_ONLY" = "1" ]; then
    python render.py -o "$WORK_DIR" --train_eval_split --eval_only
    python metrics.py -o "$WORK_DIR" --train_eval_split --eval_only
  else
    python render.py -o "$WORK_DIR" --train_eval_split
    python metrics.py -o "$WORK_DIR" --train_eval_split
  fi
else
  python render.py -o "$WORK_DIR"
  python metrics.py -o "$WORK_DIR"
fi
