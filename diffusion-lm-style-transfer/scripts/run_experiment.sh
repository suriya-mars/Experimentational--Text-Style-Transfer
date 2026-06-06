#!/usr/bin/env bash
# Full experiment: pretrain → classify → finetune → evaluate

set -euo pipefail

echo "[1/4] Downloading data..."
bash scripts/download_data.sh

echo "[2/4] Preprocessing..."
bash scripts/preprocess.sh

echo "[3/4] Running training pipeline..."
python phase6_training_pipeline/pretrain.py
python phase6_training_pipeline/train_classifier.py
python phase6_training_pipeline/finetune.py

echo "[4/4] Evaluating..."
python phase7_evaluation/evaluate.py \
  --source data/processed/test_source.txt \
  --output outputs/transferred.txt \
  --target_label 1 \
  --save_path outputs/eval_results.json

echo "Experiment complete. Results in outputs/eval_results.json"
