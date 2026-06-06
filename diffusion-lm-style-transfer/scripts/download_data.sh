#!/usr/bin/env bash
# Download and prepare style transfer datasets

set -euo pipefail

DATA_DIR="${DATA_DIR:-./data/raw}"
mkdir -p "$DATA_DIR"

echo "Downloading Yelp Polarity dataset..."
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("yelp_polarity")
ds["train"].to_json("data/raw/yelp_train.jsonl")
ds["test"].to_json("data/raw/yelp_test.jsonl")
print(f"Train: {len(ds['train'])} | Test: {len(ds['test'])}")
EOF

echo "Done. Data saved to $DATA_DIR"
