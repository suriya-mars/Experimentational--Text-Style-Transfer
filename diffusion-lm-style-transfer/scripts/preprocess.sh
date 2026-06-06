#!/usr/bin/env bash
# Tokenize and cache datasets for training

set -euo pipefail

python - <<'EOF'
import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "test"]:
    path = Path(f"data/raw/yelp_{split}.jsonl")
    if not path.exists():
        print(f"Missing {path}, run download_data.sh first")
        continue
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    enc = tokenizer(texts, max_length=64, padding="max_length", truncation=True)
    out = [{"input_ids": ids, "label": lbl} for ids, lbl in zip(enc["input_ids"], labels)]
    (out_dir / f"yelp_{split}.jsonl").write_text("\n".join(json.dumps(r) for r in out))
    print(f"Saved {len(out)} {split} examples")
EOF
