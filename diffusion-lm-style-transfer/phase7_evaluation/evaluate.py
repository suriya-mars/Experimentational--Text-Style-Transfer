"""End-to-end evaluation script."""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import ContentPreservation, EvalResults, Fluency, SelfBLEU, StyleAccuracy


def evaluate_pipeline(
    source_texts: list[str],
    output_texts: list[str],
    target_style_label: int,
    output_path: str | None = None,
) -> EvalResults:
    print("Computing style accuracy...")
    style_acc = StyleAccuracy()(output_texts, target_style_label)

    print("Computing content preservation (BERTScore)...")
    content = ContentPreservation()(source_texts, output_texts)

    print("Computing fluency (GPT-2 perplexity)...")
    fluency = Fluency().perplexity(output_texts)

    print("Computing self-BLEU...")
    diversity = SelfBLEU()(output_texts)

    results = EvalResults(
        style_accuracy=style_acc,
        content_preservation=content,
        fluency_perplexity=fluency,
        self_bleu=diversity,
    )

    print("\n" + "=" * 40)
    print(results)
    print("=" * 40)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "style_accuracy": results.style_accuracy,
                    "content_preservation": results.content_preservation,
                    "fluency_perplexity": results.fluency_perplexity,
                    "self_bleu": results.self_bleu,
                },
                f,
                indent=2,
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_label", type=int, required=True)
    parser.add_argument("--save_path", default="./outputs/eval_results.json")
    args = parser.parse_args()

    with open(args.source) as f:
        sources = [line.strip() for line in f]
    with open(args.output) as f:
        outputs = [line.strip() for line in f]

    evaluate_pipeline(sources, outputs, args.target_label, args.save_path)
