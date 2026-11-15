from __future__ import annotations

import argparse
import json

from src.evaluation.evaluate_temporal_predictor import evaluate_temporal_predictor


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen bidirectional TOPO predictor.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    summary = evaluate_temporal_predictor(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
