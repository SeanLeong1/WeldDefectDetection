from __future__ import annotations

import argparse
import json

from src.training.train_temporal_predictor import train_temporal_predictor


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a clean-only bidirectional TOPO surface predictor.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = train_temporal_predictor(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
