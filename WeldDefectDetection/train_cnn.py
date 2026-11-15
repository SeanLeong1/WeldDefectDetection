from __future__ import annotations

import argparse
import json

from src.training.train_cnn import train_cnn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the WeldDefectDetection CNN on a DataSimulation dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to the training config YAML.")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed override.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = train_cnn(args.config, seed_override=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
