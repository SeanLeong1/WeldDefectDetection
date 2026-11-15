from __future__ import annotations

import argparse
import json

from src.evaluation.tune_classical import tune_classical_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select classical TOPO morphology parameters on a non-test dataset split."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.split == "test":
        parser.error("The test split must not be used for parameter selection.")
    summary = tune_classical_baseline(
        args.config,
        split=args.split,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
