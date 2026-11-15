from __future__ import annotations

import argparse
import json

from src.evaluation.evaluate_dataset import evaluate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the morphology baseline or CNN checkpoint on a DataSimulation dataset split."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the evaluation config YAML.")
    parser.add_argument(
        "--algorithm",
        choices=["morphology", "cnn"],
        required=True,
        help="Which detector to evaluate.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split to evaluate, e.g. train, val, or test.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional CNN checkpoint path. Ignored for morphology evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to a timestamped run folder under config.eval.experiment_root.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset-root override, retained in the saved config snapshot.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = evaluate_dataset(
        config_path=args.config,
        algorithm=args.algorithm,
        split=args.split,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        dataset_root_override=args.dataset_root,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
