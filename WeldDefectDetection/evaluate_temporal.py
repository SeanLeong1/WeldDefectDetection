from __future__ import annotations

import argparse
import json

from src.evaluation.temporal_reconstruction import run_temporal_reconstruction


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate the bidirectional TOPO reconstruction-residual baseline."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--evaluation-split", default="test")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--frozen-threshold", type=float, default=None)
    parser.add_argument("--frozen-min-area", type=int, default=None)
    parser.add_argument(
        "--development-dataset-root",
        default=None,
        help="Required with --evaluation-split all to prove confirmation independence.",
    )
    args = parser.parse_args()
    summary = run_temporal_reconstruction(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        selection_split=args.selection_split,
        evaluation_split=args.evaluation_split,
        seed=args.seed,
        frozen_threshold=args.frozen_threshold,
        frozen_min_area=args.frozen_min_area,
        development_dataset_root=args.development_dataset_root,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
