from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataset_contract import validate_dataset_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a weld-topo-segmentation/v2 dataset and detect split leakage."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip opening image and mask files.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_dataset_contract(
        args.dataset_root,
        inspect_images=not args.metadata_only,
    )
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
