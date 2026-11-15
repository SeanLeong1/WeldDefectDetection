from __future__ import annotations

import argparse
import json

from src.evaluation.compare_robustness import compare_robustness_evaluations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare paired frozen-model evaluation summaries."
    )
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--variant-summary", required=True)
    parser.add_argument("--paired-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_robustness_evaluations(
        reference_summary=args.reference_summary,
        variant_summary=args.variant_summary,
        paired_audit=args.paired_audit,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
