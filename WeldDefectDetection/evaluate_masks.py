from __future__ import annotations

import json

from src.evaluation.evaluate_dataset import evaluate_dataset


def main() -> int:
    summary = evaluate_dataset(
        config_path="config.yaml",
        algorithm="morphology",
        split="val",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
