from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.static_audit import audit_static_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a static TOPO dataset and optional independence comparisons."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--compare-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_static_dataset(args.dataset_root, compare_roots=args.compare_root)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
