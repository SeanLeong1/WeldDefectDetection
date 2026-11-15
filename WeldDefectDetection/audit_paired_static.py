from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.paired_static_audit import audit_paired_static_datasets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify paired static datasets for controlled robustness evaluation."
    )
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--variant-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_paired_static_datasets(args.reference_root, args.variant_root)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
