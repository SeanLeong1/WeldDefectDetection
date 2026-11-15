from __future__ import annotations

import argparse
import json

from src.evaluation.completion_verifier import verify_completion_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the frozen synthetic-project functionality manifest."
    )
    parser.add_argument("--manifest", default="completion_manifest_20260908.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = verify_completion_manifest(args.manifest, output_path=args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
