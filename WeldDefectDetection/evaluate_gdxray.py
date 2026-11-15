from __future__ import annotations

import argparse
import json

from src.evaluation.evaluate_gdxray import evaluate_gdxray_cnn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zero-shot tiled CNN evaluation on paired GDXray W0001/W0002 images."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", default="data/raw/W0001")
    parser.add_argument("--mask-dir", default="data/raw/W0002")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=96)
    args = parser.parse_args()
    summary = evaluate_gdxray_cnn(
        config_path=args.config,
        checkpoint=args.checkpoint,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        stride=args.stride,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
