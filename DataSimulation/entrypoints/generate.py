from __future__ import annotations

import argparse
import json
from pathlib import Path

from bse_renderer.pipeline.generate import generate_batch_dataset
from bse_renderer.renderer import RENDER_MODEL_MODES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch dataset generation")
    parser.add_argument("--config-dir", required=True, help="Path to config directory")
    parser.add_argument("--num-images", required=True, type=int, help="Number of images")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--render-model", choices=RENDER_MODEL_MODES, default=None)
    parser.add_argument(
        "--disable-noise",
        action="store_true",
        help="Write noiseless images by disabling Poisson, thermal, and contamination effects.",
    )
    parser.add_argument("--voxel-depth-mm", type=float, default=None)
    parser.add_argument("--slice-offset-mm", type=float, default=None)
    parser.add_argument("--slice-tilt-deg", type=float, default=None)
    parser.add_argument("--slice-rotation-deg", type=float, default=None)
    parser.add_argument("--fixed-slice", action="store_true")
    parser.add_argument("--render-view", choices=["slice", "top_surface"], default="slice")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = generate_batch_dataset(
        config_dir=args.config_dir,
        output_dir=Path(args.output_dir),
        num_images=args.num_images,
        image_size=args.image_size,
        seed=args.seed,
        workers=args.workers,
        calibration_path=args.calibration_path,
        render_model_mode=args.render_model,
        disable_noise=args.disable_noise,
        voxel_depth_mm=args.voxel_depth_mm,
        slice_offset_mm=args.slice_offset_mm,
        slice_tilt_deg=args.slice_tilt_deg,
        slice_rotation_deg=args.slice_rotation_deg,
        randomize_slice=not args.fixed_slice,
        render_view_mode=args.render_view,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
