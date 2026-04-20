from __future__ import annotations

import argparse
import json
from pathlib import Path

from bse_renderer.gui import App
from bse_renderer.pipeline.generate import generate_batch_dataset, generate_single_sample
from bse_renderer.validation.casino_comparison import run_casino_comparison
from bse_renderer.validation.signal_tests import run_signal_tests
from bse_renderer.visualisation.image_preview import preview_channels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bse_renderer")
    sub = parser.add_subparsers(dest="mode", required=True)

    gui = sub.add_parser("gui", help="Launch interactive GUI")
    gui.add_argument("--config-dir", default=str(Path("config").resolve()))

    single = sub.add_parser("single", help="Render single sample")
    single.add_argument("--config-dir", required=True)
    single.add_argument("--seed", type=int, default=42)
    single.add_argument("--image-size", type=int, default=256)
    single.add_argument("--material", default=None, help="Material preset name override")
    single.add_argument(
        "--avg-z",
        type=float,
        default=None,
        help="Use average atomic number to estimate material parameters",
    )
    single.add_argument(
        "--defect-plan",
        default=None,
        help="Manual defect plan, e.g. keyhole_pore(diameter_mm=1.0,depth_mm=0.6)",
    )
    single.add_argument("--max-defects", type=int, default=3)
    single.add_argument("--voxel-depth-mm", type=float, default=None)
    single.add_argument("--slice-offset-mm", type=float, default=None)
    single.add_argument("--slice-tilt-deg", type=float, default=None)
    single.add_argument("--slice-rotation-deg", type=float, default=None)
    single.add_argument("--preview", action="store_true")

    batch = sub.add_parser("batch", help="Batch generation")
    batch.add_argument("--config-dir", required=True)
    batch.add_argument("--num-images", required=True, type=int)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--image-size", type=int, default=256)
    batch.add_argument("--seed", type=int, default=42)
    batch.add_argument("--workers", type=int, default=1)
    batch.add_argument("--voxel-depth-mm", type=float, default=None)
    batch.add_argument("--slice-offset-mm", type=float, default=None)
    batch.add_argument("--slice-tilt-deg", type=float, default=None)
    batch.add_argument("--slice-rotation-deg", type=float, default=None)
    batch.add_argument(
        "--fixed-slice",
        action="store_true",
        help="Use the provided/default slice controls for every sample instead of randomizing slice orientation.",
    )

    validation = sub.add_parser("validation", help="Run validation suite")
    validation.add_argument("--config-dir", required=False)
    validation.add_argument(
        "--casino-exe",
        default=None,
        help="Path to CASINO adapter executable (or set BSE_CASINO_EXE).",
    )
    validation.add_argument(
        "--validation-run-root",
        default=str((Path("data") / "output" / "validation_reports").resolve()),
        help="Directory for validation report artifacts.",
    )
    validation.add_argument(
        "--casino-budget",
        choices=["fast", "medium", "high"],
        default="medium",
        help="Runtime budget for regime sweep case count.",
    )
    validation.add_argument(
        "--casino-calibrate",
        action="store_true",
        help="Run CASINO calibration and apply coefficients during validation sweeps.",
    )
    validation.add_argument(
        "--casino-calibration-cases",
        type=int,
        default=18,
        help="Number of stratified CASINO cases to use for calibration.",
    )
    validation.add_argument(
        "--casino-timeout-s",
        type=float,
        default=240.0,
        help="Timeout per CASINO case execution in seconds.",
    )
    validation.add_argument(
        "--regime-cases",
        type=int,
        default=12,
        help="Cases per regime (used when budget override is not active).",
    )
    validation.add_argument(
        "--standards-t-mm",
        type=float,
        default=None,
        help="Override ISO thickness t (mm).",
    )
    validation.add_argument(
        "--standards-s-mm",
        type=float,
        default=None,
        help="Override ISO throat s (mm).",
    )
    validation.add_argument(
        "--contrast-threshold",
        type=float,
        default=None,
        help="Override defect detection contrast threshold for standards proxy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "gui":
        App(args.config_dir).run()
        return 0

    if args.mode == "single":
        sample = generate_single_sample(
            config_dir=args.config_dir,
            seed=args.seed,
            image_size=args.image_size,
            material_name=args.material,
            average_atomic_number=args.avg_z,
            defect_plan=args.defect_plan,
            max_defects=args.max_defects,
            voxel_depth_mm=args.voxel_depth_mm,
            slice_offset_mm=args.slice_offset_mm,
            slice_tilt_deg=args.slice_tilt_deg,
            slice_rotation_deg=args.slice_rotation_deg,
        )
        result = {
            "metadata": sample["metadata"],
            "defects": [d.defect_type for d in sample["defects"]],
        }
        print(json.dumps(result, indent=2))
        if args.preview:
            preview_channels(
                sample["channels"].compo,
                sample["channels"].topo,
                sample["defect_mask"],
                title="Single Sample",
            )
        return 0

    if args.mode == "batch":
        summary = generate_batch_dataset(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            num_images=args.num_images,
            image_size=args.image_size,
            seed=args.seed,
            workers=args.workers,
            voxel_depth_mm=args.voxel_depth_mm,
            slice_offset_mm=args.slice_offset_mm,
            slice_tilt_deg=args.slice_tilt_deg,
            slice_rotation_deg=args.slice_rotation_deg,
            randomize_slice=not args.fixed_slice,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.mode == "validation":
        out = {
            "signal_tests": run_signal_tests(),
            "casino_comparison": run_casino_comparison(
                config_dir=args.config_dir or "config",
                casino_exe=args.casino_exe,
                run_root=args.validation_run_root,
                run_budget=args.casino_budget,
                calibrate=args.casino_calibrate,
                calibration_cases=args.casino_calibration_cases,
                timeout_s=args.casino_timeout_s,
                regime_cases=args.regime_cases,
                t_mm=args.standards_t_mm,
                s_mm=args.standards_s_mm,
                contrast_threshold=args.contrast_threshold,
            ),
        }
        print(json.dumps(out, indent=2))
        return 0

    parser.error("Unknown mode")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
