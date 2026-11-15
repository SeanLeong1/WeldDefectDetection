from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bse_renderer.gui import App
from bse_renderer.domain_randomisation.dataset_forms import (
    generate_balanced_topo_dataset,
    generate_topo_sequence_dataset,
)
from bse_renderer.pipeline.generate import (
    generate_batch_dataset,
    generate_single_sample,
    render_comparison_case,
)
from bse_renderer.renderer import (
    EMISSION_B_MODE_CONSTANT,
    EMISSION_B_MODE_LAMBERTIAN,
    EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    EMISSION_B_MODE_SIN_FIT,
    EMISSION_B_MODE_QUADRATIC_FIT,
    RENDER_MODEL_FORMULA_ONLY,
    RENDER_MODEL_MODES,
)
from bse_renderer.validation.casino_benchmarks import (
    assemble_casino_benchmark_campaign,
    evaluate_casino_benchmark_fit,
    extract_casino_run_note_values,
    fit_casino_benchmark_calibration,
)
from bse_renderer.validation.casino_emission_fit import fit_casino_emission_b
from bse_renderer.validation.emission_b_render_comparison import compare_emission_b_render_modes
from bse_renderer.validation.emission_b_topo_ablation import run_emission_b_topo_ablation
from bse_renderer.validation.casino_campaign_summary import (
    summarize_casino_benchmark_campaign,
)
from bse_renderer.validation.casino_comparison import run_casino_comparison
from bse_renderer.validation.comparison_summary import summarize_comparison_cases
from bse_renderer.validation.signal_tests import run_signal_tests
from bse_renderer.validation.visibility_pilot import run_visibility_pilot
from bse_renderer.visualisation.image_preview import preview_channels

EMISSION_B_MODE_CHOICES = [
    'angular_table',
    EMISSION_B_MODE_LAMBERTIAN,
    EMISSION_B_MODE_CONSTANT,
    EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    EMISSION_B_MODE_SIN_FIT,
    EMISSION_B_MODE_QUADRATIC_FIT,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bse_renderer")
    sub = parser.add_subparsers(dest="mode", required=True)

    gui = sub.add_parser("gui", help="Launch interactive GUI")
    gui.add_argument("--config-dir", default=str(Path("config").resolve()))

    single = sub.add_parser("single", help="Render single sample")
    single.add_argument("--config-dir", required=True)
    single.add_argument("--seed", type=int, default=42)
    single.add_argument("--image-size", type=int, default=256)
    single.add_argument("--pixel-size-mm", type=float, default=0.05)
    single.add_argument("--weld-layout", default="local_patch")
    single.add_argument("--seam-rotation-deg", type=float, default=0.0)
    single.add_argument("--seam-offset-mm", type=float, default=0.0)
    single.add_argument("--circle-radius-mm", type=float, default=80.0)
    single.add_argument("--circle-position-deg", type=float, default=90.0)
    single.add_argument("--calibration-path", default=None)
    single.add_argument("--render-model", choices=RENDER_MODEL_MODES, default=None)
    single.add_argument("--emission-b-mode", choices=EMISSION_B_MODE_CHOICES, default="lambertian")
    single.add_argument("--emission-b-path", default=None)
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
    single.add_argument(
        "--fixed-defect-location",
        action="store_true",
        help="Keep generated defect placement anchored instead of randomizing position.",
    )
    single.add_argument("--preview", action="store_true")
    single.add_argument(
        "--render-view",
        choices=["slice", "top_surface"],
        default="slice",
        help="Render a cross-section slice or the externally visible top surface.",
    )

    batch = sub.add_parser("batch", help="Batch generation")
    batch.add_argument("--config-dir", required=True)
    batch.add_argument("--num-images", required=True, type=int)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--image-size", type=int, default=256)
    batch.add_argument("--seed", type=int, default=42)
    batch.add_argument("--workers", type=int, default=1)
    batch.add_argument("--pixel-size-mm", type=float, default=0.05)
    batch.add_argument("--weld-layout", default="local_patch")
    batch.add_argument("--seam-rotation-deg", type=float, default=0.0)
    batch.add_argument("--seam-offset-mm", type=float, default=0.0)
    batch.add_argument("--circle-radius-mm", type=float, default=80.0)
    batch.add_argument("--circle-position-deg", type=float, default=90.0)
    batch.add_argument("--calibration-path", default=None)
    batch.add_argument("--render-model", choices=RENDER_MODEL_MODES, default=None)
    batch.add_argument("--emission-b-mode", choices=EMISSION_B_MODE_CHOICES, default="lambertian")
    batch.add_argument("--emission-b-path", default=None)
    batch.add_argument(
        "--disable-noise",
        action="store_true",
        help="Write noiseless images by disabling Poisson, thermal, and contamination effects.",
    )
    batch.add_argument("--voxel-depth-mm", type=float, default=None)
    batch.add_argument("--slice-offset-mm", type=float, default=None)
    batch.add_argument("--slice-tilt-deg", type=float, default=None)
    batch.add_argument("--slice-rotation-deg", type=float, default=None)
    batch.add_argument(
        "--fixed-slice",
        action="store_true",
        help="Use the provided/default slice controls for every sample instead of randomizing slice orientation.",
    )
    batch.add_argument(
        "--fixed-defect-location",
        action="store_true",
        help="Keep generated defect placement anchored instead of randomizing defect position.",
    )
    batch.add_argument(
        "--render-view",
        choices=["slice", "top_surface"],
        default="slice",
        help="Use top_surface for surface-only CV datasets.",
    )

    compare = sub.add_parser(
        "compare-render",
        help="Render the same case across the formula-only, CASINO, and geometry-profile model ladder.",
    )
    compare.add_argument("--config-dir", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--calibration-path", default=None)
    compare.add_argument("--render-model", choices=RENDER_MODEL_MODES, default=None)
    compare.add_argument("--emission-b-mode", choices=EMISSION_B_MODE_CHOICES, default="lambertian")
    compare.add_argument("--emission-b-path", default=None)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--image-size", type=int, default=256)
    compare.add_argument("--pixel-size-mm", type=float, default=0.05)
    compare.add_argument("--weld-layout", default="local_patch")
    compare.add_argument("--seam-rotation-deg", type=float, default=0.0)
    compare.add_argument("--seam-offset-mm", type=float, default=0.0)
    compare.add_argument("--circle-radius-mm", type=float, default=80.0)
    compare.add_argument("--circle-position-deg", type=float, default=90.0)
    compare.add_argument("--material", default=None, help="Material preset name override")
    compare.add_argument("--avg-z", type=float, default=None)
    compare.add_argument("--defect-plan", default=None)
    compare.add_argument("--max-defects", type=int, default=3)
    compare.add_argument("--voxel-depth-mm", type=float, default=None)
    compare.add_argument("--slice-offset-mm", type=float, default=None)
    compare.add_argument("--slice-tilt-deg", type=float, default=None)
    compare.add_argument("--slice-rotation-deg", type=float, default=None)
    compare.add_argument(
        "--fixed-defect-location",
        action="store_true",
        help="Keep generated defect placement anchored instead of randomizing position.",
    )
    compare.add_argument("--stem", default=None, help="Optional output filename stem.")

    compare_summarize = sub.add_parser(
        "compare-summarize",
        help="Summarize compare-render JSON outputs into one CSV/JSON table.",
    )
    compare_summarize.add_argument(
        "--comparison-root",
        default=str((Path("data") / "output" / "comparison_cases").resolve()),
    )
    compare_summarize.add_argument("--output-dir", default=None)

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
    validation.add_argument(
        "--topo-contrast-threshold",
        type=float,
        default=None,
        help="Override the TOPO-channel contrast threshold independently.",
    )

    visibility_pilot = sub.add_parser(
        "visibility-pilot",
        help="Generate a balanced noiseless top-surface defect visibility pilot.",
    )
    visibility_pilot.add_argument("--config-dir", default=str(Path("config").resolve()))
    visibility_pilot.add_argument(
        "--output-dir",
        default=str((Path("data") / "output" / "visibility_pilot").resolve()),
    )
    visibility_pilot.add_argument("--repeats", type=int, default=6)
    visibility_pilot.add_argument("--image-size", type=int, default=128)
    visibility_pilot.add_argument("--pixel-size-mm", type=float, default=0.05)
    visibility_pilot.add_argument("--seed", type=int, default=731)
    visibility_pilot.add_argument("--calibration-path", default=None)
    visibility_pilot.add_argument("--render-model", choices=RENDER_MODEL_MODES, default=None)
    visibility_pilot.add_argument(
        "--emission-b-mode",
        choices=EMISSION_B_MODE_CHOICES,
        default="lambertian",
    )
    visibility_pilot.add_argument("--emission-b-path", default=None)
    visibility_pilot.add_argument(
        "--no-images",
        action="store_true",
        help="Write only CSV/JSON/plot evidence, without per-case image files.",
    )

    topo_static = sub.add_parser(
        "topo-static-dataset",
        help="Generate a balanced independent-image TOPO-only dataset.",
    )
    topo_static.add_argument("--config-dir", default=str(Path("config").resolve()))
    topo_static.add_argument("--output-dir", required=True)
    topo_static.add_argument("--samples-per-class", type=int, default=60)
    topo_static.add_argument("--image-size", type=int, default=128)
    topo_static.add_argument("--pixel-size-mm", type=float, default=0.05)
    topo_static.add_argument("--seed", type=int, default=1201)
    topo_static.add_argument("--fixed-energy-kev", type=float, default=None)
    topo_static.add_argument("--render-model", choices=RENDER_MODEL_MODES, default=None)
    topo_static.add_argument(
        "--emission-b-mode",
        choices=EMISSION_B_MODE_CHOICES,
        default="lambertian",
    )
    topo_static.add_argument("--emission-b-path", default=None)
    topo_static.add_argument(
        "--normalization-path",
        default=None,
        help="Reuse a frozen training-derived topo_normalization.json instead of refitting.",
    )
    topo_static.add_argument(
        "--noise-profile",
        choices=("development", "none"),
        default="development",
        help="Use realistic stochastic acquisition effects, or 'none' for paired diagnostics.",
    )

    topo_sequences = sub.add_parser(
        "topo-sequence-dataset",
        help="Generate coherent ordered TOPO-only sequences for sliding-window prediction.",
    )
    topo_sequences.add_argument("--config-dir", default=str(Path("config").resolve()))
    topo_sequences.add_argument("--output-dir", required=True)
    topo_sequences.add_argument("--sequences-per-class", type=int, default=3)
    topo_sequences.add_argument("--frames-per-sequence", type=int, default=9)
    topo_sequences.add_argument("--frame-size", type=int, default=96)
    topo_sequences.add_argument("--stride-px", type=int, default=None)
    topo_sequences.add_argument("--context-radius", type=int, default=2)
    topo_sequences.add_argument("--pixel-size-mm", type=float, default=0.05)
    topo_sequences.add_argument("--seed", type=int, default=2201)
    topo_sequences.add_argument("--fixed-energy-kev", type=float, default=None)
    topo_sequences.add_argument("--render-model", choices=RENDER_MODEL_MODES, default=None)
    topo_sequences.add_argument(
        "--emission-b-mode",
        choices=EMISSION_B_MODE_CHOICES,
        default="lambertian",
    )
    topo_sequences.add_argument("--emission-b-path", default=None)
    topo_sequences.add_argument(
        "--normalization-path",
        default=None,
        help="Reuse a frozen training-derived topo_normalization.json instead of refitting.",
    )
    topo_sequences.add_argument(
        "--weld-layout-mode",
        choices=("local_patch", "straight_seam"),
        default="local_patch",
        help="Use straight_seam for non-overlapping sequences spanning a long weld.",
    )
    topo_sequences.add_argument(
        "--noise-profile",
        choices=("development", "none"),
        default="development",
        help="Use realistic stochastic acquisition effects, or 'none' for paired diagnostics.",
    )

    b_ablation = sub.add_parser(
        "emission-b-topo-ablation",
        help="Compare pooled constant and sine B(alpha) using paired TOPO renders.",
    )
    b_ablation.add_argument("--config-dir", default=str(Path("config").resolve()))
    b_ablation.add_argument("--b-fit-path", required=True)
    b_ablation.add_argument("--output-dir", required=True)
    b_ablation.add_argument("--repeats-per-class", type=int, default=3)
    b_ablation.add_argument("--image-size", type=int, default=128)
    b_ablation.add_argument("--pixel-size-mm", type=float, default=0.05)
    b_ablation.add_argument("--seed", type=int, default=3101)

    benchmark_fit = sub.add_parser(
        "benchmark-fit",
        help="Fit renderer surrogate calibration from curated CASINO benchmark summaries.",
    )
    benchmark_fit.add_argument("--benchmark-summary", required=True)
    benchmark_fit.add_argument("--config-dir", default=str(Path("config").resolve()))
    benchmark_fit.add_argument(
        "--output-dir",
        default=str((Path("data") / "output" / "benchmark_fits").resolve()),
    )
    benchmark_fit.add_argument(
        "--pixel-size-mm",
        type=float,
        default=0.05,
        help="Reference renderer pixel size used when mapping CASINO r50/r90 to blur coefficients.",
    )

    benchmark_validate = sub.add_parser(
        "benchmark-validate",
        help="Evaluate a fitted renderer calibration against curated CASINO benchmark summaries.",
    )
    benchmark_validate.add_argument("--benchmark-summary", required=True)
    benchmark_validate.add_argument("--calibration-path", required=True)
    benchmark_validate.add_argument("--config-dir", default=str(Path("config").resolve()))
    benchmark_validate.add_argument(
        "--output-dir",
        default=str((Path("data") / "output" / "benchmark_fits").resolve()),
    )
    benchmark_validate.add_argument("--pixel-size-mm", type=float, default=0.05)

    benchmark_assemble = sub.add_parser(
        "benchmark-assemble",
        help="Assemble a curated CASINO benchmark campaign JSON from exported graph .dat files.",
    )
    benchmark_assemble.add_argument("--export-root", required=True)
    benchmark_assemble.add_argument("--assembly-config", required=True)
    benchmark_assemble.add_argument("--output-path", required=True)

    benchmark_extract_run = sub.add_parser(
        "benchmark-extract-run",
        help="Extract the note-ready values for a single CASINO export folder.",
    )
    benchmark_extract_run.add_argument("--artifact-root", required=True)
    benchmark_extract_run.add_argument("--run-id", default=None)
    benchmark_extract_run.add_argument("--material", default=None)
    benchmark_extract_run.add_argument("--geometry", default=None)
    benchmark_extract_run.add_argument("--energy-kev", type=float, default=None)
    benchmark_extract_run.add_argument("--tilt-deg", type=float, default=None)
    benchmark_extract_run.add_argument("--beam-diameter-nm", type=float, default=None)
    benchmark_extract_run.add_argument("--electrons", type=int, default=None)
    benchmark_extract_run.add_argument("--quality-flag", default="usable")
    benchmark_extract_run.add_argument("--output-path", default=None)

    benchmark_summarize = sub.add_parser(
        "benchmark-summarize",
        help="Group a benchmark campaign and summarize BSE metrics across geometries.",
    )
    benchmark_summarize.add_argument("--benchmark-summary", required=True)
    benchmark_summarize.add_argument("--output-dir", default=None)
    benchmark_summarize.add_argument(
        "--group-by",
        default="geometry",
        help="Comma-separated record fields to group by, e.g. geometry or geometry,tilt_deg.",
    )
    benchmark_summarize.add_argument(
        "--include-nonusable",
        action="store_true",
        help="Include records whose quality_flag is not 'usable'.",
    )

    surface_b = sub.add_parser(
        "benchmark-fit-b-surface",
        help="Stream existing CASINO summaries, separate landing faces and fit surface-conditioned B with held-out checks.",
    )
    surface_b.add_argument("--cas-root", required=True)
    surface_b.add_argument("--output-dir", required=True)
    surface_b.add_argument("--plane-tolerance-nm", type=float, default=1.0)

    benchmark_fit_b = sub.add_parser(
        "benchmark-fit-b",
        help="Legacy global-Backscattered-class B analysis; use benchmark-fit-b-surface for corrected local-surface selection.",
    )
    benchmark_fit_b.add_argument("--cas-root", required=True)
    benchmark_fit_b.add_argument("--export-root", required=True)
    benchmark_fit_b.add_argument(
        "--output-dir",
        required=True,
    )
    benchmark_fit_b.add_argument(
        "--grid-points",
        type=int,
        default=2001,
        help="Number of grid points used in the 1D B search.",
    )

    benchmark_compare_b_render = sub.add_parser(
        "benchmark-compare-b-render",
        help="Compare pooled fitted constant B versus quadratic fitted B(alpha) in renderer space against per-tilt observed-B reference renders.",
    )
    benchmark_compare_b_render.add_argument("--config-dir", default=str(Path("config").resolve()))
    benchmark_compare_b_render.add_argument("--b-fit-report", required=True)
    benchmark_compare_b_render.add_argument("--emission-b-fit", required=True)
    benchmark_compare_b_render.add_argument("--output-dir", required=True)
    benchmark_compare_b_render.add_argument("--tilts", default=None, help="Comma-separated tilt degrees. Defaults to all tilts in the B-fit report.")
    benchmark_compare_b_render.add_argument("--seed", type=int, default=42)
    benchmark_compare_b_render.add_argument("--image-size", type=int, default=256)
    benchmark_compare_b_render.add_argument("--pixel-size-mm", type=float, default=0.05)
    benchmark_compare_b_render.add_argument("--material", default="304SS")
    benchmark_compare_b_render.add_argument("--render-model", choices=RENDER_MODEL_MODES, default=RENDER_MODEL_FORMULA_ONLY)
    benchmark_compare_b_render.add_argument("--calibration-path", default=None)
    benchmark_compare_b_render.add_argument("--weld-layout", default="local_patch")
    benchmark_compare_b_render.add_argument("--seam-rotation-deg", type=float, default=0.0)
    benchmark_compare_b_render.add_argument("--seam-offset-mm", type=float, default=0.0)
    benchmark_compare_b_render.add_argument("--circle-radius-mm", type=float, default=80.0)
    benchmark_compare_b_render.add_argument("--circle-position-deg", type=float, default=90.0)
    benchmark_compare_b_render.add_argument("--avg-z", type=float, default=None)
    benchmark_compare_b_render.add_argument("--defect-plan", default=None)
    benchmark_compare_b_render.add_argument("--max-defects", type=int, default=1)
    benchmark_compare_b_render.add_argument("--voxel-depth-mm", type=float, default=None)
    benchmark_compare_b_render.add_argument("--slice-offset-mm", type=float, default=None)
    benchmark_compare_b_render.add_argument("--slice-tilt-deg", type=float, default=None)
    benchmark_compare_b_render.add_argument("--slice-rotation-deg", type=float, default=None)
    benchmark_compare_b_render.add_argument(
        "--fixed-defect-location",
        action="store_true",
        help="Keep generated defect placement anchored instead of randomizing position.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "gui":
        try:
            App(args.config_dir).run()
        except KeyboardInterrupt:
            return 0
        return 0

    if args.mode == "single":
        sample = generate_single_sample(
            config_dir=args.config_dir,
            seed=args.seed,
            image_size=args.image_size,
            pixel_size_mm=args.pixel_size_mm,
            weld_layout_mode=args.weld_layout,
            seam_rotation_deg=args.seam_rotation_deg,
            seam_offset_mm=args.seam_offset_mm,
            circle_radius_mm=args.circle_radius_mm,
            circle_position_deg=args.circle_position_deg,
            calibration_path=args.calibration_path,
            render_model_mode=args.render_model,
            emission_b_mode=args.emission_b_mode,
            emission_b_path=args.emission_b_path,
            material_name=args.material,
            average_atomic_number=args.avg_z,
            defect_plan=args.defect_plan,
            max_defects=args.max_defects,
            voxel_depth_mm=args.voxel_depth_mm,
            slice_offset_mm=args.slice_offset_mm,
            slice_tilt_deg=args.slice_tilt_deg,
            slice_rotation_deg=args.slice_rotation_deg,
            randomize_defect_location=not args.fixed_defect_location,
            render_view_mode=args.render_view,
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
                defect_mask=sample["defect_mask"],
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
            pixel_size_mm=args.pixel_size_mm,
            weld_layout_mode=args.weld_layout,
            seam_rotation_deg=args.seam_rotation_deg,
            seam_offset_mm=args.seam_offset_mm,
            circle_radius_mm=args.circle_radius_mm,
            circle_position_deg=args.circle_position_deg,
            calibration_path=args.calibration_path,
            render_model_mode=args.render_model,
            emission_b_mode=args.emission_b_mode,
            emission_b_path=args.emission_b_path,
            disable_noise=args.disable_noise,
            voxel_depth_mm=args.voxel_depth_mm,
            slice_offset_mm=args.slice_offset_mm,
            slice_tilt_deg=args.slice_tilt_deg,
            slice_rotation_deg=args.slice_rotation_deg,
            randomize_slice=not args.fixed_slice,
            randomize_defect_location=not args.fixed_defect_location,
            render_view_mode=args.render_view,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.mode == "compare-render":
        out = render_comparison_case(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            calibration_path=args.calibration_path,
            seed=args.seed,
            image_size=args.image_size,
            pixel_size_mm=args.pixel_size_mm,
            weld_layout_mode=args.weld_layout,
            seam_rotation_deg=args.seam_rotation_deg,
            seam_offset_mm=args.seam_offset_mm,
            circle_radius_mm=args.circle_radius_mm,
            circle_position_deg=args.circle_position_deg,
            material_name=args.material,
            average_atomic_number=args.avg_z,
            defect_plan=args.defect_plan,
            max_defects=args.max_defects,
            voxel_depth_mm=args.voxel_depth_mm,
            slice_offset_mm=args.slice_offset_mm,
            slice_tilt_deg=args.slice_tilt_deg,
            slice_rotation_deg=args.slice_rotation_deg,
            randomize_defect_location=not args.fixed_defect_location,
            render_model_mode=args.render_model,
            emission_b_mode=args.emission_b_mode,
            emission_b_path=args.emission_b_path,
            stem=args.stem,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "compare-summarize":
        out = summarize_comparison_cases(
            comparison_root=args.comparison_root,
            output_dir=args.output_dir,
        )
        print(json.dumps(out, indent=2))
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
                topo_contrast_threshold=args.topo_contrast_threshold,
            ),
        }
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "visibility-pilot":
        out = run_visibility_pilot(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            repeats_per_class_regime=args.repeats,
            image_size=args.image_size,
            pixel_size_mm=args.pixel_size_mm,
            base_seed=args.seed,
            calibration_path=args.calibration_path,
            render_model_mode=args.render_model,
            emission_b_mode=args.emission_b_mode,
            emission_b_path=args.emission_b_path,
            save_images=not args.no_images,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "topo-static-dataset":
        out = generate_balanced_topo_dataset(
            fixed_energy_kev=args.fixed_energy_kev,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            samples_per_class=args.samples_per_class,
            image_size=args.image_size,
            pixel_size_mm=args.pixel_size_mm,
            seed=args.seed,
            render_model_mode=args.render_model,
            emission_b_mode=args.emission_b_mode,
            emission_b_path=args.emission_b_path,
            normalization_path=args.normalization_path,
            noise_profile=args.noise_profile,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "topo-sequence-dataset":
        out = generate_topo_sequence_dataset(
            fixed_energy_kev=args.fixed_energy_kev,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            sequences_per_class=args.sequences_per_class,
            frames_per_sequence=args.frames_per_sequence,
            frame_size=args.frame_size,
            stride_px=args.stride_px,
            context_radius=args.context_radius,
            pixel_size_mm=args.pixel_size_mm,
            seed=args.seed,
            render_model_mode=args.render_model,
            emission_b_mode=args.emission_b_mode,
            emission_b_path=args.emission_b_path,
            normalization_path=args.normalization_path,
            weld_layout_mode=args.weld_layout_mode,
            noise_profile=args.noise_profile,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "emission-b-topo-ablation":
        out = run_emission_b_topo_ablation(
            config_dir=args.config_dir,
            b_fit_path=args.b_fit_path,
            output_dir=args.output_dir,
            repeats_per_class=args.repeats_per_class,
            image_size=args.image_size,
            pixel_size_mm=args.pixel_size_mm,
            seed=args.seed,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "benchmark-fit":
        out = fit_casino_benchmark_calibration(
            benchmark_summary_path=args.benchmark_summary,
            output_dir=args.output_dir,
            config_dir=args.config_dir,
            pixel_size_mm=args.pixel_size_mm,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "benchmark-validate":
        out = evaluate_casino_benchmark_fit(
            benchmark_summary_path=args.benchmark_summary,
            calibration_path=args.calibration_path,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            pixel_size_mm=args.pixel_size_mm,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "benchmark-assemble":
        out = assemble_casino_benchmark_campaign(
            export_root=args.export_root,
            assembly_config_path=args.assembly_config,
            output_path=args.output_path,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "benchmark-extract-run":
        out = extract_casino_run_note_values(
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            material=args.material,
            geometry=args.geometry,
            energy_kev=args.energy_kev,
            tilt_deg=args.tilt_deg,
            beam_diameter_nm=args.beam_diameter_nm,
            electrons=args.electrons,
            quality_flag=args.quality_flag,
        )
        if args.output_path:
            output_path = Path(args.output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "benchmark-summarize":
        group_by = tuple(part.strip() for part in args.group_by.split(",") if part.strip())
        out = summarize_casino_benchmark_campaign(
            benchmark_summary_path=args.benchmark_summary,
            output_dir=args.output_dir,
            group_by=group_by or ("geometry",),
            usable_only=not args.include_nonusable,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "benchmark-fit-b-surface":
        from bse_renderer.validation.casino_compact import run_reanalysis
        if not 0 < args.plane_tolerance_nm < float("inf"):
            raise ValueError("Plane tolerance must be finite and positive.")
        out = run_reanalysis(Path(args.cas_root), Path(args.output_dir), args.plane_tolerance_nm)
        print(json.dumps({"selected_exit_count": out["selected_exit_count"], "output_dir": args.output_dir}, indent=2))
        return 0

    if args.mode == "benchmark-fit-b":
        print("Legacy global exit-class analysis: mixed landing faces and omitted local-surface exits can bias B. Use benchmark-fit-b-surface for the corrected analysis.", file=sys.stderr)
        out = fit_casino_emission_b(
            cas_root=args.cas_root,
            export_root=args.export_root,
            output_dir=args.output_dir,
            grid_points=args.grid_points,
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "benchmark-compare-b-render":
        tilts = None
        if args.tilts:
            tilts = [float(part.strip()) for part in args.tilts.split(",") if part.strip()]
        out = compare_emission_b_render_modes(
            config_dir=args.config_dir,
            b_fit_report_path=args.b_fit_report,
            emission_b_fit_path=args.emission_b_fit,
            output_dir=args.output_dir,
            tilts_deg=tilts,
            seed=args.seed,
            image_size=args.image_size,
            pixel_size_mm=args.pixel_size_mm,
            material_name=args.material,
            render_model_mode=args.render_model,
            calibration_path=args.calibration_path,
            weld_layout_mode=args.weld_layout,
            seam_rotation_deg=args.seam_rotation_deg,
            seam_offset_mm=args.seam_offset_mm,
            circle_radius_mm=args.circle_radius_mm,
            circle_position_deg=args.circle_position_deg,
            average_atomic_number=args.avg_z,
            defect_plan=args.defect_plan,
            max_defects=args.max_defects,
            voxel_depth_mm=args.voxel_depth_mm,
            slice_offset_mm=args.slice_offset_mm,
            slice_tilt_deg=args.slice_tilt_deg,
            slice_rotation_deg=args.slice_rotation_deg,
            randomize_defect_location=not args.fixed_defect_location,
        )
        print(json.dumps(out, indent=2))
        return 0

    parser.error("Unknown mode")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
