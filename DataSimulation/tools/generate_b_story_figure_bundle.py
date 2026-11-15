from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon


REQUIRED_STAGE_FILES = (
    "01_constant_b_solve.png",
    "02_b_vs_tilt.png",
    "04_model_selection.png",
    "05_tilt_cutoff_stability.png",
    "06_asymmetry_by_tilt.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a chronological figure bundle for the CASINO emission-B fitting story."
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        required=True,
        help="Report directory containing b_fit, render_comparison, and optional b_calibration_graphs_6_stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the final bundle into. Defaults to <report-root>/b_story_figure_bundle.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 10,
        }
    )


def ensure_stage_graphs(report_root: Path) -> Path:
    stage_dir = report_root / "b_calibration_graphs_6_stage"
    missing = [name for name in REQUIRED_STAGE_FILES if not (stage_dir / name).exists()]
    if not missing:
        return stage_dir

    script_path = Path(__file__).with_name("generate_b_calibration_graphs.py")
    stage_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(script_path), "--report-root", str(report_root), "--output-dir", str(stage_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not generate the calibration graphs.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    still_missing = [name for name in REQUIRED_STAGE_FILES if not (stage_dir / name).exists()]
    if still_missing:
        raise FileNotFoundError(f"Calibration graphs are still missing after regeneration: {still_missing}")
    return stage_dir


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_local_frame_geometry(path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    ax = axes[0]
    ax.set_aspect("equal")
    ax.axis("off")

    plane = Polygon(
        [(-1.2, -0.6), (1.3, -0.1), (0.8, 0.7), (-1.7, 0.2)],
        closed=True,
        facecolor="#DCEAF7",
        edgecolor="#7FA8C9",
        linewidth=1.4,
    )
    ax.add_patch(plane)

    origin = np.array([0.0, 0.0])
    beam_start = np.array([-1.5, 1.7])
    normal_end = np.array([0.0, 2.0])
    t1_end = np.array([1.9, 0.45])
    t2_end = np.array([-1.1, 1.0])
    exit_end = np.array([1.25, 1.45])

    arrow_specs = [
        (beam_start, origin, "#D1495B", 2.2, "beam"),
        (origin, normal_end, "#2F6690", 2.0, "normal n"),
        (origin, t1_end, "#3A7D44", 2.0, "tangent_1"),
        (origin, t2_end, "#7A5195", 2.0, "tangent_2"),
        (origin, exit_end, "#EDAE49", 2.3, "exit direction"),
    ]
    for start, end, color, width, label in arrow_specs:
        patch = FancyArrowPatch(
            posA=tuple(start),
            posB=tuple(end),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=width,
            color=color,
        )
        ax.add_patch(patch)
        offset = np.array([0.05, 0.05])
        ax.text(*(end + offset), label, color=color, fontsize=9, weight="medium")

    alpha_arc = Arc((0.0, 0.0), 1.25, 1.25, angle=0, theta1=90, theta2=132, color="#D1495B", linewidth=1.4)
    phi_arc = Arc((0.0, 0.0), 1.65, 1.65, angle=0, theta1=49, theta2=90, color="#EDAE49", linewidth=1.4)
    ax.add_patch(alpha_arc)
    ax.add_patch(phi_arc)
    ax.text(-0.42, 0.73, "alpha", color="#D1495B", fontsize=9)
    ax.text(0.48, 0.88, "phi", color="#EDAE49", fontsize=9)
    ax.text(
        -1.65,
        -0.95,
        "Geometry sets the frame.\nB only biases emission inside that frame.",
        fontsize=9.4,
        color="#333333",
        va="top",
    )
    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(-1.2, 2.25)
    ax.set_title("Local Surface Frame")

    ax = axes[1]
    ax.set_aspect("equal")
    ax.axis("off")
    radius = 1.55
    ax.add_patch(Circle((0.0, 0.0), radius, facecolor="#F6F8FB", edgecolor="#9AA6B2", linewidth=1.2))
    ax.axhline(0.0, color="#3A7D44", linewidth=1.5)
    ax.axvline(0.0, color="#7A5195", linewidth=1.5)
    ax.text(radius + 0.12, 0.03, "+ tangent_1", color="#3A7D44", va="center", fontsize=9)
    ax.text(0.03, radius + 0.12, "+ tangent_2", color="#7A5195", fontsize=9)
    ax.text(-radius - 0.6, 0.03, "- tangent_1", color="#3A7D44", va="center", fontsize=9)
    ax.text(0.03, -radius - 0.28, "- tangent_2", color="#7A5195", fontsize=9)

    projection_angle_deg = 36.0
    projection = radius * 0.9 * np.array(
        [math.cos(math.radians(projection_angle_deg)), math.sin(math.radians(projection_angle_deg))]
    )
    ax.add_patch(
        FancyArrowPatch(
            posA=(0.0, 0.0),
            posB=tuple(projection),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.2,
            color="#EDAE49",
        )
    )
    varphi_arc = Arc((0.0, 0.0), 1.0, 1.0, angle=0, theta1=0, theta2=projection_angle_deg, color="#EDAE49", linewidth=1.4)
    ax.add_patch(varphi_arc)
    ax.text(0.5, 0.2, "varphi", color="#EDAE49", fontsize=9)
    ax.text(
        -1.75,
        -1.95,
        "CASINO exit vectors are converted to varphi\nwith atan2(u.tangent_2, u.tangent_1).",
        fontsize=9.2,
        color="#333333",
        va="top",
    )
    ax.set_xlim(-2.1, 2.2)
    ax.set_ylim(-2.05, 2.05)
    ax.set_title("Azimuth Measured in the Tangent Plane")
    return _save(fig, path)


def _add_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str, lines: list[str], color: str) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.05",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color + "22",
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2.0, xy[1] + height * 0.68, title, ha="center", va="center", fontsize=10, weight="medium")
    ax.text(xy[0] + width / 2.0, xy[1] + height * 0.33, "\n".join(lines), ha="center", va="center", fontsize=9, color="#333333")


def _add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#5B6570") -> None:
    ax.add_patch(
        FancyArrowPatch(
            posA=start,
            posB=end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=color,
        )
    )


def save_casino_pipeline(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    ax.axis("off")
    ax.set_xlim(0.0, 12.4)
    ax.set_ylim(0.0, 6.1)

    top_y = 3.7
    bottom_y = 1.0
    w = 2.0
    h = 1.35

    _add_box(ax, (0.5, top_y), w, h, "1. CASINO runs", ["tilt sweep", "repeat runs", "exit trajectories"], "#2F6690")
    _add_box(ax, (3.0, top_y), w, h, "2. Curated summary", ["campaign JSON", "run summary", "exit samples"], "#4E79A7")
    _add_box(ax, (5.5, top_y), w, h, "3. Fit local frame", ["normal n", "tangent_1", "tangent_2"], "#3A7D44")
    _add_box(ax, (8.0, top_y), w, h, "4. Convert exits", ["alpha", "phi", "varphi", "asymmetry term"], "#7A5195")

    _add_box(ax, (3.15, bottom_y), 2.35, h, "5a. Pooled constant B", ["single material-level solve", "primary baseline"], "#B07AA1")
    _add_box(ax, (6.0, bottom_y), 2.35, h, "5b. Quadratic B(alpha)", ["b0 + b1 sin(alpha)", "+ b2 sin^2(alpha)"], "#F28E2B")
    _add_box(ax, (9.0, bottom_y), 2.2, h, "6. Use in renderer", ["compare image-space impact", "pooled vs quadratic"], "#D1495B")

    _add_arrow(ax, (2.5, top_y + h / 2.0), (3.0, top_y + h / 2.0))
    _add_arrow(ax, (5.0, top_y + h / 2.0), (5.5, top_y + h / 2.0))
    _add_arrow(ax, (7.5, top_y + h / 2.0), (8.0, top_y + h / 2.0))
    _add_arrow(ax, (9.0, top_y), (4.35, bottom_y + h))
    _add_arrow(ax, (9.4, top_y), (7.15, bottom_y + h))
    _add_arrow(ax, (5.5, bottom_y + h / 2.0), (6.0, bottom_y + h / 2.0))
    _add_arrow(ax, (8.35, bottom_y + h / 2.0), (9.0, bottom_y + h / 2.0))

    ax.text(
        6.2,
        5.55,
        "Chronological B-estimation workflow",
        ha="center",
        va="center",
        fontsize=14,
        weight="medium",
        color="#1F1F1F",
    )
    ax.text(
        6.2,
        0.32,
        "The constant fit answers 'what single B best explains all exits?' while the quadratic law answers 'how does that best B shift with tilt?'",
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#444444",
    )
    return _save(fig, path)


def _polar_lobe(scale: float) -> np.ndarray:
    theta = np.linspace(-math.pi, math.pi, 400, dtype=np.float64)
    radius = 1.0 + scale * np.cos(theta)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def save_b_sign_flip(path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    titles = (
        ("B > 0", 0.42, "#4E79A7"),
        ("B < 0", -0.42, "#D1495B"),
    )
    for ax, (title, scale, color) in zip(axes, titles):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Circle((0.0, 0.0), 1.0, facecolor="#F7F8FA", edgecolor="#BCC6CF", linewidth=1.1))
        ax.axhline(0.0, color="#B0B8C1", linewidth=1.0)
        ax.axvline(0.0, color="#B0B8C1", linewidth=1.0)

        lobe = _polar_lobe(scale)
        ax.fill(lobe[:, 0], lobe[:, 1], color=color, alpha=0.28)
        ax.plot(lobe[:, 0], lobe[:, 1], color=color, linewidth=2.2)

        favored_side = np.array([1.25 if scale > 0 else -1.25, 0.0])
        ax.add_patch(
            FancyArrowPatch(
                posA=(0.0, 0.0),
                posB=tuple(favored_side),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=2.0,
                color=color,
            )
        )
        ax.text(1.35, 0.05, "+ tangent_1", color="#3A7D44", fontsize=9)
        ax.text(-1.95, 0.05, "- tangent_1", color="#3A7D44", fontsize=9)
        ax.text(
            0.0,
            1.42,
            title,
            ha="center",
            va="center",
            fontsize=13,
            weight="medium",
            color=color,
        )
        ax.text(
            0.0,
            -1.43,
            "Favored azimuth flips sides\nin the same local frame",
            ha="center",
            va="top",
            fontsize=9.2,
            color="#444444",
        )
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-1.7, 1.75)

    fig.suptitle("Changing the sign of B flips the azimuthal preference, not the frame itself", y=1.02, fontsize=13)
    return _save(fig, path)


def copy_story_graphs(stage_dir: Path, report_root: Path, output_dir: Path) -> list[dict[str, str]]:
    source_map = [
        (
            stage_dir / "01_constant_b_solve.png",
            output_dir / "04_constant_b_solve.png",
            "Solve the pooled constant B objective and mark the optimum inside the feasible interval.",
        ),
        (
            stage_dir / "06_asymmetry_by_tilt.png",
            output_dir / "05_asymmetry_by_tilt.png",
            "Show how the asymmetry factor shifts with tilt before interpreting the B sign change.",
        ),
        (
            stage_dir / "05_tilt_cutoff_stability.png",
            output_dir / "06_tilt_cutoff_stability.png",
            "Show when the pooled constant B changes sign as higher-tilt runs are admitted.",
        ),
        (
            stage_dir / "02_b_vs_tilt.png",
            output_dir / "07_b_vs_tilt.png",
            "Compare per-run and per-tilt fitted B against the constant and quadratic approximations.",
        ),
        (
            stage_dir / "04_model_selection.png",
            output_dir / "08_model_selection.png",
            "Compare constant, sin-law, and quadratic laws with holdout NLL and residual error.",
        ),
        (
            report_root / "render_comparison" / "emission_b_render_comparison.png",
            output_dir / "09_renderer_comparison.png",
            "Close the story with the image-space difference between pooled constant B and quadratic B(alpha).",
        ),
    ]
    copied: list[dict[str, str]] = []
    for source, dest, caption in source_map:
        if not source.exists():
            raise FileNotFoundError(f"Missing required source figure: {source}")
        shutil.copy2(source, dest)
        copied.append(
            {
                "file": dest.name,
                "caption": caption,
                "source": str(source),
                "kind": "copied",
            }
        )
    return copied


def save_contact_sheet(output_dir: Path, ordered_entries: list[dict[str, str]]) -> Path:
    image_entries = [entry for entry in ordered_entries if entry["file"].endswith(".png")]
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 11.0))
    axes = axes.ravel()
    for ax, entry in zip(axes, image_entries):
        image = mpimg.imread(output_dir / entry["file"])
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(entry["file"].split("_", 1)[1].replace(".png", "").replace("_", " "), fontsize=9)
    for ax in axes[len(image_entries):]:
        ax.axis("off")
    fig.suptitle("Chronological B-story figure bundle", fontsize=15, y=0.98)
    return _save(fig, output_dir / "00_bundle_overview.png")


def write_manifest(output_dir: Path, entries: list[dict[str, str]]) -> tuple[Path, Path]:
    payload = {"figures": entries}
    json_path = output_dir / "figure_manifest.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Chronological B-story figure bundle",
        "",
        "The figures are ordered to explain how B is defined, inferred from CASINO exits, approximated as either a pooled constant or a quadratic law, and finally used in rendering.",
        "",
    ]
    for entry in entries:
        lines.append(f"## {entry['file']}")
        lines.append("")
        lines.append(f"- Caption: {entry['caption']}")
        lines.append(f"- Kind: {entry['kind']}")
        if "source" in entry:
            lines.append(f"- Source: `{entry['source']}`")
        lines.append("")

    md_path = output_dir / "figure_manifest.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    output_dir = (args.output_dir or (report_root / "b_story_figure_bundle")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    stage_dir = ensure_stage_graphs(report_root)

    entries: list[dict[str, str]] = []
    entries.append(
        {
            "file": save_local_frame_geometry(output_dir / "01_local_frame_geometry.png").name,
            "caption": "Define the local surface frame used to measure alpha, phi, and varphi before B is fit.",
            "kind": "generated",
        }
    )
    entries.append(
        {
            "file": save_casino_pipeline(output_dir / "02_casino_to_samples_pipeline.png").name,
            "caption": "Show the full CASINO-to-fit workflow from trajectory exports to pooled and quadratic B models.",
            "kind": "generated",
        }
    )
    entries.append(
        {
            "file": save_b_sign_flip(output_dir / "03_b_sign_flip_schematic.png").name,
            "caption": "Clarify that flipping the sign of B flips the favored azimuthal side within the same frame.",
            "kind": "generated",
        }
    )
    entries.extend(copy_story_graphs(stage_dir, report_root, output_dir))
    overview_path = save_contact_sheet(output_dir, entries)
    entries.insert(
        0,
        {
            "file": overview_path.name,
            "caption": "Contact-sheet overview of the full chronological figure bundle.",
            "kind": "generated",
        },
    )
    manifest_json, manifest_md = write_manifest(output_dir, entries)

    print(str(output_dir))
    print(str(manifest_json))
    print(str(manifest_md))
    for entry in entries:
        print(str(output_dir / entry["file"]))


if __name__ == "__main__":
    main()
