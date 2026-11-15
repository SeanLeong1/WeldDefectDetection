from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..comparison_metrics import compute_topography_delta_metrics
from ..utils import apply_display_map, apply_sem_tone_curve, build_signed_display_map


def _mapped(
    arr: np.ndarray,
    mapping: dict[str, float | str] | None,
    default: dict[str, float | str],
) -> np.ndarray:
    return apply_sem_tone_curve(apply_display_map(arr, mapping or default))


def preview_channels(
    compo: np.ndarray,
    topo: np.ndarray,
    tx: np.ndarray | None = None,
    ty: np.ndarray | None = None,
    defect_mask: np.ndarray | None = None,
    display_mapping: dict[str, dict[str, float | str]] | None = None,
    title: str = "Rendered Preview",
) -> None:
    mapping = display_mapping or {}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(
        _mapped(topo, mapping.get("topo"), {"mode": "linear", "low": 0.0, "high": 1.0}),
        cmap="gray",
    )
    axes[0].set_title("TOPO")
    if tx is not None:
        axes[1].imshow(
            _mapped(tx, mapping.get("tx"), {"mode": "signed", "center": 0.0, "span": 1.0}),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        axes[1].imshow(np.zeros_like(topo), cmap="gray")
    axes[1].set_title("Tx")
    if ty is not None:
        axes[2].imshow(
            _mapped(ty, mapping.get("ty"), {"mode": "signed", "center": 0.0, "span": 1.0}),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        axes[2].imshow(np.zeros_like(topo), cmap="gray")
    axes[2].set_title("Ty")
    if defect_mask is not None:
        axes[3].imshow(defect_mask, cmap="gray")
        axes[3].set_title("Defect Mask")
    else:
        tx_field = np.zeros_like(topo) if tx is None else tx
        ty_field = np.zeros_like(topo) if ty is None else ty
        gradient = np.sqrt(np.square(tx_field) + np.square(ty_field))
        axes[3].imshow(gradient, cmap="magma")
        axes[3].set_title("Gradient Magnitude")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def preview_defect_impact(
    clean_compo: np.ndarray,
    defect_compo: np.ndarray,
    clean_topo: np.ndarray,
    defect_topo: np.ndarray,
    defect_tx: np.ndarray | None = None,
    defect_ty: np.ndarray | None = None,
    defect_mask: np.ndarray | None = None,
    clean_mapping: dict[str, dict[str, float | str]] | None = None,
    defect_mapping: dict[str, dict[str, float | str]] | None = None,
) -> None:
    clean_map = clean_mapping or {}
    defect_map = defect_mapping or {}
    diff_topo = defect_topo - clean_topo
    diff_topo_map = build_signed_display_map(diff_topo, quantile=0.995, pad_fraction=0.06)
    topo_span = float(diff_topo_map["span"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(
        _mapped(clean_topo, clean_map.get("topo"), {"mode": "linear", "low": 0.0, "high": 1.0}),
        cmap="gray",
    )
    axes[0, 0].set_title("Clean TOPO")
    axes[0, 1].imshow(
        _mapped(defect_topo, defect_map.get("topo"), {"mode": "linear", "low": 0.0, "high": 1.0}),
        cmap="gray",
    )
    axes[0, 1].set_title("Defected TOPO")
    axes[0, 2].imshow(
        diff_topo,
        cmap="RdBu_r",
        vmin=-topo_span,
        vmax=topo_span,
    )
    axes[0, 2].set_title("TOPO Difference")

    if defect_tx is None:
        axes[1, 0].imshow(np.zeros_like(defect_topo), cmap="gray")
    else:
        axes[1, 0].imshow(
            _mapped(defect_tx, defect_map.get("tx"), {"mode": "signed", "center": 0.0, "span": 1.0}),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    axes[1, 0].set_title("Defected Tx")

    if defect_ty is None:
        axes[1, 1].imshow(np.zeros_like(defect_topo), cmap="gray")
    else:
        axes[1, 1].imshow(
            _mapped(defect_ty, defect_map.get("ty"), {"mode": "signed", "center": 0.0, "span": 1.0}),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    axes[1, 1].set_title("Defected Ty")

    if defect_mask is None:
        axes[1, 2].imshow(np.abs(diff_topo), cmap="magma")
        axes[1, 2].set_title("|TOPO Difference|")
    else:
        axes[1, 2].imshow(defect_mask, cmap="gray")
        axes[1, 2].set_title("Defect Mask")
        for ax in (axes[0, 2],):
            ax.contour(defect_mask.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=0.8)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Clean vs Defected Topography Comparison")
    plt.tight_layout()
    plt.show()


def preview_calibration_modes(
    mode_outputs: dict[str, dict[str, object]],
    defect_mask: np.ndarray | None = None,
    title: str = "Render Model Comparison",
    diagnostic_lines: dict[str, list[str]] | None = None,
) -> None:
    if not mode_outputs:
        raise ValueError("mode_outputs must contain at least one render mode.")
    mode_items = list(mode_outputs.items())
    base_key, base_output = mode_items[0]
    base_topo = np.asarray(base_output["topo"], dtype=np.float32)
    num_cols = len(mode_items)
    diagnostics = diagnostic_lines or {}
    base_mode_label = str(base_key).replace("_", " ").title()

    fig, axes = plt.subplots(
        6,
        num_cols,
        figsize=(5.2 * num_cols, 20),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0, 1.0, 1.20]},
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]
    layouts = [
        ("topo", "TOPO"),
        ("tx", "Tx"),
        ("ty", "Ty"),
    ]
    for col_idx, (mode_key, mode_output) in enumerate(mode_items):
        mode_label = str(mode_key).replace("_", " ").title()
        for row_idx, (channel_key, label) in enumerate(layouts):
            channel = mode_output.get(channel_key)
            mapping = mode_output.get("display_mapping") or {}
            if channel_key == "topo":
                image = _mapped(
                    channel,
                    mapping.get("topo"),
                    {"mode": "linear", "low": 0.0, "high": 1.0},
                )
                axes[row_idx, col_idx].imshow(image, cmap="gray")
            elif channel is None:
                axes[row_idx, col_idx].imshow(np.zeros_like(base_topo), cmap="gray")
            else:
                image = _mapped(
                    channel,
                    mapping.get(channel_key),
                    {"mode": "signed", "center": 0.0, "span": 1.0},
                )
                axes[row_idx, col_idx].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx, col_idx].set_title(f"{mode_label} {label}")

        defect_response = np.asarray(
            mode_output.get("defect_response_topo", np.zeros_like(base_topo)),
            dtype=np.float32,
        )
        defect_response_map = build_signed_display_map(
            defect_response,
            quantile=0.995,
            pad_fraction=0.06,
        )
        defect_response_span = max(float(defect_response_map["span"]), 1e-6)
        axes[3, col_idx].imshow(
            defect_response,
            cmap="RdBu_r",
            vmin=-defect_response_span,
            vmax=defect_response_span,
        )
        axes[3, col_idx].set_title(f"{mode_label} Defected - Clean TOPO")
        if defect_mask is not None:
            axes[3, col_idx].contour(
                defect_mask.astype(np.uint8),
                levels=[0.5],
                colors="yellow",
                linewidths=0.8,
            )

        if col_idx == 0:
            if defect_mask is None:
                axes[4, col_idx].imshow(np.zeros_like(base_topo), cmap="gray")
                axes[4, col_idx].set_title("Baseline Reference")
            else:
                axes[4, col_idx].imshow(defect_mask, cmap="gray")
                axes[4, col_idx].set_title("Defect Mask")
            info_lines = [
                "Metric Guide",
                f"Baseline mode: {base_mode_label}",
                "Row 4: each model compared against its own clean render.",
                f"Row 5: each candidate model compared against {base_mode_label}.",
            ]
            if defect_mask is not None:
                info_lines.append(f"Connected defect regions: {_connected_region_count(defect_mask)}")
            defect_response_metrics = mode_output.get("defect_response_metrics")
            if isinstance(defect_response_metrics, dict):
                info_lines.append("")
                info_lines.extend(
                    _comparison_metric_lines(
                        defect_response_metrics,
                        defect_mask,
                        candidate_label=f"{mode_label} Defect Response",
                        baseline_label="this model's clean render",
                    )
                )
            info_lines.extend(str(line) for line in diagnostics.get(mode_key, []))
            axes[5, col_idx].text(
                0.02,
                0.98,
                "\n".join(info_lines),
                va="top",
                ha="left",
                family="monospace",
                fontsize=8.5,
            )
            axes[5, col_idx].set_facecolor("#f3f3f3")
            continue

        candidate_topo = np.asarray(mode_output["topo"], dtype=np.float32)
        delta = candidate_topo - base_topo
        delta_map = build_signed_display_map(delta, quantile=0.995, pad_fraction=0.06)
        span = max(float(delta_map["span"]), 1e-6)
        metrics = compute_topography_delta_metrics(
            base_topo,
            candidate_topo,
            defect_mask=defect_mask,
        )
        axes[4, col_idx].imshow(
            delta,
            cmap="RdBu_r",
            vmin=-span,
            vmax=span,
        )
        axes[4, col_idx].set_title(f"{mode_label} - {str(base_key).replace('_', ' ').title()} TOPO")
        metric_lines: list[str] = []
        defect_response_metrics = mode_output.get("defect_response_metrics")
        if isinstance(defect_response_metrics, dict):
            metric_lines.extend(
                _comparison_metric_lines(
                    defect_response_metrics,
                    defect_mask,
                    candidate_label=f"{mode_label} Defect Response",
                    baseline_label="this model's clean render",
                )
            )
            metric_lines.append("")
        metric_lines.extend(
                _comparison_metric_lines(
                    metrics,
                    defect_mask,
                    candidate_label=f"{mode_label} Model Delta",
                    baseline_label=f"{base_mode_label} baseline",
                )
            )
        extra = diagnostics.get(mode_key, [])
        if extra:
            metric_lines.append("")
            metric_lines.extend(str(line) for line in extra)
        axes[5, col_idx].text(
            0.02,
            0.98,
            "\n".join(metric_lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )
        axes[5, col_idx].set_facecolor("#f3f3f3")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    _show_scrollable_figure(fig, title=title)


def _connected_region_count(defect_mask: np.ndarray | None) -> int:
    if defect_mask is None:
        return 0
    mask = np.asarray(defect_mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return 0
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return count


def _show_scrollable_figure(fig: plt.Figure, *, title: str) -> None:
    manager = plt.get_current_fig_manager()
    root = tk._default_root
    if root is None:
        plt.show()
        return

    window = tk.Toplevel(root)
    window.title(title)
    window.geometry("1440x920")

    outer = ttk.Frame(window)
    outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(outer, highlightthickness=0)
    vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    hscroll = ttk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
    content = ttk.Frame(canvas)

    canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    vscroll.pack(side="right", fill="y")
    hscroll.pack(side="bottom", fill="x")

    window_id = canvas.create_window((0, 0), window=content, anchor="nw")

    def _sync_scrollregion(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _fit_content_width(event) -> None:
        current_width = content.winfo_reqwidth()
        if current_width < event.width:
            canvas.itemconfigure(window_id, width=event.width)

    def _on_mousewheel(event) -> None:
        if event.state & 0x0001:
            units = -1 if event.delta > 0 else 1
            canvas.xview_scroll(units, "units")
        else:
            units = -1 if event.delta > 0 else 1
            canvas.yview_scroll(units, "units")

    def _on_button4(_event) -> None:
        canvas.yview_scroll(-1, "units")

    def _on_button5(_event) -> None:
        canvas.yview_scroll(1, "units")

    content.bind("<Configure>", _sync_scrollregion)
    canvas.bind("<Configure>", _fit_content_width)
    canvas.bind("<MouseWheel>", _on_mousewheel)
    canvas.bind("<Button-4>", _on_button4)
    canvas.bind("<Button-5>", _on_button5)
    content.bind("<MouseWheel>", _on_mousewheel)
    content.bind("<Button-4>", _on_button4)
    content.bind("<Button-5>", _on_button5)

    figure_canvas = FigureCanvasTkAgg(fig, master=content)
    figure_canvas.draw()
    figure_widget = figure_canvas.get_tk_widget()
    figure_widget.pack(fill="both", expand=True)
    _sync_scrollregion()

    plt.close(fig)
    try:
        manager.destroy()
    except Exception:
        pass


def _comparison_metric_lines(
    multi_metrics: dict[str, float | int | None],
    defect_mask: np.ndarray | None,
    *,
    candidate_label: str = "Multi Weld Profile",
    baseline_label: str = "Parameter Only",
) -> list[str]:
    whole_mean = _as_float(multi_metrics.get("mean_abs_topo_delta"))
    defect_mean = _as_float(multi_metrics.get("defect_mean_abs_topo_delta"))
    background_mean = _as_float(multi_metrics.get("background_mean_abs_topo_delta"))
    ratio = _as_float(multi_metrics.get("defect_to_background_mean_ratio"))

    lines = [
        f"{candidate_label} Metrics",
        f"Reference: {baseline_label}",
        f"Defect pixels: {multi_metrics['defect_pixels']}",
        f"Connected defect regions: {_connected_region_count(defect_mask)}",
        f"Defect mean abs: {_fmt_metric(defect_mean)} ({_fmt_fixed(defect_mean)})",
        f"Background mean abs: {_fmt_metric(background_mean)} ({_fmt_fixed(background_mean)})",
        f"Defect/bg ratio: {_fmt_metric(ratio)}",
        f"Defect max abs: {_fmt_metric(multi_metrics['defect_max_abs_topo_delta'])}",
    ]
    if whole_mean is not None:
        lines.append(f"Whole-image mean abs: {_fmt_metric(whole_mean)} ({_fmt_fixed(whole_mean)})")
    lines.extend(
        [
            "Interpretation:",
            "ratio > 1 means the change is concentrated",
            "inside the defect mask more than background.",
        ]
    )
    if (
        whole_mean is not None
        and defect_mean is not None
        and background_mean is not None
        and ratio is not None
        and whole_mean < 1e-3
        and defect_mean > background_mean
        and ratio > 10.0
    ):
        lines.append("This is a localized defect response, not a global shift.")
    return lines


def _fmt_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value_float = float(value)
    if abs(value_float) >= 1000.0 or (0.0 < abs(value_float) < 1e-3):
        return f"{value_float:.3e}"
    return f"{value_float:.4f}"


def _fmt_fixed(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6f}"


def _as_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)
