from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(
        _mapped(compo, mapping.get("compo"), {"mode": "linear", "low": 0.0, "high": 1.0}),
        cmap="gray",
    )
    axes[0].set_title("COMPO")
    axes[1].imshow(
        _mapped(topo, mapping.get("topo"), {"mode": "linear", "low": 0.0, "high": 1.0}),
        cmap="gray",
    )
    axes[1].set_title("TOPO")
    if tx is not None:
        axes[2].imshow(
            _mapped(tx, mapping.get("tx"), {"mode": "signed", "center": 0.0, "span": 1.0}),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        axes[2].imshow(np.zeros_like(compo), cmap="gray")
    axes[2].set_title("Tx")
    if ty is not None:
        axes[3].imshow(
            _mapped(ty, mapping.get("ty"), {"mode": "signed", "center": 0.0, "span": 1.0}),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        axes[3].imshow(np.zeros_like(compo), cmap="gray")
    axes[3].set_title("Ty")
    if defect_mask is not None:
        axes[4].imshow(defect_mask, cmap="gray")
        axes[4].set_title("Defect Mask")
    else:
        diff = np.abs(compo - topo)
        axes[4].imshow(diff, cmap="magma")
        axes[4].set_title("|COMPO-TOPO|")
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
    diff_compo = defect_compo - clean_compo
    diff_topo = defect_topo - clean_topo
    diff_compo_map = build_signed_display_map(diff_compo, quantile=0.995, pad_fraction=0.06)
    diff_topo_map = build_signed_display_map(diff_topo, quantile=0.995, pad_fraction=0.06)
    compo_span = float(diff_compo_map["span"])
    topo_span = float(diff_topo_map["span"])
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes[0, 0].imshow(
        _mapped(clean_compo, clean_map.get("compo"), {"mode": "linear", "low": 0.0, "high": 1.0}),
        cmap="gray",
    )
    axes[0, 0].set_title("Clean COMPO")
    axes[0, 1].imshow(
        _mapped(defect_compo, defect_map.get("compo"), {"mode": "linear", "low": 0.0, "high": 1.0}),
        cmap="gray",
    )
    axes[0, 1].set_title("Defected COMPO")
    axes[0, 2].imshow(
        diff_compo,
        cmap="RdBu_r",
        vmin=-compo_span,
        vmax=compo_span,
    )
    axes[0, 2].set_title("COMPO Difference")

    axes[1, 0].imshow(
        _mapped(clean_topo, clean_map.get("topo"), {"mode": "linear", "low": 0.0, "high": 1.0}),
        cmap="gray",
    )
    axes[1, 0].set_title("Clean TOPO")
    axes[1, 1].imshow(
        _mapped(defect_topo, defect_map.get("topo"), {"mode": "linear", "low": 0.0, "high": 1.0}),
        cmap="gray",
    )
    axes[1, 1].set_title("Defected TOPO")
    axes[1, 2].imshow(
        diff_topo,
        cmap="RdBu_r",
        vmin=-topo_span,
        vmax=topo_span,
    )
    axes[1, 2].set_title("TOPO Difference")

    if defect_tx is None:
        axes[2, 0].imshow(np.zeros_like(defect_topo), cmap="gray")
    else:
        axes[2, 0].imshow(
            _mapped(defect_tx, defect_map.get("tx"), {"mode": "signed", "center": 0.0, "span": 1.0}),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    axes[2, 0].set_title("Defected Tx")

    if defect_ty is None:
        axes[2, 1].imshow(np.zeros_like(defect_topo), cmap="gray")
    else:
        axes[2, 1].imshow(
            _mapped(defect_ty, defect_map.get("ty"), {"mode": "signed", "center": 0.0, "span": 1.0}),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    axes[2, 1].set_title("Defected Ty")

    if defect_mask is None:
        axes[2, 2].imshow(np.abs(diff_topo), cmap="magma")
        axes[2, 2].set_title("|TOPO Difference|")
    else:
        axes[2, 2].imshow(defect_mask, cmap="gray")
        axes[2, 2].set_title("Defect Mask")
        for ax in (axes[0, 2], axes[1, 2]):
            ax.contour(defect_mask.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=0.8)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Clean vs Defected Render Comparison")
    plt.tight_layout()
    plt.show()
