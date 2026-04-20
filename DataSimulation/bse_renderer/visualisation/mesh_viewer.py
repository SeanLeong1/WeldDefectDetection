from __future__ import annotations

import numpy as np


def preview_surface_mesh(heightmap_mm: np.ndarray, defect_mask: np.ndarray | None = None) -> None:
    """Open 3D mesh viewer using PyVista if available; fallback to matplotlib."""
    try:
        import pyvista as pv  # pragma: no cover - optional dependency path
    except Exception:
        _preview_matplotlib(heightmap_mm, defect_mask)
        return

    h, w = heightmap_mm.shape
    y, x = np.mgrid[0:h, 0:w]
    grid = pv.StructuredGrid(x.astype(float), y.astype(float), heightmap_mm.astype(float))
    p = pv.Plotter()
    p.add_mesh(grid, scalars=heightmap_mm.ravel(order="F"), cmap="viridis", show_edges=False)
    if defect_mask is not None and defect_mask.any():
        mask_float = defect_mask.astype(float)
        dgrid = pv.StructuredGrid(x.astype(float), y.astype(float), (heightmap_mm + 0.01).astype(float))
        p.add_mesh(dgrid, scalars=mask_float.ravel(order="F"), cmap="Reds", opacity=0.35)
    p.show()


def _preview_matplotlib(heightmap_mm: np.ndarray, defect_mask: np.ndarray | None = None) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    h, w = heightmap_mm.shape
    y, x = np.mgrid[0:h, 0:w]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, heightmap_mm, cmap="viridis", linewidth=0, antialiased=True)
    if defect_mask is not None and defect_mask.any():
        ys, xs = np.where(defect_mask)
        ax.scatter(xs, ys, heightmap_mm[ys, xs], c="r", s=4, alpha=0.6)
    ax.set_title("Surface Mesh Preview")
    plt.tight_layout()
    plt.show()

