from __future__ import annotations

import numpy as np


def _coord_vectors(
    heightmap_mm: np.ndarray,
    x_coords_mm: np.ndarray | None,
    y_coords_mm: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = heightmap_mm.shape
    if x_coords_mm is None:
        x_coords_mm = np.arange(w, dtype=np.float32)
    if y_coords_mm is None:
        y_coords_mm = np.arange(h, dtype=np.float32)
    x_coords_mm = np.asarray(x_coords_mm, dtype=np.float32)
    y_coords_mm = np.asarray(y_coords_mm, dtype=np.float32)
    if x_coords_mm.shape != (w,) or y_coords_mm.shape != (h,):
        raise ValueError("Coordinate vectors must match the heightmap dimensions.")
    return x_coords_mm, y_coords_mm


def _resolve_z_exaggeration(
    heightmap_mm: np.ndarray,
    x_coords_mm: np.ndarray,
    y_coords_mm: np.ndarray,
    z_exaggeration: float | None,
) -> tuple[float, str]:
    finite = np.asarray(heightmap_mm, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    robust_span = 1e-6
    if finite.size > 0:
        q_low, q_high = np.quantile(finite, [0.01, 0.99])
        robust_span = max(float(q_high - q_low), 1e-6)
    full_span = max(float(np.ptp(heightmap_mm)), 1e-6)
    if z_exaggeration is not None:
        factor = max(float(z_exaggeration), 1.0)
    else:
        x_range = max(float(x_coords_mm.max() - x_coords_mm.min()), 1e-6)
        y_range = max(float(y_coords_mm.max() - y_coords_mm.min()), 1e-6)
        lateral_range = max(min(x_range, y_range), 1e-6)
        target_visible_z = 0.10 * lateral_range
        robust_factor = target_visible_z / robust_span
        full_factor = target_visible_z / full_span
        factor = np.clip(max(full_factor, 0.65 * robust_factor), 1.0, 36.0)
    if factor <= 1.001:
        return float(factor), "z (mm)"
    return float(factor), f"z (mm, preview x{factor:.1f})"


def preview_surface_mesh(
    heightmap_mm: np.ndarray,
    defect_mask: np.ndarray | None = None,
    *,
    x_coords_mm: np.ndarray | None = None,
    y_coords_mm: np.ndarray | None = None,
    title: str = "Surface Mesh Preview",
    z_exaggeration: float | None = None,
) -> None:
    """Open 3D mesh viewer using PyVista if available; fallback to matplotlib."""
    x_coords_mm, y_coords_mm = _coord_vectors(heightmap_mm, x_coords_mm, y_coords_mm)
    z_exaggeration, z_label = _resolve_z_exaggeration(
        heightmap_mm,
        x_coords_mm,
        y_coords_mm,
        z_exaggeration,
    )
    try:
        import pyvista as pv  # pragma: no cover - optional dependency path
    except Exception:
        _preview_matplotlib(
            heightmap_mm,
            defect_mask,
            x_coords_mm=x_coords_mm,
            y_coords_mm=y_coords_mm,
            title=title,
            z_exaggeration=z_exaggeration,
        )
        return

    y_mm, x_mm = np.meshgrid(y_coords_mm, x_coords_mm, indexing="ij")
    z_mm = heightmap_mm.astype(float) * float(z_exaggeration)
    grid = pv.StructuredGrid(x_mm.astype(float), y_mm.astype(float), z_mm)
    p = pv.Plotter()
    p.add_mesh(grid, scalars=heightmap_mm.ravel(order="F"), cmap="viridis", show_edges=False)
    if defect_mask is not None and defect_mask.any():
        mask_float = defect_mask.astype(float)
        z_pad = max(0.02 * max(float(np.ptp(heightmap_mm)), 1e-6), 1e-6)
        dgrid = pv.StructuredGrid(
            x_mm.astype(float),
            y_mm.astype(float),
            (z_mm + z_pad).astype(float),
        )
        p.add_mesh(dgrid, scalars=mask_float.ravel(order="F"), cmap="Reds", opacity=0.35)
    p.add_title(title, font_size=12)
    p.show_grid(xtitle="x (mm)", ytitle="y (mm)", ztitle=z_label)
    p.show()


def _preview_matplotlib(
    heightmap_mm: np.ndarray,
    defect_mask: np.ndarray | None = None,
    *,
    x_coords_mm: np.ndarray | None = None,
    y_coords_mm: np.ndarray | None = None,
    title: str = "Surface Mesh Preview",
    z_exaggeration: float | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x_coords_mm, y_coords_mm = _coord_vectors(heightmap_mm, x_coords_mm, y_coords_mm)
    z_exaggeration, z_label = _resolve_z_exaggeration(
        heightmap_mm,
        x_coords_mm,
        y_coords_mm,
        z_exaggeration,
    )
    y_mm, x_mm = np.meshgrid(y_coords_mm, x_coords_mm, indexing="ij")
    z_mm = heightmap_mm.astype(np.float32) * float(z_exaggeration)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_mm, y_mm, z_mm, cmap="viridis", linewidth=0, antialiased=True)
    if defect_mask is not None and defect_mask.any():
        ys, xs = np.where(defect_mask)
        ax.scatter(x_coords_mm[xs], y_coords_mm[ys], z_mm[ys, xs], c="r", s=4, alpha=0.6)
    x_range = max(float(x_coords_mm.max() - x_coords_mm.min()), 1e-6)
    y_range = max(float(y_coords_mm.max() - y_coords_mm.min()), 1e-6)
    z_range = max(float(np.ptp(z_mm)), 1e-6)
    ax.set_box_aspect((x_range, y_range, z_range))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel(z_label)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def preview_volume_geometry(
    fusion_mask: np.ndarray,
    defect_mask: np.ndarray | None = None,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    title: str = "Fusion Body Preview",
) -> None:
    fusion_mask = np.asarray(fusion_mask, dtype=bool)
    if fusion_mask.ndim != 3:
        raise ValueError("fusion_mask must be a 3D boolean volume.")
    defect_mask = None if defect_mask is None else np.asarray(defect_mask, dtype=bool)

    coords = np.argwhere(fusion_mask)
    if coords.size == 0:
        raise ValueError("Fusion mask is empty; nothing to preview.")

    sx = float(spacing_mm[0])
    sy = float(spacing_mm[1])
    sz = float(spacing_mm[2])
    max_points = 40000
    if coords.shape[0] > max_points:
        step = int(np.ceil(coords.shape[0] / max_points))
        coords = coords[::step]
    fusion_points = np.column_stack(
        [coords[:, 2] * sx, coords[:, 1] * sy, coords[:, 0] * sz]
    ).astype(np.float32)

    defect_points = None
    if defect_mask is not None and defect_mask.any():
        dcoords = np.argwhere(defect_mask)
        if dcoords.shape[0] > 12000:
            dstep = int(np.ceil(dcoords.shape[0] / 12000))
            dcoords = dcoords[::dstep]
        defect_points = np.column_stack(
            [dcoords[:, 2] * sx, dcoords[:, 1] * sy, dcoords[:, 0] * sz]
        ).astype(np.float32)

    try:
        import pyvista as pv  # pragma: no cover - optional dependency path
    except Exception:
        _preview_volume_matplotlib(fusion_points, defect_points, title)
        return

    p = pv.Plotter()
    p.add_mesh(
        pv.PolyData(fusion_points),
        color="#7fa9c9",
        render_points_as_spheres=True,
        point_size=5,
        opacity=0.16,
    )
    if defect_points is not None and len(defect_points) > 0:
        p.add_mesh(
            pv.PolyData(defect_points),
            color="#c94f4f",
            render_points_as_spheres=True,
            point_size=7,
            opacity=0.55,
        )
    p.add_title(title, font_size=12)
    p.show_grid()
    p.show()


def _preview_volume_matplotlib(
    fusion_points: np.ndarray,
    defect_points: np.ndarray | None,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        fusion_points[:, 0],
        fusion_points[:, 1],
        fusion_points[:, 2],
        c="#7fa9c9",
        s=3,
        alpha=0.10,
    )
    if defect_points is not None and len(defect_points) > 0:
        ax.scatter(
            defect_points[:, 0],
            defect_points[:, 1],
            defect_points[:, 2],
            c="#c94f4f",
            s=8,
            alpha=0.45,
        )
    ax.set_title(title)
    ax.set_xlabel("Travel x (mm)")
    ax.set_ylabel("Width y (mm)")
    ax.set_zlabel("Depth z (mm)")
    plt.tight_layout()
    plt.show()
