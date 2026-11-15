from __future__ import annotations

from pathlib import Path

import numpy as np


def load_cad_heightmap(
    cad_path: str | Path,
    image_size: int,
    scale: float = 1.0,
) -> np.ndarray:
    """Load STL/OBJ/PLY and rasterize to heightmap when trimesh is available."""
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("trimesh is required for CAD import") from exc

    mesh = trimesh.load_mesh(str(cad_path))
    if hasattr(mesh, "apply_scale"):
        mesh.apply_scale(scale)
    bounds = mesh.bounds
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    xs = np.linspace(x_min, x_max, image_size)
    ys = np.linspace(y_min, y_max, image_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_max + 1.0)])
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (points.shape[0], 1))
    locations, index_ray, _ = mesh.ray.intersects_location(points, directions)
    zmap = np.full(xx.size, z_min, dtype=np.float32)
    if len(index_ray) > 0:
        zmap[index_ray] = locations[:, 2]
    return zmap.reshape((image_size, image_size))

