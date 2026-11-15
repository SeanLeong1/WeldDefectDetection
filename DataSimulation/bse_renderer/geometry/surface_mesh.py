from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config_loader import BeamRuntime
from .weld_profile import generate_weld_profile


@dataclass
class SurfaceState:
    heightmap_mm: np.ndarray
    normals: np.ndarray
    material_z: np.ndarray
    defect_depth_um: np.ndarray
    yield_modifier: np.ndarray
    openness: np.ndarray
    section_relief_mm: np.ndarray
    section_material_z: np.ndarray
    section_yield_modifier: np.ndarray
    section_void_fraction: np.ndarray
    defect_mask: np.ndarray
    pixel_size_mm: float
    fusion_boundary: np.ndarray

    def recompute_normals(self) -> None:
        dz_dy, dz_dx = np.gradient(self.heightmap_mm.astype(np.float64), self.pixel_size_mm, self.pixel_size_mm)
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(self.heightmap_mm)
        mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
        self.normals = np.stack([nx / mag, ny / mag, nz / mag], axis=-1).astype(np.float32)

    def effective_heightmap(self, section_weight: float = 0.72) -> np.ndarray:
        weight = float(np.clip(section_weight, 0.0, 1.0))
        surface_weight = 1.0 - 0.78 * weight
        return (
            surface_weight * self.heightmap_mm.astype(np.float32)
            + self.section_relief_mm.astype(np.float32)
        ).astype(np.float32)


def _initial_normals(heightmap_mm: np.ndarray, pixel_size_mm: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(heightmap_mm.astype(np.float64), pixel_size_mm, pixel_size_mm)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(heightmap_mm)
    mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
    return np.stack([nx / mag, ny / mag, nz / mag], axis=-1).astype(np.float32)


def build_surface_from_config(
    beam: BeamRuntime,
    image_size: int = 256,
    pixel_size_mm: float = 0.05,
    base_z: float = 26.0,
    weld_z: float = 26.2,
    crown_height_mm: float = 0.01,
    seam_depth_mm: float = 0.18,
    bead_width_mm: float = 1.4,
    haz_width_mm: float = 1.0,
    beam_focus_quality: float = 1.0,
    asymmetry_offset_mm: float = 0.0,
    seed: int | None = None,
) -> SurfaceState:
    heightmap = generate_weld_profile(
        width_px=image_size,
        height_px=image_size,
        crown_height_mm=crown_height_mm,
        seam_depth_mm=seam_depth_mm,
        bead_width_mm=bead_width_mm,
        haz_width_mm=haz_width_mm,
        pixel_size_mm=pixel_size_mm,
        travel_speed_mm_min=beam.travel_speed_mm_min,
        beam_power_kw=beam.power_w / 1000.0,
        beam_focus_quality=beam_focus_quality,
        asymmetry_offset_mm=asymmetry_offset_mm,
        seed=seed,
    )
    normals = _initial_normals(heightmap, pixel_size_mm)

    y, x = np.mgrid[0:image_size, 0:image_size]
    cy = image_size / 2.0
    half_w = (bead_width_mm / pixel_size_mm) / 2.0
    fusion_boundary = np.abs(y - cy) <= half_w
    material_z = np.where(fusion_boundary, weld_z, base_z).astype(np.float32)

    defect_depth_um = np.full((image_size, image_size), np.inf, dtype=np.float32)
    yield_modifier = np.ones((image_size, image_size), dtype=np.float32)
    openness = np.zeros((image_size, image_size), dtype=np.float32)
    section_relief_mm = np.zeros((image_size, image_size), dtype=np.float32)
    section_material_z = material_z.copy().astype(np.float32)
    section_yield_modifier = np.ones((image_size, image_size), dtype=np.float32)
    section_void_fraction = np.zeros((image_size, image_size), dtype=np.float32)
    defect_mask = np.zeros((image_size, image_size), dtype=bool)
    return SurfaceState(
        heightmap_mm=heightmap.astype(np.float32),
        normals=normals,
        material_z=material_z,
        defect_depth_um=defect_depth_um,
        yield_modifier=yield_modifier,
        openness=openness,
        section_relief_mm=section_relief_mm,
        section_material_z=section_material_z,
        section_yield_modifier=section_yield_modifier,
        section_void_fraction=section_void_fraction,
        defect_mask=defect_mask,
        pixel_size_mm=pixel_size_mm,
        fusion_boundary=fusion_boundary,
    )
