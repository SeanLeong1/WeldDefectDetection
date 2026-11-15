"""Surface-only variability and optional interpolating-spline derivatives.

The historical renderer is inherited without modifying its source. A clean
surface is captured first for each latent specimen. Only its background relief
and baseline effective-material map are warped; injected pore relief, occupancy,
and geometric masks retain their physical coordinates and dimensions.
"""
from dataclasses import replace
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates
from bse_renderer.renderer.scan_engine import ScanEngine
from bse_renderer.renderer.image_formation import form_channels


def gradients(height, spacing, method="finite_difference"):
    height = np.asarray(height, dtype=np.float64)
    if height.ndim != 2 or min(height.shape) < 4 or not np.isfinite(height).all():
        raise ValueError("A finite height map with at least four samples per axis is required.")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Physical pixel spacing must be positive.")
    if method == "finite_difference":
        gy, gx = np.gradient(height, spacing, spacing)
    elif method == "cubic_spline":
        # Tensor-product interpolant derivatives evaluated at the original knots.
        # This is interpolation, with not-a-knot boundaries and no smoothing.
        x = np.arange(height.shape[1]) * spacing
        y = np.arange(height.shape[0]) * spacing
        gx = CubicSpline(x, height, axis=1, bc_type="not-a-knot")(x, 1)
        gy = CubicSpline(y, height, axis=0, bc_type="not-a-knot")(y, 1)
    else:
        raise ValueError(f"Unknown gradient method: {method}")
    return gy, gx


def normals(height, spacing, method="finite_difference"):
    gy, gx = gradients(height, spacing, method)
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)


def smooth_variations(size, seed, spacing=.05, knot_spacing_mm=6.4):
    """Independent bounded width/depth functions, continuous across all crops."""
    x = np.arange(size) * spacing
    knots = np.linspace(x[0], x[-1], max(4, int(np.ceil(x[-1]/knot_spacing_mm))+1))
    rng = np.random.default_rng(seed + 170003)
    fields = []
    for _ in range(2):
        values = rng.uniform(-1, 1, len(knots))
        f = CubicSpline(knots, values, bc_type="natural")(x)
        f /= max(float(np.max(np.abs(f))), 1e-12)
        fields.append(f.astype(np.float32))
    return fields


class VariableSurfaceEngine(ScanEngine):
    def __init__(self, *, frame_size=64, halo=16, gradient_method="finite_difference", **kwargs):
        super().__init__(**kwargs)
        if self.render_view_mode != "top_surface" or self.weld_layout.mode != "straight_seam":
            raise ValueError("This experiment supports straight top-surface welds only.")
        self.frame_size, self.halo = frame_size, halo
        self.gradient_method = gradient_method
        self.amplitude = 0.
        self.base_relief = self.base_material = None
        self.base_seed = None
        self.width_field = self.depth_field = None
        self.capture = False

    def set_variation(self, amplitude):
        if not 0 <= amplitude <= .25:
            raise ValueError("Surface variation must be in [0, 0.25].")
        self.amplitude = float(amplitude)

    def run(self, sample, session_state=None):
        if self.base_seed != sample.seed:
            if sample.defects:
                raise ValueError("Render the matching clean specimen first.")
            self.base_seed = sample.seed
            self.base_relief = self.base_material = None
            self.width_field, self.depth_field = smooth_variations(self.image_size, sample.seed, self.pixel_size_mm)
        result = super().run(sample, session_state)
        gain = float(np.interp(sample.beam.focusing_current_norm, [0., 1.], [1.10, .92]))
        self.quiet_topo = form_channels(self.quiet_quadrants).topo * gain
        self.noise = sample.noise
        result.metadata["variability"] = dict(
            amplitude=self.amplitude, width_fraction_range=[float((1+self.amplitude*self.width_field).min()), float((1+self.amplitude*self.width_field).max())],
            depth_fraction_range=[float((1+self.amplitude*self.depth_field).min()), float((1+self.amplitude*self.depth_field).max())],
            control_point_spacing_mm=6.4, gradient_method=self.gradient_method,
            warp="clean surface width/depth only; additive pore response and masks unchanged",
            physical_acceptance_standard=False, detector_strip_halo_px=self.halo)
        return result

    def _top_surface_section(self, volume, *, min_visible_area_px=6):
        section, defects = super()._top_surface_section(volume, min_visible_area_px=min_visible_area_px)
        if self.base_relief is None:
            if defects:
                raise ValueError("Baseline capture contains defects.")
            self.base_relief = section.relief_mm_map.copy()
            self.base_material = section.material_z_map.copy()
        delta = section.relief_mm_map - self.base_relief
        material_delta = section.material_z_map - self.base_material
        size = self.image_size
        # In the historical straight layout, y=0 lies at the plate centre;
        # asymmetry translates the seam slightly but is fixed within the pair.
        y, x = np.indices((size, size), dtype=np.float64)
        centre = .5*(size-1)
        width = 1 + self.amplitude*self.width_field
        coords = np.stack([centre+(y-centre)/width[None, :], x])
        if self.amplitude:
            base = map_coordinates(self.base_relief, coords, order=1, mode="nearest")
            base *= 1 + self.amplitude*self.depth_field[None, :]
            mat = map_coordinates(self.base_material, coords, order=0, mode="nearest")
        else:
            base, mat = self.base_relief.copy(), self.base_material.copy()
        section.relief_mm_map = (base+delta).astype(np.float32)
        section.material_z_map = (mat+material_delta).astype(np.float32)
        self.full_relief = section.relief_mm_map.copy()
        self.pore_delta = delta.copy()
        # Render a wide enough strip for the retained centre crops. Keep a halo
        # for gradient and blur boundaries; inspect its adequacy in pilot tests.
        start = (size-self.frame_size)//2-self.halo
        end = start+self.frame_size+2*self.halo
        self.strip_start = start
        for field in ("material_z_map", "yield_modifier_map", "void_fraction_map", "relief_mm_map", "defect_mask"):
            setattr(section, field, getattr(section, field)[start:end].copy())
        cropped = []
        for d in defects:
            mask = d.mask[start:end].copy()
            if not mask.any():
                raise ValueError("Pore fell outside the retained detector strip.")
            ys, xs = np.where(mask)
            cropped.append(replace(d, mask=mask, bbox_xyxy=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))))
        return section, cropped

    def _normals_from_heightmap(self, heightmap_mm, pixel_size_mm):
        self.last_normals = normals(heightmap_mm, pixel_size_mm, self.gradient_method)
        return self.last_normals

    def _apply_adaptive_quadrant_blur(self, quadrants, *, sigma_px, blur_scale_map):
        if 4*float(sigma_px)+2 > self.halo:
            raise ValueError("Detector strip halo is too small for the interaction blur.")
        out = super()._apply_adaptive_quadrant_blur(quadrants, sigma_px=sigma_px, blur_scale_map=blur_scale_map)
        self.quiet_quadrants = out.copy()
        return out
