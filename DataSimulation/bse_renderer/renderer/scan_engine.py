from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path

import numpy as np
try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - optional dependency fallback
    gaussian_filter = None

from ..domain_randomisation.parameter_sampler import SampledParameters
from ..geometry.voxel_volume import (
    apply_defect_bodies,
    build_section_plane,
    build_voxel_volume,
    slice_metadata,
    slice_volume,
)
from ..geometry.weld_profile import (
    WELD_GEOMETRY_RECESSED,
    validate_weld_geometry_mode,
)
from ..models import (
    DefectInstance,
    RenderChannels,
    RenderResult,
    SectionPlaneSpec,
    SectionSliceState,
    WeldLayoutRuntime,
)
from ..noise.contamination_model import apply_contamination
from ..noise.poisson_noise import apply_poisson_noise
from ..noise.thermal_drift import apply_thermal_drift
from ..noise.raster_acquisition import AcquisitionProfile, acquire, motion_coordinates, warp_image
from ..physics.interaction_volume import kanaya_okayama_range
from ..utils import (
    build_linear_display_map,
    build_signed_display_map,
)
from .calibration import (
    RendererCalibration,
    parameter_only_calibration,
    infer_render_model_mode,
    interpolate_tilt_scale,
    interpolate_tilt_scale_map,
    resolve_renderer_calibration,
    validate_render_model_mode,
)
from .emission_b import (
    EMISSION_B_MODE_ASSUMED_CONSTANT,
    EMISSION_B_MODE_LAMBERTIAN,
    EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    EMISSION_B_MODE_SIN_FIT,
    EMISSION_B_MODE_QUADRATIC_FIT,
    EmissionBFit,
    load_emission_b_fit,
    qualify_emission_b_fit,
    validate_emission_b_mode,
)
from .image_formation import form_channels
from .quadrant_detector import compute_quadrant_signals
from ..geometry.weld_profile import normalize_weld_layout


_GEOMETRY_BENCHMARK_DEFECT_MAP = {
    "cold_shut": "slit_defect_cold_shut",
    "lack_of_fusion": "slit_defect_cold_shut",
}


def _sha256_if_file(path: str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ScanEngine:
    def __init__(
        self,
        image_size: int = 256,
        pixel_size_mm: float = 0.05,
        calibration_path: str | None = None,
        calibration: RendererCalibration | None = None,
        render_model_mode: str | None = None,
        emission_b_mode: str | None = None,
        emission_b_path: str | None = None,
        weld_geometry_mode: str = WELD_GEOMETRY_RECESSED,
        weld_layout: WeldLayoutRuntime | None = None,
        render_view_mode: str = "slice",
        allow_unvalidated_emission_b: bool = False,
        acquisition_model: str = "legacy",
        acquisition_profile: AcquisitionProfile | None = None,
    ):
        self.image_size = image_size
        if acquisition_model not in {'legacy','raster_v2'}:
            raise ValueError('Unknown acquisition model.')
        self.acquisition_model = acquisition_model
        self.acquisition_profile = acquisition_profile or AcquisitionProfile()
        self.acquisition_profile.validate()
        self.pixel_size_mm = pixel_size_mm
        self.weld_geometry_mode = validate_weld_geometry_mode(weld_geometry_mode)
        self.weld_layout = normalize_weld_layout(weld_layout)
        self.render_view_mode = str(render_view_mode or "slice").strip().lower()
        if self.render_view_mode not in {"slice", "top_surface"}:
            raise ValueError(
                f"Unknown render_view_mode '{render_view_mode}'. Expected 'slice' or 'top_surface'."
            )
        self.requested_render_model_mode = validate_render_model_mode(render_model_mode)
        self.emission_b_mode = validate_emission_b_mode(emission_b_mode)
        self.emission_b_path = None if emission_b_path is None else str(emission_b_path)
        self.emission_b_fit: EmissionBFit | None = None
        if self.emission_b_mode in {
            EMISSION_B_MODE_ASSUMED_CONSTANT,
            'angular_table',
            EMISSION_B_MODE_POOLED_CONSTANT_FIT,
            EMISSION_B_MODE_SIN_FIT,
            EMISSION_B_MODE_QUADRATIC_FIT,
        }:
            if self.emission_b_path is None:
                raise ValueError(
                    "Fitted emission B modes require an emission B fit JSON path."
                )
            self.emission_b_fit = load_emission_b_fit(self.emission_b_path, self.emission_b_mode)
            if not allow_unvalidated_emission_b and self.emission_b_mode != EMISSION_B_MODE_ASSUMED_CONSTANT:
                self.emission_b_fit = qualify_emission_b_fit(self.emission_b_fit)
        self.emission_b_diagnostic_only = bool(allow_unvalidated_emission_b)
        self.requested_calibration_path = (
            None
            if calibration_path is None
            else str(calibration_path)
        )
        if calibration is not None:
            self.calibration = calibration
            self.render_model_mode = (
                infer_render_model_mode(calibration)
                if self.requested_render_model_mode is None
                else self.requested_render_model_mode
            )
        else:
            cal_path = calibration_path or os.getenv("BSE_RENDERER_CALIBRATION")
            self.calibration = resolve_renderer_calibration(
                cal_path,
                render_model_mode=self.requested_render_model_mode,
            )
            self.render_model_mode = (
                infer_render_model_mode(self.calibration)
                if self.requested_render_model_mode is None
                else self.requested_render_model_mode
            )
        if self.render_model_mode == "formula_only":
            self.calibration = RendererCalibration.identity(source=self.calibration.source)
        elif self.render_model_mode == "formula_casino":
            self.calibration = parameter_only_calibration(self.calibration)

    def _resolve_emission_b_coeff(
        self,
        normals: np.ndarray,
        material_b: float,
        incidence_tilt_deg: float,
    ) -> tuple[float | np.ndarray, dict[str, object]]:
        if self.emission_b_mode == EMISSION_B_MODE_LAMBERTIAN:
            return 0.0, {
                "mode": self.emission_b_mode,
                "source_path": None,
                "source_sha256": None,
                "source_type": "explicit_uncalibrated_lambertian_fallback",
                "formula": "B = 0 (Lambertian fallback)",
                "incidence_tilt_deg": float(incidence_tilt_deg),
                "b_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
            }
        if self.emission_b_fit is not None:
            assert self.emission_b_fit is not None
            if self.emission_b_fit.angular_table is not None:
                angles = np.degrees(np.arccos(np.clip(normals[..., 2], 0., 1.)))
                self.emission_b_fit.angular_table.validate_domain(angles)
                if self.emission_b_fit.qualified_tilt_domain is not None:
                    lo, hi = self.emission_b_fit.qualified_tilt_domain
                    if np.any(angles < lo-1e-5) or np.any(angles > hi+1e-5):
                        raise ValueError('Angular table incidence is outside the qualified domain.')
                return 0., {'mode': 'angular_table', 'source_path': self.emission_b_fit.source_path,
                            'source_sha256': _sha256_if_file(self.emission_b_fit.source_path),
                            'formula': self.emission_b_fit.formula, 'b_stats': None}
            # Constant fits still need every local incidence inside the qualified domain.
            local_angles = np.degrees(np.arccos(np.clip(normals[..., 2], 0.0, 1.0)))
            self.emission_b_fit.evaluate(local_angles)
            if self.emission_b_mode in (EMISSION_B_MODE_POOLED_CONSTANT_FIT, EMISSION_B_MODE_ASSUMED_CONSTANT):
                effective_b: float | np.ndarray = float(
                    np.asarray(self.emission_b_fit.evaluate(float(incidence_tilt_deg))).item()
                )
                b_values = np.asarray([effective_b], dtype=np.float32)
                local_tilt_deg: float | np.ndarray = float(incidence_tilt_deg)
            else:
                nz = np.clip(np.asarray(normals[..., 2], dtype=np.float32), 0.0, 1.0)
                local_tilt_deg = np.degrees(np.arccos(nz)).astype(np.float32)
                effective_b = self.emission_b_fit.evaluate(local_tilt_deg).astype(np.float32)
                b_values = np.asarray(effective_b, dtype=np.float32)
            return effective_b, {
                "mode": self.emission_b_mode,
                "source_path": self.emission_b_fit.source_path,
                "source_sha256": _sha256_if_file(self.emission_b_fit.source_path),
                "source_type": self.emission_b_fit.source_type,
                "evidence_status": ('engineering_assumption' if self.emission_b_fit.assumption is not None
                                    else 'diagnostic_unqualified' if self.emission_b_diagnostic_only else 'qualified'),
                "assumption": self.emission_b_fit.assumption,
                "formula": self.emission_b_fit.formula,
                "fit_mode": self.emission_b_fit.emission_b_mode,
                "incidence_tilt_deg": float(incidence_tilt_deg),
                "local_incidence_tilt_stats_deg": {
                    "min": float(np.min(local_tilt_deg)),
                    "max": float(np.max(local_tilt_deg)),
                    "mean": float(np.mean(local_tilt_deg)),
                },
                "b_stats": {
                    "min": float(np.min(b_values)),
                    "max": float(np.max(b_values)),
                    "mean": float(np.mean(b_values)),
                },
            }
        return float(material_b), {
            "mode": self.emission_b_mode,
            "source_path": None,
            "source_sha256": None,
            "source_type": "material_constant",
            "formula": f"B = {float(material_b):.6f}",
            "incidence_tilt_deg": float(incidence_tilt_deg),
            "b_stats": {
                "min": float(material_b),
                "max": float(material_b),
                "mean": float(material_b),
            },
        }

    @staticmethod
    def _beam_gain(power_w: float) -> float:
        ref_power_w = 18_000.0  # 60 kV * 300 mA
        return float(np.clip((max(power_w, 1e-6) / ref_power_w) ** 0.20, 0.55, 1.65))

    def _interaction_blur_px(
        self,
        focal_spot_mm: float,
        e0_kev: float,
        pixel_size_mm: float | None = None,
    ) -> float:
        # Larger beam spot and higher accelerating voltage broaden the source volume.
        spacing_mm = self.pixel_size_mm if pixel_size_mm is None else float(pixel_size_mm)
        spot_px = max(focal_spot_mm, 0.0) / max(spacing_mm, 1e-6)
        energy_term = np.clip((max(e0_kev, 1e-6) / 60.0) ** 0.6, 0.5, 2.2)
        return float(np.clip(0.10 * spot_px + 0.35 * energy_term, 0.0, 4.0))

    def _yield_scale(self, e0_kev: float) -> float:
        model = self.calibration.surrogate.yield_model
        ref = max(model.reference_energy_kev, 1e-6)
        energy_factor = (max(e0_kev, 1e-6) / ref) ** float(model.energy_exponent)
        return float(max(model.scale * energy_factor, 1e-6))

    def _yield_tilt_scale(self, tilt_deg: float = 0.0) -> float:
        model = self.calibration.surrogate.yield_model
        return float(
            max(
                interpolate_tilt_scale(
                    tilt_deg,
                    model.tilt_deg_knots,
                    model.tilt_scale_knots,
                ),
                1e-6,
            )
        )

    def _yield_tilt_scale_map(self, normals: np.ndarray) -> np.ndarray:
        model = self.calibration.surrogate.yield_model
        nz = np.clip(np.asarray(normals[..., 2], dtype=np.float32), 0.0, 1.0)
        local_tilt_deg = np.degrees(np.arccos(nz))
        return np.clip(
            interpolate_tilt_scale_map(
                local_tilt_deg,
                model.tilt_deg_knots,
                model.tilt_scale_knots,
            ),
            1e-6,
            None,
        ).astype(np.float32)

    def _blur_scale(self, tilt_deg: float = 0.0) -> float:
        model = self.calibration.surrogate.blur_model
        tilt_scale = interpolate_tilt_scale(
            tilt_deg,
            model.tilt_deg_knots,
            model.tilt_scale_knots,
        )
        return float(max(model.scale * tilt_scale, 1e-6))

    def _depth_scale(self, tilt_deg: float = 0.0) -> float:
        model = self.calibration.surrogate.depth_model
        tilt_scale = interpolate_tilt_scale(
            tilt_deg,
            model.tilt_deg_knots,
            model.tilt_scale_knots,
        )
        return float(max(model.range_scale * tilt_scale, 1e-6))

    def _effective_range_um(self, range_um: float, tilt_deg: float = 0.0) -> float:
        return float(max(range_um * self._depth_scale(tilt_deg), 0.0))

    @staticmethod
    def _apply_beam_response(
        signal: np.ndarray,
        gain: float,
        contrast_factor: float,
    ) -> np.ndarray:
        centered = signal.astype(np.float32) - float(signal.mean())
        return ((centered * contrast_factor) + float(signal.mean())) * gain

    @staticmethod
    def _apply_quadrant_blur(quadrants: np.ndarray, sigma_px: float) -> np.ndarray:
        if gaussian_filter is None or sigma_px <= 1e-3:
            return quadrants.astype(np.float32)
        blurred = np.empty_like(quadrants, dtype=np.float32)
        for idx in range(quadrants.shape[-1]):
            blurred[..., idx] = gaussian_filter(
                quadrants[..., idx],
                sigma=sigma_px,
                mode="nearest",
            )
        return blurred

    @staticmethod
    def _normalize_feature_map(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        hi = float(np.quantile(finite, 0.95))
        if hi <= 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip(arr / hi, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _defect_profile_name(defect: DefectInstance) -> str | None:
        direct = _GEOMETRY_BENCHMARK_DEFECT_MAP.get(defect.defect_type)
        if direct is not None:
            return direct
        if defect.defect_type in {"crack_solidification", "crack_liquidation"}:
            length_mm = float(defect.params.get("length_mm", 0.0))
            return "slit_defect_crack_long" if length_mm >= 2.5 else "slit_defect_crack_short"
        if defect.defect_type == "spiking":
            length_mm = float(defect.params.get("spike_length_mm", 0.0))
            width_mm = float(defect.params.get("spike_width_mm", 0.0))
            return "spiking_surface_large" if max(length_mm, 4.0 * width_mm) >= 1.25 else "spiking_surface_medium"
        return None

    @staticmethod
    def _adaptive_profile_weight_map(
        defect: DefectInstance,
        profile_name: str,
        section_relief_mm: np.ndarray,
        void_fraction_map: np.ndarray,
        slope_norm: np.ndarray,
        curvature_norm: np.ndarray,
        protrusion_norm: np.ndarray,
        depression_norm: np.ndarray,
    ) -> np.ndarray:
        mask = defect.mask.astype(np.float32)
        if gaussian_filter is not None:
            mask = gaussian_filter(mask, sigma=1.25, mode="nearest")
        if float(mask.max()) > 1e-6:
            mask = mask / float(mask.max())
        if profile_name.startswith("spiking_surface"):
            geom_signal = np.maximum(protrusion_norm, 0.55 * slope_norm + 0.45 * curvature_norm)
        elif "crack" in profile_name or "cold_shut" in profile_name:
            geom_signal = np.maximum(np.maximum(void_fraction_map, depression_norm), 0.45 * slope_norm + 0.55 * curvature_norm)
        else:
            geom_signal = np.maximum(void_fraction_map, 0.50 * slope_norm + 0.50 * curvature_norm)
        return np.clip(mask * (0.35 + 0.65 * geom_signal), 0.0, 1.0).astype(np.float32)

    def _build_geometry_response_maps(
        self,
        *,
        section_relief_mm: np.ndarray,
        void_fraction_map: np.ndarray,
        defects: list[DefectInstance],
    ) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
        shape = section_relief_mm.shape
        ones = np.ones(shape, dtype=np.float32)
        defaults = {
            "yield_scale_map": ones.copy(),
            "blur_scale_map": ones.copy(),
            "depth_scale_map": ones.copy(),
            "topo_gain_map": ones.copy(),
            "openness_scale_map": ones.copy(),
        }
        profiles = self.calibration.surrogate.geometry_profiles
        if not profiles:
            return defaults, []

        gy, gx = np.gradient(section_relief_mm.astype(np.float32))
        slope_norm = self._normalize_feature_map(np.hypot(gx, gy))
        lap = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
        curvature_norm = self._normalize_feature_map(np.abs(lap))
        centered_relief = section_relief_mm.astype(np.float32) - float(np.median(section_relief_mm))
        protrusion_norm = self._normalize_feature_map(np.maximum(centered_relief, 0.0))
        depression_norm = self._normalize_feature_map(np.maximum(-centered_relief, 0.0))

        yield_scale_map = ones.copy()
        blur_scale_map = ones.copy()
        depth_scale_map = ones.copy()
        topo_gain_map = ones.copy()
        openness_scale_map = ones.copy()
        used_profiles: list[dict[str, float]] = []

        for defect in defects:
            profile_name = self._defect_profile_name(defect)
            if profile_name is None:
                continue
            profile = profiles.get(profile_name)
            if profile is None:
                continue
            weight = self._adaptive_profile_weight_map(
                defect,
                profile_name,
                section_relief_mm,
                void_fraction_map.astype(np.float32),
                slope_norm,
                curvature_norm,
                protrusion_norm,
                depression_norm,
            )
            if float(weight.max()) <= 1e-6:
                continue
            yield_scale_map += weight * (float(profile.yield_scale) - 1.0)
            blur_scale_map += weight * (float(profile.blur_scale) - 1.0)
            depth_scale_map += weight * (float(profile.depth_scale) - 1.0)
            topo_gain_map += weight * (float(profile.topo_gain) - 1.0)
            openness_scale_map += weight * (float(profile.openness_scale) - 1.0)
            used_profiles.append(
                {
                    "geometry": profile_name,
                    "max_weight": float(weight.max()),
                    "mean_weight": float(weight.mean()),
                }
            )

        maps = {
            "yield_scale_map": np.clip(yield_scale_map, 0.25, 4.0).astype(np.float32),
            "blur_scale_map": np.clip(blur_scale_map, 0.20, 4.0).astype(np.float32),
            "depth_scale_map": np.clip(depth_scale_map, 0.25, 4.0).astype(np.float32),
            "topo_gain_map": np.clip(topo_gain_map, 0.50, 3.0).astype(np.float32),
            "openness_scale_map": np.clip(openness_scale_map, 0.50, 4.0).astype(np.float32),
        }
        return maps, used_profiles

    def _apply_adaptive_quadrant_blur(
        self,
        quadrants: np.ndarray,
        sigma_px: float,
        blur_scale_map: np.ndarray,
    ) -> np.ndarray:
        if gaussian_filter is None or sigma_px <= 1e-3:
            return quadrants.astype(np.float32)
        scale_map = np.clip(np.asarray(blur_scale_map, dtype=np.float32), 0.20, 4.0)
        if float(np.max(scale_map) - np.min(scale_map)) <= 1e-3:
            return self._apply_quadrant_blur(quadrants, sigma_px * float(scale_map.mean()))

        min_scale = float(np.clip(np.quantile(scale_map, 0.05), 0.20, 4.0))
        max_scale = float(np.clip(np.quantile(scale_map, 0.95), 0.20, 4.0))
        low = self._apply_quadrant_blur(quadrants, sigma_px * min_scale)
        mid = self._apply_quadrant_blur(quadrants, sigma_px)
        high = self._apply_quadrant_blur(quadrants, sigma_px * max_scale)

        low_w = np.zeros_like(scale_map, dtype=np.float32)
        if min_scale < 0.999:
            low_w = np.clip((1.0 - scale_map) / max(1.0 - min_scale, 1e-6), 0.0, 1.0)
        high_w = np.zeros_like(scale_map, dtype=np.float32)
        if max_scale > 1.001:
            high_w = np.clip((scale_map - 1.0) / max(max_scale - 1.0, 1e-6), 0.0, 1.0)
        mid_w = np.clip(1.0 - low_w - high_w, 0.0, 1.0)

        out = np.empty_like(quadrants, dtype=np.float32)
        for idx in range(quadrants.shape[-1]):
            out[..., idx] = (
                low_w * low[..., idx]
                + mid_w * mid[..., idx]
                + high_w * high[..., idx]
            ).astype(np.float32)
        return out

    @staticmethod
    def _apply_quadrant_poisson(
        quadrants: np.ndarray,
        beam_current_ma: float,
        dwell_time_us: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = np.empty_like(quadrants, dtype=np.float32)
        for idx in range(quadrants.shape[-1]):
            out[..., idx] = apply_poisson_noise(
                quadrants[..., idx],
                beam_current_ma,
                dwell_time_us,
                rng,
            )
        return out

    @staticmethod
    def _apply_quadrant_thermal(
        quadrants: np.ndarray,
        *,
        temperature_k: float,
        bandwidth_hz: float,
        contamination_rate: float,
        rng: np.random.Generator,
        session_thermal_baseline: float,
        session_progress: float,
    ) -> np.ndarray:
        out = np.empty_like(quadrants, dtype=np.float32)
        for idx in range(quadrants.shape[-1]):
            axis = idx % 2
            baseline_scale = 1.0 if axis == 0 else 0.7
            drift_scale = 0.14 if axis == 0 else 0.10
            out[..., idx] = apply_thermal_drift(
                quadrants[..., idx],
                temperature_k=temperature_k,
                bandwidth_hz=bandwidth_hz,
                drift_rate=contamination_rate * drift_scale,
                rng=rng,
                axis=axis,
                session_baseline=baseline_scale * session_thermal_baseline,
                session_progress=session_progress,
            )
        return out

    @staticmethod
    def _apply_channel_calibration(
        arr: np.ndarray,
        scale: float,
        bias: float,
    ) -> np.ndarray:
        return arr.astype(np.float32) * float(scale) + float(bias)

    @staticmethod
    def _normals_from_heightmap(
        heightmap_mm: np.ndarray,
        pixel_size_mm: float,
    ) -> np.ndarray:
        spacing_mm = float(pixel_size_mm)
        if not np.isfinite(spacing_mm) or spacing_mm <= 0.0:
            raise ValueError("pixel_size_mm must be a finite positive value.")
        dz_dy, dz_dx = np.gradient(
            heightmap_mm.astype(np.float64),
            spacing_mm,
            spacing_mm,
        )
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(heightmap_mm)
        mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
        return np.stack([nx / mag, ny / mag, nz / mag], axis=-1).astype(np.float32)

    @staticmethod
    def _top_surface_section(
        volume,
        *,
        min_visible_area_px: int = 6,
    ) -> tuple[SectionSliceState, list[DefectInstance]]:
        surface_idx = int(
            np.clip(
                int(volume.metadata.get("surface_plane_index", len(volume.z_mm) - 1)),
                0,
                len(volume.z_mm) - 1,
            )
        )
        z0 = max(0, surface_idx - 1)
        z1 = min(len(volume.z_mm), surface_idx + 2)
        surface_z_mm = float(volume.z_mm[surface_idx])
        half_x = 0.5 * float(max(volume.x_mm.max() - volume.x_mm.min(), 0.0))
        half_y = 0.5 * float(max(volume.y_mm.max() - volume.y_mm.min(), 0.0))

        defect_mask = np.zeros(volume.defect_mask.shape[1:], dtype=bool)
        intersections: list[DefectInstance] = []
        for channel in volume.defect_channels:
            mask = np.any(channel["mask"][z0:z1].astype(bool), axis=0)
            if int(mask.sum()) < int(min_visible_area_px):
                continue
            defect_mask |= mask
            ys, xs = np.where(mask)
            intersections.append(
                DefectInstance(
                    defect_type=str(channel["type"]),
                    params=dict(channel["params"]),
                    mask=mask.astype(bool),
                    bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    category_id=int(channel["category_id"]),
                )
            )

        section = SectionSliceState(
            material_z_map=volume.material_z[surface_idx].astype(np.float32),
            yield_modifier_map=np.clip(volume.yield_modifier[surface_idx], 0.15, 1.20).astype(
                np.float32
            ),
            void_fraction_map=np.clip(
                np.max(volume.void_fraction[z0:z1], axis=0),
                0.0,
                1.0,
            ).astype(np.float32),
            relief_mm_map=volume.relief_mm[surface_idx].astype(np.float32),
            defect_mask=defect_mask.astype(np.uint8),
            plane=SectionPlaneSpec(
                origin_mm=(0.0, 0.0, surface_z_mm),
                normal_xyz=(0.0, 0.0, 1.0),
                u_axis_xyz=(1.0, 0.0, 0.0),
                v_axis_xyz=(0.0, 1.0, 0.0),
            ),
            sampling_extent_mm=(half_x, half_y),
            defect_intersections=[
                {
                    "type": defect.defect_type,
                    "bbox_xyxy": list(defect.bbox_xyxy),
                    "area_px": int(defect.mask.sum()),
                }
                for defect in intersections
            ],
            metadata={
                "valid_fraction": 1.0,
                "volume_shape": list(volume.material_z.shape),
                "sampling_mode": "top_surface",
                "surface_plane_index": surface_idx,
            },
        )
        return section, intersections

    def run(
        self,
        sample: SampledParameters,
        session_state: dict[str, object] | None = None,
    ) -> RenderResult:
        if self.emission_b_fit is not None and self.emission_b_fit.assumption is not None:
            assumption = self.emission_b_fit.assumption
            if sample.beam.e0_kev != assumption['energy_kev'] or sample.material.name != assumption['material_name']:
                raise ValueError('Sample is outside the assumed 304SS, 60 keV campaign settings.')
        if self.emission_b_fit is not None and self.emission_b_fit.qualified_energy_domain is not None:
            lo, hi = self.emission_b_fit.qualified_energy_domain
            if not lo <= sample.beam.e0_kev <= hi or not np.isclose(sample.material.z, self.emission_b_fit.qualified_material_z):
                raise ValueError("Sample energy/material lies outside the emission fit qualification.")
        rng = np.random.default_rng(sample.seed)
        session = session_state or {}
        session_progress = float(np.clip(float(session.get("progress", 1.0)), 0.0, 1.0))
        session_q_mismatch = session.get("quadrant_mismatch", {})
        if not isinstance(session_q_mismatch, dict):
            session_q_mismatch = {}
        session_thermal_baseline = float(session.get("thermal_baseline", 0.0))

        ebw_reinforcement_mm = float(
            np.clip(
                np.interp(sample.beam.focusing_current_norm, [0.0, 1.0], [0.02, 0.0]),
                0.0,
                0.06,
            )
        )
        power_factor = np.clip((sample.beam.power_w / 18_000.0) ** 0.15, 0.75, 1.25)
        speed_factor = np.clip(
            np.interp(sample.beam.travel_speed_mm_min, [100.0, 3000.0], [1.2, 0.82]),
            0.75,
            1.25,
        )
        ebw_width_mm = float(np.clip(1.25 * power_factor * speed_factor, 0.55, 2.4))
        ebw_haz_width_mm = float(
            np.clip(
                np.interp(sample.beam.e0_kev, [30.0, 175.0], [0.7, 1.8]),
                0.5,
                2.2,
            )
        )
        ebw_focus_quality = float(np.clip(sample.beam.focusing_current_norm, 0.0, 1.0))
        ebw_asymmetry_mm = float(
            np.clip(
                np.interp(sample.beam.travel_speed_mm_min, [100.0, 3000.0], [-0.10, 0.10]),
                -0.15,
                0.15,
            )
        )
        depth_power_factor = np.clip((sample.beam.power_w / 18_000.0) ** 0.32, 0.7, 1.6)
        depth_speed_factor = np.clip(
            np.interp(sample.beam.travel_speed_mm_min, [100.0, 3000.0], [1.25, 0.75]),
            0.6,
            1.4,
        )
        depth_focus_factor = np.clip(
            np.interp(sample.beam.focusing_current_norm, [0.0, 1.0], [0.85, 1.15]),
            0.7,
            1.4,
        )
        ebw_seam_depth_mm = float(
            np.clip(0.18 * depth_power_factor * depth_speed_factor * depth_focus_factor, 0.06, 0.45)
        )

        volume = build_voxel_volume(
            beam=sample.beam,
            material=sample.material,
            image_size=self.image_size,
            pixel_size_mm=self.pixel_size_mm,
            voxel_depth_mm=sample.slice.voxel_depth_mm,
            crown_height_mm=ebw_reinforcement_mm,
            seam_depth_mm=ebw_seam_depth_mm,
            bead_width_mm=ebw_width_mm,
            haz_width_mm=ebw_haz_width_mm,
            beam_focus_quality=ebw_focus_quality,
            asymmetry_offset_mm=ebw_asymmetry_mm,
            weld_geometry_mode=self.weld_geometry_mode,
            weld_layout=self.weld_layout,
            seed=sample.seed,
        )
        effective_pixel_size_mm = float(volume.metadata.get("pixel_size_mm", self.pixel_size_mm))
        apply_defect_bodies(
            volume,
            sample.defects,
            rng,
            target_slice_x_mm=sample.slice.slice_offset_mm,
            randomize_position=sample.randomize_defect_location,
            anchor_to_surface=self.render_view_mode == "top_surface",
        )
        if self.render_view_mode == "top_surface":
            section, defects = self._top_surface_section(volume)
            render_sampling_metadata = {
                "mode": "top_surface",
                "sampling_extent_mm": list(section.sampling_extent_mm),
                "surface_plane_index": int(section.metadata.get("surface_plane_index", 0)),
                "effective_pixel_size_mm": effective_pixel_size_mm,
            }
        else:
            plane = build_section_plane(sample.slice)
            section, defects = slice_volume(
                volume,
                plane,
                image_size=self.image_size,
                pixel_size_mm=effective_pixel_size_mm,
            )
            render_sampling_metadata = slice_metadata(sample.slice, plane, volume, section)

        range_um = kanaya_okayama_range(
            a=sample.material.a,
            e0_kev=sample.beam.e0_kev,
            z=sample.material.z,
            rho=sample.material.rho,
        )
        slice_tilt_deg = float(sample.slice.slice_tilt_deg)
        tilt_blur_scale = interpolate_tilt_scale(
            slice_tilt_deg,
            self.calibration.surrogate.blur_model.tilt_deg_knots,
            self.calibration.surrogate.blur_model.tilt_scale_knots,
        )
        tilt_depth_scale = interpolate_tilt_scale(
            slice_tilt_deg,
            self.calibration.surrogate.depth_model.tilt_deg_knots,
            self.calibration.surrogate.depth_model.tilt_scale_knots,
        )
        effective_range_scale = self._depth_scale(slice_tilt_deg)
        effective_range_um = self._effective_range_um(range_um, slice_tilt_deg)
        effective_heightmap = section.relief_mm_map.astype(np.float32)
        effective_normals = self._normals_from_heightmap(
            effective_heightmap,
            pixel_size_mm=effective_pixel_size_mm,
        )
        effective_material_z = section.material_z_map.astype(np.float32)
        if self.emission_b_fit is not None and self.emission_b_fit.qualified_material_z is not None:
            if not np.all(np.isclose(effective_material_z, self.emission_b_fit.qualified_material_z)):
                raise ValueError('Local material modifiers exceed the qualified alloy domain.')
        local_yield_tilt_scale_map = self._yield_tilt_scale_map(effective_normals)
        adaptive_maps, adaptive_profiles_used = self._build_geometry_response_maps(
            section_relief_mm=effective_heightmap,
            void_fraction_map=section.void_fraction_map.astype(np.float32),
            defects=defects,
        )
        local_yield_modifier = np.clip(
            section.yield_modifier_map * (1.0 - 0.22 * section.void_fraction_map),
            0.15,
            1.15,
        )
        local_openness = np.clip(
            0.88 * section.void_fraction_map,
            0.0,
            1.0,
        ).astype(np.float32)
        local_openness = np.clip(
            local_openness * adaptive_maps["openness_scale_map"],
            0.0,
            1.0,
        ).astype(np.float32)

        planned_defect_types = [str(item.get("type", "")) for item in sample.defects]
        pore_count = sum(1 for d in planned_defect_types if d in {"keyhole_pore", "intrinsic_pore"})
        defect_model = self.calibration.surrogate.defect_model
        if pore_count > 0:
            pore_scale = defect_model.pore_visibility_scale
            if pore_count > 1:
                pore_scale *= defect_model.cluster_visibility_scale
            local_openness = np.clip(local_openness * float(max(pore_scale, 1e-6)), 0.0, 1.0)

        depth_response_factor = float(
            np.clip(np.sqrt(max(effective_range_scale, 1e-6)), 0.7, 1.35)
        )
        local_depth_factor_map = np.clip(
            depth_response_factor * np.sqrt(adaptive_maps["depth_scale_map"]),
            0.7,
            1.6,
        ).astype(np.float32)
        local_yield_modifier = np.clip(local_yield_modifier * local_depth_factor_map, 0.15, 1.20)
        emission_b_coeff, emission_b_metadata = self._resolve_emission_b_coeff(
            effective_normals,
            sample.material.b,
            slice_tilt_deg,
        )

        gain = self._beam_gain(sample.beam.power_w)
        directional_gain = float(
            np.clip(
                np.interp(sample.beam.e0_kev, [30.0, 175.0], [1.25, 0.82])
                * np.interp(sample.beam.focusing_current_norm, [0.0, 1.0], [1.08, 0.92]),
                0.65,
                1.45,
            )
        )

        quadrants = compute_quadrant_signals(
            normals=effective_normals,
            detector=sample.detector,
            material=sample.material,
            material_z_map=effective_material_z,
            yield_modifier_map=local_yield_modifier,
            openness_map=local_openness,
            directional_gain=directional_gain,
            yield_scale=(
                self._yield_scale(sample.beam.e0_kev)
                * local_yield_tilt_scale_map
                * adaptive_maps["yield_scale_map"]
            ),
            obliquity_exponent_scale=self.calibration.surrogate.yield_model.obliquity_exponent_scale,
            b_coeff=emission_b_coeff,
            angular_density=(self.emission_b_fit.angular_table.density
                             if self.emission_b_fit is not None and self.emission_b_fit.angular_table is not None else None),
        )
        quadrants *= gain

        blur_px = self._interaction_blur_px(
            focal_spot_mm=sample.beam.focal_spot_mm,
            e0_kev=sample.beam.e0_kev,
            pixel_size_mm=effective_pixel_size_mm,
        )
        effective_blur_scale = self._blur_scale(slice_tilt_deg)
        blur_px *= effective_blur_scale
        quadrants = self._apply_adaptive_quadrant_blur(
            quadrants,
            sigma_px=blur_px,
            blur_scale_map=adaptive_maps["blur_scale_map"],
        )

        acquisition_metadata = {'version':'legacy','calibrated':False}
        if self.acquisition_model == 'raster_v2':
            coords = motion_coordinates(quadrants.shape[:2], sample.beam.dwell_time_us,
                                        effective_pixel_size_mm, self.acquisition_profile)
            quadrants = warp_image(quadrants, coords)
            self.acquisition_quiet_quadrants = quadrants.copy()
            if hasattr(self, 'quiet_quadrants'):
                self.quiet_quadrants = quadrants.copy()
            if coords is not None:
                section.defect_mask = warp_image(section.defect_mask, coords, order=0)
                transformed = []
                for defect in defects:
                    mask = warp_image(defect.mask.astype(np.uint8), coords, order=0).astype(bool)
                    if mask.any():
                        ys,xs = np.where(mask)
                        transformed.append(replace(defect, mask=mask,
                            bbox_xyxy=(int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))))
                defects = transformed
            quadrants, acquisition_metadata = acquire(quadrants, sample.noise, seed=sample.seed,
                dwell_us=sample.beam.dwell_time_us, profile=self.acquisition_profile)
        if self.acquisition_model == 'legacy' and sample.noise.contamination_enabled:
            quadrants = apply_contamination(
                quadrants,
                contamination_rate=sample.noise.contamination_rate,
                quadrant_offsets=sample.noise.quadrant_offsets,
                session_progress=session_progress,
                session_mismatch=session_q_mismatch,
            )

        if self.acquisition_model == 'legacy' and sample.noise.poisson_enabled:
            quadrants = self._apply_quadrant_poisson(
                quadrants,
                sample.beam.beam_current_ma,
                sample.beam.dwell_time_us,
                rng,
            )
        if self.acquisition_model == 'legacy' and sample.noise.thermal_enabled:
            quadrants = self._apply_quadrant_thermal(
                quadrants,
                temperature_k=sample.noise.temperature_k,
                bandwidth_hz=sample.noise.bandwidth_hz,
                rng=rng,
                contamination_rate=sample.noise.contamination_rate,
                session_thermal_baseline=session_thermal_baseline,
                session_progress=session_progress,
            )
        quadrants = np.clip(quadrants, 0.0, None).astype(np.float32)

        channels = form_channels(quadrants)
        contrast_factor = float(
            np.clip((60.0 / max(sample.beam.e0_kev, 1e-6)) ** 0.10, 0.88, 1.18)
        )
        topo_focus_gain = float(
            np.interp(sample.beam.focusing_current_norm, [0.0, 1.0], [1.10, 0.92])
        )
        if "undercut" in planned_defect_types:
            topo_focus_gain *= float(
                np.clip(defect_model.groove_topography_scale, 0.5, 2.5)
            )
        topo_focus_gain_map = topo_focus_gain * adaptive_maps["topo_gain_map"]

        compo = self._apply_beam_response(
            channels.compo,
            gain=1.0,
            contrast_factor=contrast_factor,
        )
        topo = channels.tmag * topo_focus_gain_map
        tx = channels.tx * topo_focus_gain_map
        ty = channels.ty * topo_focus_gain_map

        compo = self._apply_channel_calibration(
            compo,
            scale=self.calibration.compo.scale,
            bias=self.calibration.compo.bias,
        )
        topo = self._apply_channel_calibration(
            topo,
            scale=self.calibration.topo.scale,
            bias=self.calibration.topo.bias,
        )
        tx = self._apply_channel_calibration(
            tx,
            scale=self.calibration.topo.scale,
            bias=0.0,
        )
        ty = self._apply_channel_calibration(
            ty,
            scale=self.calibration.topo.scale,
            bias=0.0,
        )

        final = RenderChannels(
            compo=compo.astype(np.float32),
            topo=topo.astype(np.float32),
            tx=tx.astype(np.float32),
            ty=ty.astype(np.float32),
            tmag=topo.astype(np.float32),
            rx=channels.rx.astype(np.float32),
        )

        display_mapping = {
            "compo": build_linear_display_map(final.compo, low_q=0.02, high_q=0.985, pad_fraction=0.10),
            "topo": build_linear_display_map(final.topo, low_q=0.01, high_q=0.99, pad_fraction=0.08, clamp_low_to_zero=True),
            "tx": build_signed_display_map(final.tx, quantile=0.985, pad_fraction=0.08),
            "ty": build_signed_display_map(final.ty, quantile=0.985, pad_fraction=0.08),
            "tmag": build_linear_display_map(final.tmag, low_q=0.01, high_q=0.99, pad_fraction=0.08, clamp_low_to_zero=True),
            "rx": build_signed_display_map(final.rx, quantile=0.985, pad_fraction=0.08),
        }
        detector_frame = {
            "quadrants_deg": dict(sample.detector.quadrants_deg),
            "tx_positive_pair": ["A", "D"],
            "tx_negative_pair": ["B", "C"],
            "ty_positive_pair": ["A", "B"],
            "ty_negative_pair": ["C", "D"],
        }
        visible_defect_types = [d.defect_type for d in defects]
        unmatched_planned = list(planned_defect_types)
        for visible_type in visible_defect_types:
            if visible_type in unmatched_planned:
                unmatched_planned.remove(visible_type)
        metadata = {
            "operating_domain": {
                "incidence_deg_quantiles": np.percentile(np.degrees(np.arccos(np.clip(effective_normals[..., 2], -1, 1))), [0, 1, 50, 95, 99, 100]).tolist(),
                "incidence_histogram_1deg": np.histogram(np.degrees(np.arccos(np.clip(effective_normals[..., 2], -1, 1))), bins=np.arange(92))[0].tolist(),
                "material_z_range": [float(effective_material_z.min()), float(effective_material_z.max())],
                "base_material_z": float(sample.material.z),
                "modified_material_fraction": float(np.mean(~np.isclose(effective_material_z, sample.material.z))),
                "detector_polar_deg": np.degrees(np.arctan2([sample.detector.inner_radius_mm, sample.detector.outer_radius_mm], sample.detector.height_mm)).tolist(),
                "topo_finite": bool(np.isfinite(final.topo).all()),
                "topo_range": [float(final.topo.min()), float(final.topo.max())],
                "zero_quadrant_fraction": float(np.mean(quadrants == 0)),
                "mask_pixels": int(np.count_nonzero(section.defect_mask)),
            },
            "seed": sample.seed,
            "beam": asdict(sample.beam),
            "material": asdict(sample.material),
            "detector": asdict(sample.detector),
            "noise": asdict(sample.noise),
            "angular_frame_version": "local-beam-tangent/v2",
            "render_coordinate_frame": "image_relief_frame_beam_minus_z",
            "section_plane_role": "geometry_sampling_only",
            "emission_b_diagnostic_only": self.emission_b_diagnostic_only,
            "depth_response_model": "surface_relief_with_empirical_calibration_gain",
            "interaction_range_role": "diagnostic_only_not_penetration_attenuation",
            "range_um": range_um,
            "effective_range_um": effective_range_um,
            "beam_gain": gain,
            "beam_blur_px": blur_px,
            "beam_blur_scale": self.calibration.surrogate.blur_model.scale,
            "tilt_blur_scale": tilt_blur_scale,
            "effective_beam_blur_scale": effective_blur_scale,
            "beam_contrast_factor": contrast_factor,
            "directional_gain": directional_gain,
            "yield_scale": self._yield_scale(sample.beam.e0_kev),
            "tilt_yield_scale": self._yield_tilt_scale(slice_tilt_deg),
            "effective_yield_scale_range": [
                float(local_yield_tilt_scale_map.min()),
                float(local_yield_tilt_scale_map.max()),
            ],
            "obliquity_exponent_scale": self.calibration.surrogate.yield_model.obliquity_exponent_scale,
            "depth_response_scale": self.calibration.surrogate.depth_model.range_scale,
            "tilt_depth_scale": tilt_depth_scale,
            "effective_depth_response_scale": effective_range_scale,
            "multi_weld_profile_profiles_used": adaptive_profiles_used,
            "multi_weld_profile_profile_count": len(self.calibration.surrogate.geometry_profiles),
            "multi_weld_profile_yield_scale_range": [
                float(adaptive_maps["yield_scale_map"].min()),
                float(adaptive_maps["yield_scale_map"].max()),
            ],
            "multi_weld_profile_blur_scale_range": [
                float(adaptive_maps["blur_scale_map"].min()),
                float(adaptive_maps["blur_scale_map"].max()),
            ],
            "multi_weld_profile_depth_scale_range": [
                float(adaptive_maps["depth_scale_map"].min()),
                float(adaptive_maps["depth_scale_map"].max()),
            ],
            "multi_weld_profile_topo_gain_range": [
                float(adaptive_maps["topo_gain_map"].min()),
                float(adaptive_maps["topo_gain_map"].max()),
            ],
            "pore_visibility_scale": defect_model.pore_visibility_scale,
            "groove_topography_scale": defect_model.groove_topography_scale,
            "cluster_visibility_scale": defect_model.cluster_visibility_scale,
            "section_plane_weight": 1.0,
            "ebw_seam_depth_mm": ebw_seam_depth_mm,
            "ebw_reinforcement_mm": ebw_reinforcement_mm,
            "ebw_width_mm": ebw_width_mm,
            "ebw_haz_width_mm": ebw_haz_width_mm,
            "ebw_focus_quality": ebw_focus_quality,
            "ebw_asymmetry_mm": ebw_asymmetry_mm,
            "weld_geometry_mode": self.weld_geometry_mode,
            "weld_layout": asdict(self.weld_layout),
            "render_model_mode": self.render_model_mode,
            "emission_b": emission_b_metadata,
            "pixel_size_mm": float(self.pixel_size_mm),
            "effective_pixel_size_mm": effective_pixel_size_mm,
            "session_progress": session_progress,
            "calibration_source": self.calibration.source,
            "calibration_sha256": _sha256_if_file(self.calibration.source),
            "requested_calibration_path": self.requested_calibration_path,
            "requested_render_model_mode": self.requested_render_model_mode,
            "calibration_compo_scale": self.calibration.compo.scale,
            "calibration_topo_scale": self.calibration.topo.scale,
            "calibration_campaign_source": self.calibration.surrogate.source_campaign,
            "geometry_profiles_active": bool(self.calibration.surrogate.geometry_profiles),
            "display_mapping": display_mapping,
            "detector_frame": detector_frame,
            "planned_defects": planned_defect_types,
            "visible_defects": visible_defect_types,
            "excluded_nonvisible_defects": unmatched_planned,
            "surface_visible_label_policy": bool(self.render_view_mode == "top_surface"),
            "defects_anchored_to_surface": bool(self.render_view_mode == "top_surface"),
            "slice_runtime": asdict(sample.slice),
            "randomize_defect_location": bool(sample.randomize_defect_location),
            "render_view_mode": self.render_view_mode,
            "slice_render": render_sampling_metadata,
            "defects": planned_defect_types,
        }
        return RenderResult(
            channels=final,
            defect_mask=section.defect_mask.astype(np.uint8),
            defects=defects,
            metadata={**metadata, 'acquisition': acquisition_metadata},
        )
