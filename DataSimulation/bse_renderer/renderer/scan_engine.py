from __future__ import annotations

from dataclasses import asdict
import os

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
from ..models import RenderChannels, RenderResult
from ..noise.contamination_model import apply_contamination
from ..noise.poisson_noise import apply_poisson_noise
from ..noise.thermal_drift import apply_thermal_drift
from ..physics.interaction_volume import kanaya_okayama_range
from ..utils import (
    build_linear_display_map,
    build_signed_display_map,
)
from .calibration import RendererCalibration, load_renderer_calibration
from .image_formation import form_channels
from .quadrant_detector import compute_quadrant_signals


class ScanEngine:
    def __init__(
        self,
        image_size: int = 256,
        pixel_size_mm: float = 0.05,
        calibration_path: str | None = None,
        calibration: RendererCalibration | None = None,
    ):
        self.image_size = image_size
        self.pixel_size_mm = pixel_size_mm
        if calibration is not None:
            self.calibration = calibration
        else:
            cal_path = calibration_path or os.getenv("BSE_RENDERER_CALIBRATION")
            self.calibration = load_renderer_calibration(cal_path)

    @staticmethod
    def _beam_gain(power_w: float) -> float:
        ref_power_w = 18_000.0  # 60 kV * 300 mA
        return float(np.clip((max(power_w, 1e-6) / ref_power_w) ** 0.20, 0.55, 1.65))

    def _interaction_blur_px(self, focal_spot_mm: float, e0_kev: float) -> float:
        # Larger beam spot and higher accelerating voltage broaden the source volume.
        spot_px = max(focal_spot_mm, 0.0) / max(self.pixel_size_mm, 1e-6)
        energy_term = np.clip((max(e0_kev, 1e-6) / 60.0) ** 0.6, 0.5, 2.2)
        return float(np.clip(0.10 * spot_px + 0.35 * energy_term, 0.0, 4.0))

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
    def _normals_from_heightmap(heightmap_mm: np.ndarray) -> np.ndarray:
        dz_dy, dz_dx = np.gradient(heightmap_mm.astype(np.float64))
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(heightmap_mm)
        mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
        return np.stack([nx / mag, ny / mag, nz / mag], axis=-1).astype(np.float32)

    def run(
        self,
        sample: SampledParameters,
        session_state: dict[str, object] | None = None,
    ) -> RenderResult:
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
            seed=sample.seed,
        )
        apply_defect_bodies(
            volume,
            sample.defects,
            rng,
            target_slice_x_mm=sample.slice.slice_offset_mm,
        )
        plane = build_section_plane(sample.slice)
        section, defects = slice_volume(
            volume,
            plane,
            image_size=self.image_size,
            pixel_size_mm=self.pixel_size_mm,
        )

        range_um = kanaya_okayama_range(
            a=sample.material.a,
            e0_kev=sample.beam.e0_kev,
            z=sample.material.z,
            rho=sample.material.rho,
        )
        effective_heightmap = section.relief_mm_map.astype(np.float32)
        effective_normals = self._normals_from_heightmap(effective_heightmap)
        effective_material_z = section.material_z_map.astype(np.float32)
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
        )
        quadrants *= gain

        blur_px = self._interaction_blur_px(
            focal_spot_mm=sample.beam.focal_spot_mm,
            e0_kev=sample.beam.e0_kev,
        )
        quadrants = self._apply_quadrant_blur(quadrants, sigma_px=blur_px)

        if sample.noise.contamination_enabled:
            quadrants = apply_contamination(
                quadrants,
                contamination_rate=sample.noise.contamination_rate,
                quadrant_offsets=sample.noise.quadrant_offsets,
                session_progress=session_progress,
                session_mismatch=session_q_mismatch,
            )

        if sample.noise.poisson_enabled:
            quadrants = self._apply_quadrant_poisson(
                quadrants,
                sample.beam.beam_current_ma,
                sample.beam.dwell_time_us,
                rng,
            )
        if sample.noise.thermal_enabled:
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

        compo = self._apply_beam_response(
            channels.compo,
            gain=1.0,
            contrast_factor=contrast_factor,
        )
        topo = channels.tmag * topo_focus_gain
        tx = channels.tx * topo_focus_gain
        ty = channels.ty * topo_focus_gain

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
        metadata = {
            "seed": sample.seed,
            "beam": asdict(sample.beam),
            "material": asdict(sample.material),
            "detector": asdict(sample.detector),
            "noise": asdict(sample.noise),
            "range_um": range_um,
            "beam_gain": gain,
            "beam_blur_px": blur_px,
            "beam_contrast_factor": contrast_factor,
            "directional_gain": directional_gain,
            "section_plane_weight": 1.0,
            "ebw_seam_depth_mm": ebw_seam_depth_mm,
            "ebw_reinforcement_mm": ebw_reinforcement_mm,
            "ebw_width_mm": ebw_width_mm,
            "ebw_haz_width_mm": ebw_haz_width_mm,
            "ebw_focus_quality": ebw_focus_quality,
            "ebw_asymmetry_mm": ebw_asymmetry_mm,
            "session_progress": session_progress,
            "calibration_source": self.calibration.source,
            "calibration_compo_scale": self.calibration.compo.scale,
            "calibration_topo_scale": self.calibration.topo.scale,
            "display_mapping": display_mapping,
            "detector_frame": detector_frame,
            "planned_defects": [str(item.get("type", "")) for item in sample.defects],
            "visible_defects": [d.defect_type for d in defects],
            "slice_runtime": asdict(sample.slice),
            "slice_render": slice_metadata(sample.slice, plane, volume, section),
            "defects": [str(item.get("type", "")) for item in sample.defects],
        }
        return RenderResult(
            channels=final,
            defect_mask=section.defect_mask.astype(np.uint8),
            defects=defects,
            metadata=metadata,
        )
