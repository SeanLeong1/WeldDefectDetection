from __future__ import annotations

from dataclasses import replace
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import yaml

from .config_loader import (
    ConfigBundle,
    load_config_bundle,
)
from .domain_randomisation.batch_generator import BatchGenerator, BatchOptions
from .domain_randomisation.parameter_sampler import SampledParameters
from .geometry.defect_injection import (
    sample_defect_plan,
    validate_defect_plan,
)
from .geometry.voxel_volume import (
    apply_defect_bodies,
    build_section_plane,
    build_voxel_volume,
    slice_volume,
)
from .materials import material_from_average_atomic_number
from .models import BeamRuntime, DetectorRuntime, NoiseRuntime, SliceRuntime
from .renderer.scan_engine import ScanEngine
from .visualisation.image_preview import preview_channels, preview_defect_impact
from .visualisation.mesh_viewer import preview_surface_mesh


_DEFECT_UI_SCHEMA = [
    {
        "type": "keyhole_pore",
        "label": "Keyhole pore",
        "size_param": "diameter_mm",
        "depth_param": "depth_mm",
        "size_default": 1.4,
        "depth_default": 0.9,
        "count_default": 2,
    },
    {
        "type": "intrinsic_pore",
        "label": "Intrinsic pore",
        "size_param": "diameter_mm",
        "depth_param": "depth_mm",
        "size_default": 1.0,
        "depth_default": 0.8,
        "count_default": 1,
    },
    {
        "type": "crack_solidification",
        "label": "Solidification crack",
        "size_param": "length_mm",
        "depth_param": "depth_mm",
        "size_default": 3.2,
        "depth_default": 0.6,
        "count_default": 1,
    },
    {
        "type": "crack_liquidation",
        "label": "Liquation crack",
        "size_param": "length_mm",
        "depth_param": "depth_mm",
        "size_default": 2.6,
        "depth_default": 0.45,
        "count_default": 1,
    },
    {
        "type": "undercut",
        "label": "Undercut",
        "size_param": "groove_depth_mm",
        "depth_param": None,
        "size_default": 0.5,
        "depth_default": None,
        "count_default": 1,
    },
    {
        "type": "lack_of_fusion",
        "label": "Lack of fusion",
        "size_param": "extent_mm",
        "depth_param": "depth_mm",
        "size_default": 2.8,
        "depth_default": 0.9,
        "count_default": 1,
    },
    {
        "type": "cold_shut",
        "label": "Cold shut",
        "size_param": "extent_mm",
        "depth_param": "depth_mm",
        "size_default": 2.0,
        "depth_default": 0.7,
        "count_default": 1,
    },
    {
        "type": "spiking",
        "label": "Spiking",
        "size_param": "spike_length_mm",
        "depth_param": None,
        "size_default": 1.2,
        "depth_default": None,
        "count_default": 1,
    },
]


class App:
    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir).resolve()
        self.bundle = load_config_bundle(self.config_dir)

        self.root = tk.Tk()
        self.root.title("BSE Renderer GUI")
        self.root.geometry("980x760")
        self.status_var = tk.StringVar(value="Ready")
        self._build_vars()
        self._build_ui()

    def _build_vars(self) -> None:
        b = self.bundle.beam
        d = self.bundle.detector
        n = self.bundle.noise
        s = self.bundle.slice

        self.v_config_dir = tk.StringVar(value=str(self.config_dir))
        self.v_material = tk.StringVar(value=self.bundle.material.default_material)
        self.v_voltage = tk.DoubleVar(value=b.accelerating_voltage_kv.default)
        self.v_current = tk.DoubleVar(value=b.beam_current_ma.default)
        self.v_focus = tk.DoubleVar(value=b.focusing_current_norm.default)
        self.v_speed = tk.DoubleVar(value=b.travel_speed_mm_min.default)
        self.v_dwell = tk.DoubleVar(value=b.dwell_time_us.default)
        self.v_focal = tk.DoubleVar(value=b.focal_spot_mm.default)
        self.v_h = tk.DoubleVar(value=d.height_mm.default)
        self.v_r = tk.DoubleVar(value=d.radius_mm.default)
        self.v_temp = tk.DoubleVar(value=n.temperature_k.default)
        self.v_bw = tk.DoubleVar(value=n.bandwidth_hz.default)
        self.v_contam = tk.DoubleVar(value=n.contamination_rate.default)
        self.v_num_images = tk.IntVar(value=32)
        self.v_image_size = tk.IntVar(value=256)
        self.v_seed = tk.IntVar(value=42)
        self.v_max_defects = tk.IntVar(value=3)
        self.v_voxel_depth = tk.DoubleVar(value=s.voxel_depth_mm.default)
        self.v_slice_offset = tk.DoubleVar(value=s.slice_offset_mm.default)
        self.v_slice_tilt = tk.DoubleVar(value=s.slice_tilt_deg.default)
        self.v_slice_rotation = tk.DoubleVar(value=s.slice_rotation_deg.default)
        self.v_fixed_slice = tk.BooleanVar(value=True)
        self.v_use_avg_z = tk.BooleanVar(value=False)
        self.v_avg_z = tk.DoubleVar(
            value=self.bundle.material.presets[self.bundle.material.default_material].z
        )
        self.v_preview_noise = tk.BooleanVar(value=False)
        self._defect_controls: dict[str, dict[str, tk.Variable]] = {}
        for item in _DEFECT_UI_SCHEMA:
            self._defect_controls[item["type"]] = {
                "enabled": tk.BooleanVar(value=False),
                "count": tk.IntVar(value=item["count_default"]),
                "size": tk.DoubleVar(value=float(item["size_default"])),
                "depth": tk.DoubleVar(
                    value=0.0
                    if item["depth_default"] is None
                    else float(item["depth_default"])
                ),
            }
        self.v_output_dir = tk.StringVar(value=str((self.config_dir.parent / "data" / "output").resolve()))

    def _row(self, parent: tk.Widget, label: str, var: tk.Variable, row: int) -> None:
        tk.Label(parent, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=6, pady=3)
        tk.Entry(parent, textvariable=var, width=22).grid(row=row, column=1, sticky="ew", padx=6, pady=3)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="Config Dir").pack(side="left")
        tk.Entry(top, textvariable=self.v_config_dir, width=70).pack(side="left", padx=6)
        tk.Button(top, text="Browse", command=self._choose_config_dir).pack(side="left")
        tk.Button(top, text="Reload", command=self._reload_bundle).pack(side="left", padx=6)
        tk.Button(top, text="Save Defaults", command=self._save_defaults).pack(side="left", padx=6)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=4)

        beam_tab = tk.Frame(notebook)
        det_tab = tk.Frame(notebook)
        noise_tab = tk.Frame(notebook)
        run_tab = tk.Frame(notebook)
        notebook.add(beam_tab, text="Beam/Material")
        notebook.add(det_tab, text="Detector")
        notebook.add(noise_tab, text="Noise")
        notebook.add(run_tab, text="Run")

        self._row(beam_tab, "Material preset", self.v_material, 0)
        tk.Checkbutton(
            beam_tab,
            text="Use average atomic number override",
            variable=self.v_use_avg_z,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        self._row(beam_tab, "Average atomic number (Zavg)", self.v_avg_z, 2)
        self._row(beam_tab, "Accelerating voltage (kV)", self.v_voltage, 3)
        self._row(beam_tab, "Beam current (mA)", self.v_current, 4)
        self._row(beam_tab, "Focusing current (norm)", self.v_focus, 5)
        self._row(beam_tab, "Travel speed (mm/min)", self.v_speed, 6)
        self._row(beam_tab, "Dwell time (us)", self.v_dwell, 7)
        self._row(beam_tab, "Focal spot (mm)", self.v_focal, 8)

        self._row(det_tab, "Detector height h (mm)", self.v_h, 0)
        self._row(det_tab, "Detector radius R (mm)", self.v_r, 1)

        self._row(noise_tab, "Temperature (K)", self.v_temp, 0)
        self._row(noise_tab, "Bandwidth (Hz)", self.v_bw, 1)
        self._row(noise_tab, "Contamination rate", self.v_contam, 2)

        self._row(run_tab, "Num images", self.v_num_images, 0)
        self._row(run_tab, "Image size", self.v_image_size, 1)
        self._row(run_tab, "Seed", self.v_seed, 2)
        self._row(run_tab, "Voxel depth (mm)", self.v_voxel_depth, 3)
        self._row(run_tab, "Slice offset (mm)", self.v_slice_offset, 4)
        self._row(run_tab, "Slice tilt (deg)", self.v_slice_tilt, 5)
        self._row(run_tab, "Slice rotation (deg)", self.v_slice_rotation, 6)
        tk.Checkbutton(
            run_tab,
            text="Use fixed slice in batch generation",
            variable=self.v_fixed_slice,
        ).grid(row=7, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        self._row(run_tab, "Max defects/sample", self.v_max_defects, 8)
        self._row(run_tab, "Output dir", self.v_output_dir, 9)
        tk.Checkbutton(
            run_tab,
            text="Apply noise in preview",
            variable=self.v_preview_noise,
        ).grid(row=10, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        defect_frame = tk.LabelFrame(run_tab, text="Manufactured Defects (EBW-relevant)")
        defect_frame.grid(row=11, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        run_tab.grid_columnconfigure(1, weight=1)
        defect_frame.grid_columnconfigure(0, weight=1)
        tk.Label(defect_frame, text="Check defects, tune count/size/depth, then preview.").grid(
            row=0, column=0, columnspan=5, sticky="w", padx=4, pady=(3, 5)
        )
        tk.Label(defect_frame, text="Defect").grid(row=1, column=0, sticky="w", padx=4)
        tk.Label(defect_frame, text="Count").grid(row=1, column=1, sticky="w", padx=4)
        tk.Label(defect_frame, text="Size (mm)").grid(row=1, column=2, sticky="w", padx=4)
        tk.Label(defect_frame, text="Depth (mm)").grid(row=1, column=3, sticky="w", padx=4)
        row = 2
        for item in _DEFECT_UI_SCHEMA:
            ctrl = self._defect_controls[item["type"]]
            tk.Checkbutton(
                defect_frame,
                text=item["label"],
                variable=ctrl["enabled"],
            ).grid(row=row, column=0, sticky="w", padx=4, pady=1)
            tk.Entry(defect_frame, width=6, textvariable=ctrl["count"]).grid(
                row=row, column=1, sticky="w", padx=4, pady=1
            )
            tk.Entry(defect_frame, width=8, textvariable=ctrl["size"]).grid(
                row=row, column=2, sticky="w", padx=4, pady=1
            )
            if item["depth_param"] is None:
                tk.Label(defect_frame, text="-").grid(row=row, column=3, sticky="w", padx=4, pady=1)
            else:
                tk.Entry(defect_frame, width=8, textvariable=ctrl["depth"]).grid(
                    row=row, column=3, sticky="w", padx=4, pady=1
                )
            row += 1

        defect_btns = tk.Frame(defect_frame)
        defect_btns.grid(row=row, column=0, columnspan=5, sticky="w", padx=2, pady=(6, 4))
        tk.Button(defect_btns, text="Select All", command=lambda: self._set_all_defects(True)).pack(side="left", padx=3)
        tk.Button(defect_btns, text="Clear All", command=lambda: self._set_all_defects(False)).pack(side="left", padx=3)
        tk.Button(defect_btns, text="Obvious Sizes", command=self._set_obvious_defects).pack(side="left", padx=3)

        actions = tk.Frame(self.root)
        actions.pack(fill="x", padx=8, pady=8)
        tk.Button(actions, text="Preview 3D Surface", command=self._preview_mesh).pack(side="left", padx=4)
        tk.Button(actions, text="Preview Render", command=self._preview_render).pack(side="left", padx=4)
        tk.Button(actions, text="Compare Clean vs Defected", command=self._preview_defect_impact).pack(side="left", padx=4)
        tk.Button(actions, text="Generate Batch", command=self._generate_batch).pack(side="right", padx=4)

        tk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=8, pady=4)

    def _choose_config_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(self.config_dir))
        if not chosen:
            return
        self.v_config_dir.set(chosen)
        self._reload_bundle()

    def _reload_bundle(self) -> None:
        try:
            self.config_dir = Path(self.v_config_dir.get()).resolve()
            self.bundle = load_config_bundle(self.config_dir)
            self.v_voxel_depth.set(self.bundle.slice.voxel_depth_mm.default)
            self.v_slice_offset.set(self.bundle.slice.slice_offset_mm.default)
            self.v_slice_tilt.set(self.bundle.slice.slice_tilt_deg.default)
            self.v_slice_rotation.set(self.bundle.slice.slice_rotation_deg.default)
            self.status_var.set(f"Loaded config from {self.config_dir}")
        except Exception as exc:
            messagebox.showerror("Config Error", str(exc))

    def _set_all_defects(self, enabled: bool) -> None:
        for ctrl in self._defect_controls.values():
            ctrl["enabled"].set(enabled)

    def _set_obvious_defects(self) -> None:
        for item in _DEFECT_UI_SCHEMA:
            ctrl = self._defect_controls[item["type"]]
            ctrl["enabled"].set(True)
            ctrl["count"].set(item["count_default"])
            ctrl["size"].set(float(item["size_default"]))
            if item["depth_default"] is not None:
                ctrl["depth"].set(float(item["depth_default"]))

    def _clamp_defect_param(self, defect_type: str, param: str, value: float) -> float:
        bounds = self.bundle.defect.parameter_ranges.get(defect_type, {}).get(param)
        if bounds is None:
            return float(value)
        lo, hi = bounds
        return float(np.clip(float(value), lo, hi))

    def _build_selected_defect_plan(self) -> list[dict]:
        plan: list[dict] = []
        for item in _DEFECT_UI_SCHEMA:
            ctrl = self._defect_controls[item["type"]]
            if not bool(ctrl["enabled"].get()):
                continue
            count = max(0, int(ctrl["count"].get()))
            if count <= 0:
                continue
            defect_type = item["type"]
            size = self._clamp_defect_param(
                defect_type,
                item["size_param"],
                float(ctrl["size"].get()),
            )
            params: dict[str, float] = {item["size_param"]: size}
            if item["depth_param"] is not None:
                params[item["depth_param"]] = self._clamp_defect_param(
                    defect_type,
                    item["depth_param"],
                    float(ctrl["depth"].get()),
                )

            if defect_type.startswith("crack_"):
                params["width_mm"] = self._clamp_defect_param(
                    defect_type,
                    "width_mm",
                    max(0.05, 0.05 * size),
                )
            elif defect_type == "undercut":
                params["toe_angle_deg"] = self._clamp_defect_param(
                    defect_type,
                    "toe_angle_deg",
                    34.0,
                )
            elif defect_type == "lack_of_fusion":
                params["irregularity"] = self._clamp_defect_param(
                    defect_type,
                    "irregularity",
                    0.7,
                )
            elif defect_type == "spiking":
                params["spike_width_mm"] = self._clamp_defect_param(
                    defect_type,
                    "spike_width_mm",
                    max(0.06, 0.10 * size),
                )

            for _ in range(count):
                plan.append({"type": defect_type, "params": dict(params)})
        validate_defect_plan(self.bundle.defect, plan)
        return plan

    def _manual_sample(self) -> tuple:
        rng = np.random.default_rng(self.v_seed.get())
        if self.v_use_avg_z.get():
            material = material_from_average_atomic_number(
                average_z=self.v_avg_z.get(),
                known_presets=self.bundle.material.presets.values(),
                name=f"Zavg_{self.v_avg_z.get():.2f}",
            )
        else:
            if self.v_material.get() not in self.bundle.material.presets:
                raise ValueError("Unknown material preset")
            material = self.bundle.material.presets[self.v_material.get()]

        beam = BeamRuntime(
            accelerating_voltage_kv=self.v_voltage.get(),
            beam_current_ma=self.v_current.get(),
            focusing_current_norm=self.v_focus.get(),
            travel_speed_mm_min=self.v_speed.get(),
            dwell_time_us=self.v_dwell.get(),
            focal_spot_mm=self.v_focal.get(),
            e0_kev=self.v_voltage.get(),
            power_w=(self.v_voltage.get() * 1000.0) * (self.v_current.get() / 1000.0),
        )
        detector = DetectorRuntime(
            height_mm=self.v_h.get(),
            radius_mm=self.v_r.get(),
            inner_radius_mm=self.bundle.detector.inner_radius_mm.default,
            outer_radius_mm=self.bundle.detector.outer_radius_mm.default,
            quadrants_deg=dict(self.bundle.detector.quadrants_deg),
            sensitivity={q: 1.0 for q in ("A", "B", "C", "D")},
            psi0_rad=float(np.arctan2(self.v_h.get(), self.v_r.get())),
        )
        noise = NoiseRuntime(
            poisson_enabled=self.bundle.noise.poisson_enabled,
            thermal_enabled=self.bundle.noise.thermal_enabled,
            contamination_enabled=self.bundle.noise.contamination_enabled,
            temperature_k=self.v_temp.get(),
            bandwidth_hz=self.v_bw.get(),
            contamination_rate=self.v_contam.get(),
            quadrant_offsets={q: 0.0 for q in ("A", "B", "C", "D")},
        )
        slice_runtime = SliceRuntime(
            voxel_depth_mm=float(self.v_voxel_depth.get()),
            slice_offset_mm=float(self.v_slice_offset.get()),
            slice_tilt_deg=float(self.v_slice_tilt.get()),
            slice_rotation_deg=float(self.v_slice_rotation.get()),
        )
        defect_plan = self._build_selected_defect_plan()
        if not defect_plan:
            defect_plan = sample_defect_plan(
                self.bundle.defect, rng, max_defects=self.v_max_defects.get()
            )
        return beam, material, detector, noise, slice_runtime, defect_plan

    def _build_preview_section(self) -> tuple:
        beam, material, _, _, slice_runtime, defects = self._manual_sample()
        power_factor = np.clip((beam.power_w / 18_000.0) ** 0.15, 0.75, 1.25)
        speed_factor = np.clip(
            np.interp(beam.travel_speed_mm_min, [100.0, 3000.0], [1.2, 0.82]),
            0.75,
            1.25,
        )
        depth_power_factor = np.clip((beam.power_w / 18_000.0) ** 0.32, 0.7, 1.6)
        depth_speed_factor = np.clip(
            np.interp(beam.travel_speed_mm_min, [100.0, 3000.0], [1.25, 0.75]),
            0.6,
            1.4,
        )
        depth_focus_factor = np.clip(
            np.interp(beam.focusing_current_norm, [0.0, 1.0], [0.85, 1.15]),
            0.7,
            1.4,
        )
        reinforcement_mm = float(
            np.clip(np.interp(beam.focusing_current_norm, [0.0, 1.0], [0.02, 0.0]), 0.0, 0.06)
        )
        width_mm = float(np.clip(1.25 * power_factor * speed_factor, 0.55, 2.4))
        haz_width_mm = float(
            np.clip(np.interp(beam.e0_kev, [30.0, 175.0], [0.7, 1.8]), 0.5, 2.2)
        )
        seam_depth_mm = float(
            np.clip(0.18 * depth_power_factor * depth_speed_factor * depth_focus_factor, 0.06, 0.45)
        )
        asymmetry_mm = float(
            np.clip(
                np.interp(beam.travel_speed_mm_min, [100.0, 3000.0], [-0.10, 0.10]),
                -0.15,
                0.15,
            )
        )
        volume = build_voxel_volume(
            beam=beam,
            material=material,
            image_size=self.v_image_size.get(),
            pixel_size_mm=0.05,
            voxel_depth_mm=slice_runtime.voxel_depth_mm,
            crown_height_mm=reinforcement_mm,
            seam_depth_mm=seam_depth_mm,
            bead_width_mm=width_mm,
            haz_width_mm=haz_width_mm,
            beam_focus_quality=float(np.clip(beam.focusing_current_norm, 0.0, 1.0)),
            asymmetry_offset_mm=asymmetry_mm,
            seed=self.v_seed.get(),
        )
        apply_defect_bodies(
            volume,
            defects,
            np.random.default_rng(self.v_seed.get()),
            target_slice_x_mm=slice_runtime.slice_offset_mm,
        )
        plane = build_section_plane(slice_runtime)
        section, intersections = slice_volume(
            volume,
            plane,
            image_size=self.v_image_size.get(),
            pixel_size_mm=0.05,
        )
        return section, intersections

    def _preview_mesh(self) -> None:
        try:
            section, _ = self._build_preview_section()
            preview_surface_mesh(section.relief_mm_map, section.defect_mask.astype(bool))
            self.status_var.set("Slice mesh preview complete")
        except Exception as exc:
            messagebox.showerror("Preview Error", str(exc))

    def _preview_render(self) -> None:
        try:
            beam, material, detector, noise, slice_runtime, defects = self._manual_sample()
            if not self.v_preview_noise.get():
                noise = replace(
                    noise,
                    poisson_enabled=False,
                    thermal_enabled=False,
                    contamination_enabled=False,
                    contamination_rate=0.0,
                    quadrant_offsets={q: 0.0 for q in ("A", "B", "C", "D")},
                )
            sample = SampledParameters(
                seed=self.v_seed.get(),
                beam=beam,
                material=material,
                detector=detector,
                noise=noise,
                slice=slice_runtime,
                defects=defects,
            )
            result = ScanEngine(image_size=self.v_image_size.get()).run(sample)
            preview_channels(
                result.channels.compo,
                result.channels.topo,
                result.channels.tx,
                result.channels.ty,
                result.defect_mask,
                display_mapping=result.metadata.get("display_mapping"),
                title="GUI Render Preview",
            )
            mode = "with noise" if self.v_preview_noise.get() else "no noise"
            self.status_var.set(f"Render preview complete ({mode})")
        except Exception as exc:
            messagebox.showerror("Preview Error", str(exc))

    def _preview_defect_impact(self) -> None:
        try:
            beam, material, detector, noise, slice_runtime, defects = self._manual_sample()
            if not self.v_preview_noise.get():
                noise = replace(
                    noise,
                    poisson_enabled=False,
                    thermal_enabled=False,
                    contamination_enabled=False,
                    contamination_rate=0.0,
                    quadrant_offsets={q: 0.0 for q in ("A", "B", "C", "D")},
                )
            sample_defect = SampledParameters(
                seed=self.v_seed.get(),
                beam=beam,
                material=material,
                detector=detector,
                noise=noise,
                slice=slice_runtime,
                defects=defects,
            )
            sample_clean = replace(sample_defect, defects=[])
            engine = ScanEngine(image_size=self.v_image_size.get())
            clean = engine.run(sample_clean)
            defected = engine.run(sample_defect)
            preview_defect_impact(
                clean_compo=clean.channels.compo,
                defect_compo=defected.channels.compo,
                clean_topo=clean.channels.topo,
                defect_topo=defected.channels.topo,
                defect_tx=defected.channels.tx,
                defect_ty=defected.channels.ty,
                defect_mask=defected.defect_mask,
                clean_mapping=clean.metadata.get("display_mapping"),
                defect_mapping=defected.metadata.get("display_mapping"),
            )
            self.status_var.set("Defect impact comparison complete")
        except Exception as exc:
            messagebox.showerror("Preview Error", str(exc))

    def _generate_batch(self) -> None:
        def worker() -> None:
            try:
                opts = BatchOptions(
                    num_images=self.v_num_images.get(),
                    output_dir=Path(self.v_output_dir.get()),
                    image_size=self.v_image_size.get(),
                    seed=self.v_seed.get(),
                    max_defects=self.v_max_defects.get(),
                    voxel_depth_mm=float(self.v_voxel_depth.get()),
                    slice_offset_mm=float(self.v_slice_offset.get()),
                    slice_tilt_deg=float(self.v_slice_tilt.get()),
                    slice_rotation_deg=float(self.v_slice_rotation.get()),
                    randomize_slice=not bool(self.v_fixed_slice.get()),
                )
                summary = BatchGenerator(self.bundle, opts).run()
                self.status_var.set(
                    f"Generated {summary['num_images']} images in {summary['output_dir']}"
                )
            except Exception as exc:
                self.status_var.set(f"Generation failed: {exc}")

        self.status_var.set("Generating dataset...")
        threading.Thread(target=worker, daemon=True).start()

    def _save_defaults(self) -> None:
        try:
            self._write_default(self.config_dir / "beam_params.yaml", "accelerating_voltage_kv", self.v_voltage.get())
            self._write_default(self.config_dir / "beam_params.yaml", "beam_current_ma", self.v_current.get())
            self._write_default(self.config_dir / "beam_params.yaml", "focusing_current_norm", self.v_focus.get())
            self._write_default(self.config_dir / "beam_params.yaml", "travel_speed_mm_min", self.v_speed.get())
            self._write_default(self.config_dir / "beam_params.yaml", "dwell_time_us", self.v_dwell.get())
            self._write_default(self.config_dir / "beam_params.yaml", "focal_spot_mm", self.v_focal.get())
            self._write_default(self.config_dir / "detector_params.yaml", "height_mm", self.v_h.get())
            self._write_default(self.config_dir / "detector_params.yaml", "radius_mm", self.v_r.get())
            self._write_default(self.config_dir / "noise_params.yaml", "temperature_k", self.v_temp.get())
            self._write_default(self.config_dir / "noise_params.yaml", "bandwidth_hz", self.v_bw.get())
            self._write_default(self.config_dir / "noise_params.yaml", "contamination_rate", self.v_contam.get())
            self._write_default(self.config_dir / "slice_params.yaml", "voxel_depth_mm", self.v_voxel_depth.get())
            self._write_default(self.config_dir / "slice_params.yaml", "slice_offset_mm", self.v_slice_offset.get())
            self._write_default(self.config_dir / "slice_params.yaml", "slice_tilt_deg", self.v_slice_tilt.get())
            self._write_default(self.config_dir / "slice_params.yaml", "slice_rotation_deg", self.v_slice_rotation.get())
            mat_path = self.config_dir / "material_params.yaml"
            mat_data = yaml.safe_load(mat_path.read_text(encoding="utf-8"))
            mat_data["default_material"] = self.v_material.get()
            mat_path.write_text(yaml.safe_dump(mat_data, sort_keys=False), encoding="utf-8")
            self.status_var.set("Saved defaults to YAML files")
            self._reload_bundle()
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    @staticmethod
    def _write_default(path: Path, key: str, value: float) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data[key]["default"] = float(value)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def run(self) -> None:
        self.root.mainloop()
