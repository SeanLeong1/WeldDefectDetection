from __future__ import annotations

from dataclasses import replace
import threading
import queue
import sys
import time
import uuid
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
    validate_defect_plan,
)
from .geometry.voxel_volume import (
    apply_defect_bodies,
    build_section_plane,
    build_voxel_volume,
    slice_volume,
)
from .geometry.weld_profile import (
    WELD_GEOMETRY_CLOSED,
    WELD_GEOMETRY_RECESSED,
    WELD_LAYOUT_CIRCULAR_WELD,
    WELD_LAYOUT_LOCAL_PATCH,
    WELD_LAYOUT_STRAIGHT_SEAM,
)
from .materials import material_from_average_atomic_number
from .models import BeamRuntime, DetectorRuntime, NoiseRuntime, SliceRuntime, WeldLayoutRuntime
from .renderer.calibration import (
    RENDER_MODEL_FORMULA_CASINO,
    RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
    RENDER_MODEL_FORMULA_ONLY,
)
from .renderer.emission_b import (
    EMISSION_B_MODE_CONSTANT,
    EMISSION_B_MODE_LAMBERTIAN,
    EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    EMISSION_B_MODE_SIN_FIT,
    EMISSION_B_MODE_QUADRATIC_FIT,
)
from .renderer.scan_engine import ScanEngine
from .comparison_metrics import compute_topography_delta_metrics
from .visualisation.image_preview import (
    preview_calibration_modes,
    preview_channels,
    preview_defect_impact,
)
from .visualisation.mesh_viewer import preview_surface_mesh, preview_volume_geometry


_DEFECT_UI_SCHEMA = [
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
    {
        "type": "keyhole_pore",
        "label": "Keyhole pore (procedural)",
        "size_param": "diameter_mm",
        "depth_param": "depth_mm",
        "size_default": 1.4,
        "depth_default": 0.9,
        "count_default": 2,
    },
    {
        "type": "intrinsic_pore",
        "label": "Intrinsic pore (procedural)",
        "size_param": "diameter_mm",
        "depth_param": "depth_mm",
        "size_default": 1.0,
        "depth_default": 0.8,
        "count_default": 1,
    },
    {
        "type": "crack_liquidation",
        "label": "Liquation crack (legacy)",
        "size_param": "length_mm",
        "depth_param": "depth_mm",
        "size_default": 2.6,
        "depth_default": 0.45,
        "count_default": 1,
    },
    {
        "type": "undercut",
        "label": "Undercut (legacy)",
        "size_param": "groove_depth_mm",
        "depth_param": None,
        "size_default": 0.5,
        "depth_default": None,
        "count_default": 1,
    },
    {
        "type": "lack_of_fusion",
        "label": "Lack of fusion (legacy)",
        "size_param": "extent_mm",
        "depth_param": "depth_mm",
        "size_default": 2.8,
        "depth_default": 0.9,
        "count_default": 1,
    },
]

_WELD_GEOMETRY_LABELS = {
    "Closed fusion zone (recommended)": WELD_GEOMETRY_CLOSED,
    "Legacy open-seam approximation (legacy)": WELD_GEOMETRY_RECESSED,
}
_RENDER_MODEL_LABELS = {
    "Formula only": RENDER_MODEL_FORMULA_ONLY,
    "Formula + CASINO": RENDER_MODEL_FORMULA_CASINO,
    "Formula + CASINO + geometry profiles": RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
}
_EMISSION_B_LABELS = {
    "Constant B (engineering approximation)": EMISSION_B_MODE_CONSTANT,
    "Lambertian B=0 (uncalibrated fallback)": EMISSION_B_MODE_LAMBERTIAN,
    "Constant material B": EMISSION_B_MODE_CONSTANT,
    "Pooled fitted constant B": EMISSION_B_MODE_POOLED_CONSTANT_FIT,
    "Sine fitted B(alpha) (experimental)": EMISSION_B_MODE_SIN_FIT,
    "Quadratic fitted B(alpha)": EMISSION_B_MODE_QUADRATIC_FIT,
}
_DEFAULT_WELD_GEOMETRY_LABEL = "Closed fusion zone (recommended)"
_LEGACY_WELD_GEOMETRY_LABEL = "Legacy open-seam approximation (legacy)"
_SURFACE_CAMPAIGN_DEFECT_TYPES = {"crack_solidification", "cold_shut", "spiking"}
_LEGACY_DEFECT_TYPES = {
    "keyhole_pore",
    "intrinsic_pore",
    "crack_liquidation",
    "undercut",
    "lack_of_fusion",
}
_PREVIEW_MODE_LABELS = {
    "Top weld surface": "top_surface",
    "Cross-section relief slice": "surface_relief",
    "Fusion body volume": "fusion_body_volume",
}
_WELD_LAYOUT_LABELS = {
    "Local patch": WELD_LAYOUT_LOCAL_PATCH,
    "Straight seam": WELD_LAYOUT_STRAIGHT_SEAM,
    "Circular weld": WELD_LAYOUT_CIRCULAR_WELD,
}
_SCALE_PRESET_VALUES = {
    "Fine detail (10 um/px)": 0.01,
    "Default local patch (50 um/px)": 0.05,
    "Wide patch (100 um/px)": 0.10,
    "Macro survey (250 um/px)": 0.25,
    "Custom": None,
}


class App:
    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir).resolve()
        self.bundle = load_config_bundle(self.config_dir)
        self._calibration_options: dict[str, str | None] = {}
        self._emission_b_options: dict[str, str | None] = {}
        self.calibration_combo: ttk.Combobox | None = None
        self.emission_b_combo: ttk.Combobox | None = None
        self.weld_geometry_combo: ttk.Combobox | None = None
        self._run_field_widgets: dict[str, tuple[tk.Label, tk.Entry]] = {}
        self._defect_row_widgets: dict[str, list[tk.Widget]] = {}
        self._legacy_defect_button: tk.Button | None = None

        self.root = tk.Tk()
        repo = Path(__file__).resolve().parents[2]
        if str(repo / 'WeldDefectDetection') not in sys.path:
            sys.path.insert(0, str(repo / 'WeldDefectDetection'))
        from inspection.gui import apply_theme
        apply_theme(self.root)
        self.root.title("Weld Vision | TOPO research studio")
        self.root.geometry("1220x900")
        self.root.minsize(1000, 760)
        self._batch_events = queue.Queue()
        self._batch_running = False
        self.status_var = tk.StringVar(value="Ready")
        self._active_mousewheel_canvas: tk.Canvas | None = None
        self._build_vars()
        self._build_ui()
        self.root.after(150, self._poll_batch)

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
        self.v_inner_r = tk.DoubleVar(value=d.inner_radius_mm.default)
        self.v_outer_r = tk.DoubleVar(value=d.outer_radius_mm.default)
        self.v_temp = tk.DoubleVar(value=n.temperature_k.default)
        self.v_bw = tk.DoubleVar(value=n.bandwidth_hz.default)
        self.v_contam = tk.DoubleVar(value=n.contamination_rate.default)
        self.v_num_images = tk.IntVar(value=32)
        self.v_image_size = tk.IntVar(value=256)
        self.v_scale_preset = tk.StringVar(value="Default local patch (50 um/px)")
        self.v_pixel_size_mm = tk.DoubleVar(value=0.05)
        self.v_seed = tk.IntVar(value=42)
        self.v_max_defects = tk.IntVar(value=3)
        self.v_voxel_depth = tk.DoubleVar(value=s.voxel_depth_mm.default)
        self.v_slice_offset = tk.DoubleVar(value=s.slice_offset_mm.default)
        self.v_slice_tilt = tk.DoubleVar(value=s.slice_tilt_deg.default)
        self.v_slice_rotation = tk.DoubleVar(value=s.slice_rotation_deg.default)
        self.v_fixed_slice = tk.BooleanVar(value=True)
        self.v_random_defect_location = tk.BooleanVar(value=True)
        self.v_use_avg_z = tk.BooleanVar(value=False)
        self.v_avg_z = tk.DoubleVar(
            value=self.bundle.material.presets[self.bundle.material.default_material].z
        )
        self.v_preview_noise = tk.BooleanVar(value=False)
        self.v_disable_batch_noise = tk.BooleanVar(value=False)
        self.v_show_legacy_options = tk.BooleanVar(value=True)
        self.v_control_hint = tk.StringVar(value="")
        self.v_weld_geometry_label = tk.StringVar(value=_DEFAULT_WELD_GEOMETRY_LABEL)
        self.v_weld_layout_label = tk.StringVar(value="Local patch")
        self.v_seam_rotation = tk.DoubleVar(value=0.0)
        self.v_seam_offset = tk.DoubleVar(value=0.0)
        self.v_circle_radius = tk.DoubleVar(value=40.0)
        self.v_circle_position = tk.DoubleVar(value=90.0)
        self.v_geometry_preview_label = tk.StringVar(value="Top weld surface")
        self.v_render_model_label = tk.StringVar(value="Formula only")
        self.v_calibration_label = tk.StringVar(value="No calibration selected")
        self.v_emission_b_label = tk.StringVar(value="Constant B (engineering approximation)")
        self.v_constant_b = tk.DoubleVar(value=0.9206948532270666)
        self.v_emission_b_fit_label = tk.StringVar(value="No emission B fit selected")
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
        self.v_scale_preset.trace_add("write", self._on_scale_preset_changed)
        self.v_show_legacy_options.trace_add("write", self._refresh_legacy_option_visibility)
        self.v_weld_layout_label.trace_add("write", self._refresh_parameter_control_states)
        self.v_slice_tilt.trace_add("write", self._refresh_parameter_control_states)
        self.v_geometry_preview_label.trace_add("write", self._refresh_parameter_control_states)
        self._refresh_calibration_options(initial=True)
        self._refresh_emission_b_fit_options(initial=True)

    def _row(self, parent: tk.Widget, label: str, var: tk.Variable, row: int) -> tuple[tk.Label, tk.Entry]:
        label_widget = tk.Label(parent, text=label, anchor="w")
        label_widget.grid(row=row, column=0, sticky="w", padx=6, pady=3)
        entry_widget = tk.Entry(parent, textvariable=var, width=22)
        entry_widget.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        return label_widget, entry_widget

    def _on_scale_preset_changed(self, *_args) -> None:
        value = _SCALE_PRESET_VALUES.get(self.v_scale_preset.get())
        if value is not None:
            self.v_pixel_size_mm.set(float(value))

    def _make_scrollable_tab(self, parent: tk.Widget) -> tuple[tk.Frame, tk.Frame, tk.Canvas]:
        outer = tk.Frame(parent)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas)

        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure("content", width=event.width),
        )
        canvas.create_window((0, 0), window=content, anchor="nw", tags=("content",))
        canvas.configure(yscrollcommand=scrollbar.set)

        def _activate_mousewheel(_event) -> None:
            self._active_mousewheel_canvas = canvas

        def _deactivate_mousewheel(_event) -> None:
            if self._active_mousewheel_canvas is canvas:
                self._active_mousewheel_canvas = None

        canvas.bind("<Enter>", _activate_mousewheel)
        canvas.bind("<Leave>", _deactivate_mousewheel)
        content.bind("<Enter>", _activate_mousewheel)
        content.bind("<Leave>", _deactivate_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return outer, content, canvas

    def _on_mousewheel(self, event) -> None:
        canvas = self._active_mousewheel_canvas
        if canvas is None:
            return
        if getattr(event, "delta", 0):
            units = -int(event.delta / 120)
            if units != 0:
                canvas.yview_scroll(units, "units")
        elif getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")

    def _build_ui(self) -> None:
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")

        header = ttk.Frame(self.root, padding=(20, 14))
        header.pack(fill='x')
        ttk.Label(header, text='Weld Vision', style='Heading.TLabel').pack(side='left')
        ttk.Label(header, text='TOPO generation & defect inspection  /  Research prototype',
                  style='Muted.TLabel').pack(side='left', padx=20)
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="Config Dir").pack(side="left")
        tk.Entry(top, textvariable=self.v_config_dir, width=70).pack(side="left", padx=6)
        tk.Button(top, text="Browse", command=self._choose_config_dir).pack(side="left")
        tk.Button(top, text="Reload", command=self._reload_bundle).pack(side="left", padx=6)
        tk.Button(top, text="Save Defaults", command=self._save_defaults).pack(side="left", padx=6)

        notebook = ttk.Notebook(self.root)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True, padx=8, pady=4)

        beam_tab = tk.Frame(notebook)
        det_tab = tk.Frame(notebook)
        noise_tab = tk.Frame(notebook)
        run_tab_outer, run_tab, _run_canvas = self._make_scrollable_tab(notebook)
        notebook.add(beam_tab, text="Beam/Material")
        notebook.add(det_tab, text="Detector")
        notebook.add(noise_tab, text="Noise")
        notebook.add(run_tab_outer, text="Run")
        notebook.tab(run_tab_outer, text='Generate images')
        notebook.insert(0, run_tab_outer)
        from inspection.gui import InspectionPanel
        self.inspection_panel = InspectionPanel(notebook)
        notebook.add(self.inspection_panel, text='Inspect images')
        notebook.insert(1, self.inspection_panel)
        self._build_overview(notebook, run_tab_outer)
        self._row(beam_tab, 'Constant B (explicit assumption)', self.v_constant_b, 9)

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
        self._row(det_tab, "Detector inner radius (mm)", self.v_inner_r, 1)
        self._row(det_tab, "Detector outer radius (mm)", self.v_outer_r, 2)

        tk.Label(noise_tab, text="Raster acquisition v2 | assumed, not measured\n"
                 "Preview noise enables counting noise, readout and slow electronic drift.\n"
                 "All detector channels share one scan clock. Specimen motion defaults to zero.",
                 justify="left", wraplength=420).grid(row=0, column=0, columnspan=2, padx=6, pady=8)
        self._row(noise_tab, "Legacy temperature (K)*", self.v_temp, 1)
        self._row(noise_tab, "Legacy bandwidth (Hz)*", self.v_bw, 2)
        self._row(noise_tab, "Legacy contamination rate*", self.v_contam, 3)
        tk.Label(noise_tab, text="* Saved for legacy reproduction only; unused by v2.\n"
                 "V2 profile: bse_renderer/noise/raster_acquisition.py",
                 justify="left", wraplength=420).grid(row=4, column=0, columnspan=2, padx=6, pady=8)

        self._row(run_tab, "Num images", self.v_num_images, 0)
        self._row(run_tab, "Image size", self.v_image_size, 1)
        tk.Label(run_tab, text="Scale preset", anchor="w").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(
            run_tab,
            textvariable=self.v_scale_preset,
            state="readonly",
            values=list(_SCALE_PRESET_VALUES.keys()),
            width=28,
        ).grid(row=2, column=1, sticky="ew", padx=6, pady=3)
        self._row(run_tab, "Pixel size (mm/px)", self.v_pixel_size_mm, 3)
        tk.Label(run_tab, text="Macro weld layout", anchor="w").grid(
            row=4, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Combobox(
            run_tab,
            textvariable=self.v_weld_layout_label,
            state="readonly",
            values=list(_WELD_LAYOUT_LABELS.keys()),
            width=28,
        ).grid(row=4, column=1, sticky="ew", padx=6, pady=3)
        self._run_field_widgets["seam_rotation"] = self._row(run_tab, "Seam rotation (deg)", self.v_seam_rotation, 5)
        self._run_field_widgets["seam_offset"] = self._row(run_tab, "Seam offset (mm)", self.v_seam_offset, 6)
        self._run_field_widgets["circle_radius"] = self._row(run_tab, "Circle radius (mm), 40 = 80 mm diameter", self.v_circle_radius, 7)
        self._run_field_widgets["circle_position"] = self._row(run_tab, "Circle sample angle (deg)", self.v_circle_position, 8)
        self._row(run_tab, "Seed", self.v_seed, 9)
        self._run_field_widgets["voxel_depth"] = self._row(run_tab, "Voxel depth (mm)", self.v_voxel_depth, 10)
        self._run_field_widgets["slice_offset"] = self._row(run_tab, "Slice offset (mm)", self.v_slice_offset, 11)
        self._run_field_widgets["slice_tilt"] = self._row(run_tab, "Slice tilt (deg)", self.v_slice_tilt, 12)
        self._run_field_widgets["slice_rotation"] = self._row(run_tab, "Slice rotation (deg)", self.v_slice_rotation, 13)
        tk.Label(
            run_tab,
            textvariable=self.v_control_hint,
            anchor="w",
            justify="left",
            fg="#555555",
            wraplength=360,
        ).grid(row=14, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))
        tk.Checkbutton(
            run_tab,
            text="Use fixed slice in batch generation",
            variable=self.v_fixed_slice,
        ).grid(row=15, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        tk.Checkbutton(
            run_tab,
            text="Randomize defect location",
            variable=self.v_random_defect_location,
        ).grid(row=16, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        self._row(run_tab, "Max defects/sample", self.v_max_defects, 17)
        self._row(run_tab, "Output dir", self.v_output_dir, 18)
        tk.Checkbutton(
            run_tab,
            text="Apply noise in preview",
            variable=self.v_preview_noise,
        ).grid(row=19, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        tk.Checkbutton(
            run_tab,
            text="Disable noise in generated batch output",
            variable=self.v_disable_batch_noise,
        ).grid(row=20, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        tk.Label(run_tab, text="Weld geometry mode", anchor="w").grid(
            row=21, column=0, sticky="w", padx=6, pady=3
        )
        self.weld_geometry_combo = ttk.Combobox(
            run_tab,
            textvariable=self.v_weld_geometry_label,
            state="readonly",
            values=[_DEFAULT_WELD_GEOMETRY_LABEL],
            width=28,
        )
        self.weld_geometry_combo.grid(row=21, column=1, sticky="ew", padx=6, pady=3)
        tk.Label(run_tab, text="3D preview mode", anchor="w").grid(
            row=22, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Combobox(
            run_tab,
            textvariable=self.v_geometry_preview_label,
            state="readonly",
            values=list(_PREVIEW_MODE_LABELS.keys()),
            width=28,
        ).grid(row=22, column=1, sticky="ew", padx=6, pady=3)
        tk.Label(run_tab, text="Render model", anchor="w").grid(
            row=23, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Combobox(
            run_tab,
            textvariable=self.v_render_model_label,
            state="readonly",
            values=list(_RENDER_MODEL_LABELS.keys()),
            width=28,
        ).grid(row=23, column=1, sticky="ew", padx=6, pady=3)
        tk.Label(run_tab, text="CASINO fit source", anchor="w").grid(
            row=24, column=0, sticky="w", padx=6, pady=3
        )
        self.calibration_combo = ttk.Combobox(
            run_tab,
            textvariable=self.v_calibration_label,
            state="readonly",
            values=list(self._calibration_options.keys()),
            width=28,
        )
        self.calibration_combo.grid(row=24, column=1, sticky="ew", padx=6, pady=3)
        calibration_btns = tk.Frame(run_tab)
        calibration_btns.grid(row=25, column=1, sticky="w", padx=3, pady=(0, 4))
        tk.Button(
            calibration_btns,
            text="Refresh calibrations",
            command=self._refresh_calibration_options,
        ).pack(side="left", padx=3)
        tk.Button(
            calibration_btns,
            text="Browse calibration...",
            command=self._choose_calibration_file,
        ).pack(side="left", padx=3)
        tk.Label(run_tab, text="Emission B mode", anchor="w").grid(
            row=26, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Combobox(
            run_tab,
            textvariable=self.v_emission_b_label,
            state="readonly",
            values=list(_EMISSION_B_LABELS.keys()),
            width=28,
        ).grid(row=26, column=1, sticky="ew", padx=6, pady=3)
        tk.Label(run_tab, text="Emission B fit source", anchor="w").grid(
            row=27, column=0, sticky="w", padx=6, pady=3
        )
        self.emission_b_combo = ttk.Combobox(
            run_tab,
            textvariable=self.v_emission_b_fit_label,
            state="readonly",
            values=list(self._emission_b_options.keys()),
            width=28,
        )
        self.emission_b_combo.grid(row=27, column=1, sticky="ew", padx=6, pady=3)
        emission_b_btns = tk.Frame(run_tab)
        emission_b_btns.grid(row=28, column=1, sticky="w", padx=3, pady=(0, 4))
        tk.Button(
            emission_b_btns,
            text="Refresh B fits",
            command=self._refresh_emission_b_fit_options,
        ).pack(side="left", padx=3)
        tk.Button(
            emission_b_btns,
            text="Browse B fit...",
            command=self._choose_emission_b_fit_file,
        ).pack(side="left", padx=3)
        tk.Checkbutton(
            run_tab,
            text="Show additional geometry and defect options",
            variable=self.v_show_legacy_options,
        ).grid(row=29, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        defect_frame = tk.LabelFrame(run_tab, text="Manufactured Defects")
        defect_frame.grid(row=30, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        run_tab.grid_columnconfigure(1, weight=1)
        defect_frame.grid_columnconfigure(0, weight=1)
        tk.Label(
            defect_frame,
            text="Select surface defects for this run. Pore size is not the process keyhole width or depth.",
        ).grid(
            row=0, column=0, columnspan=5, sticky="w", padx=4, pady=(3, 5)
        )
        tk.Label(defect_frame, text="Defect").grid(row=1, column=0, sticky="w", padx=4)
        tk.Label(defect_frame, text="Count").grid(row=1, column=1, sticky="w", padx=4)
        tk.Label(defect_frame, text="Size (mm)").grid(row=1, column=2, sticky="w", padx=4)
        tk.Label(defect_frame, text="Depth (mm)").grid(row=1, column=3, sticky="w", padx=4)
        row = 2
        for item in _DEFECT_UI_SCHEMA:
            ctrl = self._defect_controls[item["type"]]
            widgets: list[tk.Widget] = []
            widgets.append(
                tk.Checkbutton(
                defect_frame,
                text=item["label"],
                variable=ctrl["enabled"],
                )
            )
            widgets[-1].grid(row=row, column=0, sticky="w", padx=4, pady=1)
            widgets.append(tk.Entry(defect_frame, width=6, textvariable=ctrl["count"]))
            widgets[-1].grid(row=row, column=1, sticky="w", padx=4, pady=1)
            widgets.append(tk.Entry(defect_frame, width=8, textvariable=ctrl["size"]))
            widgets[-1].grid(row=row, column=2, sticky="w", padx=4, pady=1)
            if item["depth_param"] is None:
                widgets.append(tk.Label(defect_frame, text="-"))
                widgets[-1].grid(row=row, column=3, sticky="w", padx=4, pady=1)
            else:
                widgets.append(tk.Entry(defect_frame, width=8, textvariable=ctrl["depth"]))
                widgets[-1].grid(row=row, column=3, sticky="w", padx=4, pady=1)
            self._defect_row_widgets[item["type"]] = widgets
            row += 1

        defect_btns = tk.Frame(defect_frame)
        defect_btns.grid(row=row, column=0, columnspan=5, sticky="w", padx=2, pady=(6, 4))
        tk.Button(defect_btns, text="Select All", command=lambda: self._set_all_defects(True)).pack(side="left", padx=3)
        tk.Button(defect_btns, text="Clear All", command=lambda: self._set_all_defects(False)).pack(side="left", padx=3)
        tk.Button(defect_btns, text="Surface Campaign", command=self._set_surface_campaign_defects).pack(side="left", padx=3)
        self._legacy_defect_button = tk.Button(defect_btns, text="Legacy Study Set", command=self._set_legacy_defects)
        self._legacy_defect_button.pack(side="left", padx=3)
        tk.Button(defect_btns, text="Obvious Sizes", command=self._set_obvious_defects).pack(side="left", padx=3)

        actions = tk.Frame(run_tab)
        actions.grid(row=31, column=0, columnspan=2, sticky='ew', padx=8, pady=12)
        tk.Button(actions, text="Preview 3D Surface", command=self._preview_mesh).pack(side="left", padx=4)
        tk.Button(actions, text="Preview Render", command=self._preview_render).pack(side="left", padx=4)
        tk.Button(actions, text="Compare Clean vs Defected", command=self._preview_defect_impact).pack(side="left", padx=4)
        tk.Button(
            actions,
            text="Compare B models",
            command=self._preview_emission_b_comparison,
        ).pack(side="left", padx=4)
        tk.Button(
            actions,
            text="Compare render models",
            command=self._preview_calibration_comparison,
        ).pack(side="left", padx=4)
        self.generate_button = tk.Button(actions, text="Generate batch", command=self._generate_batch,
                                        bg='#087f8c', fg='white')
        self.generate_button.pack(side="right", padx=4)

        tk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=8, pady=4)
        self._refresh_legacy_option_visibility()
        self._refresh_parameter_control_states()

    def _build_overview(self, notebook, run_tab):
        page = ttk.Frame(notebook, padding=28)
        notebook.add(page, text='Overview')
        notebook.insert(0, page)
        notebook.select(page)
        ttk.Label(page, text='From a surface model to defect candidates', style='Heading.TLabel').pack(anchor='w')
        ttk.Label(page, text='A local workbench for controlled simulation, computer vision and evidence review.',
                  style='Muted.TLabel').pack(anchor='w', pady=(6, 24))
        for title, body, action, label in [
            ('01   Generate TOPO images', 'Configure the beam, detector and surface defects. Preview geometry and render channels, or export a labelled batch.\nConstant B = 0.920695 is an engineering approximation; generated images are synthetic.',
             lambda: notebook.select(run_tab), 'Open generator'),
            ('02   Inspect images', 'Choose an image or folder. Check files, run the frozen CNN and review overlays.\nCompare median and threshold baselines when their input requirements are met.',
             lambda: notebook.select(self.inspection_panel), 'Open CV inspection'),
        ]:
            card = ttk.LabelFrame(page, text=title, padding=18)
            card.pack(fill='x', pady=(0, 18))
            ttk.Label(card, text=body, wraplength=940, justify='left').pack(anchor='w', pady=(0, 12))
            ttk.Button(card, text=label, style='Accent.TButton', command=action).pack(anchor='w')
        scope = ttk.LabelFrame(page, text='Specimen context and model limits', padding=18)
        scope.pack(fill='x')
        ttk.Label(scope, text='Reported weld diameter: 80 mm  •  Keyhole depth: 8–10 mm  •  Narrow keyhole width: 8–10 mm\n\n'
                  'The current renderer models a small procedural weld region. Its width/depth parameters do not yet represent\n'
                  'the reported process keyhole. Circular layouts use a 40 mm radius for an 80 mm diameter; fitting the full\n'
                  'circle changes the effective pixel size. The frozen CNN expects 50 µm/pixel.\n\n'
                  'Real images: initial 3–5 image case study. Expert labels and acquisition matching remain necessary.',
                  wraplength=1020, justify='left').pack(anchor='w')

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
            self._refresh_calibration_options()
            self._refresh_legacy_option_visibility()
            self._refresh_parameter_control_states()
            self.status_var.set(f"Loaded config from {self.config_dir}")
        except Exception as exc:
            messagebox.showerror("Config Error", str(exc))

    def _discover_calibration_options(self) -> dict[str, str | None]:
        options: dict[str, str | None] = {"No calibration selected": None}
        output_root = (self.config_dir.parent / "data" / "output").resolve()
        if not output_root.exists():
            return options
        for path in sorted(output_root.rglob("renderer_calibration.json")):
            label = f"CASINO fit: {path.parent.relative_to(output_root)}"
            options[label] = str(path)
        return options

    def _discover_emission_b_fit_options(self) -> dict[str, str | None]:
        options: dict[str, str | None] = {"No emission B fit selected": None}
        output_root = (self.config_dir.parent / "data" / "output").resolve()
        if not output_root.exists():
            return options
        for path in sorted(output_root.rglob("b_fit_gui_summary.json")):
            label = f"Emission B fit: {path.parent.relative_to(output_root)}"
            options[label] = str(path)
        return options

    def _refresh_calibration_options(self, initial: bool = False) -> None:
        current_label = self.v_calibration_label.get()
        self._calibration_options = self._discover_calibration_options()
        if current_label not in self._calibration_options:
            current_label = "No calibration selected"
        self.v_calibration_label.set(current_label)
        if self.calibration_combo is not None:
            self.calibration_combo.configure(values=list(self._calibration_options.keys()))
        if not initial:
            self.status_var.set("Calibration list refreshed")

    def _refresh_emission_b_fit_options(self, initial: bool = False) -> None:
        current_label = self.v_emission_b_fit_label.get()
        self._emission_b_options = self._discover_emission_b_fit_options()
        if current_label not in self._emission_b_options:
            current_label = "No emission B fit selected"
        self.v_emission_b_fit_label.set(current_label)
        if self.emission_b_combo is not None:
            self.emission_b_combo.configure(values=list(self._emission_b_options.keys()))
        if not initial:
            self.status_var.set("Emission B fit list refreshed")

    def _choose_calibration_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose renderer calibration JSON",
            initialdir=str((self.config_dir.parent / "data" / "output").resolve()),
            filetypes=[("Calibration JSON", "*.json"), ("All files", "*.*")],
        )
        if not chosen:
            return
        path = str(Path(chosen).resolve())
        label = f"Custom: {Path(path).parent.name}"
        self._calibration_options[label] = path
        if self.calibration_combo is not None:
            self.calibration_combo.configure(values=list(self._calibration_options.keys()))
        self.v_calibration_label.set(label)
        self.status_var.set(f"Selected calibration: {label}")

    def _choose_emission_b_fit_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose emission B fit JSON",
            initialdir=str((self.config_dir.parent / "data" / "output").resolve()),
            filetypes=[("Emission B fit JSON", "*.json"), ("All files", "*.*")],
        )
        if not chosen:
            return
        path = str(Path(chosen).resolve())
        label = f"Custom B fit: {Path(path).parent.name}"
        self._emission_b_options[label] = path
        if self.emission_b_combo is not None:
            self.emission_b_combo.configure(values=list(self._emission_b_options.keys()))
        self.v_emission_b_fit_label.set(label)
        self.status_var.set(f"Selected emission B fit: {label}")

    def _selected_calibration_path(self) -> str | None:
        return self._calibration_options.get(self.v_calibration_label.get())

    def _selected_emission_b_fit_path(self) -> str | None:
        return self._emission_b_options.get(self.v_emission_b_fit_label.get())

    @staticmethod
    def _is_legacy_defect(defect_type: str) -> bool:
        return defect_type in _LEGACY_DEFECT_TYPES

    def _visible_weld_geometry_labels(self) -> list[str]:
        labels = [_DEFAULT_WELD_GEOMETRY_LABEL]
        if self.v_show_legacy_options.get():
            labels.append(_LEGACY_WELD_GEOMETRY_LABEL)
        return labels

    def _set_entry_enabled(self, key: str, enabled: bool) -> None:
        widgets = self._run_field_widgets.get(key)
        if widgets is None:
            return
        state = "normal" if enabled else "disabled"
        _, entry_widget = widgets
        entry_widget.configure(state=state)

    def _refresh_legacy_option_visibility(self, *_args) -> None:
        visible_geometry_labels = self._visible_weld_geometry_labels()
        if self.weld_geometry_combo is not None:
            self.weld_geometry_combo.configure(values=visible_geometry_labels)
        if self.v_weld_geometry_label.get() not in visible_geometry_labels:
            self.v_weld_geometry_label.set(_DEFAULT_WELD_GEOMETRY_LABEL)

        show_legacy = bool(self.v_show_legacy_options.get())
        for defect_type, widgets in self._defect_row_widgets.items():
            is_legacy = self._is_legacy_defect(defect_type)
            if is_legacy and not show_legacy:
                self._defect_controls[defect_type]["enabled"].set(False)
                for widget in widgets:
                    widget.grid_remove()
            else:
                for widget in widgets:
                    widget.grid()
        if self._legacy_defect_button is not None:
            if show_legacy:
                self._legacy_defect_button.pack(side="left", padx=3)
            else:
                self._legacy_defect_button.pack_forget()

    def _refresh_parameter_control_states(self, *_args) -> None:
        is_circular = (
            _WELD_LAYOUT_LABELS.get(self.v_weld_layout_label.get(), WELD_LAYOUT_LOCAL_PATCH)
            == WELD_LAYOUT_CIRCULAR_WELD
        )
        self._set_entry_enabled("seam_rotation", not is_circular)
        self._set_entry_enabled("circle_radius", is_circular)
        self._set_entry_enabled("circle_position", is_circular)
        tilt_deg = float(self.v_slice_tilt.get())
        self._set_entry_enabled("slice_rotation", abs(tilt_deg) > 1e-6)

        hints: list[str] = []
        if is_circular:
            hints.append("Circular weld uses seam offset, circle radius, and circle sample angle. Seam rotation is ignored.")
        else:
            hints.append("Straight-seam and local-patch layouts use seam rotation and seam offset. Circle controls are ignored.")
        if self._selected_preview_mode() == "top_surface":
            hints.append("Top weld surface preview ignores slice offset, slice tilt, and slice rotation. Those controls still affect slice previews and batch generation.")
        elif abs(tilt_deg) <= 1e-6:
            hints.append("Slice rotation has no effect until slice tilt is non-zero.")
        self.v_control_hint.set(" ".join(hints))

    def _selected_render_model_mode(self) -> str:
        return _RENDER_MODEL_LABELS.get(
            self.v_render_model_label.get(),
            RENDER_MODEL_FORMULA_ONLY,
        )

    def _selected_emission_b_mode(self) -> str:
        return _EMISSION_B_LABELS.get(
            self.v_emission_b_label.get(),
            EMISSION_B_MODE_CONSTANT,
        )

    def _selected_render_settings(self) -> tuple[str | None, str, str, str | None]:
        calibration_path = self._selected_calibration_path()
        render_model_mode = self._selected_render_model_mode()
        if render_model_mode != RENDER_MODEL_FORMULA_ONLY and calibration_path is None:
            raise ValueError(
                "Select a renderer calibration JSON before using Formula + CASINO or Formula + CASINO + geometry profiles."
            )
        emission_b_mode = self._selected_emission_b_mode()
        emission_b_path = self._selected_emission_b_fit_path()
        if emission_b_mode in {
            EMISSION_B_MODE_POOLED_CONSTANT_FIT,
            EMISSION_B_MODE_SIN_FIT,
            EMISSION_B_MODE_QUADRATIC_FIT,
        } and emission_b_path is None:
            raise ValueError(
                "Select an emission B fit JSON before using a fitted B mode."
            )
        return calibration_path, render_model_mode, emission_b_mode, emission_b_path

    def _set_all_defects(self, enabled: bool) -> None:
        show_legacy = bool(self.v_show_legacy_options.get())
        for defect_type, ctrl in self._defect_controls.items():
            if self._is_legacy_defect(defect_type) and not show_legacy:
                ctrl["enabled"].set(False)
                continue
            ctrl["enabled"].set(enabled)

    def _set_obvious_defects(self) -> None:
        show_legacy = bool(self.v_show_legacy_options.get())
        for item in _DEFECT_UI_SCHEMA:
            if self._is_legacy_defect(item["type"]) and not show_legacy:
                self._defect_controls[item["type"]]["enabled"].set(False)
                continue
            ctrl = self._defect_controls[item["type"]]
            ctrl["enabled"].set(True)
            ctrl["count"].set(item["count_default"])
            ctrl["size"].set(float(item["size_default"]))
            if item["depth_default"] is not None:
                ctrl["depth"].set(float(item["depth_default"]))

    def _set_surface_campaign_defects(self) -> None:
        active = _SURFACE_CAMPAIGN_DEFECT_TYPES
        for item in _DEFECT_UI_SCHEMA:
            ctrl = self._defect_controls[item["type"]]
            enabled = item["type"] in active
            ctrl["enabled"].set(enabled)
            ctrl["count"].set(item["count_default"])
            ctrl["size"].set(float(item["size_default"]))
            if item["depth_default"] is not None:
                ctrl["depth"].set(float(item["depth_default"]))

    def _set_legacy_defects(self) -> None:
        active = _LEGACY_DEFECT_TYPES
        for item in _DEFECT_UI_SCHEMA:
            ctrl = self._defect_controls[item["type"]]
            enabled = item["type"] in active
            ctrl["enabled"].set(enabled)
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

    def _snap_defect_controls_to_clamped_values(self) -> list[str]:
        updates: list[str] = []
        for item in _DEFECT_UI_SCHEMA:
            defect_type = item["type"]
            ctrl = self._defect_controls[defect_type]

            raw_count = int(ctrl["count"].get())
            snapped_count = max(0, raw_count)
            if snapped_count != raw_count:
                ctrl["count"].set(snapped_count)
                updates.append(f"{item['label']}: count {raw_count} -> {snapped_count}")

            raw_size = float(ctrl["size"].get())
            snapped_size = self._clamp_defect_param(
                defect_type,
                item["size_param"],
                raw_size,
            )
            if not np.isclose(snapped_size, raw_size):
                ctrl["size"].set(snapped_size)
                updates.append(
                    f"{item['label']}: {item['size_param']} {raw_size:g} -> {snapped_size:g}"
                )

            if item["depth_param"] is not None:
                raw_depth = float(ctrl["depth"].get())
                snapped_depth = self._clamp_defect_param(
                    defect_type,
                    item["depth_param"],
                    raw_depth,
                )
                if not np.isclose(snapped_depth, raw_depth):
                    ctrl["depth"].set(snapped_depth)
                    updates.append(
                        f"{item['label']}: {item['depth_param']} {raw_depth:g} -> {snapped_depth:g}"
                    )
        return updates

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
        size, spacing, dz = int(self.v_image_size.get()), float(self.v_pixel_size_mm.get()), float(self.v_voxel_depth.get())
        if not 16 <= size <= 512 or not np.isfinite([spacing, dz]).all() or min(spacing, dz) <= 0:
            raise ValueError('Use 16–512 pixels and positive finite pixel/voxel spacing.')
        from .geometry.weld_profile import resolve_layout_sampling_grid
        _, _, grid = resolve_layout_sampling_grid(width_px=size, height_px=size, pixel_size_mm=spacing,
                                                  layout=self._selected_weld_layout())
        extent = (size-1)*grid['layout_pixel_size_mm']
        layers = max(16, int(np.ceil(max(.55*extent, 3.2*max(2.4, grid['layout_pixel_size_mm']), 2.4)/dz))+1)
        estimated = size*size*layers*32
        if estimated > 512*1024**2:
            raise ValueError(f'Estimated voxel storage exceeds 512 MiB ({estimated/1024**2:.0f} MiB). '
                             'Reduce image size or increase voxel spacing. Full-circle views change the effective pixel size.')
        for variable, name in [(self.v_voltage, 'accelerating_voltage_kv'), (self.v_current, 'beam_current_ma'),
                               (self.v_focus, 'focusing_current_norm'), (self.v_speed, 'travel_speed_mm_min'),
                               (self.v_dwell, 'dwell_time_us'), (self.v_focal, 'focal_spot_mm')]:
            value = float(variable.get())
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite.')
            getattr(self.bundle.beam, name).validate(value, name)
        for item in _DEFECT_UI_SCHEMA:
            control = self._defect_controls[item['type']]
            if not control['enabled'].get():
                continue
            if control['count'].get() < 1:
                raise ValueError('Selected defect count must be positive.')
            for key, param in [('size', item['size_param']), ('depth', item['depth_param'])]:
                if param is not None:
                    value = float(control[key].get())
                    lo, hi = self.bundle.defect.parameter_ranges[item['type']][param]
                    if not np.isfinite(value) or not lo <= value <= hi:
                        raise ValueError(f'{item["label"]}: {param} must be in [{lo}, {hi}] mm. '
                                         'Process keyhole dimensions are not pore defect dimensions.')
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
        if self._selected_emission_b_mode() == EMISSION_B_MODE_CONSTANT:
            value = float(self.v_constant_b.get())
            if not np.isfinite(value) or abs(value) > 1:
                raise ValueError('Constant B must be finite and in [-1, 1] for global positivity.')
            material = replace(material, b=value)

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
            inner_radius_mm=self.v_inner_r.get(),
            outer_radius_mm=self.v_outer_r.get(),
            quadrants_deg=dict(self.bundle.detector.quadrants_deg),
            sensitivity={q: 1.0 for q in ("A", "B", "C", "D")},
            psi0_rad=float(np.arctan2(self.v_h.get(), self.v_r.get())),
        )
        noise = NoiseRuntime(
            poisson_enabled=bool(self.v_preview_noise.get()),
            thermal_enabled=bool(self.v_preview_noise.get()),
            contamination_enabled=bool(self.v_preview_noise.get()),
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
        return beam, material, detector, noise, slice_runtime, defect_plan

    def _selected_weld_geometry_mode(self) -> str:
        return _WELD_GEOMETRY_LABELS.get(
            self.v_weld_geometry_label.get(),
            WELD_GEOMETRY_CLOSED,
        )

    def _selected_weld_layout(self) -> WeldLayoutRuntime:
        values = [float(v.get()) for v in (self.v_seam_rotation, self.v_seam_offset, self.v_circle_radius, self.v_circle_position)]
        if not np.isfinite(values).all() or values[2] <= 0:
            raise ValueError('Layout parameters must be finite and circle radius must be positive.')
        return WeldLayoutRuntime(
            mode=_WELD_LAYOUT_LABELS.get(
                self.v_weld_layout_label.get(),
                WELD_LAYOUT_LOCAL_PATCH,
            ),
            seam_rotation_deg=float(self.v_seam_rotation.get()),
            seam_offset_mm=float(self.v_seam_offset.get()),
            circle_radius_mm=float(self.v_circle_radius.get()),
            circle_position_deg=float(self.v_circle_position.get()),
        )

    def _selected_preview_mode(self) -> str:
        return _PREVIEW_MODE_LABELS.get(
            self.v_geometry_preview_label.get(),
            "fusion_body_volume",
        )

    def _selected_render_view_mode(self) -> str:
        return "top_surface" if self._selected_preview_mode() == "top_surface" else "slice"

    def _build_preview_geometry(self) -> tuple:
        beam, material, _, _, slice_runtime, defects = self._manual_sample()
        pixel_size_mm = float(self.v_pixel_size_mm.get())
        weld_layout = self._selected_weld_layout()
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
            pixel_size_mm=pixel_size_mm,
            voxel_depth_mm=slice_runtime.voxel_depth_mm,
            crown_height_mm=reinforcement_mm,
            seam_depth_mm=seam_depth_mm,
            bead_width_mm=width_mm,
            haz_width_mm=haz_width_mm,
            beam_focus_quality=float(np.clip(beam.focusing_current_norm, 0.0, 1.0)),
            asymmetry_offset_mm=asymmetry_mm,
            weld_geometry_mode=self._selected_weld_geometry_mode(),
            weld_layout=weld_layout,
            seed=self.v_seed.get(),
        )
        apply_defect_bodies(
            volume,
            defects,
            np.random.default_rng(self.v_seed.get()),
            target_slice_x_mm=slice_runtime.slice_offset_mm,
            randomize_position=bool(self.v_random_defect_location.get()),
        )
        plane = build_section_plane(slice_runtime)
        section, intersections = slice_volume(
            volume,
            plane,
            image_size=self.v_image_size.get(),
            pixel_size_mm=float(volume.metadata.get("pixel_size_mm", pixel_size_mm)),
        )
        return volume, section, intersections

    @staticmethod
    def _top_surface_preview(volume) -> tuple[np.ndarray, np.ndarray]:
        surface_idx = int(
            volume.metadata.get(
                "surface_plane_index",
                volume.relief_mm.shape[0] - 1,
            )
        )
        surface_idx = int(np.clip(surface_idx, 0, volume.relief_mm.shape[0] - 1))
        z0 = max(0, surface_idx - 1)
        z1 = min(volume.relief_mm.shape[0], surface_idx + 2)
        surface_map = volume.relief_mm[surface_idx].astype(np.float32)
        surface_defects = np.any(volume.defect_mask[z0:z1], axis=0)
        return surface_map, surface_defects.astype(bool)

    @staticmethod
    def _section_preview_coords(section) -> tuple[np.ndarray, np.ndarray]:
        half_u, half_v = section.sampling_extent_mm
        h, w = section.relief_mm_map.shape
        x_coords_mm = np.linspace(-float(half_u), float(half_u), w, dtype=np.float32)
        y_coords_mm = np.linspace(-float(half_v), float(half_v), h, dtype=np.float32)
        return x_coords_mm, y_coords_mm

    def _preview_mesh(self) -> None:
        try:
            volume, section, _ = self._build_preview_geometry()
            if self._selected_preview_mode() == "fusion_body_volume":
                preview_volume_geometry(
                    volume.fusion_mask,
                    volume.defect_mask,
                    spacing_mm=volume.spacing_mm,
                    title=f"Fusion Body Preview ({self.v_weld_geometry_label.get()})",
                )
                self.status_var.set("Fusion body preview complete")
            elif self._selected_preview_mode() == "top_surface":
                surface_map, surface_defects = self._top_surface_preview(volume)
                preview_surface_mesh(
                    surface_map,
                    surface_defects,
                    x_coords_mm=volume.x_mm,
                    y_coords_mm=volume.y_mm,
                    title=f"Top Weld Surface Preview ({self.v_weld_layout_label.get()})",
                )
                self.status_var.set("Top weld surface preview complete")
            else:
                x_coords_mm, y_coords_mm = self._section_preview_coords(section)
                preview_surface_mesh(
                    section.relief_mm_map,
                    section.defect_mask.astype(bool),
                    x_coords_mm=x_coords_mm,
                    y_coords_mm=y_coords_mm,
                    title="Cross-section Relief Preview",
                )
                self.status_var.set("Cross-section relief preview complete")
        except Exception as exc:
            messagebox.showerror("Preview Error", str(exc))

    def _preview_render(self) -> None:
        try:
            calibration_path, render_model_mode, emission_b_mode, emission_b_path = self._selected_render_settings()
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
                randomize_defect_location=bool(self.v_random_defect_location.get()),
            )
            result = ScanEngine(
                acquisition_model="raster_v2",
                image_size=self.v_image_size.get(),
                pixel_size_mm=float(self.v_pixel_size_mm.get()),
                calibration_path=calibration_path,
                render_model_mode=render_model_mode,
                emission_b_mode=emission_b_mode,
                emission_b_path=emission_b_path,
                weld_geometry_mode=self._selected_weld_geometry_mode(),
                weld_layout=self._selected_weld_layout(),
                render_view_mode=self._selected_render_view_mode(),
            ).run(sample)
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
            self.status_var.set(
                f"Render preview complete ({self.v_render_model_label.get()}, {mode})"
            )
        except Exception as exc:
            messagebox.showerror("Preview Error", str(exc))

    def _preview_defect_impact(self) -> None:
        try:
            calibration_path, render_model_mode, emission_b_mode, emission_b_path = self._selected_render_settings()
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
                randomize_defect_location=bool(self.v_random_defect_location.get()),
            )
            sample_clean = replace(sample_defect, defects=[])
            engine = ScanEngine(
                acquisition_model="raster_v2",
                image_size=self.v_image_size.get(),
                pixel_size_mm=float(self.v_pixel_size_mm.get()),
                calibration_path=calibration_path,
                render_model_mode=render_model_mode,
                emission_b_mode=emission_b_mode,
                emission_b_path=emission_b_path,
                weld_geometry_mode=self._selected_weld_geometry_mode(),
                weld_layout=self._selected_weld_layout(),
                render_view_mode=self._selected_render_view_mode(),
            )
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

    def _preview_calibration_comparison(self) -> None:
        try:
            calibration_path = self._selected_calibration_path()
            if calibration_path is None:
                raise ValueError(
                    "Select a renderer calibration JSON to compare formula-only, Formula + CASINO, and Formula + CASINO + geometry profiles."
                )
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
                randomize_defect_location=bool(self.v_random_defect_location.get()),
            )
            sample_clean = replace(sample, defects=[])
            engine_kwargs = {
                "acquisition_model": "raster_v2",
                "image_size": self.v_image_size.get(),
                "pixel_size_mm": float(self.v_pixel_size_mm.get()),
                "calibration_path": calibration_path,
                "emission_b_mode": self._selected_emission_b_mode(),
                "emission_b_path": self._selected_emission_b_fit_path(),
                "weld_geometry_mode": self._selected_weld_geometry_mode(),
                "weld_layout": self._selected_weld_layout(),
                "render_view_mode": self._selected_render_view_mode(),
            }
            formula_only = ScanEngine(
                render_model_mode=RENDER_MODEL_FORMULA_ONLY,
                **engine_kwargs,
            ).run(sample)
            formula_only_clean = ScanEngine(
                render_model_mode=RENDER_MODEL_FORMULA_ONLY,
                **engine_kwargs,
            ).run(sample_clean)
            formula_casino = ScanEngine(
                render_model_mode=RENDER_MODEL_FORMULA_CASINO,
                **engine_kwargs,
            ).run(sample)
            formula_casino_clean = ScanEngine(
                render_model_mode=RENDER_MODEL_FORMULA_CASINO,
                **engine_kwargs,
            ).run(sample_clean)
            geometry = ScanEngine(
                render_model_mode=RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
                **engine_kwargs,
            ).run(sample)
            geometry_clean = ScanEngine(
                render_model_mode=RENDER_MODEL_FORMULA_CASINO_GEOMETRY,
                **engine_kwargs,
            ).run(sample_clean)
            diagnostic_lines: dict[str, list[str]] = {}
            if not geometry.metadata.get("geometry_profiles_active"):
                diagnostic_lines[RENDER_MODEL_FORMULA_CASINO_GEOMETRY] = [
                    "Selected calibration has no geometry profiles.",
                    "Formula + CASINO + geometry profiles will collapse onto Formula + CASINO.",
                ]
            elif not geometry.metadata.get("multi_weld_profile_profiles_used"):
                diagnostic_lines[RENDER_MODEL_FORMULA_CASINO_GEOMETRY] = [
                    "No geometry response profiles were activated.",
                    "Current defects may not map to this calibration set.",
                ]
            preview_calibration_modes(
                {
                    RENDER_MODEL_FORMULA_ONLY: {
                        "topo": formula_only.channels.topo,
                        "tx": formula_only.channels.tx,
                        "ty": formula_only.channels.ty,
                        "display_mapping": formula_only.metadata.get("display_mapping"),
                        "defect_response_topo": formula_only.channels.topo - formula_only_clean.channels.topo,
                        "defect_response_metrics": compute_topography_delta_metrics(
                            formula_only_clean.channels.topo,
                            formula_only.channels.topo,
                            defect_mask=formula_only.defect_mask,
                        ),
                    },
                    RENDER_MODEL_FORMULA_CASINO: {
                        "topo": formula_casino.channels.topo,
                        "tx": formula_casino.channels.tx,
                        "ty": formula_casino.channels.ty,
                        "display_mapping": formula_casino.metadata.get("display_mapping"),
                        "defect_response_topo": formula_casino.channels.topo - formula_casino_clean.channels.topo,
                        "defect_response_metrics": compute_topography_delta_metrics(
                            formula_casino_clean.channels.topo,
                            formula_casino.channels.topo,
                            defect_mask=formula_casino.defect_mask,
                        ),
                    },
                    RENDER_MODEL_FORMULA_CASINO_GEOMETRY: {
                        "topo": geometry.channels.topo,
                        "tx": geometry.channels.tx,
                        "ty": geometry.channels.ty,
                        "display_mapping": geometry.metadata.get("display_mapping"),
                        "defect_response_topo": geometry.channels.topo - geometry_clean.channels.topo,
                        "defect_response_metrics": compute_topography_delta_metrics(
                            geometry_clean.channels.topo,
                            geometry.channels.topo,
                            defect_mask=geometry.defect_mask,
                        ),
                    },
                },
                defect_mask=geometry.defect_mask,
                title="Formula-only vs CASINO vs CASINO + Geometry Topography and Gradients",
                diagnostic_lines=diagnostic_lines,
            )
            self.status_var.set("Three-model comparison complete")
        except Exception as exc:
            messagebox.showerror("Preview Error", str(exc))

    def _preview_emission_b_comparison(self) -> None:
        try:
            calibration_path, render_model_mode, _emission_b_mode, emission_b_path = self._selected_render_settings()
            if emission_b_path is None:
                raise ValueError(
                    "Select an emission B fit JSON to compare pooled fitted constant B against quadratic fitted B(alpha)."
                )
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
                randomize_defect_location=bool(self.v_random_defect_location.get()),
            )
            sample_clean = replace(sample, defects=[])
            engine_kwargs = {
                "acquisition_model": "raster_v2",
                "image_size": self.v_image_size.get(),
                "pixel_size_mm": float(self.v_pixel_size_mm.get()),
                "calibration_path": calibration_path,
                "render_model_mode": render_model_mode,
                "weld_geometry_mode": self._selected_weld_geometry_mode(),
                "weld_layout": self._selected_weld_layout(),
                "render_view_mode": self._selected_render_view_mode(),
            }
            pooled_constant_engine = ScanEngine(
                emission_b_mode=EMISSION_B_MODE_POOLED_CONSTANT_FIT,
                emission_b_path=emission_b_path,
                **engine_kwargs,
            )
            quadratic_engine = ScanEngine(
                emission_b_mode=EMISSION_B_MODE_QUADRATIC_FIT,
                emission_b_path=emission_b_path,
                **engine_kwargs,
            )
            pooled_constant = pooled_constant_engine.run(sample)
            pooled_constant_clean = pooled_constant_engine.run(sample_clean)
            quadratic = quadratic_engine.run(sample)
            quadratic_clean = quadratic_engine.run(sample_clean)
            preview_calibration_modes(
                {
                    "pooled_constant_b": {
                        "topo": pooled_constant.channels.topo,
                        "tx": pooled_constant.channels.tx,
                        "ty": pooled_constant.channels.ty,
                        "display_mapping": pooled_constant.metadata.get("display_mapping"),
                        "defect_response_topo": pooled_constant.channels.topo - pooled_constant_clean.channels.topo,
                        "defect_response_metrics": compute_topography_delta_metrics(
                            pooled_constant_clean.channels.topo,
                            pooled_constant.channels.topo,
                            defect_mask=pooled_constant.defect_mask,
                        ),
                    },
                    "quadratic_b": {
                        "topo": quadratic.channels.topo,
                        "tx": quadratic.channels.tx,
                        "ty": quadratic.channels.ty,
                        "display_mapping": quadratic.metadata.get("display_mapping"),
                        "defect_response_topo": quadratic.channels.topo - quadratic_clean.channels.topo,
                        "defect_response_metrics": compute_topography_delta_metrics(
                            quadratic_clean.channels.topo,
                            quadratic.channels.topo,
                            defect_mask=quadratic.defect_mask,
                        ),
                    },
                },
                defect_mask=quadratic.defect_mask,
                title="Pooled fitted constant B vs fitted quadratic B(alpha)",
                diagnostic_lines={
                    "pooled_constant_b": [
                        f"Render model: {render_model_mode}",
                        f"Emission B fit: {Path(emission_b_path).name}",
                        str((pooled_constant.metadata.get("emission_b") or {}).get("formula", "")),
                    ],
                    "quadratic_b": [
                        f"Render model: {render_model_mode}",
                        f"Emission B fit: {Path(emission_b_path).name}",
                        str((quadratic.metadata.get("emission_b") or {}).get("formula", "")),
                    ],
                },
            )
            self.status_var.set("Pooled constant B vs fitted B(alpha) comparison complete")
        except Exception as exc:
            messagebox.showerror("Preview Error", str(exc))

    def _generate_batch(self) -> None:
        if self._batch_running:
            return
        try:
            calibration_path, render_model_mode, emission_b_mode, emission_b_path = self._selected_render_settings()
            beam, material, detector, noise, slice_runtime, defects = self._manual_sample()
            image_size = int(self.v_image_size.get())
            pixel_size = float(self.v_pixel_size_mm.get())
            count = int(self.v_num_images.get())
            if not 16 <= image_size <= 512 or not np.isfinite(pixel_size) or pixel_size <= 0 or not 1 <= count <= 10000:
                raise ValueError('Use 16–512 pixels, a positive finite pixel size and 1–10,000 images.')
            output = Path(self.v_output_dir.get()).resolve() / (time.strftime('studio_%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
            opts = BatchOptions(
                num_images=count, output_dir=output, image_size=image_size, seed=self.v_seed.get(),
                pixel_size_mm=pixel_size, weld_layout_mode=self._selected_weld_layout().mode,
                seam_rotation_deg=float(self.v_seam_rotation.get()), seam_offset_mm=float(self.v_seam_offset.get()),
                circle_radius_mm=float(self.v_circle_radius.get()), circle_position_deg=float(self.v_circle_position.get()),
                max_defects=self.v_max_defects.get(), voxel_depth_mm=float(self.v_voxel_depth.get()),
                slice_offset_mm=float(self.v_slice_offset.get()), slice_tilt_deg=float(self.v_slice_tilt.get()),
                slice_rotation_deg=float(self.v_slice_rotation.get()), randomize_slice=not bool(self.v_fixed_slice.get()),
                randomize_defect_location=bool(self.v_random_defect_location.get()), calibration_path=calibration_path,
                render_model_mode=render_model_mode, emission_b_mode=emission_b_mode, emission_b_path=emission_b_path,
                disable_noise=bool(self.v_disable_batch_noise.get()), render_view_mode=self._selected_render_view_mode(),
                weld_geometry_mode=self._selected_weld_geometry_mode(),
                sample_override=SampledParameters(self.v_seed.get(), beam, material, detector, noise, slice_runtime, defects))
            bundle = self.bundle
        except Exception as exc:
            messagebox.showerror('Check generation settings', str(exc), parent=self.root)
            return
        def worker() -> None:
            try:
                summary = BatchGenerator(bundle, opts).run()
                self._batch_events.put(summary)
            except Exception as exc:
                self._batch_events.put(f"Generation failed: {exc}")

        self._batch_running = True
        self.generate_button.configure(state='disabled')
        self.status_var.set("Generating dataset...")
        threading.Thread(target=worker, daemon=True).start()

    def _poll_batch(self):
        try:
            message = self._batch_events.get_nowait()
            if isinstance(message, dict):
                output_dir = message['output_dir']
                self.inspection_panel.source.set(str(Path(output_dir)/'raw/topo'))
                self.inspection_panel.mode.set('Raw TOPO (training units)')
                self.inspection_panel.spacing.set('')  # Use each generated file's actual scale metadata.
                self.inspection_panel.recursive.set(True)
                self.inspection_panel.method.set('CNN')
                message = f"Generated {message['num_images']} images in {output_dir}. Ready in Inspect images."
            self.status_var.set(message)
            self._batch_running = False
            self.generate_button.configure(state='normal')
        except queue.Empty:
            pass
        self.root.after(150, self._poll_batch)

    def _save_defaults(self) -> None:
        try:
            self._write_default(self.config_dir / "beam_params.yaml", "accelerating_voltage_kv", self.v_voltage.get())
            self._write_default(self.config_dir / "beam_params.yaml", "beam_current_ma", self.v_current.get())
            self._write_default(self.config_dir / "beam_params.yaml", "focusing_current_norm", self.v_focus.get())
            self._write_default(self.config_dir / "beam_params.yaml", "travel_speed_mm_min", self.v_speed.get())
            self._write_default(self.config_dir / "beam_params.yaml", "dwell_time_us", self.v_dwell.get())
            self._write_default(self.config_dir / "beam_params.yaml", "focal_spot_mm", self.v_focal.get())
            self._write_default(self.config_dir / "detector_params.yaml", "height_mm", self.v_h.get())
            self._write_default(self.config_dir / "detector_params.yaml", "inner_radius_mm", self.v_inner_r.get())
            self._write_default(self.config_dir / "detector_params.yaml", "outer_radius_mm", self.v_outer_r.get())
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
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            try:
                if self.root.winfo_exists():
                    self.root.after_idle(self.root.destroy)
                    self.root.update_idletasks()
            except tk.TclError:
                pass
