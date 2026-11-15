from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
import numpy as np
from tqdm import tqdm

from ..config_loader import ConfigBundle, default_slice_runtime
from ..geometry.defect_injection import CATEGORY_IDS
from ..models import SliceRuntime, WeldLayoutRuntime
from ..output.image_writer import save_channel_images, save_mask
from ..output.label_generator import CocoBuilder, write_yolo_labels
from ..renderer.scan_engine import ScanEngine
from ..utils import ensure_dir
from .parameter_sampler import SampledParameters, sample_parameters


@dataclass(frozen=True)
class BatchOptions:
    num_images: int
    output_dir: Path
    image_size: int = 256
    seed: int = 42
    workers: int = 1
    pixel_size_mm: float = 0.05
    weld_layout_mode: str = "local_patch"
    seam_rotation_deg: float = 0.0
    seam_offset_mm: float = 0.0
    circle_radius_mm: float = 80.0
    circle_position_deg: float = 90.0
    train_frac: float = 0.7
    val_frac: float = 0.2
    test_frac: float = 0.1
    max_defects: int = 3
    voxel_depth_mm: float | None = None
    slice_offset_mm: float | None = None
    slice_tilt_deg: float | None = None
    slice_rotation_deg: float | None = None
    randomize_slice: bool = True
    randomize_defect_location: bool = True
    calibration_path: str | None = None
    render_model_mode: str | None = None
    emission_b_mode: str | None = None
    emission_b_path: str | None = None
    disable_noise: bool = False
    render_view_mode: str = "slice"
    weld_geometry_mode: str = "recessed_seam"
    sample_override: SampledParameters | None = None
    acquisition_model: str = "raster_v2"


class BatchGenerator:
    def __init__(self, bundle: ConfigBundle, options: BatchOptions):
        self.bundle = bundle
        self.options = options
        if options.workers != 1:
            raise ValueError("Generic batch generation currently supports workers=1 only.")
        if options.num_images < 1 or min(options.train_frac, options.val_frac, options.test_frac) < 0:
            raise ValueError("Image count must be positive and split fractions nonnegative.")
        total = options.train_frac + options.val_frac + options.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    def _split_for_index(self, idx: int) -> str:
        n = self.options.num_images
        train_n = round(n * self.options.train_frac)
        val_n = round(n * self.options.val_frac)
        if idx < train_n:
            return "train"
        if idx < train_n + val_n:
            return "val"
        return "test"

    @staticmethod
    def _relative_artifact(path: str, root: Path) -> str:
        return Path(path).resolve().relative_to(root).as_posix()

    def _contract_record(
        self,
        *,
        index: int,
        split: str,
        seed: int,
        artifacts: dict[str, str],
        result,
        root: Path,
    ) -> dict:
        visible_defects = [d.defect_type for d in result.defects]
        emission_b = result.metadata.get("emission_b") or {}
        b_stats = emission_b.get("b_stats") or {}
        emission_b_value = float(
            b_stats.get("mean", result.metadata.get("material", {}).get("b", 0.0))
        )
        noise = result.metadata.get("noise") or {}
        noise_enabled = bool(
            noise.get("poisson_enabled")
            or noise.get("thermal_enabled")
            or noise.get("contamination_enabled")
        )
        sample_id = f"sample-{index:05d}"
        return {
            "contract_version": "weld-preview/v1",
            "sample_id": sample_id,
            "sequence_id": f"static-{sample_id}",
            "frame_index": 0,
            "split": split,
            "channels": {
                name: self._relative_artifact(artifacts[f"{name}_u16"], root)
                for name in ("topo", "tx", "ty")
            },
            "mask_path": self._relative_artifact(artifacts["mask"], root),
            "defect_type": (
                "clean"
                if not visible_defects
                else visible_defects[0]
                if len(visible_defects) == 1
                else "multiple"
            ),
            "defect_types": visible_defects,
            "is_defective": bool(visible_defects),
            "seed": int(seed),
            "beam_regime": str(result.metadata.get("beam_regime", "randomized")),
            "emission_b_mode": str(emission_b.get("mode", "constant")),
            "emission_b_value": emission_b_value,
            "emission_b_source_path": emission_b.get("source_path"),
            "emission_b_source_sha256": emission_b.get("source_sha256"),
            "calibration_source": result.metadata.get("calibration_source"),
            "calibration_sha256": result.metadata.get("calibration_sha256"),
            "render_view_mode": str(result.metadata.get("render_view_mode", "slice")),
            "pixel_size_mm": float(result.metadata.get("effective_pixel_size_mm", self.options.pixel_size_mm)),
            "noise_enabled": noise_enabled,
            "planned_defects": list(result.metadata.get("planned_defects", [])),
            "excluded_nonvisible_defects": list(
                result.metadata.get("excluded_nonvisible_defects", [])
            ),
        }

    def run(self) -> dict:
        out_root = self.options.output_dir.resolve()
        if out_root.exists() and any(out_root.iterdir()):
            raise FileExistsError("Batch output must be a new empty directory.")
        ensure_dir(out_root)
        engine = ScanEngine(
            acquisition_model=self.options.acquisition_model,
            image_size=self.options.image_size,
            pixel_size_mm=self.options.pixel_size_mm,
            calibration_path=self.options.calibration_path,
            render_model_mode=self.options.render_model_mode,
            emission_b_mode=self.options.emission_b_mode,
            emission_b_path=self.options.emission_b_path,
            weld_layout=WeldLayoutRuntime(
                mode=self.options.weld_layout_mode,
                seam_rotation_deg=self.options.seam_rotation_deg,
                seam_offset_mm=self.options.seam_offset_mm,
                circle_radius_mm=self.options.circle_radius_mm,
                circle_position_deg=self.options.circle_position_deg,
            ),
            render_view_mode=self.options.render_view_mode,
            weld_geometry_mode=self.options.weld_geometry_mode,
        )
        coco = CocoBuilder()
        coco.add_categories(CATEGORY_IDS)
        manifest = []
        contract_records = []

        for i in tqdm(range(self.options.num_images), desc="Generating", unit="img"):
            split = self._split_for_index(i)
            seed = self.options.seed * 1000 + i
            sampled = sample_parameters(
                self.bundle,
                seed=seed,
                max_defects=self.options.max_defects,
                randomize_defect_location=self.options.randomize_defect_location,
            )
            if self.options.sample_override is not None:
                sampled = replace(self.options.sample_override, seed=seed,
                                  randomize_defect_location=self.options.randomize_defect_location)
            if not self.options.randomize_slice:
                sampled = replace(sampled, slice=default_slice_runtime(self.bundle.slice))
            if not self.options.randomize_slice or any(
                value is not None
                for value in (
                    self.options.voxel_depth_mm,
                    self.options.slice_offset_mm,
                    self.options.slice_tilt_deg,
                    self.options.slice_rotation_deg,
                )
            ):
                sampled = replace(
                    sampled,
                    slice=SliceRuntime(
                        voxel_depth_mm=sampled.slice.voxel_depth_mm
                        if self.options.voxel_depth_mm is None
                        else float(self.options.voxel_depth_mm),
                        slice_offset_mm=sampled.slice.slice_offset_mm
                        if self.options.slice_offset_mm is None
                        else float(self.options.slice_offset_mm),
                        slice_tilt_deg=sampled.slice.slice_tilt_deg
                        if self.options.slice_tilt_deg is None
                        else float(self.options.slice_tilt_deg),
                        slice_rotation_deg=sampled.slice.slice_rotation_deg
                        if self.options.slice_rotation_deg is None
                        else float(self.options.slice_rotation_deg),
                    ),
                )
            if self.options.disable_noise:
                sampled = replace(
                    sampled,
                    noise=replace(
                        sampled.noise,
                        poisson_enabled=False,
                        thermal_enabled=False,
                        contamination_enabled=False,
                        contamination_rate=0.0,
                        quadrant_offsets={q: 0.0 for q in ("A", "B", "C", "D")},
                    ),
                )
            result = engine.run(sampled)
            stem = f"{i:05d}"
            artifacts = save_channel_images(
                output_root=out_root,
                split=split,
                stem=stem,
                compo=None,
                topo=result.channels.topo,
                tx=result.channels.tx,
                ty=result.channels.ty,
                display_mapping=result.metadata.get("display_mapping"),
            )
            raw_dir = out_root / 'raw' / 'topo' / split
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f'{stem}.npy'
            np.save(raw_path, np.asarray(result.channels.topo, dtype=np.float32), allow_pickle=False)
            sidecar = raw_path.with_name(raw_path.name + '.json')
            sidecar.write_text(json.dumps(dict(
                pixel_size_mm=float(result.metadata.get('effective_pixel_size_mm', self.options.pixel_size_mm)),
                channel='topo_magnitude', source_kind='synthetic_preview', physical_validation=False,
                material=sampled.material.name, energy_kev=sampled.beam.e0_kev), indent=2), encoding='utf-8')
            artifacts['topo_raw'] = str(raw_path)
            artifacts['topo_raw_metadata'] = str(sidecar)
            artifacts["mask"] = save_mask(out_root, split, stem, result.defect_mask)
            artifacts["yolo"] = write_yolo_labels(
                out_root=out_root,
                split=split,
                stem=stem,
                defects=result.defects,
                width=result.channels.topo.shape[1],
                height=result.channels.topo.shape[0],
            )
            topo_rel = str(Path("images") / "topo" / split / f"{stem}.png")
            coco.add_sample(
                image_id=i + 1,
                file_name=topo_rel,
                width=result.channels.topo.shape[1],
                height=result.channels.topo.shape[0],
                defects=result.defects,
            )
            manifest.append(
                {
                    "index": i,
                    "split": split,
                    "seed": seed,
                    "defects": [d.defect_type for d in result.defects],
                    "planned_defects": list(result.metadata.get("planned_defects", [])),
                    "visible_defects": list(result.metadata.get("visible_defects", [])),
                    "excluded_nonvisible_defects": list(
                        result.metadata.get("excluded_nonvisible_defects", [])
                    ),
                    "metadata": result.metadata,
                    "artifacts": artifacts,
                }
            )
            contract_records.append(
                self._contract_record(
                    index=i,
                    split=split,
                    seed=seed,
                    artifacts=artifacts,
                    result=result,
                    root=out_root,
                )
            )

        coco_path = coco.save(out_root)
        manifest_path = out_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        records_path = out_root / "samples.jsonl"
        records_path.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in contract_records),
            encoding="utf-8",
        )

        contract_descriptor = {
            "contract_version": "weld-preview/v1",
            "task": "binary_surface_defect_segmentation",
            "dataset_kind": "static",
            "records": "samples.jsonl",
            "required_channels": ["topo"],
            "optional_channels": ["tx", "ty"],
            "mask_semantics": "binary_surface_visible_defect",
            "splits": ["train", "val", "test"],
            "split_policy": "group_by_sequence_id",
            "noise_policy": "disabled" if self.options.disable_noise else "configuration_controlled",
            "render_view_mode": self.options.render_view_mode,
            "sequence_semantics": (
                "Each static image is a one-frame sequence. Future sequential datasets reuse the "
                "same record fields with shared sequence_id and increasing frame_index."
            ),
        }
        contract_path = out_root / "dataset_contract.yaml"
        contract_path.write_text(
            yaml.safe_dump(contract_descriptor, sort_keys=False), encoding="utf-8"
        )

        ds_yaml = {
            "path": str(out_root),
            "train": "images/topo/train",
            "val": "images/topo/val",
            "test": "images/topo/test",
            "nc": len(CATEGORY_IDS),
            "names": [name for name, _ in sorted(CATEGORY_IDS.items(), key=lambda x: x[1])],
            "render_view_mode": self.options.render_view_mode,
            "label_scope": (
                "surface_visible_intersections_only"
                if self.options.render_view_mode == "top_surface"
                else "slice_intersections_only"
            ),
            "contract": "dataset_contract.yaml",
            "records": "samples.jsonl",
        }
        dataset_yaml_path = out_root / "dataset.yaml"
        dataset_yaml_path.write_text(yaml.safe_dump(ds_yaml, sort_keys=False), encoding="utf-8")
        return {
            "output_dir": str(out_root),
            "manifest": str(manifest_path),
            "coco": coco_path,
            "dataset_yaml": str(dataset_yaml_path),
            "dataset_contract": str(contract_path),
            "records": str(records_path),
            "num_images": self.options.num_images,
            "render_view_mode": self.options.render_view_mode,
        }
