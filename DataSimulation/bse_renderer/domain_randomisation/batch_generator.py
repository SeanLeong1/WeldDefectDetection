from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from tqdm import tqdm

from ..config_loader import ConfigBundle
from ..geometry.defect_injection import CATEGORY_IDS
from ..models import SliceRuntime
from ..output.image_writer import save_channel_images, save_mask
from ..output.label_generator import CocoBuilder, write_yolo_labels
from ..renderer.scan_engine import ScanEngine
from ..utils import ensure_dir
from .parameter_sampler import sample_parameters


@dataclass(frozen=True)
class BatchOptions:
    num_images: int
    output_dir: Path
    image_size: int = 256
    seed: int = 42
    workers: int = 1
    train_frac: float = 0.7
    val_frac: float = 0.2
    test_frac: float = 0.1
    max_defects: int = 3
    voxel_depth_mm: float | None = None
    slice_offset_mm: float | None = None
    slice_tilt_deg: float | None = None
    slice_rotation_deg: float | None = None
    randomize_slice: bool = True


class BatchGenerator:
    def __init__(self, bundle: ConfigBundle, options: BatchOptions):
        self.bundle = bundle
        self.options = options
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

    def run(self) -> dict:
        out_root = self.options.output_dir.resolve()
        ensure_dir(out_root)
        engine = ScanEngine(image_size=self.options.image_size)
        coco = CocoBuilder()
        coco.add_categories(CATEGORY_IDS)
        manifest = []

        for i in tqdm(range(self.options.num_images), desc="Generating", unit="img"):
            split = self._split_for_index(i)
            seed = self.options.seed * 1000 + i
            sampled = sample_parameters(
                self.bundle,
                seed=seed,
                max_defects=self.options.max_defects,
            )
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
            result = engine.run(sampled)
            stem = f"{i:05d}"
            artifacts = save_channel_images(
                output_root=out_root,
                split=split,
                stem=stem,
                compo=result.channels.compo,
                topo=result.channels.topo,
                tx=result.channels.tx,
                ty=result.channels.ty,
                display_mapping=result.metadata.get("display_mapping"),
            )
            artifacts["mask"] = save_mask(out_root, split, stem, result.defect_mask)
            artifacts["yolo"] = write_yolo_labels(
                out_root=out_root,
                split=split,
                stem=stem,
                defects=result.defects,
                width=result.channels.compo.shape[1],
                height=result.channels.compo.shape[0],
            )
            compo_rel = str(Path("images") / "compo" / split / f"{stem}.png")
            coco.add_sample(
                image_id=i + 1,
                file_name=compo_rel,
                width=result.channels.compo.shape[1],
                height=result.channels.compo.shape[0],
                defects=result.defects,
            )
            manifest.append(
                {
                    "index": i,
                    "split": split,
                    "seed": seed,
                    "defects": [d.defect_type for d in result.defects],
                    "metadata": result.metadata,
                    "artifacts": artifacts,
                }
            )

        coco_path = coco.save(out_root)
        manifest_path = out_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        ds_yaml = {
            "path": str(out_root),
            "train": "images/compo/train",
            "val": "images/compo/val",
            "test": "images/compo/test",
            "nc": len(CATEGORY_IDS),
            "names": [name for name, _ in sorted(CATEGORY_IDS.items(), key=lambda x: x[1])],
        }
        dataset_yaml_path = out_root / "dataset.yaml"
        dataset_yaml_path.write_text(yaml.safe_dump(ds_yaml, sort_keys=False), encoding="utf-8")
        return {
            "output_dir": str(out_root),
            "manifest": str(manifest_path),
            "coco": coco_path,
            "dataset_yaml": str(dataset_yaml_path),
            "num_images": self.options.num_images,
        }
