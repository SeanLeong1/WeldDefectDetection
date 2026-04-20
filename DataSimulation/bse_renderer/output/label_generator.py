from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..models import DefectInstance
from ..utils import ensure_dir


def _bbox_xyxy_to_coco(x0: int, y0: int, x1: int, y1: int) -> list[float]:
    w = max(0, x1 - x0 + 1)
    h = max(0, y1 - y0 + 1)
    return [float(x0), float(y0), float(w), float(h)]


def _bbox_xyxy_to_yolo(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[float, float, float, float]:
    bw = max(0, x1 - x0 + 1)
    bh = max(0, y1 - y0 + 1)
    cx = x0 + bw / 2.0
    cy = y0 + bh / 2.0
    return (cx / w, cy / h, bw / w, bh / h)


def write_yolo_labels(
    out_root: Path,
    split: str,
    stem: str,
    defects: list[DefectInstance],
    width: int,
    height: int,
) -> str:
    label_dir = out_root / "labels" / "yolo" / split
    ensure_dir(label_dir)
    path = label_dir / f"{stem}.txt"
    rows = []
    for defect in defects:
        x0, y0, x1, y1 = defect.bbox_xyxy
        cx, cy, bw, bh = _bbox_xyxy_to_yolo(x0, y0, x1, y1, width, height)
        rows.append(f"{defect.category_id - 1} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return str(path)


@dataclass
class CocoBuilder:
    images: list[dict] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)
    categories: list[dict] = field(default_factory=list)
    _ann_id: int = 1

    def add_categories(self, mapping: dict[str, int]) -> None:
        self.categories = [
            {"id": cid, "name": name, "supercategory": "defect"}
            for name, cid in sorted(mapping.items(), key=lambda x: x[1])
        ]

    def add_sample(
        self,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
        defects: list[DefectInstance],
    ) -> None:
        self.images.append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height}
        )
        for defect in defects:
            x0, y0, x1, y1 = defect.bbox_xyxy
            bbox = _bbox_xyxy_to_coco(x0, y0, x1, y1)
            area = bbox[2] * bbox[3]
            self.annotations.append(
                {
                    "id": self._ann_id,
                    "image_id": image_id,
                    "category_id": defect.category_id,
                    "bbox": bbox,
                    "area": float(area),
                    "iscrowd": 0,
                }
            )
            self._ann_id += 1

    def save(self, out_root: Path) -> str:
        coco_dir = out_root / "labels" / "coco"
        ensure_dir(coco_dir)
        path = coco_dir / "annotations.json"
        payload = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

