from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.algorithms.classical.cnn_detector import CNNDetector
from src.utils.file_io import load_config, write_csv, write_json
from src.utils.metrics import aggregate_segmentation_metrics, evaluate_segmentation_masks


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_tiled_probability(
    detector: CNNDetector,
    image: np.ndarray,
    *,
    tile_size: int = 128,
    stride: int = 96,
) -> np.ndarray:
    """Run a square-trained CNN across a large image without aspect-ratio distortion."""
    if tile_size <= 0 or stride <= 0 or stride > tile_size:
        raise ValueError("Require tile_size > 0 and 0 < stride <= tile_size.")
    gray = detector._normalize_grayscale(image)
    height, width = gray.shape
    padded_height = max(height, tile_size)
    padded_width = max(width, tile_size)
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    padded = cv2.copyMakeBorder(
        gray,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    probability_sum = np.zeros(padded.shape, dtype=np.float32)
    weight = np.zeros(padded.shape, dtype=np.float32)
    for y in _tile_starts(padded_height, tile_size, stride):
        for x in _tile_starts(padded_width, tile_size, stride):
            tile = padded[y : y + tile_size, x : x + tile_size]
            probability = detector.predict_mask(tile)
            if probability.shape != tile.shape:
                raise ValueError(
                    f"Detector returned tile shape {probability.shape}, expected {tile.shape}."
                )
            probability_sum[y : y + tile_size, x : x + tile_size] += probability
            weight[y : y + tile_size, x : x + tile_size] += 1.0
    if np.any(weight == 0):
        raise RuntimeError("Tiled inference left uncovered pixels.")
    return (probability_sum / weight)[:height, :width]


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read GDXray image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _save_overlay(
    path: Path,
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    max_width: int = 2400,
) -> None:
    gray = CNNDetector._normalize_grayscale(image)
    rgb = np.stack([gray, gray, gray], axis=-1)
    truth = target.astype(bool)
    pred = prediction.astype(bool)
    rgb[truth] = 0.60 * rgb[truth] + 0.40 * np.array([0.0, 1.0, 0.0])
    rgb[pred & ~truth] = 0.55 * rgb[pred & ~truth] + 0.45 * np.array([1.0, 0.0, 0.0])
    rgb[pred & truth] = 0.50 * rgb[pred & truth] + 0.50 * np.array([1.0, 1.0, 0.0])
    output = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    if output.shape[1] > max_width:
        scale = max_width / output.shape[1]
        output = cv2.resize(
            output,
            (max_width, max(1, int(round(output.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


def _bootstrap_by_image(
    rows: list[dict[str, Any]], *, seed: int, iterations: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ious: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[int(index)] for index in indices]
        ious.append(float(aggregate_segmentation_metrics(sample)["iou"]))
    return {
        "iterations": iterations,
        "resampling_unit": "radiograph",
        "pixel_iou": {
            "mean": float(np.mean(ious)),
            "ci_lower": float(np.quantile(ious, 0.025)),
            "ci_upper": float(np.quantile(ious, 0.975)),
        },
    }


def evaluate_gdxray_cnn(
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    image_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
    tile_size: int = 128,
    stride: int = 96,
    bootstrap_iterations: int = 500,
    bootstrap_seed: int = 20260901,
) -> dict[str, Any]:
    config = load_config(config_path)
    config.setdefault("cnn", {})["checkpoint_path"] = str(
        Path(checkpoint).expanduser().resolve()
    )
    detector = CNNDetector(config)
    images_root = Path(image_dir).expanduser().resolve()
    masks_root = Path(mask_dir).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    overlays = root / "overlays"
    probabilities = root / "probabilities"
    overlays.mkdir(parents=True, exist_ok=True)
    probabilities.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    image_paths = sorted(images_root.glob("W0001_*.png"))
    if not image_paths:
        raise ValueError(f"No W0001 PNG images found under {images_root}.")
    for image_path in image_paths:
        suffix = image_path.stem.removeprefix("W0001_")
        mask_path = masks_root / f"W0002_{suffix}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing paired GDXray mask: {mask_path}")
        image = _read_gray(image_path)
        target = (_read_gray(mask_path) > 0).astype(np.uint8)
        if target.shape != image.shape:
            raise ValueError(
                f"Pair shape mismatch for {image_path.name}: {image.shape} vs {target.shape}."
            )
        probability = predict_tiled_probability(
            detector, image, tile_size=tile_size, stride=stride
        )
        prediction = (probability >= detector.threshold).astype(np.uint8)
        metrics = evaluate_segmentation_masks(prediction, target)
        rows.append(
            {
                "sample_id": image_path.stem,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "target_positive_pixels": int(target.sum()),
                "predicted_positive_pixels": int(prediction.sum()),
                "target_is_defective": bool(np.any(target)),
                "predicted_is_defective": bool(np.any(prediction)),
                **metrics,
            }
        )
        cv2.imwrite(
            str(probabilities / f"{image_path.stem}.png"),
            (np.clip(probability, 0.0, 1.0) * 65535).astype(np.uint16),
        )
        _save_overlay(overlays / f"{image_path.stem}.png", image, target, prediction)

    pixel = aggregate_segmentation_metrics(rows)
    image_recall = sum(bool(row["predicted_is_defective"]) for row in rows) / len(rows)
    macro = {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in ("precision", "recall", "f1", "iou")
    }
    summary = {
        "status": "ok",
        "evaluation": "zero_shot_cross_modality",
        "source_training_modality": "synthetic_bse_topography",
        "target_modality": "real_xray_radiography",
        "warning": (
            "GDXray W0001/W0002 is X-ray radiography, not SEM. This measures severe "
            "cross-modality transfer and must not be reported as real-SEM validation."
        ),
        "image_count": len(rows),
        "all_images_have_positive_masks": all(
            bool(row["target_is_defective"]) for row in rows
        ),
        "image_detection_recall": float(image_recall),
        "image_specificity": None,
        "pixel": pixel,
        "macro_per_image": macro,
        "bootstrap": _bootstrap_by_image(
            rows, seed=bootstrap_seed, iterations=bootstrap_iterations
        ),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "checkpoint_threshold": float(detector.threshold),
        "tile_size": int(tile_size),
        "stride": int(stride),
        "output_dir": str(root),
    }
    write_csv(root / "metrics.csv", rows)
    write_json(root / "summary.json", summary)
    return summary
