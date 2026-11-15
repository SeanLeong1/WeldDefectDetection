from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import sys
from dataclasses import replace, asdict
from importlib.metadata import distributions
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from ..config_loader import default_slice_runtime, load_config_bundle
from ..models import WeldLayoutRuntime
from ..output.image_writer import save_mask
from ..renderer.scan_engine import ScanEngine
from ..utils import apply_sem_tone_curve
from .parameter_sampler import sample_parameters


CONTRACT_VERSION = "weld-topo-segmentation/v2"
CORE_DEFECT_CLASSES = (
    "keyhole_pore",
    "intrinsic_pore",
    "crack_solidification",
    "crack_liquidation",
    "lack_of_fusion",
    "undercut",
)
DATASET_CLASSES = ("clean",) + CORE_DEFECT_CLASSES
BEAM_REGIMES = ("low", "mid", "high")
NOISE_PROFILES = ("development", "none")


def _prepare_new_output_root(output_dir: str | Path) -> Path:
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Dataset output directory is not empty: {root}. "
            "Use a new versioned directory to avoid mixing or overwriting datasets."
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _preflight_frozen_normalization(
    normalization_path: str | Path | None,
) -> Path | None:
    if normalization_path is None:
        return None
    path = Path(normalization_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen TOPO normalization not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Frozen TOPO normalization is not valid JSON: {path}") from exc
    low = payload.get("raw_low")
    high = payload.get("raw_high")
    if (
        not isinstance(low, (int, float))
        or not isinstance(high, (int, float))
        or not np.isfinite([low, high]).all()
        or float(high) <= float(low)
    ):
        raise ValueError("Frozen TOPO normalization requires numeric raw_high > raw_low.")
    if payload.get("source_split") != "train":
        raise ValueError("Frozen TOPO normalization must originate from the training split.")
    if payload.get("tone_curve") != "sem_sqrt":
        raise ValueError("Frozen TOPO normalization must use the sem_sqrt tone curve.")
    return path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _freeze_generation_provenance(
    root: Path,
    *,
    config_dir: str | Path,
    engine: ScanEngine,
    seed: int,
    pixel_size_mm: float,
    generation_parameters: dict[str, Any],
) -> dict[str, Any]:
    config_root = load_config_bundle(config_dir).config_dir
    source_root = Path(__file__).resolve().parents[2]
    config_snapshot = []
    for path in sorted(config_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}:
            config_snapshot.append(
                {
                    "path": path.relative_to(config_root).as_posix(),
                    "sha256": _sha256_path(path),
                    "text": path.read_text(encoding="utf-8"),
                }
            )
    repo_root = source_root.parent
    renderer_paths = sorted((source_root / "bse_renderer").rglob("*.py"))
    source_paths = set(renderer_paths)
    for directory in (source_root / "entrypoints", source_root / "tools", repo_root / "WeldDefectDetection" / "src"):
        source_paths.update(directory.rglob("*.py"))
    for directory in (source_root, repo_root / "WeldDefectDetection"):
        for suffix in ("*.py", "*.yaml", "requirements*.txt"):
            source_paths.update(directory.glob(suffix))
    source_paths.update(path for path in config_root.rglob("*") if path.is_file() and path.suffix in (".yaml", ".yml", ".json"))
    archive = root / "generation_source.zip"
    source_hashes = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as frozen:
        for source in sorted(source_paths):
            name = source.relative_to(repo_root).as_posix() if source.is_relative_to(repo_root) else "resolved_config/" + source.relative_to(config_root).as_posix()
            content = source.read_bytes()
            frozen.writestr(name, content)
            source_hashes[name] = hashlib.sha256(content).hexdigest()
    dependencies = {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]}
    payload = {
        "git_revision": _git_revision(source_root),
        "source_identity_policy": "archived_working_tree_including_uncommitted_source",
        "source_archive": archive.name,
        "source_archive_sha256": _sha256_path(archive),
        "source_hashes": source_hashes,
        "dependency_versions": dict(sorted(dependencies.items())),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": int(seed),
        "pixel_size_mm": float(pixel_size_mm),
        "generation_parameters": generation_parameters,
        "render_model_mode": engine.render_model_mode,
        "emission_b_mode": engine.emission_b_mode,
        "emission_assumption": (None if engine.emission_b_fit is None else engine.emission_b_fit.assumption),
        "emission_b_source_path": engine.emission_b_path,
        "emission_b_source_sha256": (
            None
            if engine.emission_b_path is None
            else _sha256_path(Path(engine.emission_b_path).resolve())
        ),
        "calibration_source_path": engine.calibration.source,
        "calibration_source_sha256": (
            None
            if engine.calibration.source is None
            else _sha256_path(Path(engine.calibration.source).resolve())
        ),
        "renderer_source_hashes": {
            path.relative_to(source_root).as_posix(): _sha256_path(path)
            for path in renderer_paths
        },
        "configuration_snapshot": config_snapshot,
    }
    path = root / "generation_provenance.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"path": str(path), "sha256": _sha256_path(path), **payload}


def _beam_overrides(regime: str) -> dict[str, float]:
    if regime == "low":
        return {
            "accelerating_voltage_kv": 45.0,
            "beam_current_ma": 90.0,
            "focusing_current_norm": 0.35,
            "travel_speed_mm_min": 1600.0,
        }
    if regime == "high":
        return {
            "accelerating_voltage_kv": 120.0,
            "beam_current_ma": 420.0,
            "focusing_current_norm": 0.85,
            "travel_speed_mm_min": 500.0,
        }
    return {
        "accelerating_voltage_kv": 70.0,
        "beam_current_ma": 220.0,
        "focusing_current_norm": 0.60,
        "travel_speed_mm_min": 900.0,
    }


def _set_regime(sample, regime: str, fixed_energy_kev: float | None = None):
    beam = replace(sample.beam, **_beam_overrides(regime))
    if fixed_energy_kev is not None:
        if not np.isfinite(fixed_energy_kev) or fixed_energy_kev <= 0:
            raise ValueError("Fixed beam energy must be finite and positive.")
        beam = replace(beam, accelerating_voltage_kv=float(fixed_energy_kev))
    beam = replace(
        beam,
        e0_kev=beam.accelerating_voltage_kv,
        power_w=(beam.accelerating_voltage_kv * 1000.0)
        * (beam.beam_current_ma / 1000.0),
    )
    return replace(sample, beam=beam)


def _disable_noise(sample):
    return replace(
        sample,
        noise=replace(
            sample.noise,
            poisson_enabled=False,
            thermal_enabled=False,
            contamination_enabled=False,
            contamination_rate=0.0,
            quadrant_offsets={q: 0.0 for q in ("A", "B", "C", "D")},
        ),
    )


def _apply_noise_profile(sample, noise_profile: str):
    profile = str(noise_profile).strip().lower()
    if profile not in NOISE_PROFILES:
        raise ValueError(
            f"Unknown noise_profile '{noise_profile}'. Expected one of: {', '.join(NOISE_PROFILES)}."
        )
    if profile == "none":
        return _disable_noise(sample)
    return replace(
        sample,
        noise=replace(
            sample.noise,
            poisson_enabled=True,
            thermal_enabled=True,
            contamination_enabled=True,
        ),
    )


def _sample_defect(defect_type: str, rng: np.random.Generator) -> list[dict[str, Any]]:
    if defect_type == "clean":
        return []
    if defect_type == "keyhole_pore":
        params = {
            "diameter_mm": float(rng.uniform(0.50, 1.20)),
            "depth_mm": float(rng.uniform(0.25, 0.60)),
        }
    elif defect_type == "intrinsic_pore":
        params = {
            "diameter_mm": float(rng.uniform(0.50, 1.00)),
            "depth_mm": float(rng.uniform(0.25, 0.65)),
        }
    elif defect_type == "crack_solidification":
        params = {
            "length_mm": float(rng.uniform(0.80, 2.20)),
            "width_mm": float(rng.uniform(0.08, 0.16)),
            "depth_mm": float(rng.uniform(0.20, 0.55)),
        }
    elif defect_type == "crack_liquidation":
        params = {
            "length_mm": float(rng.uniform(0.80, 2.00)),
            "width_mm": float(rng.uniform(0.08, 0.18)),
            "depth_mm": float(rng.uniform(0.18, 0.50)),
        }
    elif defect_type == "lack_of_fusion":
        params = {
            "extent_mm": float(rng.uniform(0.80, 2.20)),
            "depth_mm": float(rng.uniform(0.30, 0.70)),
            "irregularity": float(rng.uniform(0.25, 0.65)),
        }
    elif defect_type == "undercut":
        params = {
            "groove_depth_mm": float(rng.uniform(0.30, 0.75)),
            "toe_angle_deg": float(rng.uniform(22.0, 44.0)),
        }
    else:
        raise ValueError(f"Unsupported balanced defect class '{defect_type}'.")
    return [{"type": defect_type, "params": params}]


def _split_for_repetition(repetition: int, total: int) -> str:
    test_n = max(1, round(total * 0.20))
    val_n = max(1, round(total * 0.20))
    if val_n + test_n >= total:
        val_n = 1
        test_n = 1
    train_n = total - val_n - test_n
    if repetition < train_n:
        return "train"
    if repetition < train_n + val_n:
        return "val"
    return "test"


def _regime_for_sample(repetition: int, class_index: int) -> str:
    """Cycle regimes with a class offset so split and regime are not confounded."""
    return BEAM_REGIMES[(int(repetition) + int(class_index)) % len(BEAM_REGIMES)]


def _planned_target_frame(
    repetition: int,
    class_index: int,
    *,
    frames_per_sequence: int,
    context_radius: int,
) -> int:
    eligible = tuple(range(context_radius, frames_per_sequence - context_radius))
    if not eligible:
        raise ValueError("No target frames remain after applying context_radius.")
    return eligible[(int(repetition) + int(class_index)) % len(eligible)]


def _relative(path: str | Path, root: Path) -> str:
    return Path(path).resolve().relative_to(root).as_posix()


def _raw_topo_path(root: Path, split: str, stem: str) -> Path:
    path = root / "raw" / "topo" / split / f"{stem}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _normalised_topo_path(root: Path, split: str, stem: str) -> Path:
    return root / "images" / "topo" / split / f"{stem}.png"


def _save_raw_topo(root: Path, split: str, stem: str, topo: np.ndarray) -> tuple[Path, Path]:
    raw_path = _raw_topo_path(root, split, stem)
    np.save(raw_path, np.asarray(topo, dtype=np.float32), allow_pickle=False)
    return raw_path, _normalised_topo_path(root, split, stem)


def _write_u16(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)).save(path)


def _write_u8(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def _apply_training_global_topo_normalisation(
    root: Path,
    records: list[dict[str, Any]],
    *,
    low_quantile: float = 0.01,
    high_quantile: float = 0.99,
    pad_fraction: float = 0.08,
    frozen_normalization_path: str | Path | None = None,
) -> dict[str, Any]:
    frozen_path: Path | None = None
    if frozen_normalization_path is not None:
        frozen_path = Path(frozen_normalization_path).expanduser().resolve()
        source_bytes = frozen_path.read_bytes()
        source_payload = json.loads(source_bytes.decode("utf-8"))
        low = float(source_payload["raw_low"])
        high = float(source_payload["raw_high"])
        if high <= low:
            raise ValueError("Frozen TOPO normalization must have raw_high > raw_low.")
        if source_payload.get("tone_curve") != "sem_sqrt":
            raise ValueError("Frozen TOPO normalization must use the sem_sqrt tone curve.")
        payload = {
            **source_payload,
            "application_policy": "frozen_external_training_mapping",
            "fit_performed_on_this_dataset": False,
            "frozen_from_dataset": frozen_path.parent.name,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        }
    else:
        train_values = [
            np.load(root / record["raw_channels"]["topo"], allow_pickle=False).reshape(-1)
            for record in records
            if record["split"] == "train"
        ]
        if not train_values:
            raise ValueError("Global TOPO normalization requires training records.")
        values = np.concatenate(train_values).astype(np.float32, copy=False)
        low = float(np.quantile(values, low_quantile))
        high = float(np.quantile(values, high_quantile))
        span = max(high - low, 1e-12)
        low = max(0.0, low - pad_fraction * span)
        high = high + pad_fraction * span
        if high <= low:
            high = low + 1.0
        payload = {
            "method": "training_split_global_percentile",
            "source_split": "train",
            "low_quantile": float(low_quantile),
            "high_quantile": float(high_quantile),
            "pad_fraction": float(pad_fraction),
            "raw_low": low,
            "raw_high": high,
            "tone_curve": "sem_sqrt",
            "raw_format": "float32_npy",
            "application_policy": "fit_on_this_dataset_training_split",
            "fit_performed_on_this_dataset": True,
        }

    for record in records:
        raw = np.load(root / record["raw_channels"]["topo"], allow_pickle=False)
        scaled = apply_sem_tone_curve(np.clip((raw - low) / (high - low), 0.0, 1.0))
        image_path = root / record["channels"]["topo"]
        preview_path = (
            root
            / "images_8bit"
            / "topo"
            / str(record["split"])
            / f"{record['sample_id']}.png"
        )
        _write_u16(image_path, scaled)
        _write_u8(preview_path, scaled)

    path = root / "topo_normalization.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "path": str(path)}


def _emission_fields(metadata: dict[str, Any]) -> tuple[str, float]:
    emission = metadata.get("emission_b") or {}
    stats = emission.get("b_stats") or {}
    value = float(stats.get("mean", (metadata.get("material") or {}).get("b", 0.0)))
    return str(emission.get("mode", "constant")), value


def _write_contract_bundle(
    root: Path,
    records: list[dict[str, Any]],
    *,
    dataset_kind: str,
    extra_descriptor: dict[str, Any] | None = None,
) -> dict[str, str]:
    records_path = root / "samples.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    descriptor: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "task": "binary_surface_defect_segmentation",
        "dataset_kind": dataset_kind,
        "records": "samples.jsonl",
        "required_channels": ["topo"],
        "prohibited_channels": ["compo"],
        "mask_semantics": "binary_surface_visible_defect",
        "splits": ["train", "val", "test"],
        "split_policy": "group_by_sequence_id",
        "noise_policy": "disabled",
        "render_view_mode": "top_surface",
    }
    descriptor.update(extra_descriptor or {})
    contract_path = root / "dataset_contract.yaml"
    contract_path.write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")
    dataset_path = root / "dataset.yaml"
    dataset_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "contract": "dataset_contract.yaml",
                "records": "samples.jsonl",
                "image_channel": "topo",
                "mask": "masks",
                "task": "binary_surface_defect_segmentation",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "records": str(records_path),
        "dataset_contract": str(contract_path),
        "dataset_yaml": str(dataset_path),
    }


def generate_balanced_topo_dataset(
    *,
    config_dir: str | Path,
    output_dir: str | Path,
    samples_per_class: int = 60,
    image_size: int = 128,
    pixel_size_mm: float = 0.05,
    seed: int = 1201,
    render_model_mode: str | None = None,
    emission_b_mode: str | None = None,
    emission_b_path: str | None = None,
    normalization_path: str | Path | None = None,
    noise_profile: str = "development",
    fixed_energy_kev: float | None = None,
) -> dict[str, Any]:
    if samples_per_class < 3:
        raise ValueError("samples_per_class must be at least 3 to populate every split.")
    frozen_normalization = _preflight_frozen_normalization(normalization_path)
    cfg = load_config_bundle(config_dir)
    root = _prepare_new_output_root(output_dir)
    engine = ScanEngine(
        image_size=image_size,
        pixel_size_mm=pixel_size_mm,
        render_model_mode=render_model_mode,
        emission_b_mode=emission_b_mode,
        emission_b_path=emission_b_path,
        render_view_mode="top_surface",
        weld_layout=WeldLayoutRuntime(mode="local_patch"),
    )
    provenance = _freeze_generation_provenance(
        root,
        config_dir=config_dir,
        engine=engine,
        seed=seed,
        pixel_size_mm=pixel_size_mm,
        generation_parameters={
            "dataset_kind": "static_balanced",
            "fixed_energy_kev": fixed_energy_kev,
            "samples_per_class": int(samples_per_class),
            "image_size": int(image_size),
            "noise_profile": noise_profile,
            "normalization_path": (
                None if frozen_normalization is None else str(frozen_normalization)
            ),
        },
    )
    controlled_slice = default_slice_runtime(cfg.slice)
    records: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    resampled_degenerate_count = 0
    maximum_generation_attempt = 1
    for class_index, defect_type in enumerate(DATASET_CLASSES):
        class_counts[defect_type] = 0
        for repetition in range(samples_per_class):
            regime = _regime_for_sample(repetition, class_index)
            split = _split_for_repetition(repetition, samples_per_class)
            base_sample_seed = int(seed + 10000 * class_index + repetition)
            for attempt in range(25):
                sample_seed = int(base_sample_seed + 1_000_000 * attempt)
                rng = np.random.default_rng(sample_seed)
                sampled = sample_parameters(
                    cfg,
                    seed=sample_seed,
                    max_defects=0,
                    randomize_defect_location=True,
                )
                sampled = replace(
                    _set_regime(_apply_noise_profile(sampled, noise_profile), regime, fixed_energy_kev),
                    slice=controlled_slice,
                    defects=_sample_defect(defect_type, rng),
                )
                result = engine.run(sampled)
                visible_types = [d.defect_type for d in result.defects]
                visible_ok = defect_type == "clean" or defect_type in visible_types
                topo = np.asarray(result.channels.topo, dtype=np.float32)
                signal_ok = bool(
                    np.all(np.isfinite(topo))
                    and float(np.max(topo) - np.min(topo)) > 1e-12
                )
                if visible_ok and signal_ok:
                    break
            else:
                raise RuntimeError(
                    f"Unable to generate a non-degenerate surface-visible '{defect_type}' "
                    f"sample after 25 attempts from base seed {base_sample_seed}."
                )
            generation_attempt = attempt + 1
            maximum_generation_attempt = max(maximum_generation_attempt, generation_attempt)
            if generation_attempt > 1:
                resampled_degenerate_count += 1
            sample_id = f"{defect_type}-{repetition:04d}"
            raw_topo_path, image_topo_path = _save_raw_topo(
                root, split, sample_id, result.channels.topo
            )
            mask_path = save_mask(root, split, sample_id, result.defect_mask)
            b_mode, b_value = _emission_fields(result.metadata)
            records.append(
                {
                    "contract_version": CONTRACT_VERSION,
                    "sample_id": sample_id,
                    "sequence_id": f"static-{sample_id}",
                    "frame_index": 0,
                    "split": split,
                    "channels": {"topo": _relative(image_topo_path, root)},
                    "raw_channels": {"topo": _relative(raw_topo_path, root)},
                    "mask_path": _relative(mask_path, root),
                    "defect_type": defect_type,
                    "defect_types": visible_types,
                    "is_defective": bool(visible_types),
                    "seed": sample_seed,
                    "base_seed": base_sample_seed,
                    "generation_attempt": generation_attempt,
                    "raw_topo_dynamic_range": float(np.max(topo) - np.min(topo)),
                    "beam_regime": regime,
                    "emission_b_mode": b_mode,
                    "emission_b_value": b_value,
                    "emission_b_source_path": (result.metadata.get("emission_b") or {}).get("source_path"),
                    "emission_b_source_sha256": (result.metadata.get("emission_b") or {}).get("source_sha256"),
                    "calibration_source": result.metadata.get("calibration_source"),
                    "calibration_sha256": result.metadata.get("calibration_sha256"),
                    "render_view_mode": "top_surface",
                    "pixel_size_mm": float(result.metadata["effective_pixel_size_mm"]),
                    "noise_enabled": noise_profile != "none",
                    "noise_profile": noise_profile,
                    "planned_defects": list(result.metadata.get("planned_defects", [])),
                    "planned_defect_parameters": list(sampled.defects),
                    "realized_non_noise_parameters": {key: value for key, value in asdict(sampled).items() if key != "noise"},
                    "renderer_parameters": {"image_size": engine.image_size, "pixel_size_mm": engine.pixel_size_mm,
                                            "weld_layout": asdict(engine.weld_layout), "render_model_mode": engine.render_model_mode,
                                            "emission_b_mode": engine.emission_b_mode, "render_view_mode": engine.render_view_mode},
                    "excluded_nonvisible_defects": list(
                        result.metadata.get("excluded_nonvisible_defects", [])
                    ),
                }
            )
            class_counts[defect_type] += 1

    normalization = _apply_training_global_topo_normalisation(
        root, records, frozen_normalization_path=frozen_normalization
    )
    artifacts = _write_contract_bundle(
        root,
        records,
        dataset_kind="static_balanced",
        extra_descriptor={
            "classes": list(DATASET_CLASSES),
            "samples_per_class": int(samples_per_class),
            "sequence_semantics": "one independent image per sequence",
            "normalization": "topo_normalization.json",
            "noise_profile": noise_profile,
            "noise_policy": noise_profile,
            "provenance": "generation_provenance.json",
        },
    )
    summary = {
        "status": "ok",
        "output_dir": str(root),
        "record_count": len(records),
        "sequence_count": len(records),
        "class_counts": class_counts,
        "resampled_degenerate_count": resampled_degenerate_count,
        "maximum_generation_attempt": maximum_generation_attempt,
        "normalization": normalization,
        "provenance": provenance,
        **artifacts,
    }
    (root / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def generate_topo_sequence_dataset(
    *,
    config_dir: str | Path,
    output_dir: str | Path,
    sequences_per_class: int = 3,
    frames_per_sequence: int = 9,
    frame_size: int = 96,
    stride_px: int | None = None,
    context_radius: int = 2,
    pixel_size_mm: float = 0.05,
    seed: int = 2201,
    render_model_mode: str | None = None,
    emission_b_mode: str | None = None,
    emission_b_path: str | None = None,
    normalization_path: str | Path | None = None,
    weld_layout_mode: str = "local_patch",
    noise_profile: str = "development",
    fixed_energy_kev: float | None = None,
) -> dict[str, Any]:
    if sequences_per_class < 3:
        raise ValueError("sequences_per_class must be at least 3 to populate every split.")
    if frames_per_sequence < 3:
        raise ValueError("frames_per_sequence must be at least 3 for sliding windows.")
    if context_radius < 1 or frames_per_sequence < 2 * context_radius + 1:
        raise ValueError(
            "frames_per_sequence must be at least 2 * context_radius + 1."
        )
    stride_px = frame_size if stride_px is None else stride_px
    if stride_px < frame_size:
        raise ValueError("Unseen-region prediction requires stride_px >= frame_size.")
    if frame_size < 32 or stride_px < 1:
        raise ValueError("frame_size must be >=32 and stride_px must be positive.")
    scene_size = int(frame_size + stride_px * (frames_per_sequence - 1))
    frozen_normalization = _preflight_frozen_normalization(normalization_path)
    cfg = load_config_bundle(config_dir)
    root = _prepare_new_output_root(output_dir)
    engine = ScanEngine(
        image_size=scene_size,
        pixel_size_mm=pixel_size_mm,
        render_model_mode=render_model_mode,
        emission_b_mode=emission_b_mode,
        emission_b_path=emission_b_path,
        render_view_mode="top_surface",
        weld_layout=WeldLayoutRuntime(mode=weld_layout_mode),
    )
    provenance = _freeze_generation_provenance(
        root,
        config_dir=config_dir,
        engine=engine,
        seed=seed,
        pixel_size_mm=pixel_size_mm,
        generation_parameters={
            "dataset_kind": "sequential",
            "fixed_energy_kev": fixed_energy_kev,
            "prediction_task": "unseen_region",
            "sequences_per_class": int(sequences_per_class),
            "frames_per_sequence": int(frames_per_sequence),
            "frame_size": int(frame_size),
            "stride_px": int(stride_px),
            "context_radius": int(context_radius),
            "weld_layout_mode": weld_layout_mode,
            "noise_profile": noise_profile,
            "normalization_path": (
                None if frozen_normalization is None else str(frozen_normalization)
            ),
        },
    )
    controlled_slice = default_slice_runtime(cfg.slice)
    records: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    resampled_degenerate_count = 0
    maximum_generation_attempt = 1
    for class_index, defect_type in enumerate(DATASET_CLASSES):
        for repetition in range(sequences_per_class):
            sequence_number = class_index * sequences_per_class + repetition
            sequence_id = f"sequence-{sequence_number:04d}"
            split = _split_for_repetition(repetition, sequences_per_class)
            base_sequence_seed = int(seed + 10000 * class_index + repetition)
            regime = _regime_for_sample(repetition, class_index)
            planned_target_frame_index = (
                None
                if defect_type == "clean"
                else _planned_target_frame(
                    repetition,
                    class_index,
                    frames_per_sequence=frames_per_sequence,
                    context_radius=context_radius,
                )
            )
            planned_target_x_mm = (
                0.0
                if planned_target_frame_index is None
                else (
                    planned_target_frame_index
                    - 0.5 * (frames_per_sequence - 1)
                )
                * stride_px
                * pixel_size_mm
            )
            planned_defect_types = [defect_type] if defect_type != "clean" else []
            if defect_type != "clean" and repetition % 3 == 2:
                primary_index = CORE_DEFECT_CLASSES.index(defect_type)
                planned_defect_types.append(
                    CORE_DEFECT_CLASSES[(primary_index + 1) % len(CORE_DEFECT_CLASSES)]
                )
            y0 = (scene_size - frame_size) // 2
            y1 = y0 + frame_size
            for attempt in range(25):
                sequence_seed = int(base_sequence_seed + 1_000_000 * attempt)
                rng = np.random.default_rng(sequence_seed)
                sampled = sample_parameters(
                    cfg,
                    seed=sequence_seed,
                    max_defects=0,
                    randomize_defect_location=True,
                )
                sampled = replace(
                    _set_regime(_apply_noise_profile(sampled, noise_profile), regime, fixed_energy_kev),
                    slice=replace(
                        controlled_slice,
                        slice_offset_mm=float(planned_target_x_mm),
                    ),
                    defects=[
                        defect
                        for planned_type in planned_defect_types
                        for defect in _sample_defect(planned_type, rng)
                    ],
                )
                result = engine.run(sampled)
                full_visible_types = [d.defect_type for d in result.defects]
                visible_ok = all(
                    planned_type in full_visible_types
                    for planned_type in planned_defect_types
                )
                crops = [
                    np.asarray(
                        result.channels.topo[
                            y0:y1,
                            frame_index * stride_px:frame_index * stride_px + frame_size,
                        ],
                        dtype=np.float32,
                    )
                    for frame_index in range(frames_per_sequence)
                ]
                mask_crops = [
                    result.defect_mask[
                        y0:y1,
                        frame_index * stride_px:frame_index * stride_px + frame_size,
                    ]
                    for frame_index in range(frames_per_sequence)
                ]
                target_visible_ok = (
                    defect_type == "clean"
                    or planned_target_frame_index is not None
                    and np.any(mask_crops[planned_target_frame_index])
                )
                signal_ok = all(
                    np.all(np.isfinite(crop))
                    and float(np.max(crop) - np.min(crop)) > 1e-12
                    for crop in crops
                )
                if visible_ok and target_visible_ok and signal_ok:
                    break
            else:
                raise RuntimeError(
                    f"Unable to generate a non-degenerate surface-visible sequence for "
                    f"'{defect_type}' after 25 attempts from base seed {base_sequence_seed}."
                )
            generation_attempt = attempt + 1
            maximum_generation_attempt = max(maximum_generation_attempt, generation_attempt)
            if generation_attempt > 1:
                resampled_degenerate_count += 1
            defect_frame_count = 0
            b_mode, b_value = _emission_fields(result.metadata)
            for frame_index in range(frames_per_sequence):
                x0 = int(frame_index * stride_px)
                x1 = x0 + frame_size
                topo_crop = crops[frame_index]
                mask_crop = mask_crops[frame_index]
                frame_types = [
                    defect.defect_type
                    for defect in result.defects
                    if np.any(defect.mask[y0:y1, x0:x1])
                ]
                if frame_types:
                    defect_frame_count += 1
                sample_id = f"{sequence_id}-frame-{frame_index:04d}"
                raw_topo_path, image_topo_path = _save_raw_topo(
                    root, split, sample_id, topo_crop
                )
                mask_path = save_mask(root, split, sample_id, mask_crop)
                records.append(
                    {
                        "contract_version": CONTRACT_VERSION,
                        "sample_id": sample_id,
                        "sequence_id": sequence_id,
                        "frame_index": frame_index,
                        "split": split,
                        "channels": {"topo": _relative(image_topo_path, root)},
                        "raw_channels": {"topo": _relative(raw_topo_path, root)},
                        "mask_path": _relative(mask_path, root),
                        "defect_type": (
                            "clean"
                            if not frame_types
                            else frame_types[0]
                            if len(frame_types) == 1
                            else "multiple"
                        ),
                        "defect_types": frame_types,
                        "is_defective": bool(frame_types),
                        "seed": sequence_seed,
                        "base_seed": base_sequence_seed,
                        "generation_attempt": generation_attempt,
                        "raw_topo_dynamic_range": float(
                            np.max(topo_crop) - np.min(topo_crop)
                        ),
                        "beam_regime": regime,
                        "emission_b_mode": b_mode,
                        "emission_b_value": b_value,
                        "emission_b_source_path": (result.metadata.get("emission_b") or {}).get("source_path"),
                        "emission_b_source_sha256": (result.metadata.get("emission_b") or {}).get("source_sha256"),
                        "calibration_source": result.metadata.get("calibration_source"),
                        "calibration_sha256": result.metadata.get("calibration_sha256"),
                        "render_view_mode": "top_surface",
                        "pixel_size_mm": float(result.metadata["effective_pixel_size_mm"]),
                        "noise_enabled": noise_profile != "none",
                        "noise_profile": noise_profile,
                        "planned_target_frame_index": planned_target_frame_index,
                        "planned_defect_parameters": list(sampled.defects),
                    "realized_non_noise_parameters": {key: value for key, value in asdict(sampled).items() if key != "noise"},
                    "renderer_parameters": {"image_size": engine.image_size, "pixel_size_mm": engine.pixel_size_mm,
                                            "weld_layout": asdict(engine.weld_layout), "render_model_mode": engine.render_model_mode,
                                            "emission_b_mode": engine.emission_b_mode, "render_view_mode": engine.render_view_mode},
                        "source_scene_size_px": scene_size,
                        "crop_x0_px": x0,
                        "crop_y0_px": y0,
                    }
                )
            sequence_rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": split,
                    "planned_class": defect_type,
                    "planned_defect_types": planned_defect_types,
                    "seed": sequence_seed,
                    "base_seed": base_sequence_seed,
                    "generation_attempt": generation_attempt,
                    "beam_regime": regime,
                    "frames": frames_per_sequence,
                    "defect_frames": defect_frame_count,
                    "planned_target_frame_index": planned_target_frame_index,
                    "planned_target_x_mm": float(planned_target_x_mm),
                }
            )
            sequence_records = [
                record for record in records if record["sequence_id"] == sequence_id
            ]
            sequence_records.sort(key=lambda record: int(record["frame_index"]))
            for center in range(context_radius, frames_per_sequence - context_radius):
                previous = sequence_records[center - context_radius:center]
                future = sequence_records[center + 1:center + context_radius + 1]
                target = sequence_records[center]
                window_rows.append(
                    {
                        "window_id": f"{sequence_id}-window-{center:04d}",
                        "sequence_id": sequence_id,
                        "split": split,
                        "target_sample_id": target["sample_id"],
                        "target_frame_index": center,
                        "previous_sample_ids": [record["sample_id"] for record in previous],
                        "future_sample_ids": [record["sample_id"] for record in future],
                        "target_is_defective": bool(target["is_defective"]),
                    }
                )

    normalization = _apply_training_global_topo_normalisation(
        root, records, frozen_normalization_path=frozen_normalization
    )
    artifacts = _write_contract_bundle(
        root,
        records,
        dataset_kind="sequential",
        extra_descriptor={
            "classes": list(DATASET_CLASSES),
            "sequences_per_class": int(sequences_per_class),
            "frames_per_sequence": int(frames_per_sequence),
            "prediction_task": "unseen_region",
            "frame_size_px": int(frame_size),
            "stride_px": int(stride_px),
            "context_radius": int(context_radius),
            "noise_profile": noise_profile,
            "noise_policy": noise_profile,
            "provenance": "generation_provenance.json",
            "windows": "windows.jsonl",
            "sequence_manifest": "sequence_manifest.json",
            "source_scene_size_px": scene_size,
            "sequence_semantics": (
                "non-overlapping adjacent crops from one coherent rendered weld surface"
                if stride_px >= frame_size
                else "overlapping crops from one coherent rendered weld surface"
            ),
            "context_overlap_policy": (
                "non_overlapping" if stride_px >= frame_size else "overlapping_target_leakage_risk"
            ),
            "weld_layout_mode": str(weld_layout_mode),
            "normalization": "topo_normalization.json",
        },
    )
    sequence_manifest_path = root / "sequence_manifest.json"
    sequence_manifest_path.write_text(json.dumps(sequence_rows, indent=2), encoding="utf-8")
    windows_path = root / "windows.jsonl"
    windows_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in window_rows),
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "output_dir": str(root),
        "record_count": len(records),
        "sequence_count": len(sequence_rows),
        "frames_per_sequence": frames_per_sequence,
        "sequence_manifest": str(sequence_manifest_path),
        "windows": str(windows_path),
        "window_count": len(window_rows),
        "defect_frame_count": int(sum(row["defect_frames"] for row in sequence_rows)),
        "resampled_degenerate_count": resampled_degenerate_count,
        "maximum_generation_attempt": maximum_generation_attempt,
        "normalization": normalization,
        "provenance": provenance,
        **artifacts,
    }
    (root / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
