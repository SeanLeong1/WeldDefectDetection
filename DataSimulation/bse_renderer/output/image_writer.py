from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..utils import apply_display_map, apply_sem_tone_curve, ensure_dir


def _save_u16(path: Path, arr: np.ndarray) -> None:
    data = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((data * 65535).astype(np.uint16))
    img.save(path)


def _save_u8(path: Path, arr: np.ndarray) -> None:
    data = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((data * 255).astype(np.uint8), mode="L")
    img.save(path)


def save_channel_images(
    output_root: Path,
    split: str,
    stem: str,
    compo: np.ndarray,
    topo: np.ndarray,
    tx: np.ndarray | None = None,
    ty: np.ndarray | None = None,
    display_mapping: dict[str, dict[str, float | str]] | None = None,
    save_u8_preview: bool = True,
) -> dict[str, str]:
    compo_dir = output_root / "images" / "compo" / split
    topo_dir = output_root / "images" / "topo" / split
    tx_dir = output_root / "images" / "tx" / split
    ty_dir = output_root / "images" / "ty" / split
    ensure_dir(compo_dir)
    ensure_dir(topo_dir)
    if tx is not None:
        ensure_dir(tx_dir)
    if ty is not None:
        ensure_dir(ty_dir)

    mapping = display_mapping or {}
    compo_disp = apply_sem_tone_curve(
        apply_display_map(compo, mapping.get("compo", {"mode": "linear", "low": 0.0, "high": 1.0}))
    )
    topo_disp = apply_sem_tone_curve(
        apply_display_map(topo, mapping.get("topo", {"mode": "linear", "low": 0.0, "high": 1.0}))
    )

    compo_u16 = compo_dir / f"{stem}.png"
    topo_u16 = topo_dir / f"{stem}.png"
    _save_u16(compo_u16, compo_disp)
    _save_u16(topo_u16, topo_disp)

    out = {
        "compo_u16": str(compo_u16),
        "topo_u16": str(topo_u16),
    }
    if tx is not None:
        tx_u16 = tx_dir / f"{stem}.png"
        _save_u16(
            tx_u16,
            apply_sem_tone_curve(
                apply_display_map(tx, mapping.get("tx", {"mode": "signed", "center": 0.0, "span": 1.0}))
            ),
        )
        out["tx_u16"] = str(tx_u16)
    if ty is not None:
        ty_u16 = ty_dir / f"{stem}.png"
        _save_u16(
            ty_u16,
            apply_sem_tone_curve(
                apply_display_map(ty, mapping.get("ty", {"mode": "signed", "center": 0.0, "span": 1.0}))
            ),
        )
        out["ty_u16"] = str(ty_u16)
    if save_u8_preview:
        compo_u8_dir = output_root / "images_8bit" / "compo" / split
        topo_u8_dir = output_root / "images_8bit" / "topo" / split
        tx_u8_dir = output_root / "images_8bit" / "tx" / split
        ty_u8_dir = output_root / "images_8bit" / "ty" / split
        ensure_dir(compo_u8_dir)
        ensure_dir(topo_u8_dir)
        if tx is not None:
            ensure_dir(tx_u8_dir)
        if ty is not None:
            ensure_dir(ty_u8_dir)
        compo_u8 = compo_u8_dir / f"{stem}.png"
        topo_u8 = topo_u8_dir / f"{stem}.png"
        _save_u8(compo_u8, compo_disp)
        _save_u8(topo_u8, topo_disp)
        out["compo_u8"] = str(compo_u8)
        out["topo_u8"] = str(topo_u8)
        if tx is not None:
            tx_u8 = tx_u8_dir / f"{stem}.png"
            _save_u8(
                tx_u8,
                apply_sem_tone_curve(
                    apply_display_map(tx, mapping.get("tx", {"mode": "signed", "center": 0.0, "span": 1.0}))
                ),
            )
            out["tx_u8"] = str(tx_u8)
        if ty is not None:
            ty_u8 = ty_u8_dir / f"{stem}.png"
            _save_u8(
                ty_u8,
                apply_sem_tone_curve(
                    apply_display_map(ty, mapping.get("ty", {"mode": "signed", "center": 0.0, "span": 1.0}))
                ),
            )
            out["ty_u8"] = str(ty_u8)
    return out


def save_mask(output_root: Path, split: str, stem: str, mask: np.ndarray) -> str:
    mask_dir = output_root / "masks" / split
    ensure_dir(mask_dir)
    path = mask_dir / f"{stem}.png"
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img.save(path)
    return str(path)
