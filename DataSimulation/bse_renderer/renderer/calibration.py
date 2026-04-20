from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChannelCalibration:
    scale: float = 1.0
    bias: float = 0.0


@dataclass(frozen=True)
class RendererCalibration:
    compo: ChannelCalibration
    topo: ChannelCalibration
    source: str | None = None

    @staticmethod
    def identity(source: str | None = None) -> "RendererCalibration":
        return RendererCalibration(
            compo=ChannelCalibration(1.0, 0.0),
            topo=ChannelCalibration(1.0, 0.0),
            source=source,
        )


def load_renderer_calibration(path: str | Path | None) -> RendererCalibration:
    if path is None:
        return RendererCalibration.identity(source=None)
    p = Path(path).resolve()
    if not p.exists():
        return RendererCalibration.identity(source=str(p))
    raw = json.loads(p.read_text(encoding="utf-8"))
    compo = raw.get("compo", {})
    topo = raw.get("topo", {})
    return RendererCalibration(
        compo=ChannelCalibration(
            scale=float(compo.get("scale", 1.0)),
            bias=float(compo.get("bias", 0.0)),
        ),
        topo=ChannelCalibration(
            scale=float(topo.get("scale", 1.0)),
            bias=float(topo.get("bias", 0.0)),
        ),
        source=str(p),
    )


def save_renderer_calibration(
    path: str | Path,
    *,
    compo_scale: float,
    compo_bias: float,
    topo_scale: float,
    topo_bias: float,
    metadata: dict[str, Any] | None = None,
) -> str:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "compo": {"scale": float(compo_scale), "bias": float(compo_bias)},
        "topo": {"scale": float(topo_scale), "bias": float(topo_bias)},
    }
    if metadata:
        payload["metadata"] = metadata
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(p)
