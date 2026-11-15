from __future__ import annotations

import numpy as np

from ..models import RenderChannels


def form_channels(quadrants: np.ndarray) -> RenderChannels:
    a = quadrants[..., 0]
    b = quadrants[..., 1]
    c = quadrants[..., 2]
    d = quadrants[..., 3]

    compo = a + b + c + d
    tx = (a + d) - (b + c)
    ty = (a + b) - (c + d)
    tmag = np.sqrt(tx**2 + ty**2)
    denom = np.maximum(a + c, 1e-8)
    rx = (a - c) / denom
    topo = tmag
    return RenderChannels(
        compo=compo.astype(np.float32),
        topo=topo.astype(np.float32),
        tx=tx.astype(np.float32),
        ty=ty.astype(np.float32),
        tmag=tmag.astype(np.float32),
        rx=rx.astype(np.float32),
    )
