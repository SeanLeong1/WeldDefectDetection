from __future__ import annotations

import numpy as np

from ..models import DetectorRuntime, MaterialPreset
from ..physics.bse_yield import eta_oblique, eta_z
from ..physics.emission_model import emission_distribution


def _wrap_pi(angle_rad: np.ndarray) -> np.ndarray:
    return ((angle_rad + np.pi) % (2.0 * np.pi)) - np.pi


def _gauss_legendre_interval(lower: float, upper: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 2:
        raise ValueError("Quadrature order must be at least 2.")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    return midpoint + half_width * nodes, half_width * weights


def _integrated_quadrant_directional_term(
    normals: np.ndarray,
    alpha: np.ndarray,
    tilt_azimuth: np.ndarray,
    center_rad: float,
    phi_inner: float,
    phi_outer: float,
    b_coeff: float | np.ndarray,
    material_n: float,
    directional_gain: float,
    order_phi: int,
    order_varphi: int,
    angular_density=None,
) -> np.ndarray:
    """Integrate the hybrid directional response over one annular quadrant."""
    phi_nodes, phi_weights = _gauss_legendre_interval(phi_inner, phi_outer, order_phi)
    azimuth_nodes, azimuth_weights = _gauss_legendre_interval(
        center_rad - np.pi / 4.0,
        center_rad + np.pi / 4.0,
        order_varphi,
    )
    phi = phi_nodes[:, np.newaxis, np.newaxis, np.newaxis]
    azimuth = azimuth_nodes[np.newaxis, :, np.newaxis, np.newaxis]
    alpha_grid = alpha[np.newaxis, np.newaxis, ...]
    b_grid = np.asarray(b_coeff, dtype=np.float64)
    if b_grid.ndim > 0:
        b_grid = b_grid[np.newaxis, np.newaxis, ...]

    detector_x = np.cos(azimuth) * np.sin(phi)
    detector_y = np.sin(azimuth) * np.sin(phi)
    detector_z = np.cos(phi)
    local_mu = (
        normals[np.newaxis, np.newaxis, ..., 0] * detector_x
        + normals[np.newaxis, np.newaxis, ..., 1] * detector_y
        + normals[np.newaxis, np.newaxis, ..., 2] * detector_z
    )
    # Match the CASINO frame: incident beam b=(0,0,-1), t1 is its
    # projection onto the tangent plane and t2=n cross t1.
    tangent = normals * normals[..., 2, None]
    tangent[..., 2] -= 1.0
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = np.divide(tangent, length, out=np.zeros_like(tangent), where=length > 1e-10)
    tangent = np.where(length > 1e-10, tangent, np.array([1.0, 0.0, 0.0]))
    tangent2 = np.cross(normals, tangent)
    def project(axis):
        return (axis[None, None, ..., 0] * detector_x
                + axis[None, None, ..., 1] * detector_y
                + axis[None, None, ..., 2] * detector_z)
    local_azimuth = np.arctan2(project(tangent2), project(tangent))
    view_term = np.clip(local_mu, 0.0, 1.0)
    emission = (emission_distribution(np.arccos(view_term), local_azimuth, alpha_grid, b_grid)
                if angular_density is None else np.pi*angular_density(view_term, local_azimuth, np.degrees(alpha_grid)))
    gain = float(np.clip(directional_gain, 0.2, 2.5))
    directional = (1.0 - gain) + gain * (
        0.58 * emission + 0.42 * np.power(view_term, max(material_n, 1.0))
    )
    weights = (
        phi_weights[:, np.newaxis]
        * azimuth_weights[np.newaxis, :]
        * np.sin(phi_nodes)[:, np.newaxis]
    )
    return np.sum(
        np.clip(directional, 0.05, None)
        * (local_mu > 0.0)
        * weights[..., np.newaxis, np.newaxis],
        axis=(0, 1),
    )


def compute_quadrant_signals(
    normals: np.ndarray,
    detector: DetectorRuntime,
    material: MaterialPreset,
    material_z_map: np.ndarray,
    yield_modifier_map: np.ndarray | None = None,
    openness_map: np.ndarray | None = None,
    directional_gain: float = 1.0,
    yield_scale: float | np.ndarray = 1.0,
    obliquity_exponent_scale: float = 1.0,
    b_coeff: float | np.ndarray | None = None,
    quadrature_order_phi: int = 6,
    quadrature_order_varphi: int = 8,
    angular_density=None,
) -> np.ndarray:
    """Return raw quadrant channels with shape (H, W, 4)."""
    normals = np.asarray(normals, dtype=np.float64)
    if normals.ndim != 3 or normals.shape[-1] != 3 or not np.all(np.isfinite(normals)):
        raise ValueError("Normals must be a finite (H, W, 3) array.")
    if not np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-5):
        raise ValueError("Surface normals must have unit length.")
    if np.any(normals[..., 2] < 0):
        raise ValueError("The surface must face the incident beam (normal z >= 0).")
    if not (np.isfinite([detector.height_mm, detector.inner_radius_mm, detector.outer_radius_mm]).all()
            and detector.height_mm > 0 and 0 <= detector.inner_radius_mm < detector.outer_radius_mm):
        raise ValueError("Detector requires positive height and 0 <= inner radius < outer radius.")
    nz = np.clip(normals[..., 2], -1.0, 1.0)
    alpha = np.arccos(np.clip(nz, 0.0, 1.0))
    tilt_azimuth = np.arctan2(normals[..., 1], normals[..., 0] + 1e-12)
    eta0 = eta_z(material_z_map)
    eta = eta_oblique(
        eta0,
        alpha,
        np.maximum(material_z_map, 1e-3),
        exponent_scale=obliquity_exponent_scale,
    )
    yield_modifier = (
        np.ones_like(eta, dtype=np.float32)
        if yield_modifier_map is None
        else np.clip(yield_modifier_map.astype(np.float32), 0.15, 1.20)
    )
    openness = (
        np.zeros_like(eta, dtype=np.float32)
        if openness_map is None
        else np.clip(openness_map.astype(np.float32), 0.0, 1.0)
    )
    if np.isscalar(yield_scale):
        yield_scale_map = np.full_like(eta, float(max(float(yield_scale), 1e-6)), dtype=np.float32)
    else:
        yield_scale_map = np.clip(np.asarray(yield_scale, dtype=np.float32), 1e-6, None)
    eta = eta * yield_scale_map * yield_modifier * (1.0 - 0.35 * openness)
    b_coeff_map = material.b if b_coeff is None else b_coeff

    phi_inner = np.arctan2(detector.inner_radius_mm, max(detector.height_mm, 1e-6))
    phi_outer = np.arctan2(detector.outer_radius_mm, max(detector.height_mm, 1e-6))
    if not phi_inner < phi_outer:
        raise ValueError("Detector inner radius must be smaller than outer radius.")

    quad_order = ["A", "B", "C", "D"]
    channels = []
    for q in quad_order:
        center = np.radians(detector.quadrants_deg[q])
        integrated_directional = _integrated_quadrant_directional_term(
            normals=normals,
            alpha=alpha,
            tilt_azimuth=tilt_azimuth,
            center_rad=float(center),
            phi_inner=float(phi_inner),
            phi_outer=float(phi_outer),
            b_coeff=b_coeff_map,
            material_n=material.n,
            directional_gain=directional_gain,
            order_phi=quadrature_order_phi,
            order_varphi=quadrature_order_varphi,
            angular_density=angular_density,
        )
        sensitivity = float(detector.sensitivity.get(q, 1.0))
        if not np.isfinite(sensitivity) or sensitivity <= 0.0:
            raise ValueError(f"Detector sensitivity for quadrant {q} must be finite and positive.")
        channel = eta * integrated_directional * sensitivity
        channels.append(channel.astype(np.float32))

    return np.stack(channels, axis=-1)
