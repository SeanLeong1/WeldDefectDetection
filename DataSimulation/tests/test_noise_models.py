import numpy as np

from bse_renderer.noise.contamination_model import apply_contamination
from bse_renderer.noise.poisson_noise import apply_poisson_noise
from bse_renderer.noise.thermal_drift import apply_thermal_drift


def test_poisson_noise_shape():
    rng = np.random.default_rng(0)
    s = np.full((20, 20), 0.5, dtype=np.float32)
    out = apply_poisson_noise(s, beam_current_ma=300, dwell_time_us=1.0, rng=rng)
    assert out.shape == s.shape
    assert np.all(out >= 0.0)


def test_thermal_drift_changes_signal():
    rng = np.random.default_rng(1)
    s = np.zeros((20, 20), dtype=np.float32)
    out = apply_thermal_drift(
        s, temperature_k=300.0, bandwidth_hz=5000.0, drift_rate=0.1, rng=rng
    )
    assert not np.allclose(s, out)
    assert float(np.std(out)) < 0.05


def test_contamination_quadrants():
    q = np.ones((10, 10, 4), dtype=np.float32)
    out = apply_contamination(q, contamination_rate=0.2, quadrant_offsets={"A": 0.1, "B": 0, "C": 0, "D": 0})
    assert out.shape == q.shape
    assert out[..., 0].mean() != out[..., 1].mean()


def test_session_progress_and_mismatch_shift_quadrants_directionally():
    q = np.ones((32, 32, 4), dtype=np.float32)
    out = apply_contamination(
        q,
        contamination_rate=0.15,
        quadrant_offsets={"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
        session_progress=1.0,
        session_mismatch={"A": 0.10, "B": -0.10, "C": 0.0, "D": 0.0},
    )
    assert float(out[..., 0].mean()) > float(out[..., 1].mean())


def test_thermal_session_baseline_accumulates_and_resets():
    base = np.zeros((32, 32), dtype=np.float32)
    seed = 7
    rng0 = np.random.default_rng(seed)
    rng1 = np.random.default_rng(seed)
    early = apply_thermal_drift(
        base,
        temperature_k=300.0,
        bandwidth_hz=5000.0,
        drift_rate=0.1,
        rng=rng0,
        session_baseline=0.02,
        session_progress=0.0,
    )
    late = apply_thermal_drift(
        base,
        temperature_k=300.0,
        bandwidth_hz=5000.0,
        drift_rate=0.1,
        rng=rng1,
        session_baseline=0.02,
        session_progress=1.0,
    )
    assert float(late.mean()) > float(early.mean()) + 0.01
