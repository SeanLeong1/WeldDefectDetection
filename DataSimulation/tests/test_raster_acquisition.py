from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from bse_renderer.noise.raster_acquisition import (
    AcquisitionProfile, acquire, stream, stationary_ou, pixel_times, motion_coordinates, warp_image,
)
from bse_renderer.renderer.image_formation import form_channels


def flags(shot=False, electronic=False, contamination=False):
    return SimpleNamespace(poisson_enabled=shot, thermal_enabled=electronic,
                           contamination_enabled=contamination, quadrant_offsets={})


def test_raster_clock_and_motion_keep_signal_and_mask_aligned():
    p = replace(AcquisitionProfile(), flyback_us=0, velocity_x_mm_s=1000.)
    t, line = pixel_times((8, 8), 125, 0)
    assert line == .001
    assert t[1, 0] == .001
    coords = motion_coordinates((8, 8), 125, 1., p)
    signal = np.zeros((8, 8), np.float32); signal[:, 2:5] = 1.
    moved = warp_image(signal, coords, order=0)
    label = warp_image(signal.astype(bool), coords, order=0)
    assert np.array_equal(moved > 0, label)
    assert not np.array_equal(moved, signal)
    assert motion_coordinates((8, 8), 8, .05, AcquisitionProfile()) is None


def test_stationary_drift_has_bounded_variance_and_expected_correlation():
    x = stationary_ou(200000, .02, .1, stream(5, 'test'))
    assert abs(x.mean()) < .03
    assert .94 < x.var() < 1.06
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(np.exp(-.2), abs=.008)
    assert .90 < x[:20000].var() / x[-20000:].var() < 1.10


def test_counting_statistics_scale_with_dwell():
    q = np.full((256, 256, 4), .2)
    p = AcquisitionProfile()
    a, _ = acquire(q, flags(shot=True), seed=10, dwell_us=8, profile=p)
    b, _ = acquire(q, flags(shot=True), seed=10, dwell_us=32, profile=p)
    assert a.mean() == pytest.approx(.2, abs=.0001)
    assert np.var(a) == pytest.approx(.2/(1e9*8e-6), rel=.025)
    assert np.var(a)/np.var(b) == pytest.approx(4., rel=.025)


def test_common_mode_cancels_and_independent_electronics_do_not_move_geometry():
    q = np.full((128, 128, 4), .2)
    p = replace(AcquisitionProfile(), baseline_rms=0, readout_rms=0)
    out, meta = acquire(q, flags(electronic=True), seed=4, dwell_us=8, profile=p)
    assert np.max(form_channels(out).topo) == 0
    assert meta['calibrated'] is False
    assert meta['version'] == 'raster_v2'


def test_geometry_random_draws_and_shot_toggle_do_not_change_electronic_streams():
    q = np.full((64, 64, 4), .2)
    p = AcquisitionProfile()
    a, _ = acquire(q, flags(electronic=True), seed=33, dwell_us=8, profile=p)
    np.random.default_rng(33).normal(size=100000)
    acquire(q, flags(shot=True, electronic=True), seed=33, dwell_us=8, profile=p)
    b, _ = acquire(q, flags(electronic=True), seed=33, dwell_us=8, profile=p)
    assert np.array_equal(a, b)
    changed, _ = acquire(q+.1, flags(electronic=True), seed=33, dwell_us=8,
                         profile=replace(p, common_gain_rms=0))
    base, _ = acquire(q, flags(electronic=True), seed=33, dwell_us=8,
                      profile=replace(p, common_gain_rms=0))
    assert np.allclose(changed-base, .1, atol=3e-8)


@pytest.mark.parametrize('kwargs', [{'effective_count_rate': 0}, {'readout_rms': -1},
                                   {'velocity_x_mm_s': float('nan')}, {'correlation_time_s': 0}])
def test_invalid_profiles_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(AcquisitionProfile(), **kwargs).validate()


def test_disabled_acquisition_is_identity():
    q = np.random.default_rng(7).uniform(.01, .3, (16, 24, 4)).astype(np.float32)
    a, _ = acquire(q, flags(), seed=1, dwell_us=8, profile=AcquisitionProfile())
    assert np.array_equal(a, q)
