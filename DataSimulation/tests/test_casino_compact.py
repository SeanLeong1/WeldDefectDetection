from pathlib import Path
import math

import numpy as np
import pytest

from bse_renderer.validation.casino_compact import (
    compact_summary, select_surface_exits, fit_scalar_b, _histogram, analyze_samples,
)
from bse_renderer.validation.casino_emission_fit import _parse_summary_file


def _row(position, direction, energy=40):
    values = [*position, *direction, 10, energy]
    return "\t".join(f"{x:.10f}" for x in values) + "\tRegion\t1"


def _summary(path, tilts=(20, 70), intermediate=0, off_face=False):
    with path.open("w", encoding="utf-8") as handle:
        for block, tilt in enumerate(tilts):
            handle.write(f'"Scan Point {block} Results : "\n')
            alpha = np.radians(tilt)
            normal = np.array([np.sin(alpha), 0, -np.cos(alpha)])
            tangent = np.array([np.cos(alpha), 0, np.sin(alpha)])
            second = np.cross(normal, tangent)
            for i in range(64):
                first = (i % 8 - 3.5) * tangent + (i // 8 - 3.5) * second
                az = 0 if i % 2 else np.pi
                direction = .5 * normal + np.sqrt(.75) * (np.cos(az) * tangent + np.sin(az) * second)
                kind = "Transmitted" if direction[2] > 0 else "Backscattered"
                handle.write(f'"Trajectory"\t{i}\n"Type : "\n\t{kind}\n')
                handle.write(_row(first, [0, 0, 1], 60) + "\n")
                for j in range(intermediate):
                    handle.write(_row(first - normal * (j + 1), [0, 0, 1]) + "\n")
                final = first + (i - 32) * 100 * tangent
                if off_face and i == 0:
                    final = final - normal * 1000
                handle.write(_row(final, direction) + "\n")


def test_scan_point_frames_and_global_transmitted_recovery(tmp_path):
    source, cache = tmp_path / "summary.dat", tmp_path / "compact.npz"
    _summary(source, off_face=True)
    meta = compact_summary(source, cache)
    samples, blocks = select_surface_exits(cache, 70, 1)
    assert meta["trajectory_count"] == 128
    assert blocks[0]["status"] == "excluded_landing_patch"
    assert blocks[0]["measured_tilt_deg"] == pytest.approx(20, abs=1e-6)
    assert blocks[1]["measured_tilt_deg"] == pytest.approx(70, abs=1e-6)
    assert blocks[1]["recovered_global_transmitted"] == 32
    assert blocks[1]["off_face_exits"] == 1
    assert len(samples) == 63
    assert np.all(samples[:, 1] > 0)


def test_rounded_small_spot_refinement_recovers_flat_exits_but_not_other_face():
    from bse_renderer.validation.casino_compact import surface_frame, _plane
    alpha=np.radians(7.5)
    normal=np.array([np.sin(alpha),0,-np.cos(alpha)])
    tangent=np.array([np.cos(alpha),0,np.sin(alpha)])
    origin=np.array([12500*np.sin(alpha),0,-12500*np.cos(alpha)])
    rng=np.random.default_rng(18)
    first=np.zeros((500,8))
    first[:,:3]=(origin+rng.normal(0,1.8,(500,1))*tangent
                 +rng.normal(0,1.8,(500,1))*np.array([0,1,0]))
    def round_export(points):
        return np.array([[float(f'{v:.6g}') for v in row] for row in points])
    first[:,:3]=round_export(first[:,:3])
    first[:,5]=1
    first[:,6]=60
    final=first.copy()
    final[:116,:3]=round_export(origin+rng.normal(0,1500,(116,1))*tangent
                               +rng.normal(0,1500,(116,1))*np.array([0,1,0]))
    final[116,:3]=origin+1000*normal
    has_exit=np.arange(500)<117
    _,seed_normal,_=_plane(first[:,:3],np.array([0,0,1]))
    assert np.degrees(np.arccos(seed_normal@normal))>.1
    center,fitted,_,_,diagnostic=surface_frame(first,final,has_exit)
    assert .1<diagnostic['refinement_max_normal_change_deg']<=.5
    assert diagnostic['refinement_endpoint_count']==116
    assert np.abs((final[:116,:3]-center)@fitted).max()<.1
    assert abs((final[116,:3]-center)@fitted)>999
    assert diagnostic['measured_tilt_deg']==pytest.approx(7.5,abs=.001)


def test_streaming_keeps_last_row_and_never_reads_whole_file(tmp_path, monkeypatch):
    source, cache = tmp_path / "summary.dat", tmp_path / "compact.npz"
    _summary(source, tilts=(30,), intermediate=300)
    def forbid(*args, **kwargs):
        raise AssertionError("Whole-file read is forbidden for trajectory exports")
    monkeypatch.setattr(Path, "read_text", forbid)
    meta = compact_summary(source, cache)
    assert meta["max_rows_per_history"] == 302
    with np.load(cache, allow_pickle=False) as data:
        assert data["first"].shape == (64, 8)
        assert data["final"].shape == (64, 8)
        assert np.all(data["final"][:, 6] == 40)
    old_samples, _ = _parse_summary_file(source, "30deg-r1", 30, 1)
    assert all(s.exit_energy_kev == 40 for s in old_samples)


def test_cache_reuse_checks_source_content(tmp_path):
    source, cache = tmp_path / "summary.dat", tmp_path / "compact.npz"
    _summary(source, tilts=(30,))
    before = compact_summary(source, cache)
    timestamp = cache.stat().st_mtime_ns
    assert compact_summary(source, cache) == before
    assert cache.stat().st_mtime_ns == timestamp
    with source.open("a") as handle:
        handle.write('\n"extra metadata"\t1\n')
    after = compact_summary(source, cache)
    assert after["sha256"] != before["sha256"]


def test_scalar_fit_rejects_invalid_domain_and_zero_tilt_is_unidentifiable():
    sample = np.array([[0, .5, 0, 1, 0], [0, .8, 1, 1, 0]], dtype=float)
    assert fit_scalar_b(sample)["b"] is None
    sample[0, 1] = -.1
    with pytest.raises(ValueError, match="outward"):
        fit_scalar_b(sample)


def test_fit_recovers_known_density_and_histogram_mass():
    rng = np.random.default_rng(731)
    alpha, expected = np.radians(45), .6
    mu = np.sqrt(rng.random(80000))
    az = rng.uniform(-np.pi, np.pi, len(mu))
    bracket = 1 + expected * np.sin(alpha) * np.sqrt(1 - mu * mu) * np.cos(az)
    keep = rng.random(len(mu)) < bracket / (1 + abs(expected * np.sin(alpha)))
    samples = np.column_stack([np.full(keep.sum(), alpha), mu[keep], az[keep],
                               np.ones(keep.sum()), np.full(keep.sum(), 45)])
    fit = fit_scalar_b(samples)
    assert fit["b"] == pytest.approx(expected, abs=.04)
    assert fit["max_abs_b_sin_alpha"] < 1
    mass = _histogram(samples)
    assert mass.shape == (12, 24)
    assert np.all(mass > 0)
    assert math.isclose(float(mass.sum()), 1, abs_tol=1e-12)


def test_single_run_reports_missing_holdouts_without_inventing_validation():
    rng = np.random.default_rng(73)
    sample = np.column_stack([np.zeros(500), np.sqrt(rng.random(500)),
                              rng.uniform(-np.pi, np.pi, 500), np.ones(500), np.zeros(500)])
    report = analyze_samples(sample)
    assert report["model_holdouts"] == []
    assert report["repeat_holdouts"] == []
    assert report["per_tilt"][0]["b"] is None


def test_reported_count_mismatch_excludes_incomplete_export(tmp_path):
    source, cache = tmp_path / "summary.dat", tmp_path / "compact.npz"
    _summary(source, tilts=(30,))
    with source.open("a") as handle:
        handle.write('\n"nb backscattered"\t9999\n')
    compact_summary(source, cache)
    samples, blocks = select_surface_exits(cache, 30, 1)
    assert len(samples) == 0
    assert blocks[0]["count_status"] == "mismatch"
