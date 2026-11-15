from __future__ import annotations

from bse_renderer.validation.emission_b_render_comparison import (
    _per_tilt_b_rows,
    _winner_from_combined_delta,
)


def test_per_tilt_b_rows_indexes_by_nominal_tilt():
    report = {
        "fit": {
            "per_tilt": [
                {"nominal_tilt_deg": 30.0, "fitted_b": 0.25},
                {"nominal_tilt_deg": 40.0, "fitted_b": -0.1},
            ]
        }
    }

    rows = _per_tilt_b_rows(report)

    assert sorted(rows) == [30, 40]
    assert rows[30]["fitted_b"] == 0.25
    assert rows[40]["fitted_b"] == -0.1


def test_winner_from_combined_delta_prefers_lower_delta():
    assert _winner_from_combined_delta(0.4, 0.2) == "quadratic"
    assert _winner_from_combined_delta(0.2, 0.4) == "pooled_constant"
    assert _winner_from_combined_delta(0.3, 0.3) == "tie"
