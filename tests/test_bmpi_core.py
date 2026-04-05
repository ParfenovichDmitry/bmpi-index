# -*- coding: utf-8 -*-
"""
tests/test_bmpi_core.py
=======================
Unit tests for the BMPI core computation engine.

Run: pytest tests/ -v
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmpi.bmpi_core import (
    BmpiResult,
    PressureZone,
    PRESSURE_ZONES,
    CALIB_TRAIN,
    CALIB_FULL,
    EXTREME_THRESHOLD,
    compute_bmpi,
    compute_bmpi_oos,
    compute_bmpi_series,
    get_pressure_zone,
    get_percentile,
    zone_label_series,
    zone_short_label_series,
)


class TestComputeBmpi:

    def test_returns_bmpi_result(self):
        r = compute_bmpi(312.4, -0.8834)
        assert isinstance(r, BmpiResult)

    def test_bmpi_in_unit_interval(self):
        for m in [0, 100, 312, 800, 5000]:
            for t in [-5.0, -0.9, 0.0, 1.5]:
                r = compute_bmpi(m, t)
                assert 0.0 < r.bmpi < 1.0

    def test_calibration_midpoint(self):
        """At TRAIN means the result should be ≈ 0.5."""
        r = compute_bmpi(CALIB_TRAIN["mu_mentions"], CALIB_TRAIN["mu_tone"])
        assert abs(r.bmpi - 0.5) < 0.01

    def test_train_calib_is_default(self):
        r = compute_bmpi(400, -1.0)
        assert r.calib_label == "TRAIN"

    def test_full_calib_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = compute_bmpi(400, -1.0, calib=CALIB_FULL)
            assert any("look-ahead" in str(x.message).lower() for x in w)
        assert r.calib_label == "FULL"

    def test_high_mentions_extreme_zone(self):
        r = compute_bmpi(5000, 1.0)
        assert r.zone.label == "EXTREME"
        assert r.is_extreme is True

    def test_zero_mentions_minimal_zone(self):
        r = compute_bmpi(0, -5.0)
        assert r.zone.label == "MINIMAL"

    def test_zone_contains_bmpi_value(self):
        for m in [0, 200, 400, 800, 2000]:
            for t in [-2.0, -0.5, 0.5]:
                r = compute_bmpi(m, t)
                assert r.zone.lower <= r.bmpi < r.zone.upper

    def test_z_clipped_at_3(self):
        r = compute_bmpi(999_999, 100.0)
        assert r.z_volume == pytest.approx(3.0)
        assert r.z_tone   == pytest.approx(3.0)

    def test_is_anomalous(self):
        r_high    = compute_bmpi(5000, 1.0)
        r_normal  = compute_bmpi(312,  -0.88)
        assert r_high.is_anomalous   is True
        assert r_normal.is_anomalous is False

    def test_top_pct_complement(self):
        r = compute_bmpi(400, -1.0)
        assert r.top_pct == 100 - r.percentile

    def test_str_repr(self):
        r   = compute_bmpi(500, -1.0)
        txt = str(r)
        assert "BMPI" in txt
        assert "zone" in txt.lower()

    def test_oos_uses_train(self):
        r1 = compute_bmpi_oos(500, -1.0)
        r2 = compute_bmpi(500, -1.0, calib=CALIB_TRAIN)
        assert r1.bmpi == r2.bmpi
        assert r1.calib_label == "TRAIN"


class TestPressureZones:

    def test_five_zones(self):
        assert len(PRESSURE_ZONES) == 5

    def test_zone_labels(self):
        labels = [z.label for z in PRESSURE_ZONES]
        assert labels == ["MINIMAL", "BASELINE", "ELEVATED", "HIGH", "EXTREME"]

    def test_starts_at_zero(self):
        assert PRESSURE_ZONES[0].lower == 0.0

    def test_ends_above_one(self):
        assert PRESSURE_ZONES[-1].upper > 1.0

    def test_contiguous(self):
        for a, b in zip(PRESSURE_ZONES, PRESSURE_ZONES[1:]):
            assert abs(a.upper - b.lower) < 1e-9

    def test_zone_lookup(self):
        assert get_pressure_zone(0.30).label  == "MINIMAL"
        assert get_pressure_zone(0.50).label  == "BASELINE"
        assert get_pressure_zone(0.56).label  == "ELEVATED"
        assert get_pressure_zone(0.62).label  == "HIGH"
        assert get_pressure_zone(0.75).label  == "EXTREME"

    def test_boundary_inclusive_lower(self):
        assert get_pressure_zone(0.470).label == "BASELINE"
        assert get_pressure_zone(0.530).label == "ELEVATED"
        assert get_pressure_zone(0.590).label == "HIGH"
        assert get_pressure_zone(0.650).label == "EXTREME"

    def test_extreme_threshold_constant(self):
        assert EXTREME_THRESHOLD == 0.650

    def test_zones_have_academic_note(self):
        for z in PRESSURE_ZONES:
            assert len(z.academic_note) > 20

    def test_zones_have_short_label(self):
        for z in PRESSURE_ZONES:
            assert len(z.label_short) >= 4


class TestGetPercentile:

    def test_low_bmpi_low_percentile(self):
        assert get_percentile(0.30) == 5

    def test_median(self):
        assert get_percentile(0.478) == 50

    def test_high_bmpi(self):
        assert get_percentile(0.80) == 99

    def test_monotone(self):
        vals = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75]
        pcts = [get_percentile(v) for v in vals]
        assert pcts == sorted(pcts)


class TestComputeBmpiSeries:

    def test_shape(self):
        out = compute_bmpi_series([200, 400, 800], [-1.0, -0.5, 0.5])
        assert out.shape == (3,)

    def test_all_in_unit_interval(self):
        rng  = np.random.default_rng(42)
        ment = rng.uniform(0, 3000, 500)
        tone = rng.uniform(-5, 2, 500)
        out  = compute_bmpi_series(ment, tone)
        assert np.all((out > 0) & (out < 1))

    def test_consistent_with_scalar(self):
        ms  = [100.0, 500.0, 2000.0]
        ts  = [-2.0, -0.9, 0.5]
        vec = compute_bmpi_series(ms, ts)
        for i, (m, t) in enumerate(zip(ms, ts)):
            scalar = compute_bmpi(m, t).bmpi
            assert abs(vec[i] - scalar) < 1e-6

    def test_nan_propagates(self):
        ment = np.array([300.0, float("nan"), 400.0])
        tone = np.array([-1.0, -0.9, float("nan")])
        out  = compute_bmpi_series(ment, tone)
        assert math.isnan(out[1])
        assert math.isnan(out[2])
        assert not math.isnan(out[0])

    def test_uses_train_by_default(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_bmpi_series([400], [-1.0])
            assert not any("look-ahead" in str(x.message).lower() for x in w)


class TestZoneLabelSeries:

    def test_correct_labels(self):
        bmpi   = np.array([0.30, 0.50, 0.56, 0.62, 0.75])
        labels = zone_label_series(bmpi)
        assert list(labels) == ["MINIMAL", "BASELINE", "ELEVATED", "HIGH", "EXTREME"]

    def test_short_labels(self):
        bmpi   = np.array([0.30, 0.75])
        labels = zone_short_label_series(bmpi)
        assert list(labels) == ["Minimal", "Extreme"]
