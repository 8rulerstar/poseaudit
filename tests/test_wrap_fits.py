"""Angles whose reference runs across the wrap (179 and -179, or 359 and 1):
the fits must not take the two sides for a period apart."""

import warnings

import numpy as np
import pytest

import poseaudit as pa


def _audit(pred, ref, unit="deg"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pa.audit_values(pred, ref, 5, name="k", unit=unit, resamples=200)


def _data(centre, period=360.0, seed=0):
    rng = np.random.default_rng(seed)
    ref = centre + rng.uniform(-10, 10, 300) * period / 360
    pred = ref + rng.normal(0, 5, 300) * period / 360
    half = period / 2
    return (pred + half) % period - half, (ref + half) % period - half


@pytest.mark.parametrize("unit,period", [("deg", 360.0), ("rad", 2 * np.pi)])
def test_fits_across_the_wrap_match_the_same_data_away_from_it(unit, period):
    across = _audit(*_data(period / 2, period), unit=unit)
    away = _audit(*_data(0.0, period), unit=unit)
    for name in ("icc", "ccc", "gain", "robust_gain", "pearson"):
        assert getattr(across, name) == pytest.approx(getattr(away, name), abs=1e-9)
    assert across.icc < 0.9  # not pushed towards 1 by the wrap
    assert across.mean_abs_error == pytest.approx(away.mean_abs_error)


def test_zero_to_360_gives_the_fits_of_the_same_data():
    pred, ref = _data(0.0)
    shifted = _audit(pred % 360, ref % 360)
    plain = _audit(pred, ref)
    assert shifted.icc == pytest.approx(plain.icc, abs=1e-9)
    assert shifted.gain == pytest.approx(plain.gain, abs=1e-9)


def test_a_reference_round_the_whole_circle_is_flagged():
    rng = np.random.default_rng(1)
    ref = rng.uniform(-180, 180, 300)
    result = _audit(ref + rng.normal(0, 5, 300), ref)
    assert any("of the circle" in w for w in result.warnings)


def test_no_flag_for_a_reference_in_one_part_of_the_circle():
    result = _audit(*_data(180.0))
    assert not any("of the circle" in w for w in result.warnings)
