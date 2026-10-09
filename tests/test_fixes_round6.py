"""Round 6: integer coordinates, measures listed the other way round, and
properties of whole audits checked on random data."""

import re
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from poseaudit import Instance, angle, audit, compare, length, pair, ratio, tilt
from poseaudit.cli import UsageError, _measures


def _box(kp):
    return np.array([0, 0, 1000, 1000])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32])
def test_unsigned_and_narrow_coordinates_read_as_float64(dtype) -> None:
    kp = np.array([[0, 0], [10, 10], [20, 0], [5, 0]])
    instance = Instance(_box(kp), kp.astype(dtype), np.ones(4, bool))
    assert instance.keypoints.dtype == np.float64
    for measure in (angle(0, 1, 2), tilt(0, 1), ratio(0, 1, 1, 2), length(1, 0)):
        expected = measure.read(kp.astype(float))
        assert measure.read(kp.astype(dtype)) == pytest.approx(expected)
        assert measure.read(instance.keypoints) == pytest.approx(expected)
    assert angle(0, 1, 2).read_many(kp.astype(dtype)[None])[0] == pytest.approx(90)
    assert tilt(0, 1).read_many(kp.astype(dtype)[None])[0] == pytest.approx(-45)


@pytest.mark.parametrize(
    "given_twice",
    [
        [("angle", "5,7,9"), ("angle", "9,7,5")],
        [("tilt", "5,7"), ("tilt", "7,5")],
        [("length", "1,2"), ("length", "2,1")],
        [("ratio", "1,2,3,4"), ("ratio", "2,1,4,3")],
    ],
)
def test_a_measure_listed_the_other_way_round_is_a_duplicate(given_twice) -> None:
    class Args:
        measures = given_twice

    with pytest.raises(UsageError, match="given twice"):
        _measures(Args())


def test_measures_that_differ_are_not_duplicates() -> None:
    class Args:
        measures = [("angle", "5,7,9"), ("angle", "7,5,9"), ("ratio", "1,2,3,4"),
                    ("ratio", "3,4,1,2")]  # fmt: skip

    assert len(_measures(Args())) == 4


def test_the_python_gate_in_the_json_docs_survives_nothing_measurable() -> None:
    text = (Path(__file__).parents[1] / "docs" / "json.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    code = code.replace('json.load(open("figures.json", encoding="utf-8"))', "R")
    for r, fails in (
        ({"mean_abs_error_ci": [None, None], "n": 0, "measurable": 0}, True),
        ({"mean_abs_error_ci": [1.0, 2.0], "n": 0, "measurable": 0}, True),
        ({"mean_abs_error_ci": [1.0, 2.0], "n": 10, "measurable": 10}, False),
    ):
        if fails:
            with pytest.raises(SystemExit):
                exec(code, {"R": r})
        else:
            exec(code, {"R": r})
    assert ".measurable > 0 and .n / .measurable" in text


# ---- properties of whole audits ------------------------------------------

K = 3


def _people(seed: int, count: int, noise: float):
    rng = np.random.default_rng(seed)
    truth, predicted = {}, {}
    for i in range(count):
        xy = rng.uniform(100, 500, (K, 2))
        moved = xy + rng.normal(0, noise, xy.shape)
        truth[f"im{i}"] = [Instance(_box(xy), xy, np.ones(K, bool))]
        predicted[f"im{i}"] = [Instance(_box(moved), moved, np.ones(K, bool))]
    return truth, predicted


QUICK = {"resamples": 50, "jitter_repeats": 0, "bands": 2, "size_bands": 1}
SLOW = settings(
    max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@SLOW
@given(st.integers(0, 10_000), st.floats(1, 20))
def test_comparing_a_with_b_is_the_negative_of_b_with_a(seed, noise) -> None:
    truth, a = _people(seed, 12, noise)
    rng = np.random.default_rng(seed + 1)
    b = {}  # the same images, other errors
    for name, (t,) in truth.items():
        xy = t.keypoints + rng.normal(0, noise, (K, 2))
        b[name] = [Instance(_box(xy), xy, np.ones(K, bool))]
    m = angle(0, 1, 2)
    ab = compare({"a": a, "b": b}, truth, m, 10, resamples=50, jitter_repeats=0)
    ba = compare({"b": b, "a": a}, truth, m, 10, resamples=50, jitter_repeats=0)
    (dab,), (dba,) = ab.differences, ba.differences
    assert (dab.a, dab.b) == ("a", "b") and (dba.a, dba.b) == ("b", "a")
    assert dab.n_shared == dba.n_shared
    assert dab.mean_abs_error_diff == pytest.approx(-dba.mean_abs_error_diff)


@SLOW
@given(st.integers(0, 10_000), st.randoms(use_true_random=False))
def test_the_order_of_images_does_not_change_the_readings(seed, shuffler) -> None:
    truth, predicted = _people(seed, 10, 5)
    names = list(truth)
    shuffler.shuffle(names)
    m = angle(0, 1, 2)
    one = audit(pair(truth, predicted), m, 10, **QUICK)
    two = audit(
        pair({n: truth[n] for n in names}, {n: predicted[n] for n in names}),
        m, 10, **QUICK,
    )  # fmt: skip
    assert one.n == two.n
    for key in ("mean_abs_error", "bias", "rmse", "big_error_rate"):
        assert getattr(one, key) == pytest.approx(getattr(two, key))


@SLOW
@given(
    st.integers(0, 10_000),
    st.floats(0.05, 20),
    st.tuples(st.floats(-1e4, 1e4), st.floats(-1e4, 1e4)),
)
def test_angles_do_not_change_when_the_picture_is_scaled_or_moved(
    seed, scale, shift
) -> None:
    truth, predicted = _people(seed, 10, 5)

    def moved(data):
        return {
            k: [Instance(_box(None), i.keypoints * scale + shift, i.visible)
                for i in v]
            for k, v in data.items()
        }  # fmt: skip

    m = angle(0, 1, 2)
    one = audit(pair(truth, predicted), m, 10, **QUICK)
    two = audit(pair(moved(truth), moved(predicted)), m, 10, **QUICK)
    assert one.n == two.n
    assert two.mean_abs_error == pytest.approx(one.mean_abs_error, rel=1e-6, abs=1e-9)
    assert two.bias == pytest.approx(one.bias, rel=1e-6, abs=1e-9)


@SLOW
@given(st.integers(0, 10_000), st.lists(st.floats(0.1, 60), min_size=2, max_size=4))
def test_rates_fall_as_the_thresholds_rise(seed, cuts) -> None:
    truth, predicted = _people(seed, 15, 8)
    paired = pair(truth, predicted)
    m = angle(0, 1, 2)
    cuts = sorted(cuts)
    rates = [audit(paired, m, c, **QUICK).big_error_rate for c in cuts]
    assert all(x >= y - 1e-12 for x, y in zip(rates[:-1], rates[1:], strict=True))
    truth_angles = sorted(np.random.default_rng(seed).uniform(20, 160, 3))
    result = audit(
        paired, m, 10, thresholds=truth_angles, threshold_side="above", **QUICK
    )
    flagged = [t.true_positive + t.false_negative for t in result.thresholds]
    assert all(x >= y for x, y in zip(flagged[:-1], flagged[1:], strict=True))
