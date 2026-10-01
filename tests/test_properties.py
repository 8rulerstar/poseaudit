"""Geometric properties of the measures, checked on random inputs."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from poseaudit import angle, ratio, tilt

coordinate = st.floats(-500, 500, allow_nan=False)
point = st.tuples(coordinate, coordinate)


def rotated(kp: np.ndarray, degrees: float, scale: float, shift) -> np.ndarray:
    """Turn the picture clockwise as seen on screen (y grows downward)."""
    t = np.radians(degrees)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return kp @ r.T * scale + np.asarray(shift)


@settings(max_examples=300, deadline=None)
@given(
    st.lists(point, min_size=4, max_size=4),
    st.floats(-180, 180),
    st.floats(0.1, 10),
    point,
)
def test_angles_and_ratios_ignore_where_and_how_big(pts, turn, scale, shift) -> None:
    kp = np.array(pts)
    moved = rotated(kp, turn, scale, shift)
    for measure in (angle(0, 1, 2), ratio(0, 1, 2, 3)):
        before, after = measure.read(kp), measure.read(moved)
        if np.isfinite(before) and min(np.linalg.norm(kp[i] - kp[j]) for i, j in
                                       ((0, 1), (1, 2), (2, 3))) > 1e-3:  # fmt: skip
            assert after == pytest.approx(before, rel=1e-6, abs=1e-6)


@settings(max_examples=300, deadline=None)
@given(point, point, st.floats(-80, 80))
def test_turning_the_picture_clockwise_adds_to_the_tilt(a, b, turn) -> None:
    kp = np.array([a, b])
    if np.linalg.norm(kp[1] - kp[0]) < 1e-3:
        return
    before = tilt(0, 1).read(kp)
    after = tilt(0, 1).read(rotated(kp, turn, 1.0, (0, 0)))
    assert tilt(0, 1).difference(after, before) == pytest.approx(turn, abs=1e-6)


@settings(max_examples=300, deadline=None)
@given(st.floats(-90, 89.9), st.floats(-90, 89.9))
def test_tilt_errors_stay_within_a_quarter_turn(p, t) -> None:
    error = tilt(0, 1).difference(p, t)
    assert -90.0 <= error < 90.0
    assert abs(error) <= abs(p - t) + 1e-9
