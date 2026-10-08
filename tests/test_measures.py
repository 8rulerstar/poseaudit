import numpy as np
import pytest

from poseaudit.measures import angle, length, ratio, tilt


def points(*xy: tuple[float, float]) -> np.ndarray:
    return np.array(xy, float)


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        ((0, 10), (0, 0), 0.0),  # upright
        ((0, 10), (10, 0), 45.0),  # upper end to the right
        ((0, 10), (-10, 0), -45.0),  # upper end to the left
    ],
)
def test_tilt_is_positive_when_the_upper_end_leans_right(lower, upper, expected):
    assert tilt(0, 1).read(points(lower, upper)) == pytest.approx(expected)
    # the axis has no direction
    assert tilt(0, 1).read(points(upper, lower)) == pytest.approx(expected)


def test_tilt_reads_within_minus_90_to_90() -> None:
    lying = tilt(0, 1).read(points((0, 0), (10, 0)))
    assert -90.0 <= lying < 90.0


def test_tilt_error_ignores_which_end_is_which() -> None:
    """An axis read as 89 and as -89 is 2 degrees off, not 178."""
    assert tilt(0, 1).difference(89.0, -89.0) == pytest.approx(-2.0)
    assert tilt(0, 1).difference(-89.0, 89.0) == pytest.approx(2.0)


def test_angle_is_the_interior_angle_at_the_middle_point() -> None:
    right = points((10, 0), (0, 0), (0, 10))
    straight = points((-10, 0), (0, 0), (10, 0))
    assert angle(0, 1, 2).read(right) == pytest.approx(90.0)
    assert angle(0, 1, 2).read(straight) == pytest.approx(180.0)


def test_length_and_ratio() -> None:
    kp = points((0, 0), (3, 4), (0, 0), (0, 10))
    assert length(0, 1).read(kp) == pytest.approx(5.0)
    assert ratio(0, 1, 2, 3).read(kp) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("measure", "kp"),
    [
        (tilt(0, 1), points((5, 5), (5, 5))),
        (angle(0, 1, 2), points((5, 5), (5, 5), (9, 9))),
        (ratio(0, 1, 2, 3), points((0, 0), (1, 1), (4, 4), (4, 4))),
    ],
)
def test_unmeasurable_geometry_reads_nan_quietly(measure, kp) -> None:
    with np.errstate(all="raise"):
        assert np.isnan(measure.read(kp))


def test_angles_near_straight_and_near_closed_keep_their_digits() -> None:
    """arccos loses about half its digits near 0 and 180 degrees."""
    nearly_straight = points((-1000, 0), (0, 0), (1000, 0.001))
    expected = 180.0 - np.degrees(np.arctan2(0.001, 1000))
    assert angle(0, 1, 2).read(nearly_straight) == pytest.approx(expected, abs=1e-9)
    folded = points((3, 0), (0, 3), (3, 0))
    assert angle(0, 1, 2).read(folded) == 0.0


@pytest.mark.parametrize(
    "make",
    [
        lambda: angle(25, 25, 27),
        lambda: angle(25, 27, 25),
        lambda: tilt(3, 3),
        lambda: length(1, 1),
        lambda: ratio(0, 0, 1, 2),
        lambda: ratio(0, 1, 1, 0),
    ],
)
def test_a_repeated_keypoint_is_refused_at_once(make) -> None:
    """angle(25, 25, 27) has no angle to read: every instance would end up
    unmeasurable, with no word why."""
    with pytest.raises(ValueError, match="repeated"):
        make()


def test_a_ratio_may_share_a_point_between_its_segments() -> None:
    upper_over_fore = ratio(5, 7, 7, 9)
    kp = np.zeros((10, 2))
    kp[5], kp[7], kp[9] = (0, 0), (0, 2), (0, 3)
    assert upper_over_fore.read(kp) == pytest.approx(2.0)
