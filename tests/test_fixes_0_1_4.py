"""Regressions found after 0.1.3: each failed before its fix."""

import numpy as np
import pytest

from poseaudit import audit, pair
from poseaudit.measures import angle
from poseaudit.report import summary
from poseaudit.types import Instance

BOX = np.array([0.0, 0.0, 100.0, 100.0])
BENT = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])
SEEN = np.ones(3, bool)


def boxed() -> list[Instance]:
    return [Instance(BOX, BENT, SEEN)]


def both_orders(names: list[str]) -> list[dict]:
    return [{n: boxed() for n in names}, {n: boxed() for n in reversed(names)}]


def test_names_that_differ_only_by_folder_are_warned_about() -> None:
    truth = {n: boxed() for n in ["imgs/b.jpg", "imgs/a.jpg", "c.jpg"]}
    predicted = {n: boxed() for n in ["a", "C:\\x\\b.jpg", "c.jpg"]}
    result = pair(truth, predicted)
    assert len(result.pairs) == 1  # folders still keep images apart
    notes = [w for w in result.warnings if "folders differ" in w]
    assert len(notes) == 1
    assert notes[0].startswith("2 truth images")
    assert "'imgs/a.jpg'" in notes[0] and "Path(p).name" in notes[0]


def test_the_folder_warning_does_not_depend_on_order() -> None:
    names_t = ["imgs/b.jpg", "imgs/a.jpg", "c.jpg"]
    names_p = ["b", "a", "c.jpg"]
    seen = {
        tuple(pair(t, p).warnings)
        for t in both_orders(names_t)
        for p in both_orders(names_p)
    }
    assert len(seen) == 1


def test_no_overlap_at_all_names_the_folders_too() -> None:
    result = pair({"imgs/a.jpg": boxed()}, {"/x/a.jpg": boxed()})
    (note,) = [w for w in result.warnings if "No image name" in w]
    assert "folders differ" in note


def test_no_folder_warning_when_last_names_differ() -> None:
    result = pair({"imgs/a.jpg": boxed(), "b.jpg": boxed()}, {"b.jpg": boxed()})
    assert not any("folders differ" in w for w in result.warnings)


@pytest.mark.parametrize("reverse", [False, True])
def test_the_extension_example_does_not_depend_on_order(reverse) -> None:
    names = ["y.jpg", "x.jpg"]
    predicted = {n: boxed() for n in (reversed(names) if reverse else names)}
    result = pair({"x": boxed(), "y": boxed()}, predicted)
    (note,) = [w for w in result.warnings if "without their extensions" in w]
    assert "'x.jpg'" in note


@pytest.mark.parametrize(("side", "blind"), [("truth", 0), ("predictions", 1)])
def test_no_visible_point_is_named_as_the_cause(side, blind) -> None:
    hidden = np.zeros(3, bool)
    sides = [
        {f"im{i}": [Instance.from_keypoints(BENT, SEEN)] for i in range(3)}
        for _ in range(2)
    ]
    sides[blind] = {f"im{i}": [Instance.from_keypoints(BENT, hidden)] for i in range(3)}
    warnings = pair(*sides).warnings
    assert not any("refer to the same images" in w for w in warnings)
    assert any(f"no point is visible in the {side}" in w for w in warnings)


def test_one_reading_prints_in_the_singular_and_without_nan_limits() -> None:
    result = audit(pair({"a": boxed()}, {"a": boxed()}), angle(0, 1, 2), big_error=10)
    text = summary(result, full=True)
    assert "nan°" not in text
    assert "limits       n/a" in text and "normal       n/a" in text
    assert "Only 1 reading:" in text
    assert "Only 0" not in text and "No large errors" in text
