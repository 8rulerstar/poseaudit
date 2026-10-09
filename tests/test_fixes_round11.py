"""Round 11: a faster jitter reference with the same numbers, `python -m
poseaudit`, and a count of the crowd regions left out."""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from poseaudit import reference
from poseaudit.io.coco import load_coco
from poseaudit.measures import Angle, Length, Ratio, Tilt

DEMO = Path(__file__).parents[1] / "examples" / "coco_elbow"

MEASURES = [
    Angle("a", (5, 7, 9), "deg"),
    Tilt("t", (5, 11), "deg"),
    Length("l", (5, 7), "px"),
    Ratio("r", (5, 7, 7, 9), ""),
    Ratio("r2", (7, 5, 9, 7), ""),
]


def _old_draw(measure, truth_points, sizes, members, scatter, rng):
    """The rebuild as it was before round 11: frames recomputed, points laid
    out at their keypoint indices and read through the general reader."""
    headings = reference._headings(measure, truth_points)
    hands = reference._hands(measure, truth_points)
    donors = np.empty(len(truth_points), int)
    for m in members:
        donors[m] = m[rng.integers(0, len(m), size=len(m))]
    shifts = reference._mirrored(scatter[donors], hands)
    rebuilt = truth_points + reference._turned(shifts, headings) * sizes[:, None, None]
    full = np.zeros((len(rebuilt), max(measure.points) + 1, 2))
    for slot, index in enumerate(measure.points):
        full[:, index] = rebuilt[:, slot]
    if isinstance(measure, Angle):
        return _old_angles(measure, full)
    if isinstance(measure, Tilt):
        return measure.read_many(full)
    return np.array([measure.read(k) for k in full])


def _old_angles(measure, keypoints):
    """Angle.read_many before round 11, with reductions along an axis of two."""
    a, b, c = (np.asarray(keypoints, float)[:, i] for i in measure.points)
    u, v = a - b, c - b
    empty = ~u.any(axis=1) | ~v.any(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = u / np.abs(u).max(axis=1, keepdims=True)
        v = v / np.abs(v).max(axis=1, keepdims=True)
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = np.degrees(np.arctan2(np.abs(cross), (u * v).sum(axis=1)))
    return np.where(empty, np.nan, out)


@pytest.mark.parametrize("measure", MEASURES, ids=lambda m: m.name)
def test_the_faster_rebuild_gives_the_same_numbers(measure) -> None:
    rng = np.random.default_rng(3)
    n = 400
    truth = rng.normal(size=(n, len(measure.points), 2)) * 40 + 300
    if isinstance(measure, Ratio):  # the shared keypoint is one point
        a, b, c, d = measure.points
        for i, j in ((0, 2), (0, 3), (1, 2), (1, 3)):
            if measure.points[i] == measure.points[j]:
                truth[:, j] = truth[:, i]
    predicted = truth + rng.normal(size=truth.shape) * 4
    sizes = rng.uniform(50, 200, n)
    units = reference.local_shifts(measure, truth, predicted, sizes)
    groups = reference.strata(rng.normal(size=n))
    idx = rng.integers(0, n, n)  # a resample, frames taken by row
    framed = tuple(part[idx] for part in reference.frames(measure, truth))
    prepared = reference.prepare(measure, truth[idx], units[idx], groups, framed)
    members, scatter, _ = prepared
    new = reference.draw(
        measure, truth[idx], sizes[idx], prepared, np.random.default_rng(9)
    )
    old = _old_draw(
        measure, truth[idx], sizes[idx], members, scatter, np.random.default_rng(9)
    )
    assert np.array_equal(new, old, equal_nan=True)


@pytest.mark.parametrize("measure", MEASURES, ids=lambda m: m.name)
def test_the_faster_readers_give_the_same_numbers(measure) -> None:
    rng = np.random.default_rng(5)
    points = rng.normal(size=(500, 10, 2)) * 100
    points[:20, 7] = points[:20, 5]  # some unmeasurable
    points[20:40, 9] = points[20:40, 7]
    if isinstance(measure, Angle):
        before = _old_angles(measure, points)
    elif isinstance(measure, Tilt):
        return  # its reader did not change
    else:  # these read one object at a time before
        before = np.array([measure.read(k) for k in points])
    assert np.array_equal(measure.read_many(points), before, equal_nan=True)


def test_python_dash_m_runs_the_command(tmp_path) -> None:
    out = tmp_path / "r.json"
    done = subprocess.run(
        [
            sys.executable, "-m", "poseaudit", "audit", "--format", "coco",
            "--gt", str(DEMO / "gt_200.json"),
            "--pred", str(DEMO / "pred_yolo11n.json"),
            "--angle", "5,7,9", "--big-error", "15", "--resamples", "50",
            "--jitter-repeats", "0", "--json", str(out),
        ],
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["n"] == 322
    usage = subprocess.run(
        [sys.executable, "-m", "poseaudit", "--help"],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert usage.returncode == 0 and usage.stdout.startswith("usage: poseaudit")


def _with_crowds(tmp_path, category=1) -> Path:
    data = json.loads((DEMO / "gt_200.json").read_text(encoding="utf-8"))
    for ann in data["annotations"][:2]:
        ann["iscrowd"] = 1
        ann["category_id"] = category
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_crowd_regions_left_out_are_counted(tmp_path) -> None:
    path = _with_crowds(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_coco(path)
    texts = [str(w.message) for w in caught]
    assert any("2 crowd regions (iscrowd 1) were left out" in t for t in texts)
    assert any(
        "counts as unmatched, unless its box overlaps a labelled person" in t
        for t in texts
    )


def test_crowd_regions_of_another_class_are_not_counted(tmp_path) -> None:
    path = _with_crowds(tmp_path, category=99)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_coco(path, classes=[1])
    assert not any("crowd" in str(w.message) for w in caught)


def test_no_crowd_no_note() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_coco(DEMO / "gt_200.json")
    assert not any("crowd" in str(w.message) for w in caught)
