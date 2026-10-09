"""Round 9: a NaN supervision class, a stale num_keypoints of 0, a class
filter on results with no category_id, and the documented public names."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import poseaudit
from poseaudit.io import from_supervision, load_coco, load_coco_results


class _KeyPoints:
    def __init__(self, xy, class_id):
        self.xy, self.class_id = xy, class_id
        self.visible = None
        self.keypoint_confidence = None


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_supervision_class_is_named(bad) -> None:
    xy = np.array([[[10, 10], [20, 20], [30, 15]]], float)
    with pytest.raises(ValueError, match="whole number"):
        from_supervision({"a": _KeyPoints(xy, np.array([bad]))})


def _gt(tmp_path, num_keypoints, flags=(2, 2, 2)):
    kp = []
    for x, v in zip((10, 20, 30), flags, strict=True):
        kp += [x, x, v]
    data = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 40, 40],
                "keypoints": kp, "num_keypoints": num_keypoints, "iscrowd": 0,
            }
        ],
    }  # fmt: skip
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_a_stale_num_keypoints_of_zero_keeps_labelled_points(tmp_path) -> None:
    assert len(load_coco(_gt(tmp_path, 0))["a.jpg"]) == 1
    assert load_coco(_gt(tmp_path, 0, (0, 0, 0)))["a.jpg"] == []


def test_a_class_filter_on_results_without_category_id(tmp_path) -> None:
    gt = _gt(tmp_path, 3)
    res = tmp_path / "res.json"
    res.write_text(
        json.dumps([{"image_id": 1, "keypoints": [10, 10, 1] * 3, "score": 1}]),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="no entry has a category_id"):
        load_coco_results(res, gt, classes=[1])


def test_every_public_name_is_documented() -> None:
    doc = (Path(__file__).parents[1] / "docs" / "python.md").read_text("utf-8")
    missing = [n for n in poseaudit.__all__ if not re.search(rf"\b{n}\b", doc)]
    assert missing == []
