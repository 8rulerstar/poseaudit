"""Regressions found in review after 0.1.1: each failed before its fix."""

import json
import warnings

import numpy as np
import pytest

from poseaudit import load_coco, load_coco_results, pair
from poseaudit.cli import main
from poseaudit.measures import angle, length, tilt
from poseaudit.types import Instance

BOX = np.array([0.0, 0.0, 100.0, 100.0])
SHIFTED = BOX + [1, 0, 1, 0]
BENT = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]])  # 90 degrees at 1
STRAIGHT = np.array([[0.0, 10.0], [10.0, 10.0], [20.0, 10.0]])  # 180 degrees
SEEN = np.ones(3, bool)


def test_people_with_near_identical_boxes_pair_by_their_keypoints() -> None:
    """Box IoU alone pairs each person with the other's nearly identical box."""
    truth = {"im": [Instance(BOX, BENT, SEEN), Instance(SHIFTED, STRAIGHT, SEEN)]}
    predicted = {"im": [Instance(SHIFTED, BENT, SEEN), Instance(BOX, STRAIGHT, SEEN)]}
    measure = angle(0, 1, 2)
    for p in pair(truth, predicted).pairs:
        assert measure.read(p.predicted.keypoints) == measure.read(p.truth.keypoints)


def test_a_prediction_with_wrong_keypoints_still_pairs_on_its_box() -> None:
    far = BENT + 60.0
    result = pair(
        {"im": [Instance(BOX, BENT, SEEN)]}, {"im": [Instance(BOX, far, SEEN)]}
    )
    assert len(result.pairs) == 1


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_tied_boxes_pair_the_same_way_in_either_order(order) -> None:
    candidates = [Instance(BOX, BENT, SEEN), Instance(BOX, STRAIGHT, SEEN)]
    predicted = {"im": [candidates[k] for k in order]}
    result = pair({"im": [Instance(BOX, BENT, SEEN)]}, predicted)
    assert angle(0, 1, 2).read(result.pairs[0].predicted.keypoints) == 90.0


def test_an_exact_tie_between_different_predictions_is_reported() -> None:
    mirrored = BENT * [1, -1] + [0, 20]  # same distance from the truth, mirrored
    truth = {"im": [Instance(BOX, STRAIGHT, SEEN)]}
    result = pair(
        truth, {"im": [Instance(BOX, BENT, SEEN), Instance(BOX, mirrored, SEEN)]}
    )
    assert any("arbitrary" in w for w in result.warnings)


def test_a_length_between_coincident_points_is_unmeasurable() -> None:
    assert np.isnan(length(0, 1).read(np.array([[5.0, 5.0], [5.0, 5.0]])))
    assert np.isnan(length(0, 1).read_many(np.zeros((1, 2, 2)))[0])


@pytest.mark.parametrize("make", [lambda: tilt("left_hip", 1), lambda: tilt(0.0, 1)])
def test_keypoints_named_instead_of_indexed_say_so(make) -> None:
    with pytest.raises(TypeError, match="by index"):
        make()


def test_numpy_indices_are_still_indices() -> None:
    assert tilt(np.int64(0), np.int64(1)).points == (0, 1)


def coco_files(tmp_path):
    gt = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 10],
             "keypoints": [0, 0, 2, 10, 10, 2, 20, 0, 2], "num_keypoints": 3}
        ],
    }  # fmt: skip
    pred = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 10], "score": 0.9,
         "keypoints": [0, 0, 1, 10, 10, 1, 20, 0, 1]}
    ]  # fmt: skip
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "pred.json").write_text(json.dumps(pred))
    return tmp_path / "gt.json", tmp_path / "pred.json"


def test_swapped_coco_files_are_named_as_swapped(tmp_path) -> None:
    gt, pred = coco_files(tmp_path)
    with pytest.raises(ValueError, match="swapped"):
        load_coco(pred)
    with pytest.raises(ValueError, match="swapped"):
        load_coco_results(gt, pred)


def test_the_cli_says_when_gt_and_pred_are_swapped(tmp_path) -> None:
    gt, pred = coco_files(tmp_path)
    args = ["audit", "--format", "coco", "--gt", str(pred), "--pred", str(gt),
            "--angle", "0,1,2", "--big-error", "15"]  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        main(args)
    assert "swapped" in str(stop.value)


def test_no_match_despite_shared_images_is_warned() -> None:
    truth = {"im": [Instance(BOX, BENT, SEEN)]}
    scaled = {"im": [Instance(BOX / 1000, BENT / 1000, SEEN)]}  # 0 to 1 by mistake
    assert any("No truth instance matched" in w for w in pair(truth, scaled).warnings)


def test_predictions_for_few_of_the_images_are_warned() -> None:
    truth = {f"im{i}": [Instance(BOX, BENT, SEEN)] for i in range(20)}
    predicted = {"im0": [Instance(BOX, BENT, SEEN)]}
    assert any("Only 1 of 20" in w for w in pair(truth, predicted).warnings)


def test_a_complete_match_is_not_warned() -> None:
    truth = {f"im{i}": [Instance(BOX, BENT, SEEN)] for i in range(20)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pair(truth, truth).warnings == []
