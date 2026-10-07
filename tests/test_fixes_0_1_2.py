"""Regressions found in review after 0.1.1: each failed before its fix."""

import json
import warnings

import numpy as np
import pytest

from poseaudit import audit, load_coco, load_coco_results, pair
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
