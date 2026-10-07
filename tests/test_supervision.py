import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from poseaudit import from_supervision

sv = pytest.importorskip("supervision")


def key_points(xy, **fields):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # opencv fallback notice
        return sv.KeyPoints(xy=np.asarray(xy, float), **fields)


def test_the_containers_visible_mask_is_honoured() -> None:
    kp = key_points(
        [[[10, 10], [20, 20], [30, 30]]],
        keypoint_confidence=np.array([[0.9, 0.9, 0.9]]),
        visible=np.array([[True, False, True]]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")  # supervision's own warning class is not
        inst = from_supervision({"img": kp})["img"][0]  # a DeprecationWarning
    assert inst.visible.tolist() == [True, False, True]
    assert not [w for w in caught if "confidence" in str(w.message)]


def test_points_supervision_treats_as_missing_are_missing() -> None:
    kp = key_points([[[0, 0], [np.nan, 5], [30, 30]]])
    assert from_supervision({"img": kp})["img"][0].visible.tolist() == [
        False,
        False,
        True,
    ]


def test_detections_supply_boxes_and_classes() -> None:
    kp = key_points([[[10, 10], [10, 90]]], class_id=np.array([2]))
    detections = sv.Detections(xyxy=np.array([[0.0, 0.0, 50.0, 100.0]]))
    inst = from_supervision({"img": kp}, {"img": detections})["img"][0]
    assert inst.boxed and inst.bbox.tolist() == [0, 0, 50, 100]
    assert inst.class_id == 2


def test_detections_in_a_different_row_order_are_caught() -> None:
    kp = key_points([[[10, 10], [10, 40]], [[210, 10], [210, 40]]])
    swapped = sv.Detections(xyxy=np.array([[200.0, 0, 220, 50], [0.0, 0, 20, 50]]))
    with pytest.warns(UserWarning, match="sorted apart"):
        from_supervision({"img": kp}, {"img": swapped})


def test_overlapping_people_swapped_between_rows_are_caught() -> None:
    """Two people whose boxes overlap: every point lies inside both boxes, so
    only which box fits best can tell the rows apart."""
    kp = key_points(
        [
            [[10, 10], [60, 10], [10, 90], [60, 90]],
            [[40, 10], [90, 10], [40, 90], [90, 90]],
        ]
    )
    right = sv.Detections(xyxy=np.array([[5.0, 5, 65, 95], [35.0, 5, 95, 95]]))
    assert len(from_supervision({"img": kp}, {"img": right})["img"]) == 2
    swapped = sv.Detections(xyxy=np.array([[35.0, 5, 95, 95], [5.0, 5, 65, 95]]))
    with pytest.warns(UserWarning, match="sorted apart"):
        from_supervision({"img": kp}, {"img": swapped})


def test_misaligned_inputs_fail_with_a_readable_message() -> None:
    kp = key_points([[[10, 10], [10, 90]]])
    with pytest.raises(ValueError, match="row for row"):
        from_supervision({"img": kp}, {"img": sv.Detections(xyxy=np.zeros((2, 4)))})
    with pytest.raises(TypeError, match="sv.KeyPoints"):
        from_supervision({"img": np.zeros((1, 2, 2))})


def test_confidence_without_a_visible_mask_is_flagged() -> None:
    kp = key_points([[[10, 10], [20, 20]]], keypoint_confidence=np.array([[0.9, 0.1]]))
    with pytest.warns(UserWarning, match="no visible mask"):
        from_supervision({"img": kp})


def test_an_empty_container_gives_no_instances() -> None:
    assert from_supervision({"img": sv.KeyPoints.empty()})["img"] == []


DEMO = Path(__file__).parents[1] / "examples" / "coco_elbow"


def _rows(name: str) -> list[dict]:
    data = json.loads((DEMO / name).read_text())
    if isinstance(data, dict):  # annotations
        return [a for a in data["annotations"] if not a.get("iscrowd")]
    return data


@pytest.mark.skipif(not DEMO.exists(), reason="demo data is not in this checkout")
@pytest.mark.parametrize("name", ["gt_200.json", "pred_yolo11n.json"])
def test_correctly_ordered_real_rows_load(name) -> None:
    """Duplicate or crowded boxes let a row's points fit a neighbour's box a
    little better; that must not read as rows sorted apart."""
    by_image: dict[int, list] = {}
    for r in _rows(name):
        by_image.setdefault(r["image_id"], []).append(r)
    kps, dets = {}, {}
    for image, rows in by_image.items():
        raw = np.array([np.reshape(r["keypoints"], (-1, 3)) for r in rows])
        kps[image] = key_points(raw[..., :2], visible=raw[..., 2] > 0)
        boxes = np.array([r["bbox"] for r in rows], float)
        boxes[:, 2:] += boxes[:, :2]
        dets[image] = sv.Detections(xyxy=boxes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert len(from_supervision(kps, dets)) == len(by_image)
    suspects = [w for w in caught if "sorted apart" in str(w.message)]
    # human labels of crowds: one image in 200 looks swapped, the model's rows none
    assert len(suspects) == (1 if name == "gt_200.json" else 0)


def test_scores_that_differ_row_for_row_are_caught() -> None:
    """With scores on both sides the check is exact, even for boxes that look
    alike."""
    xy = [[[10, 10], [10, 40]], [[12, 10], [12, 40]]]
    kp = key_points(xy, detection_confidence=np.array([0.9, 0.8]))
    same = sv.Detections(
        xyxy=np.array([[0.0, 0, 20, 50], [2.0, 0, 22, 50]]),
        confidence=np.array([0.9, 0.8]),
    )
    assert len(from_supervision({"i": kp}, {"i": same})["i"]) == 2
    swapped = sv.Detections(xyxy=same.xyxy, confidence=np.array([0.8, 0.9]))
    with pytest.raises(ValueError, match="rank their scores"):
        from_supervision({"i": kp}, {"i": swapped})


def test_images_without_detections_are_flagged() -> None:
    kp = key_points([[[10, 10], [10, 90]]])
    detections = {"a": sv.Detections(xyxy=np.array([[0.0, 0, 50, 100]]))}
    with pytest.warns(UserWarning, match="no detections"):
        from_supervision({"a": kp, "b": kp}, detections)


def test_scores_rounded_on_one_side_still_load() -> None:
    kp = key_points(
        [[[10, 10], [10, 40]], [[80, 10], [80, 40]]],
        detection_confidence=np.array([0.912345, 0.834567]),
    )
    rounded = sv.Detections(
        xyxy=np.array([[0.0, 0, 20, 50], [70.0, 0, 90, 50]]),
        confidence=np.array([0.9123, 0.8346]),
    )
    assert len(from_supervision({"i": kp}, {"i": rounded})["i"]) == 2


def test_nearly_equal_scores_swapped_are_caught() -> None:
    kp = key_points(
        [[[10, 10], [10, 40]], [[80, 10], [80, 40]]],
        detection_confidence=np.array([0.800001, 0.8000019]),
    )
    swapped = sv.Detections(
        xyxy=np.array([[70.0, 0, 90, 50], [0.0, 0, 20, 50]]),
        confidence=np.array([0.8000019, 0.800001]),
    )
    with pytest.raises(ValueError, match="rank their scores"):
        from_supervision({"i": kp}, {"i": swapped})


def test_a_visible_mask_that_is_not_boolean_is_refused() -> None:
    kp = key_points([[[10, 10], [10, 40]]])
    kp.visible = np.array([[0.9, 0.2]])
    with pytest.raises(ValueError, match="boolean"):
        from_supervision({"i": kp})


def test_a_swap_next_to_a_sparse_row_is_flagged() -> None:
    """One row with a single visible point cannot be fitted; the other row
    alone still shows the swap."""
    kp = key_points(
        [[[10, 10], [10, 40], [20, 40]], [[210, 10], [0, 0], [0, 0]]],
    )
    swapped = sv.Detections(xyxy=np.array([[200.0, 0, 220, 50], [0.0, 0, 30, 50]]))
    with pytest.warns(UserWarning, match="sorted apart"):
        from_supervision({"i": kp}, {"i": swapped})


def test_boxes_carried_in_the_container_are_used() -> None:
    """RF-DETR keeps boxes in kp.data["xyxy"]; without them matching falls back
    to keypoints."""
    kp = key_points([[[10, 10], [10, 40]]])
    kp.data["xyxy"] = np.array([[0.0, 0.0, 30.0, 50.0]])
    inst = from_supervision({"i": kp})["i"][0]
    assert inst.boxed and inst.bbox.tolist() == [0, 0, 30, 50]


def test_carried_boxes_with_detections_for_other_images_only() -> None:
    kp = key_points([[[10, 10], [10, 40]]])
    kp.data["xyxy"] = np.array([[0.0, 0.0, 30.0, 50.0]])
    other = key_points([[[10, 10], [10, 40]]])
    detections = {"b": sv.Detections(xyxy=np.array([[0.0, 0, 30, 50]]))}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = from_supervision({"a": kp, "b": other}, detections)
    assert out["a"][0].boxed and out["b"][0].boxed
    assert not [w for w in caught if "no detections" in str(w.message)]


def test_carried_boxes_are_checked() -> None:
    kp = key_points([[[10, 10], [10, 40]]])
    kp.data["xyxy"] = np.array([[0.0, 0.0, np.nan, 50.0]])
    with pytest.raises(ValueError, match="non-finite"):
        from_supervision({"i": kp})
    kp.data["xyxy"] = np.zeros((2, 4))
    with pytest.raises(ValueError, match="shape"):
        from_supervision({"i": kp})


def test_detection_boxes_must_be_finite() -> None:
    kp = key_points([[[10, 10], [10, 40]]])
    bad = sv.Detections(xyxy=np.array([[0.0, 0.0, np.inf, 50.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        from_supervision({"i": kp}, {"i": bad})


def test_an_image_with_no_rows_and_empty_carried_boxes_loads() -> None:
    kp = key_points(np.zeros((0, 2, 2)))
    kp.data["xyxy"] = np.zeros((0,))
    assert from_supervision({"i": kp})["i"] == []


def test_a_bare_key_points_container_asks_for_a_mapping() -> None:
    kp = key_points([[[1, 2], [3, 4]]])
    with pytest.raises(TypeError, match="map image name"):
        from_supervision(kp)
    with pytest.raises(TypeError, match="detections_by_image"):
        from_supervision({"a": kp}, sv.Detections(xyxy=np.array([[0.0, 0, 5, 5]])))
