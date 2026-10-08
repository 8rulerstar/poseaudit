import json

import numpy as np
import pytest

from poseaudit import load_coco, load_coco_results, load_yolo, pair
from poseaudit.types import Instance


def test_yolo_is_read_in_pixels_so_angles_survive_a_wide_image(tmp_path) -> None:
    """On a 200x100 image a 45 degree axis spans 0.25 of the width and 0.5 of
    the height; read in normalised units it would look like 26.6 degrees."""
    (tmp_path / "a.b.txt").write_text("0 0.5 0.5 0.5 1.0 0.25 0.0 2 0.5 0.5 2\n")
    data = load_yolo(tmp_path, (200, 100), num_keypoints=2)

    kp = data["a.b"][0].keypoints  # the dot in the name survives
    assert kp.tolist() == [[50.0, 0.0], [100.0, 50.0]]


def test_yolo_visibility_threshold_and_trailing_score(tmp_path) -> None:
    (tmp_path / "x.txt").write_text("0 .5 .5 .2 .2 .1 .1 0.3 .2 .2 0.9 0.87\n")
    with pytest.warns(UserWarning, match="look like confidences"):
        loose = load_yolo(tmp_path, (100, 100), 2)
    assert loose["x"][0].visible.tolist() == [True, True]
    strict = load_yolo(tmp_path, (100, 100), 2, min_confidence=0.5)
    assert strict["x"][0].visible.tolist() == [False, True]


def test_yolo_keeps_only_the_classes_asked_for(tmp_path) -> None:
    (tmp_path / "x.txt").write_text(
        "0 .5 .5 .2 .2 .1 .1 2 .2 .2 2\n1 .5 .5 .2 .2 .1 .1 2 .2 .2 2\n"
    )
    (tmp_path / "classes.txt").write_text("part_a\npart_b\n")
    assert len(load_yolo(tmp_path, (100, 100), 2, classes=[1])["x"]) == 1


@pytest.mark.parametrize(
    ("k", "values"),
    [
        (2, 3),  # fits neither 2 x 2 (+1) nor 2 x 3 (+1)
        (17, 50),  # one value lost from 17 x 3
    ],
)
def test_yolo_rows_that_fit_no_layout_are_refused(tmp_path, k, values) -> None:
    (tmp_path / "x.txt").write_text("0 .5 .5 .2 .2 " + " ".join(["0.1"] * values))
    with pytest.raises(ValueError, match="x.txt:1"):
        load_yolo(tmp_path, (100, 100), k)


def test_a_stated_layout_refuses_rows_that_only_fit_the_other(tmp_path) -> None:
    """17 x 3 values also parse as 25 x 2 plus a score: only `per_point` can
    tell them apart."""
    (tmp_path / "x.txt").write_text("0 .5 .5 .2 .2 " + " ".join(["0.1"] * 51))
    assert load_yolo(tmp_path, (100, 100), 25)["x"]  # silently fits
    with pytest.raises(ValueError):
        load_yolo(tmp_path, (100, 100), 25, per_point=3)


def test_yolo_single_point_rows_must_say_their_layout(tmp_path) -> None:
    (tmp_path / "x.txt").write_text("0 .5 .5 .2 .2 .1 .1 0.0\n")
    with pytest.raises(ValueError, match="per_point"):
        load_yolo(tmp_path, (100, 100), 1)
    assert load_yolo(tmp_path, (100, 100), 1, per_point=2)["x"][0].visible.all()


def _coco(tmp_path, results):
    gt = {
        "images": [{"id": 7, "file_name": "seven.jpg"}],
        "annotations": [
            {"image_id": 7, "bbox": [0, 0, 10, 10], "keypoints": [1, 2, 2, 3, 4, 2]},
            {"image_id": 7, "bbox": [0, 0, 99, 99], "keypoints": [0] * 6, "iscrowd": 1},
        ],
    }
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "res.json").write_text(json.dumps(results))
    return tmp_path / "gt.json", tmp_path / "res.json"


def test_coco_results_are_named_through_the_annotations(tmp_path) -> None:
    gt, res = _coco(tmp_path, [{"image_id": 7, "keypoints": [1, 2, 0.9, 3, 5, 0.3]}])
    truth = load_coco(gt)
    predicted = load_coco_results(res, gt, min_confidence=0.5)

    assert len(truth["seven.jpg"]) == 1  # the crowd region is left out
    assert predicted["seven.jpg"][0].visible.tolist() == [True, False]


def test_coco_result_for_an_unknown_image_is_refused(tmp_path) -> None:
    gt, res = _coco(tmp_path, [{"image_id": 99, "keypoints": [0] * 6}])
    with pytest.raises(ValueError, match="99"):
        load_coco_results(res, gt)


def box(x1, y1, x2, y2) -> Instance:
    corners = np.array([x1, y1, x2, y2], float)
    return Instance(corners, np.zeros((1, 2)), np.ones(1, bool))


def test_pairs_follow_boxes_not_list_order() -> None:
    left, right = box(0, 0, 10, 10), box(100, 0, 110, 10)
    result = pair({"img": [left, right]}, {"img": [right, left]})
    assert {(id(p.truth), id(p.predicted)) for p in result.pairs} == {
        (id(left), id(left)),
        (id(right), id(right)),
    }


def test_the_best_overlap_wins_whatever_the_truth_order() -> None:
    """Truth `b` fits the only prediction better; listing `a` first must not
    let `a` take it."""
    a, b = box(3, 0, 13, 10), box(0, 0, 10, 10)
    prediction = box(0, 0, 10, 10)
    for order in ([a, b], [b, a]):
        result = pair({"img": order}, {"img": [prediction]})
        assert result.pairs[0].truth is b


def test_unmatched_truth_predictions_and_images_are_reported() -> None:
    result = pair(
        {"a": [box(0, 0, 10, 10)], "b": [box(0, 0, 10, 10)]},
        {"a": [box(500, 500, 510, 510)], "c": [box(0, 0, 1, 1)]},
    )
    assert len(result.pairs) == 0
    assert len(result.missed) == 2
    assert len(result.extra) == 2
    assert result.images_without_prediction == ["b"]
    assert result.images_without_truth == ["c"]


def test_two_point_instances_without_boxes_still_pair() -> None:
    """A box drawn around two points on a vertical line has no area."""
    truth = Instance(
        np.array([0, 0, 100, 200], float),
        np.array([[50.0, 10.0], [50.0, 190.0]]),
        np.ones(2, bool),
    )
    predicted = Instance.from_keypoints(
        np.array([[51.0, 10.0], [52.0, 190.0]]), np.ones(2, bool)
    )
    assert len(pair({"img": [truth]}, {"img": [predicted]}).pairs) == 1


def test_class_ids_are_kept_and_never_paired_across(tmp_path) -> None:
    (tmp_path / "x.txt").write_text("3 .5 .5 .2 .2 .1 .1 2 .2 .2 2\n")
    assert load_yolo(tmp_path, (100, 100), 2)["x"][0].class_id == 3

    part = Instance(
        np.array([0, 0, 10, 10], float), np.zeros((1, 2)), np.ones(1, bool), class_id=0
    )
    other = Instance(
        np.array([0, 0, 10, 10], float), np.zeros((1, 2)), np.ones(1, bool), class_id=1
    )
    also = Instance(
        np.array([50, 50, 60, 60], float),
        np.zeros((1, 2)),
        np.ones(1, bool),
        class_id=0,
    )
    # both sides number classes 0 and 1: a class-0 truth never takes a class-1 box
    result = pair({"img": [part], "b": [also]}, {"img": [other], "b": [also]})
    assert len(result.pairs) == 1 and result.pairs[0].image == "b"


def test_class_numberings_that_share_nothing_are_matched_anyway() -> None:
    """COCO calls a person 1, Ultralytics 0."""
    coco = Instance(
        np.array([0, 0, 10, 10], float), np.zeros((1, 2)), np.ones(1, bool), class_id=1
    )
    yolo = Instance(
        np.array([0, 0, 10, 10], float), np.zeros((1, 2)), np.ones(1, bool), class_id=0
    )
    result = pair({"img": [coco]}, {"img": [yolo]})
    assert len(result.pairs) == 1
    assert any("share no value" in w for w in result.warnings)


def test_skeletons_of_different_sizes_are_refused() -> None:
    three = Instance(np.zeros(4), np.zeros((3, 2)), np.ones(3, bool))
    two = Instance(np.zeros(4), np.zeros((2, 2)), np.ones(2, bool))
    with pytest.raises(ValueError, match="skeletons differ"):
        pair({"img": [three]}, {"img": [two]})


def test_yolo_two_value_layout_reads_zero_zero_as_missing(tmp_path) -> None:
    (tmp_path / "x.txt").write_text("0 .5 .5 .2 .2 .1 .1 0 0\n")
    assert load_yolo(tmp_path, (100, 100), 2)["x"][0].visible.tolist() == [True, False]


def test_names_that_differ_only_by_extension_are_aligned_with_a_warning() -> None:
    result = pair({"x.jpg": [box(0, 0, 10, 10)]}, {"x": [box(0, 0, 10, 10)]})
    assert len(result.pairs) == 1
    assert any("extensions" in w for w in result.warnings)


def test_names_where_only_some_lack_the_extension_are_aligned() -> None:
    truth = {"a.jpg": [box(0, 0, 10, 10)], "b.jpg": [box(0, 0, 10, 10)]}
    predicted = {"a.jpg": [box(0, 0, 10, 10)], "b": [box(0, 0, 10, 10)]}
    result = pair(truth, predicted)
    assert len(result.pairs) == 2 and not result.missed
    assert any("'b'" in w for w in result.warnings)


def test_dropping_the_extension_keeps_the_folder() -> None:
    """cam1/0001.jpg and cam2/0001.jpg are different images."""
    result = pair(
        {"cam1/0001.jpg": [box(0, 0, 10, 10)]}, {"cam2/0001.jpg": [box(0, 0, 10, 10)]}
    )
    assert not result.pairs
    aligned = pair(
        {"cam1/0001.jpg": [box(0, 0, 10, 10)]}, {"cam1/0001": [box(0, 0, 10, 10)]}
    )
    assert len(aligned.pairs) == 1


def test_dropping_the_extension_keeps_the_name_as_given() -> None:
    """On Windows a path object would turn cam1/0001.jpg into cam1\\0001."""
    from poseaudit.pairing import _drop_ext

    assert _drop_ext("cam1/0001.jpg") == "cam1/0001"
    assert _drop_ext("cam1\\0001.JPG") == "cam1\\0001"
    assert _drop_ext("155010.457_x") == "155010.457_x"


def test_no_shared_image_name_is_reported() -> None:
    result = pair({"a.jpg": [box(0, 0, 1, 1)]}, {"b": [box(0, 0, 1, 1)]})
    assert any("No image name" in w for w in result.warnings)


def test_coco_carries_categories(tmp_path) -> None:
    gt, res = _coco(
        tmp_path, [{"image_id": 7, "category_id": 1, "keypoints": [1, 2, 2, 3, 4, 2]}]
    )
    truth = load_coco(gt)
    assert truth["seven.jpg"][0].class_id is None  # no category in that annotation
    assert load_coco_results(res, gt)["seven.jpg"][0].class_id == 1


def test_keypoint_matching_needs_two_shared_points() -> None:
    """One visible point is too little evidence that two instances are the same
    object; in a crowd it steals another's match."""
    truth = Instance(
        np.array([0, 0, 100, 100], float),
        np.array([[50.0, 10.0], [50.0, 90.0]]),
        np.ones(2, bool),
    )
    lonely = Instance.from_keypoints(
        np.array([[50.0, 10.0], [0.0, 0.0]]), np.array([True, False])
    )
    assert not pair({"img": [truth]}, {"img": [lonely]}).pairs


def test_coco_class_filter_keeps_only_those_categories(tmp_path) -> None:
    import json

    from poseaudit import load_coco

    data = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": c, "bbox": [0, 0, 10, 10],
             "keypoints": [1, 1, 2, 5, 5, 2]}
            for c in (1, 2)
        ],
    }  # fmt: skip
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data))
    assert [i.class_id for i in load_coco(path, classes=[2])["a.jpg"]] == [2]
    with pytest.warns(UserWarning, match="categories present"):
        load_coco(path, classes=[7])


def test_a_missing_yolo_folder_is_an_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no such folder"):
        load_yolo(tmp_path / "nope", (100, 100), 2)


def test_rows_that_also_read_as_flagged_triples_are_questioned(tmp_path) -> None:
    """Two points of x y v with flags 0 or 1 (2 would land off the image and
    stop the load), read as three points of x y."""
    (tmp_path / "x.txt").write_text("0 .5 .5 .2 .2 .1 .1 1 .2 .2 1\n")
    with pytest.warns(UserWarning, match="--keypoints"):
        load_yolo(tmp_path, (100, 100), 3)


def test_low_scoring_detections_can_be_dropped(tmp_path) -> None:
    kp = [1, 2, 0.9, 3, 5, 0.9]
    gt, res = _coco(
        tmp_path,
        [
            {"image_id": 7, "keypoints": kp, "score": 0.9},
            {"image_id": 7, "keypoints": kp, "score": 0.1},
        ],
    )
    assert len(load_coco_results(res, gt, 0.5)["seven.jpg"]) == 2
    assert len(load_coco_results(res, gt, 0.5, min_score=0.5)["seven.jpg"]) == 1
    yolo = tmp_path / "yolo"
    yolo.mkdir()
    (yolo / "x.txt").write_text(
        "0 .5 .5 .2 .2 .1 .1 .9 .2 .2 .9 0.95\n0 .5 .5 .2 .2 .1 .1 .9 .2 .2 .9 0.05\n"
    )
    assert (
        len(load_yolo(yolo, (100, 100), 2, min_confidence=0.5, min_score=0.5)["x"]) == 1
    )


def test_non_finite_coordinates_are_refused(tmp_path) -> None:
    gt, res = _coco(tmp_path, [{"image_id": 7, "keypoints": [1, 2, 0.9, 3, 5, 0.9]}])
    (tmp_path / "res.json").write_text(
        '[{"image_id": 7, "keypoints": [Infinity, 2, 0.9, 3, 5, 0.9]}]'
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_coco_results(res, gt)
    (tmp_path / "y").mkdir()
    (tmp_path / "y" / "x.txt").write_text("0 .5 .5 .2 .2 nan .1 .2 .2\n")
    with pytest.raises(ValueError, match="not finite"):
        load_yolo(tmp_path / "y", (100, 100), 2)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"keypoints": [1, 2, 0.9, 3, 5, 0.9], "bbox": [0, 0, "Infinity"]}, "bbox"),
        ({"keypoints": [1, 2, 0.9, 3, 5, 0.9], "bbox": [0, 0, 1e400, 5]}, "non-finite"),
        ({"keypoints": [1, 2, 0.9, 3, 5]}, "triples"),
    ],
)
def test_bad_result_rows_name_the_file_and_image(tmp_path, result, message) -> None:
    gt, res = _coco(tmp_path, [{"image_id": 7, **result}])
    with pytest.raises(ValueError, match=message) as error:
        load_coco_results(res, gt)
    assert "image 7" in str(error.value)


def test_a_nan_score_never_passes_a_score_filter(tmp_path) -> None:
    kp = [1, 2, 0.9, 3, 5, 0.9]
    gt, res = _coco(tmp_path, [{"image_id": 7, "keypoints": kp, "score": 0.9}])
    (tmp_path / "res.json").write_text(
        '[{"image_id": 7, "keypoints": [1, 2, 0.9, 3, 5, 0.9], "score": NaN}]'
    )
    assert load_coco_results(res, gt, 0.5, min_score=0.1).get("seven.jpg", []) == []


@pytest.mark.parametrize("label", ["inf", "1.5", "x"])
def test_yolo_classes_must_be_whole_numbers(tmp_path, label) -> None:
    (tmp_path / "x.txt").write_text(f"{label} .5 .5 .2 .2 .1 .1 .2 .2\n")
    with pytest.raises(ValueError, match="whole number"):
        load_yolo(tmp_path, (100, 100), 2)


def test_a_nan_yolo_score_is_dropped_by_a_score_filter(tmp_path) -> None:
    (tmp_path / "x.txt").write_text(
        "0 .5 .5 .2 .2 .1 .1 .9 .2 .2 .9 nan\n0 .5 .5 .2 .2 .1 .1 .9 .2 .2 .9 .8\n"
    )
    with pytest.warns(UserWarning, match="1 rows with a NaN score"):
        found = load_yolo(tmp_path, (100, 100), 2, min_confidence=0.5, min_score=0.5)
    assert len(found["x"]) == 1
