"""Round 8: COCO image ids written as a number in one file and a string in
the other."""

import json

import pytest

from poseaudit.io import load_coco_results


def _files(tmp_path, gt_id, result_id):
    gt = {
        "images": [{"id": gt_id, "file_name": "a.jpg"}],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }
    result = [{"image_id": result_id, "category_id": 1, "keypoints": [1, 2, 1]}]
    (tmp_path / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
    (tmp_path / "res.json").write_text(json.dumps(result), encoding="utf-8")
    return tmp_path / "res.json", tmp_path / "gt.json"


@pytest.mark.parametrize(("gt_id", "result_id"), [(7, "7"), ("7", 7)])
def test_an_image_id_of_another_type_is_named_as_such(
    tmp_path, gt_id, result_id
) -> None:
    with pytest.raises(ValueError, match="different types"):
        load_coco_results(*_files(tmp_path, gt_id, result_id))


def test_a_missing_image_id_still_says_it_is_not_listed(tmp_path) -> None:
    with pytest.raises(ValueError, match="image_id 8 is not listed") as error:
        load_coco_results(*_files(tmp_path, 7, 8))
    assert "types" not in str(error.value)
