"""0.7.2: a label file saved with a byte-order mark reads like one without,
and malformed COCO fields name the file, the entry and the field instead of
raising a bare TypeError."""

import json

import numpy as np
import pytest

from poseaudit.io.coco import load_coco, load_coco_results
from poseaudit.io.yolo import load_yolo

ROW = "0 .5 .5 .2 .2 .1 .1 2 .2 .2 2\n"


def test_a_byte_order_mark_does_not_change_a_yolo_row(tmp_path):
    plain, marked = tmp_path / "plain", tmp_path / "marked"
    plain.mkdir()
    marked.mkdir()
    (plain / "x.txt").write_text(ROW, encoding="utf-8")
    (marked / "x.txt").write_text(ROW, encoding="utf-8-sig")
    a = load_yolo(plain, (100, 100), 2)["x"]
    b = load_yolo(marked, (100, 100), 2)["x"]
    assert len(a) == len(b) == 1
    assert np.array_equal(a[0].keypoints, b[0].keypoints)


GT = {
    "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
    "annotations": [
        {
            "id": 7,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 10, 10],
            "keypoints": [1, 1, 2] * 17,
            "num_keypoints": 17,
        }
    ],
    "categories": [{"id": 1, "name": "person"}],
}
RESULT = {"image_id": 1, "category_id": 1, "keypoints": [1, 1, 2] * 17, "score": 0.9}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"bbox": None}, "bbox must be a list of x, y, width, height; got None"),
        ({"bbox": [0, 0, "10", 10]}, "bbox must hold numbers"),
        ({"score": "0.9"}, "result 0: score must be a number; got '0.9'"),
        ({"score": None}, "result 0: score must be a number; got None"),
        ({"image_id": [1]}, "result 0: image_id must be a number or a string"),
    ],
)
def test_a_malformed_result_field_is_named(tmp_path, change, message):
    (tmp_path / "gt.json").write_text(json.dumps(GT))
    (tmp_path / "pred.json").write_text(json.dumps([RESULT | change]))
    with pytest.raises(ValueError, match=message.replace("[", r"\[")):
        load_coco_results(tmp_path / "pred.json", tmp_path / "gt.json")


def test_a_null_annotation_box_is_named(tmp_path):
    gt = json.loads(json.dumps(GT))
    gt["annotations"][0]["bbox"] = None
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    with pytest.raises(ValueError, match="annotation 7: bbox must be a list"):
        load_coco(tmp_path / "gt.json")
