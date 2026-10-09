import json
from pathlib import Path

import numpy as np
import pytest

from poseaudit.cli import main
from poseaudit.measures import angle

DEMO = Path(__file__).parent.parent / "examples" / "coco_elbow"
# the demo data is in the repository, not in the sdist: tests that read it
# skip there rather than fail
needs_demo = pytest.mark.skipif(
    not (DEMO / "gt_200.json").exists(), reason="demo data is not in this checkout"
)


@pytest.mark.parametrize(
    ("content", "said"), [("", "is empty"), ("{bad", "not valid JSON")]
)
def test_unreadable_json_names_the_file(tmp_path, content, said) -> None:
    gt = tmp_path / "truth.json"
    gt.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as stop:
        main(["audit", "--format", "coco", "--gt", str(gt),
              "--pred", str(DEMO / "pred_yolo11n.json"),
              "--angle", "5,7,9", "--big-error", "15"])  # fmt: skip
    assert "truth.json" in str(stop.value) and said in str(stop.value)


def test_angle_of_huge_coordinates_stays_finite() -> None:
    k = np.array([[1e300, 0.0], [0.0, 0.0], [0.0, 1e300]])
    assert angle(0, 1, 2).read(k) == pytest.approx(90.0)
    with np.errstate(all="raise"):
        out = angle(0, 1, 2).read_many(np.stack([k, k * 1e-300, np.zeros((3, 2))]))
    assert out[:2] == pytest.approx([90.0, 90.0]) and np.isnan(out[2])


@needs_demo
def test_a_repeated_warning_prints_once(tmp_path, capsys) -> None:
    gt = json.loads((DEMO / "gt_200.json").read_text(encoding="utf-8"))
    for a in gt["annotations"]:
        a["keypoints"] = [v * 1e300 if i % 3 != 2 else v
                          for i, v in enumerate(a["keypoints"])]  # fmt: skip
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt), encoding="utf-8")
    try:
        main(["audit", "--format", "coco", "--gt", str(path),
              "--pred", str(DEMO / "pred_yolo11n.json"),
              "--angle", "5,7,9", "--big-error", "15"])  # fmt: skip
    except SystemExit:
        pass
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == len(set(lines))


def test_cli_doc_commands_have_no_literal_escapes() -> None:
    text = (Path(__file__).parent.parent / "docs" / "cli.md").read_text("utf-8")
    assert "\\n" not in text
