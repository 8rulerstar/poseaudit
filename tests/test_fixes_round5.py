import json
import warnings
from pathlib import Path

import pytest

from poseaudit import cli

DEMO = Path(__file__).parents[1] / "examples" / "coco_elbow"


@pytest.mark.skipif(not (DEMO / "gt_200.json").exists(), reason="no demo data")
def test_compare_prints_each_warning_once_and_keeps_it_in_the_json(
    tmp_path, capsys, monkeypatch
) -> None:
    original = cli._run_compare

    def warned(args, models):
        warnings.warn("a note from the loader", stacklevel=1)
        return original(args, models)

    monkeypatch.setattr(cli, "_run_compare", warned)
    monkeypatch.chdir(DEMO)
    out = tmp_path / "c.json"
    cli.main(
        "audit --format coco --gt gt_200.json --pred a=pred_yolo11n.json "
        "--pred b=pred_yolo11n.json --angle 5,7,9 --big-error 15 "
        f"--resamples 50 --json {out}".split()
    )
    err = capsys.readouterr().err
    assert err.count("a note from the loader") == 1
    assert "a note from the loader" in json.loads(out.read_text("utf-8"))["warnings"]
