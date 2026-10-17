"""What a first run shows: the README's Python block, a missing input named by
its option, and which files were written."""

from pathlib import Path

import pytest

from poseaudit.cli import main

ROOT = Path(__file__).parents[1]
DEMO = ROOT / "examples" / "coco_elbow"
BASE = ["audit", "--format", "coco", "--gt", "gt_200.json", "--pred",
        "pred_yolo11n.json", "--angle", "5,7,9", "--big-error", "15"]  # fmt: skip


def test_the_readme_python_block_prints_what_the_quick_start_prints(
    capsys, monkeypatch
):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (block,) = [
        b.split("```")[0]
        for b in readme.split("```python\n")[1:]
        if 'load_coco("gt_200.json")' in b
    ]
    monkeypatch.chdir(DEMO)
    exec(block, {})
    from_python = capsys.readouterr().out
    main([*BASE, "--size-bands", "30,60"])
    assert from_python == capsys.readouterr().out


def test_a_missing_input_is_named_by_its_option(capsys, monkeypatch):
    monkeypatch.chdir(DEMO)
    gt = BASE.index("--gt") + 1
    with pytest.raises(SystemExit) as stop:
        main([*BASE[:gt], "gt_2000.json", *BASE[gt + 1 :]])
    assert str(stop.value.code) == "poseaudit: --gt: gt_2000.json not found"


def test_written_files_are_named_on_stderr_only(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(DEMO)
    report, data = tmp_path / "report.md", tmp_path / "figures.json"
    main([*BASE, "--report", str(report), "--json", str(data)])
    out, err = capsys.readouterr()
    assert report.exists() and data.exists()
    assert f"poseaudit: wrote {data}, {report}" in err
    assert "wrote" not in out
