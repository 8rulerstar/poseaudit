"""The README quotes the demo's output; it must stay what the code prints."""

from pathlib import Path

import pytest

from poseaudit.cli import main

ROOT = Path(__file__).parents[1]
DEMO = ROOT / "examples" / "coco_elbow"
BASE = "audit --format coco --gt gt_200.json --pred pred_yolo11n.json".split() + [
    "--angle", "5,7,9", "--big-error", "15",
]  # fmt: skip

pytestmark = pytest.mark.skipif(
    not (DEMO / "gt_200.json").exists(), reason="demo data is not in this checkout"
)


def printed(capsys, monkeypatch, *extra: str) -> list[str]:
    monkeypatch.chdir(DEMO)
    main(BASE + list(extra))
    return capsys.readouterr().out.rstrip("\n").splitlines()


def test_the_headline_output_is_what_the_demo_prints(capsys, monkeypatch) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for line in printed(capsys, monkeypatch, "--size-bands", "30,60"):
        assert line in readme


def test_the_decision_lines_are_what_the_demo_prints(capsys, monkeypatch) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = printed(
        capsys, monkeypatch, "--threshold", "90", "--side", "below",
        "--pred-threshold", "100",
    )  # fmt: skip
    for line in (x for x in lines if x.strip().startswith("below")):
        assert line in readme


def test_the_size_table_matches_the_report(capsys, monkeypatch, tmp_path) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    printed(
        capsys, monkeypatch, "--size-bands", "30,60", "--report", str(tmp_path / "r.md")
    )
    report = (tmp_path / "r.md").read_text(encoding="utf-8")
    section = report.split("## Error by size")[1].split("\n## ")[0]
    rows = [x for x in section.splitlines() if " px |" in x]
    assert len(rows) == 3
    for row in rows:
        cells = row.split("|")[2:]  # n onwards: the README names the sizes in words
        assert "|".join(cells) in readme
