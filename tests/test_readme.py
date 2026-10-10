"""The README and docs/ quote the demo's output; it must stay what the code prints."""

import importlib.util
import re
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


def doc(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def printed(capsys, monkeypatch, *extra: str) -> list[str]:
    monkeypatch.chdir(DEMO)
    main(BASE + list(extra))
    return capsys.readouterr().out.rstrip("\n").splitlines()


def quick_start(readme: str) -> list[str]:
    """The demo command in the README's Quick start, word by word."""
    (line,) = [x for x in readme.splitlines() if x.startswith("poseaudit audit")
               and "gt_200.json" in x]  # fmt: skip
    return line.split()


def test_the_quick_start_runs_the_demo_command() -> None:
    expected = ["poseaudit", *BASE, "--size-bands", "30,60"]
    assert quick_start(doc("README.md")) == expected


def test_the_readme_quotes_what_the_demo_prints(capsys, monkeypatch) -> None:
    readme = " ".join(doc("README.md").split())
    out = "\n".join(printed(capsys, monkeypatch, "--size-bands", "30,60"))
    mean = re.search(r"\|error\| +mean (\S+)° ", out)
    assert mean and f"from {mean.group(1)}° to" in readme
    bands = re.search(r"0-30 px (\d+)% .* 60\+ px (\d+)% ", out)
    assert bands
    small, large = bands.groups()
    assert f"{small}% of elbows disagree with COCO's labels by 15° or more" in readme
    assert f"against {large}% above 60 px" in readme


def test_the_demo_gif_shows_what_the_quick_start_prints(capsys, monkeypatch):
    """docs/demo.gif is drawn from gifs.printed(): the README's command, and
    what the demo prints for it, fitted to the GIF's terminal width."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    spec = importlib.util.spec_from_file_location("gifs", DEMO / "gifs.py")
    assert spec and spec.loader
    gifs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gifs)
    assert " ".join(gifs.COMMAND).split() == quick_start(doc("README.md"))
    piped = printed(capsys, monkeypatch, "--size-bands", "30,60")
    shown = gifs.printed(gifs.COLUMNS)
    assert all(len(line) < gifs.COLUMNS for line in shown)
    assert " ".join(shown).split() == " ".join(piped).split()


def test_the_decision_lines_are_what_the_demo_prints(capsys, monkeypatch) -> None:
    readme = doc("docs/cli.md")
    lines = printed(
        capsys, monkeypatch, "--threshold", "90", "--side", "below",
        "--pred-threshold", "100",
    )  # fmt: skip
    for line in (x for x in lines if x.strip().startswith("below")):
        assert line in readme


def test_the_size_table_matches_the_report(capsys, monkeypatch, tmp_path) -> None:
    readme = doc("docs/statistics.md")
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


def test_the_paper_recipe_quotes_the_demo(tmp_path, capsys, monkeypatch) -> None:
    import json

    readme = doc("docs/statistics.md")
    printed(capsys, monkeypatch, "--json", str(tmp_path / "f.json"))
    d = json.loads((tmp_path / "f.json").read_text(encoding="utf-8"))
    (low, high), (blo, bhi) = d["percentile_limits"], d["bias_ci"]
    mlo, mhi = d["mean_abs_error_ci"]
    ilo, ihi = d["icc_ci"]
    rlo, rhi = d["big_error_rate_ci"]
    sentence = (
        f"read on {d['n']} of {d['measurable']} labelled arms. It had a bias of "
        f"{d['bias']:+.1f}° (95% CI {blo:.1f} to {bhi:+.1f}) with limits of "
        f"agreement of {low:.1f}° to {high:+.1f}° (2.5th to 97.5th percentiles), "
        f"a mean absolute error of {d['mean_abs_error']:.1f}° (95% CI {mlo:.1f} "
        f"to {mhi:.1f}), ICC(A,1) {d['icc']:.2f} (95% CI {ilo:.2f} to "
        f"{ihi:.2f}), and {d['big_error_rate']:.0%} (95% CI {rlo * 100:.0f} to "
        f"{rhi:.0%}) of readings off"
    )
    assert sentence in " ".join(readme.split())
    assert d["mean_abs_error_ci"][1] <= 25 and d["big_error_rate_ci"][1] <= 0.5


def _blocks(readme: str) -> list[str]:
    """The text blocks that show the demo command and what it prints."""
    blocks = [b.split("```")[0] for b in readme.split("```text\n")[1:]]
    return [b for b in blocks if b.startswith("$ poseaudit audit --format coco")]


def test_the_full_block_is_exactly_what_the_demo_prints(capsys, monkeypatch) -> None:
    block = _blocks(doc("docs/cli.md"))[0]
    command, _, output = block.partition("\n    ")[2].partition("\n")
    assert command.split() == ["--angle", "5,7,9", "--big-error", "15",
                               "--size-bands", "30,60", "--full"]  # fmt: skip
    lines = printed(capsys, monkeypatch, "--size-bands", "30,60", "--full")
    assert output == "\n".join(lines) + "\n"


def test_the_arrays_snippet_runs() -> None:
    readme = doc("docs/python.md")
    (snippet,) = [
        b.split("```")[0] for b in readme.split("```python\n")[1:] if "xy_true = " in b
    ]
    space: dict = {}
    exec(snippet, space)
    assert space["result"].n == 1


def test_the_several_joints_block_is_what_the_demo_prints(capsys, monkeypatch):
    readme = doc("docs/cli.md")
    (block,) = [b for b in _blocks(readme) if "--angle 6,8,10" in b]
    command, _, output = block.partition("\n    ")[2].partition("\n")
    monkeypatch.chdir(DEMO)
    main(BASE[:-4] + command.split())
    lines = capsys.readouterr().out.rstrip("\n").splitlines()
    assert output == "\n".join(lines) + "\n"


def test_the_score_filter_comparison_is_what_the_demo_prints(capsys, monkeypatch):
    text = doc("docs/python.md")
    (snippet,) = [
        b.split("```")[0] for b in text.split("```python\n")[1:] if "min_score=s" in b
    ]
    shown = text.split(snippet)[1].split("```text\n")[1].split("```")[0]
    monkeypatch.chdir(DEMO)
    exec(snippet, {})
    assert capsys.readouterr().out.rstrip("\n") == shown.rstrip("\n")
    readme = doc("README.md")
    for figure in ("19.48°", "17.32°", "-0.06° [-0.20 to +0.00]", "254 people"):
        assert figure in readme
    assert "dropped 68 people" in readme
