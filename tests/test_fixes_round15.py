"""Round 15: big_error in Python refused clearly in every wrong form, a
comparison as Markdown tables and as flat rows, the same settings whichever
way a comparison is run, tests that skip without the optional extras, and
docs a newcomer can find their way through."""

import argparse
import csv
import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import poseaudit as pa
from poseaudit import angle, audit, length, pair, tilt
from poseaudit.cli import _parser, main

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"


def _sets(n: int = 30, seed: int = 0, noise: float = 3.0):
    rng = np.random.default_rng(seed)

    def person(points):
        return [pa.Instance.from_keypoints(points, np.ones(len(points), bool))]

    truth, predicted = {}, {}
    for k in range(n):
        points = rng.uniform(0, 100, (4, 2))
        truth[f"i{k}"] = person(points)
        predicted[f"i{k}"] = person(points + rng.normal(0, noise, (4, 2)))
    return truth, predicted


FAST = {"resamples": 50, "jitter_repeats": 0}


TWO = (angle(0, 1, 2), length(0, 1))


def _comparison(measures=TWO, big=None):
    truth, a = _sets(seed=0)
    _, b = _sets(seed=0, noise=6.0)
    _, c = _sets(seed=0, noise=1.0)
    big = big or {"angle": 15, "length": 5}
    return pa.compare({"a": a, "b": b, "c": c}, truth, list(measures), big, **FAST)


# --- big_error in Python ------------------------------------------------------


def test_audit_takes_the_mapping_compare_takes() -> None:
    truth, predicted = _sets()
    pairing = pair(truth, predicted)
    by_map = audit(pairing, angle(0, 1, 2), {"angle": 12, "length": 3}, **FAST)
    by_unit = audit(pairing, angle(0, 1, 2), {"deg": 12}, **FAST)
    plain = audit(pairing, angle(0, 1, 2), 12, **FAST)
    assert by_map.big_error == by_unit.big_error == plain.big_error == 12.0
    assert isinstance(plain.big_error, float)  # 15 and 15.0 print alike in JSON
    assert by_map.big_error_rate == plain.big_error_rate


@pytest.mark.parametrize(
    ("given", "error", "says"),
    [
        ("angle:15,length:10", TypeError, "KIND:VALUE is the command line's form"),
        ("15", TypeError, "big_error is a number, or a mapping"),
        ({"angle": "15"}, TypeError, "got '15'"),
        ({"angel": 15}, ValueError, "'angel', which is neither a kind"),
        ({"ANGLE": 15}, ValueError, "use angle, tilt, length, ratio, deg, px"),
        ({"length": 5}, ValueError, "no value for angle"),
        ({"angle": -1}, ValueError, "above 0"),
        (True, TypeError, "big_error is a number"),
    ],
)
def test_a_big_error_in_the_wrong_form_says_what_to_pass(given, error, says) -> None:
    truth, predicted = _sets()
    with pytest.raises(error, match=re.escape(says)):
        audit(pair(truth, predicted), angle(0, 1, 2), given, **FAST)
    with pytest.raises(error, match=re.escape(says)):
        pa.compare({"a": predicted, "b": predicted}, truth, angle(0, 1, 2), given,
                   **FAST)  # fmt: skip


def test_a_key_that_is_no_kind_is_refused_even_when_unused() -> None:
    """The command line refuses `lenght:10`; so does Python, where a typo
    was dropped silently when no length was audited."""
    truth, predicted = _sets()
    with pytest.raises(ValueError, match="'lenght'"):
        audit(pair(truth, predicted), angle(0, 1, 2), {"angle": 15, "lenght": 10})


def test_several_measures_given_to_audit_are_named() -> None:
    truth, predicted = _sets()
    with pytest.raises(TypeError, match="once for each measure"):
        audit(pair(truth, predicted), [angle(0, 1, 2), tilt(0, 1)], 15)  # type: ignore[arg-type]


def test_compare_still_refuses_one_number_across_units() -> None:
    truth, predicted = _sets()
    with pytest.raises(ValueError, match="degrees and px alike"):
        pa.compare({"a": predicted, "b": predicted}, truth,
                   [angle(0, 1, 2), length(0, 1)], 15, **FAST)  # fmt: skip


# --- a comparison as a table --------------------------------------------------


def test_the_differences_put_each_interval_beside_its_estimate() -> None:
    rows = _comparison().table()
    keys = list(rows[0])
    assert keys[:3] == ["measure", "a", "b"]
    for name in ("mean_abs_error", "big_error_rate"):
        k = keys.index(f"{name}_diff")
        assert keys[k + 1 : k + 3] == [f"{name}_diff_ci_low", f"{name}_diff_ci_high"]
    assert [r["big_error"] for r in rows] == [15.0] * 3 + [5.0] * 3


def test_to_rows_gives_each_models_figures_measure_by_measure() -> None:
    c = _comparison()
    rows = c.to_rows()
    assert [(r["measure"], r["model"]) for r in rows] == [
        ("angle 0,1,2", "a"), ("angle 0,1,2", "b"), ("angle 0,1,2", "c"),
        ("length 0,1", "a"), ("length 0,1", "b"), ("length 0,1", "c"),
    ]  # fmt: skip
    assert rows[0] == {"model": "a", **c.results["a"][0].to_rows()[0]}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    assert buffer.getvalue().startswith("model,measure,unit,n,bias,")


def _table(text: str, heading: str) -> list[list[str]]:
    section = text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("|")]
    return [[c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]] for line in lines]


def test_to_markdown_gives_a_table_of_every_model_and_one_of_the_differences(
    tmp_path,
) -> None:
    c = _comparison()
    text = c.to_markdown(str(tmp_path / "c.md"))
    assert (tmp_path / "c.md").read_text(encoding="utf-8") == text
    assert text.startswith(f"# poseaudit {pa.__version__}: a, b and c compared")
    models = _table(text, "Each model")
    head = models[0]
    assert head[:2] == ["measure", "model"] and "large if" in head
    assert len(models) == 2 + 6  # header, alignment, 2 measures x 3 models
    assert all(len(row) == len(head) for row in models)
    assert [row[:2] for row in models[2:5]] == [
        ["angle 0,1,2", "a"], ["angle 0,1,2", "b"], ["angle 0,1,2", "c"],
    ]  # fmt: skip
    assert models[2][head.index("large if")] == ">= 15°"
    differences = _table(text, "Differences on shared readings")
    assert len(differences) == 2 + 6 and differences[2][1] == "a - b"
    assert "rank models on the" in text and "## Settings" in text


def test_one_shared_threshold_is_named_in_the_header() -> None:
    text = _comparison([angle(0, 1, 2), tilt(0, 1)], {"deg": 10}).to_markdown()
    head = _table(text, "Each model")[0]
    assert ">= 10°" in head and "large if" not in head
    assert "An error of 10 degrees or more counts as large." in text


def test_the_command_line_writes_the_comparison_as_markdown(tmp_path, capsys) -> None:
    from test_several import _labels

    _labels(tmp_path)
    shutil.copytree(tmp_path / "pred", tmp_path / "pred2")
    base = ["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
            "--image-size", "640", "480", "--keypoints", "3", "--big-error", "5",
            "--resamples", "50", "--jitter-repeats", "0", "--angle", "0,1,2",
            "--pred", f"one={tmp_path / 'pred'}",
            "--pred", f"two={tmp_path / 'pred2'}"]  # fmt: skip
    main(
        [*base, "--report", str(tmp_path / "c.md"), "--json", str(tmp_path / "c.json")]
    )
    text = (tmp_path / "c.md").read_text(encoding="utf-8")
    assert "| angle 0,1,2 | one |" in text and "| angle 0,1,2 | two |" in text
    assert re.search(r"\| pred \| one: .*pred; two: .*pred2 \|", text)
    data = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert "model" not in data["settings"] and "pred" not in data["settings"]
    assert data["models"]["two"][0]["settings"]["model"] == "two"
    with pytest.raises(SystemExit, match="--plot draws one model"):
        main([*base, "--plot", str(tmp_path / "p.png")])


def test_compare_in_python_records_each_model_as_the_command_line_does() -> None:
    c = _comparison()
    assert [r[0].settings["model"] for r in c.results.values()] == ["a", "b", "c"]
    assert "model" not in c.to_dict()["settings"]
    assert c.settings() == c.to_dict()["settings"]


def test_the_command_line_and_python_write_the_same_json(tmp_path) -> None:
    """One audit, two paths: every figure and the threshold's type alike."""
    from test_several import _labels

    _labels(tmp_path)
    main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
          "--pred", str(tmp_path / "pred"), "--image-size", "640", "480",
          "--keypoints", "3", "--big-error", "5", "--resamples", "50",
          "--jitter-repeats", "0", "--tilt", "0,1",
          "--json", str(tmp_path / "cli.json")])  # fmt: skip
    truth = pa.load_yolo(tmp_path / "gt", (640, 480), 3)
    predicted = pa.load_yolo(tmp_path / "pred", (640, 480), 3)
    result = audit(pair(truth, predicted), tilt(0, 1), 5, **FAST)
    from poseaudit.cli import _finite

    cli = json.loads((tmp_path / "cli.json").read_text(encoding="utf-8"))
    api = json.loads(json.dumps(_finite(result.to_dict()), default=str))
    for data in (cli, api):
        data.pop("settings")
        data.pop("warnings")
    assert json.dumps(cli, sort_keys=True) == json.dumps(api, sort_keys=True)


# --- the report and the messages ----------------------------------------------


def test_the_report_says_large_error_as_every_other_line_does() -> None:
    truth, predicted = _sets()
    text = audit(pair(truth, predicted), angle(0, 1, 2), 15, **FAST).to_markdown()
    assert "big error >=" not in text
    assert "An error of 15 degrees or more counts as large (`--big-error`" in text


def test_a_skeleton_mismatch_gives_plain_counts() -> None:
    truth, predicted = _sets()
    five = {"i0": [pa.Instance.from_keypoints(np.ones((5, 2)), np.ones(5, bool))]}
    with pytest.raises(ValueError, match=r"truth has 4 keypoints, predictions 5$"):
        pair(truth, five)


def test_plot_with_two_measures_is_named_before_matplotlib_is_needed(
    tmp_path, monkeypatch
) -> None:
    """What the line itself gets wrong comes first: installing Matplotlib
    would not have made two measures drawable."""
    from test_several import _labels, _run

    import poseaudit.plot

    def missing() -> None:
        raise ImportError('plotting needs matplotlib: pip install "poseaudit[plot]"')

    monkeypatch.setattr(poseaudit.plot, "needs_matplotlib", missing)
    _labels(tmp_path)
    with pytest.raises(SystemExit, match="--plot draws one measure"):
        _run(tmp_path, "--angle", "0,1,2", "--tilt", "0,1", "--plot", "p.png")
    with pytest.raises(SystemExit, match=re.escape('"poseaudit[plot]"')):
        _run(tmp_path, "--angle", "0,1,2", "--plot", str(tmp_path / "p.png"))


def test_images_without_pillow_name_the_extra(tmp_path, monkeypatch) -> None:
    from test_several import _labels

    _labels(tmp_path)
    (tmp_path / "img").mkdir()
    monkeypatch.setitem(sys.modules, "PIL", None)  # import PIL now fails
    with pytest.raises(SystemExit, match=re.escape('pip install "poseaudit[images]"')):
        main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
              "--pred", str(tmp_path / "pred"), "--images", str(tmp_path / "img"),
              "--keypoints", "3", "--big-error", "5", "--angle", "0,1,2"])  # fmt: skip


# --- the docs -----------------------------------------------------------------


def _options() -> set[str]:
    parser = _parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {
        option
        for action in sub.choices["audit"]._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }


def test_every_option_is_in_the_docs_table_and_nothing_else() -> None:
    text = (DOCS / "cli.md").read_text(encoding="utf-8")
    section = text.split("## Every option\n", 1)[1].split("\n## ", 1)[0]
    listed = set(re.findall(r"^\| `(--[a-z-]+)", section, flags=re.M))
    assert listed == _options()


def test_the_names_table_gives_keys_the_json_has() -> None:
    truth, predicted = _sets()
    result = audit(pair(truth, predicted), angle(0, 1, 2), 15, noise_ratio=1, **FAST)
    keys = set(result.to_dict())
    columns = set(result.to_rows()[0])
    text = (DOCS / "json.md").read_text(encoding="utf-8")
    section = text.split("## Names\n", 1)[1].split("\n## ", 1)[0]
    rows = [r for r in section.splitlines() if r.startswith("| ") and "---" not in r]
    for row in rows[1:]:
        cells = re.split(r"(?<!\\)\|", row)[1:-1]
        assert set(re.findall(r"`([a-z_0-9]+)`", cells[1])) <= keys, row
        python = re.findall(r"`([a-z_0-9]+)`", cells[2])
        assert all(hasattr(result, name) for name in python), row
        assert set(re.findall(r"`([a-z_0-9]+)`", cells[3])) <= columns, row


def _slug(heading: str) -> str:
    text = re.sub(r"`|\*", "", heading.strip()).lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    return {_slug(h) for h in re.findall(r"^#{1,6} (.*)$", text, flags=re.M)}


REPO = re.compile(
    r"https://(?:github\.com/8rulerstar/poseaudit/(?:blob|tree)/main/"
    r"|raw\.githubusercontent\.com/8rulerstar/poseaudit/main/)(.*)"
)


@pytest.mark.skipif(
    not (ROOT / "examples").is_dir(),
    reason="the source distribution leaves out what some links point to",
)
def test_every_link_in_the_docs_reaches_its_file_and_heading() -> None:
    pages = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *DOCS.glob("*.md")]
    broken = []
    for page in pages:
        text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            repo = REPO.match(target)
            if repo:
                target = str(ROOT / repo.group(1))
            elif re.match(r"https?://|mailto:", target):
                continue
            file, _, anchor = target.partition("#")
            path = (page.parent / file).resolve() if file else page
            if not path.exists():
                broken.append(f"{page.name}: {target}")
            elif anchor and path.suffix == ".md" and anchor not in _anchors(path):
                broken.append(f"{page.name}: {target}")
    assert not broken


def test_the_docs_have_an_index_and_every_page_links_to_the_others() -> None:
    pages = ["cli.md", "python.md", "statistics.md", "json.md"]
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    assert all(f"]({page}" in index for page in pages)
    for page in pages:
        nav = (DOCS / page).read_text(encoding="utf-8").split("\n")[2]
        assert "[index](README.md)" in nav and "[README](../README.md)" in nav
        assert all(f"]({other})" in nav for other in pages if other != page)


def test_the_readme_names_the_extras_its_own_steps_need() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'pip install "poseaudit[plot]"' in text
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for extra in ("plot", "images", "supervision"):
        assert f'pip install "poseaudit[{extra}]"' in index
