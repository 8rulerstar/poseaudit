"""Round 13: results that show their summary in a notebook, output that fits
an 80-column terminal, a help that says what each measure reads, warnings
printed as they happen, a report written whatever was read, a long-run note
that counts what costs time, and plot labels and ticks at the edges."""

import io
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from poseaudit import Instance, angle, audit, cli, compare, length, pair, ratio, tilt
from poseaudit.cli import _fit, _say_if_slow, main
from poseaudit.report import layout, table

ROOT = Path(__file__).parents[1]
DEMO = ROOT / "examples" / "coco_elbow"


def _sets(n: int, seed: int = 0, noise: float = 3.0, scale: float = 100.0):
    rng = np.random.default_rng(seed)

    def person(points):
        return [Instance.from_keypoints(points, np.ones(len(points), bool))]

    truth, predicted = {}, {}
    for k in range(n):
        points = rng.uniform(0, scale, (4, 2))
        truth[f"i{k}"] = person(points)
        predicted[f"i{k}"] = person(points + rng.normal(0, noise, (4, 2)))
    return truth, predicted


def _result(n: int = 40, measure=None, **kw):
    truth, predicted = _sets(n, **kw)
    return audit(pair(truth, predicted), measure or angle(0, 1, 2), 10,
                 resamples=50, jitter_repeats=0)  # fmt: skip


class _Printer:
    """What IPython hands `_repr_pretty_`."""

    def __init__(self) -> None:
        self.out = ""

    def text(self, value: str) -> None:
        self.out += value


class _Terminal(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


# --- notebooks ---------------------------------------------------------------


def test_a_result_shows_its_summary_in_a_notebook_and_a_short_repr() -> None:
    """The dataclass repr listed every reading: 80,000 characters for the
    demo's 322, which a notebook printed in full."""
    r = _result(300)
    assert len(repr(r)) < 150
    assert repr(r).startswith(
        f"<AuditResult angle (0, 1, 2): read {r.n} of {r.measurable}, mean |error| "
    )
    printer = _Printer()
    r._repr_pretty_(printer, cycle=False)
    assert printer.out == r.summary()
    empty = audit(pair({"a": []}, {"a": []}), angle(0, 1, 2), 10)
    assert repr(empty) == "<AuditResult angle (0, 1, 2): read 0 of 0>"


def test_a_comparison_and_a_pairing_show_counts_not_every_array() -> None:
    truth, predicted = _sets(200)
    pairing = pair(truth, predicted)
    assert repr(pairing) == (
        "<Pairing: 200 pairs, 0 truths and 0 predictions unmatched, 0 warnings>"
    )
    c = compare({"a": predicted, "b": truth}, truth, angle(0, 1, 2), 10,
                resamples=50, jitter_repeats=0)  # fmt: skip
    assert repr(c) == "<Comparison of a, b: 1 measure, 1 difference>"
    printer = _Printer()
    c._repr_pretty_(printer, cycle=False)
    assert printer.out == c.summary()


# --- the help ----------------------------------------------------------------


def test_the_help_says_what_each_measure_reads(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["audit", "--help"])
    out = capsys.readouterr().out
    assert "I,J,..." not in out
    for option, meaning in [
        ("--angle A,B,C", "angle at B between B-A and B-C"),
        ("--tilt A,B", "from vertical"),
        ("--length A,B", "in pixels"),
        ("--ratio A,B,C,D", "length A-B over length C-D"),
    ]:
        assert f"{option} " in out and meaning in out
    assert "repeat these for several measures" in out
    assert max(len(line) for line in out.splitlines() if "poseaudit" not in line) <= 80


def test_poseaudit_alone_prints_the_help(capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        main([])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "How far off are the joint angles" in err and "audit" in err
    assert "error:" not in err


# --- the terminal ------------------------------------------------------------


def test_a_wide_table_is_cut_into_blocks_that_fit() -> None:
    measures = (angle(0, 1, 2), tilt(0, 1), length(0, 1), ratio(0, 1, 2, 3))
    results = [_result(40, m) for m in measures]
    whole = table(results).splitlines()
    assert max(len(line) for line in whole) > 80  # what was printed before
    cut = table(results, width=80)
    rows = [line for line in cut.splitlines() if not line.startswith("  !")]
    assert max(len(line) for line in rows) <= 80
    for head in ("bias", "limits", "mean |error|", "RMSE", "slope", "ICC(A,1)"):
        assert sum(head in line for line in rows) == 1
    assert sum(line.startswith("measure") for line in rows) >= 2  # each block's
    assert sum(line.startswith("ratio 0,1,2,3") for line in rows) >= 2


def test_figures_line_up_on_the_decimal_point() -> None:
    rows = [["measure", "n", "RMSE"], ["a", "98", "29.57°"], ["b", "1234", "7.61 px"]]
    lines = layout(rows, frozenset({1, 2}))
    assert lines == [
        "measure     n  RMSE",
        "a          98  29.57°",
        "b        1234   7.61 px",
    ]
    assert layout([["x", "slope"], ["a", "n/a"], ["b", "-0.12"]],
                  frozenset({1}))[1:] == ["a  n/a", "b  -0.12"]  # fmt: skip


def test_measures_in_different_units_show_each_threshold_in_its_row() -> None:
    # round 14: the header used to read ">= 10°, 10 px", which cannot say
    # which row is held to which; see test_fixes_round14 for the column
    results = [_result(30, m) for m in (angle(0, 1, 2), tilt(0, 1), length(0, 1))]
    lines = table(results).splitlines()
    assert "large if" in lines[0] and ">= 10°, 10 px" not in lines[0]
    assert ">= 10°" in lines[1] and ">= 10 px" in lines[3]


def test_long_lines_break_between_words_under_their_label() -> None:
    text = "\n".join([
        "  limits       -50.90° to +79.99° (percentile, 2.5th to 97.5th; JSON "
        "percentile_limits)",
        "  ! " + "a long warning " * 10,
        "measure      n  bias    limits              mean |error|             RMSE  x",
        "  below 90°, predicted 98.40° (same count, fitted here): caught 61 of 84 "
        "(73%)",
    ])  # fmt: skip
    lines = _fit(text, 60).splitlines()
    assert lines[0].startswith("  limits       -50.90°") and len(lines[0]) <= 59
    assert lines[1] == " " * 15 + "97.5th; JSON percentile_limits)"
    assert lines[2].startswith("  ! a long warning")
    assert lines[3].startswith("    warning") and lines[4].startswith("    long")
    assert any(x.startswith("measure      n") and len(x) > 60 for x in lines)
    assert lines[-1].startswith("    ") and "(73%)" in lines[-1]
    assert _fit(text, None) == text  # a pipe or a file: lines left whole


def test_only_a_terminal_gets_a_width(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "93")
    assert cli._columns(_Terminal()) == 93
    assert cli._columns(io.StringIO()) is None
    monkeypatch.setenv("COLUMNS", "12")
    assert cli._columns(_Terminal()) == 40  # never narrower than this


@pytest.mark.skipif(not (DEMO / "gt_200.json").exists(), reason="no demo data")
def test_several_measures_fit_an_80_column_terminal(capsys, monkeypatch) -> None:
    monkeypatch.chdir(DEMO)
    monkeypatch.setattr(cli, "_columns", lambda stream: 80)
    main(["audit", "--format", "coco", "--gt", "gt_200.json", "--pred",
          "pred_yolo11n.json", "--angle", "5,7,9", "--tilt", "5,11",
          "--length", "5,7", "--big-error", "15,length:10", "--resamples", "50",
          "--jitter-repeats", "0"])  # fmt: skip
    out = capsys.readouterr().out
    assert max(len(line) for line in out.splitlines()) <= 79
    assert "-50.90° to +79.99°" in out


# --- warnings as they happen -------------------------------------------------


def test_loader_warnings_print_as_raised_and_survive_a_failed_run(
    monkeypatch, capsys
) -> None:
    def run(args):
        warnings.warn("a note from the loader", stacklevel=1)
        print("the audit starts", file=__import__("sys").stderr)
        raise cli.UsageError("then it fails")

    monkeypatch.setattr(cli, "_run", run)
    with pytest.raises(SystemExit) as stop:
        main(["audit", "--gt", "g", "--pred", "p", "--big-error", "1"])
    err = capsys.readouterr().err
    assert err.index("warning: a note from the loader") < err.index("the audit")
    assert "then it fails" in str(stop.value.code)


def test_the_long_run_note_counts_rebuilds_and_measures(capsys) -> None:
    """At 4,900 pairs a measure took 17 s with the default 500 rebuilds and
    3 s without: five measures ran a minute and a half in silence, while
    5,000 pairs without rebuilds were warned of minutes."""
    _say_if_slow(4_900, 2000, 500, 5)
    err = capsys.readouterr().err
    assert "5 measures" in err and "a minute or more" in err
    assert "--jitter-repeats 0" in err
    _say_if_slow(5_000, 2000, 0, 1)
    assert capsys.readouterr().err == ""
    _say_if_slow(4_900, 2000, 500, 1)
    assert capsys.readouterr().err == ""


# --- the report ----------------------------------------------------------------


def _yolo(tmp_path) -> list[str]:
    for folder, shift in (("gt", 0.0), ("pred", 0.1)):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "a.txt").write_text(
            f"0 0.5 0.5 1 1 0.5 0.0 2 {0.5 + shift} 1.0 2 0.2 0.2 2\n"
        )
    return ["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"), "--pred",
            str(tmp_path / "pred"), "--image-size", "100", "100", "--keypoints",
            "3", "--big-error", "5", "--classes", "5"]  # fmt: skip


def test_several_measures_with_nothing_read_write_a_report_not_an_empty_file(
    tmp_path, capsys
) -> None:
    with pytest.raises(SystemExit) as stop:
        main(_yolo(tmp_path) + ["--tilt", "0,1", "--length", "0,2",
                                "--big-error", "length:5",
                                "--report", str(tmp_path / "r.md")])  # fmt: skip
    assert "nothing was read" in str(stop.value.code)
    report = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "tilt (0, 1): read 0 of" in report and "length (0, 2): read 0 of" in report


def test_nothing_read_says_no_plot_was_drawn_and_names_a_stale_one(tmp_path):
    pytest.importorskip("matplotlib")
    old = tmp_path / "p.png"
    old.write_bytes(b"an earlier run")
    with pytest.raises(SystemExit) as stop:
        main(_yolo(tmp_path) + ["--tilt", "0,1", "--plot", str(old)])
    message = str(stop.value.code)
    assert "no plot was drawn" in message and "from an earlier run" in message
    assert old.read_bytes() == b"an earlier run"


def test_the_report_lead_names_the_labels_and_the_count_read() -> None:
    r = _result(40)
    lead = r.to_markdown().split("\n")[2]
    assert " off the labels; " in lead
    assert f"({r.n} of {r.measurable} labelled instances read)" in lead
    assert "agreement with the labels, not error against the true value" in lead


def test_a_class_filter_and_a_crowd_of_that_class_do_not_contradict(
    tmp_path,
) -> None:
    from poseaudit import load_coco

    points = [10, 10, 2, 20, 20, 2]
    data = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 30, 30],
             "keypoints": points, "num_keypoints": 2},
            {"id": 2, "image_id": 1, "category_id": 2, "iscrowd": 1,
             "bbox": [50, 50, 30, 30], "keypoints": [0] * 6, "num_keypoints": 0},
        ],
    }  # fmt: skip
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_coco(path, classes=[2])
    texts = [str(w.message) for w in caught]
    assert any("1 crowd region" in t for t in texts)
    assert not any("no entry has category" in t for t in texts)
    assert any("no labelled person has category [2]" in t for t in texts)


# --- the plot at its edges -----------------------------------------------------


def _figure(result):
    matplotlib = pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from poseaudit import plot

    with matplotlib.rc_context(plot.STYLE):
        fig = plot._figure(result)
        FigureCanvasAgg(fig)
        fig.canvas.draw()
    return fig


def test_lines_at_one_value_share_one_label() -> None:
    """All errors 0: bias and both pairs of limits at 0 printed five
    "+0.0°" labels stacked beside one visible line."""
    truth, _ = _sets(40)
    r = audit(pair(truth, truth), angle(0, 1, 2), 10, resamples=50,
              jitter_repeats=0)  # fmt: skip
    right = _figure(r).axes[1]
    assert [t.get_text() for t in right.texts] == ["+0.0°"]


def test_ticks_over_a_few_degrees_are_whole_steps() -> None:
    r = _result(200, tilt(0, 1), noise=1.0)
    right = _figure(r).axes[1]
    low, high = right.get_ylim()
    assert high - low < 90
    steps = np.diff(right.get_yticks())
    assert np.allclose(steps, steps[0]) and steps[0] not in (1.5, 4.5, 45)


def test_tick_labels_carry_their_value_without_an_offset() -> None:
    """Lengths near 512 px drew "+5.12e2" in the corner; millions drew "1e6"
    over the axis label."""
    rng = np.random.default_rng(0)

    def bar(size):
        points = np.array([[0.0, 0.0], [size, 0.0], [0.0, 1.0], [1.0, 1.0]])
        return [Instance.from_keypoints(points, np.ones(4, bool))]

    for base, prefix in ((512.0, ""), (2e6, "M")):
        sizes = base + rng.uniform(0, 1 if base < 1e3 else 1e6, 60)
        truth = {f"i{k}": bar(s) for k, s in enumerate(sizes)}
        predicted = {f"i{k}": bar(s + rng.normal(0, 0.2)) for k, s in enumerate(sizes)}
        r = audit(pair(truth, predicted, min_keypoint_similarity=0.01), length(0, 1),
                  1, resamples=50, jitter_repeats=0)  # fmt: skip
        fig = _figure(r)
        for ax in fig.axes[:2]:
            for axis in (ax.xaxis, ax.yaxis):
                assert axis.get_offset_text().get_text() == ""
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        if prefix:
            assert any(prefix in t for t in labels)
        else:
            assert any(t.startswith("512") for t in labels)


def test_a_few_readings_are_drawn_over_the_lines() -> None:
    few, many = _figure(_result(3)), _figure(_result(40))
    assert few.axes[1].collections[0].get_zorder() > 3
    assert many.axes[1].collections[0].get_zorder() < 2
