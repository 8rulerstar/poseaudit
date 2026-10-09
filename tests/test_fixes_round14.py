"""Round 14: a large error for each kind of measure, warnings that several
measures share printed once, a figure a notebook shows inline, ticks that
match across the panels, and loader errors that name the file and the field."""

import json
from pathlib import Path

import numpy as np
import pytest

import poseaudit as pa
from poseaudit import Instance, angle, audit, length, pair, ratio, tilt
from poseaudit.cli import UsageError, _big_errors, main
from poseaudit.report import table, warning_lines

ROOT = Path(__file__).parents[1]
DEMO = ROOT / "examples" / "coco_elbow"
needs_demo = pytest.mark.skipif(
    not (DEMO / "gt_200.json").exists(), reason="no demo data"
)
DEMO_ARGS = ["audit", "--format", "coco", "--gt", str(DEMO / "gt_200.json"),
             "--pred", str(DEMO / "pred_yolo11n.json"), "--resamples", "50",
             "--jitter-repeats", "0"]  # fmt: skip


def _sets(n: int, seed: int = 0, noise: float = 3.0):
    rng = np.random.default_rng(seed)

    def person(points):
        return [Instance.from_keypoints(points, np.ones(len(points), bool))]

    truth, predicted = {}, {}
    for k in range(n):
        points = rng.uniform(0, 100, (4, 2))
        truth[f"i{k}"] = person(points)
        predicted[f"i{k}"] = person(points + rng.normal(0, noise, (4, 2)))
    return truth, predicted


def _result(measure, big=10.0, n=40):
    truth, predicted = _sets(n)
    return audit(pair(truth, predicted), measure, big, resamples=50,
                 jitter_repeats=0)  # fmt: skip


# --- a large error for each kind of measure ----------------------------------

MIXED = [angle(5, 7, 9), tilt(5, 11), length(5, 7), ratio(5, 7, 7, 9)]


def test_big_error_by_kind_by_unit_and_bare() -> None:
    assert _big_errors(["angle:15,tilt:5,length:10,ratio:0.1"], MIXED) == [
        15, 5, 10, 0.1,
    ]  # fmt: skip
    # a unit covers its kinds, a kind wins over its unit, repeating the flag
    # adds to it, and a bare value covers the measures nothing else names
    assert _big_errors(["deg:15,px:10", "ratio:0.5", "tilt:5"], MIXED) == [
        15, 5, 10, 0.5,
    ]  # fmt: skip
    assert _big_errors(["15", "length:10, ratio : 0.1"], MIXED[:1] + MIXED[2:]) == [
        15, 10, 0.1,
    ]  # fmt: skip
    assert _big_errors(["15"], [angle(5, 7, 9), tilt(5, 11)]) == [15, 15]


@pytest.mark.parametrize(
    "given, says",
    [
        # 15 counted as large on a ratio means nothing ever is: the table
        # showed 0.0% for every ratio
        (["15"], "would count an error of 15 degrees on an angle or a tilt, "
                 "15 px on a length and 15 on a ratio as large alike"),
        (["15,ratio:0.1"], "15 degrees on an angle or a tilt and 15 px on a length"),
        (["angle:15,length:10,ratio:0.1"], "no value for tilt 5,11: add tilt:VALUE"),
        (["angel:15"], "KIND one of angle, tilt, length, ratio, deg, px"),
        (["angle:0"], "a number above 0"),
        (["angle:nan"], "a number above 0"),
        (["15,15"], "two values without a kind"),
        (["deg:15", "deg:20"], "gives deg twice"),
    ],
)  # fmt: skip
def test_big_error_refuses_what_cannot_fit(given, says) -> None:
    with pytest.raises(UsageError, match=None) as caught:
        _big_errors(given, MIXED)
    assert says in str(caught.value)


@needs_demo
def test_measures_in_different_units_each_get_their_own(tmp_path, capsys) -> None:
    out = tmp_path / "f.json"
    main(DEMO_ARGS + ["--angle", "5,7,9", "--length", "5,7", "--ratio", "5,7,7,9",
                      "--big-error", "angle:15,length:10,ratio:0.5",
                      "--json", str(out)])  # fmt: skip
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [m["big_error"] for m in data["measures"]] == [15, 10, 0.5]
    ratio_rate = data["measures"][2]["big_error_rate"]
    assert 0.05 < ratio_rate < 0.95  # was 0 at a shared 15
    lines = capsys.readouterr().out.splitlines()
    assert "large if" in lines[0] and "large errors" in lines[0]
    assert ">= 0.5" in lines[3] and ">= 10 px" in lines[2]


@needs_demo
def test_a_bare_big_error_across_units_stops_before_loading(capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        main(DEMO_ARGS + ["--angle", "5,7,9", "--ratio", "5,7,7,9",
                          "--big-error", "15"])  # fmt: skip
    assert "as in --big-error angle:15,length:10,ratio:0.1" in str(stop.value.code)


@needs_demo
@pytest.mark.parametrize("option", [["--threshold", "90"], ["--bands", "0,90,180"]])
def test_options_in_the_measures_unit_are_refused_across_units(option) -> None:
    with pytest.raises(SystemExit) as stop:
        main(DEMO_ARGS + ["--angle", "5,7,9", "--length", "5,7",
                          "--big-error", "angle:15,length:10", *option])  # fmt: skip
    assert "mix degrees and px: audit each unit in a run of its own" in str(
        stop.value.code
    )


@needs_demo
def test_an_option_in_degrees_still_fits_angles_and_tilts(capsys) -> None:
    main(DEMO_ARGS + ["--angle", "5,7,9", "--tilt", "5,11", "--big-error", "15",
                      "--threshold", "30", "--full"])  # fmt: skip
    assert "above 30" in capsys.readouterr().out


def test_compare_takes_a_big_error_for_each_kind() -> None:
    truth, predicted = _sets(30)
    models = {"a": predicted, "b": truth}
    measures = [angle(0, 1, 2), length(0, 1), ratio(0, 1, 2, 3)]
    with pytest.raises(ValueError, match="in degrees, px and ratios alike"):
        pa.compare(models, truth, measures, 15)
    with pytest.raises(ValueError, match="no value for ratio"):
        pa.compare(models, truth, measures, {"angle": 15, "px": 5})
    c = pa.compare(models, truth, measures, {"angle": 15, "px": 5, "ratio": 0.1},
                   resamples=50, jitter_repeats=0)  # fmt: skip
    assert [r.big_error for r in c.results["a"]] == [15, 5, 0.1]
    assert pa.compare(models, truth, [angle(0, 1, 2), tilt(0, 1)], 15,
                      resamples=50, jitter_repeats=0)  # fmt: skip


# --- warnings that several measures share -------------------------------------


def test_shared_advice_is_printed_once_after_each_measures_count() -> None:
    from poseaudit.audit import NEAR_MISS_BOXES

    one, two, three = (_result(m) for m in (angle(0, 1, 2), tilt(0, 1), length(0, 1)))
    one.warnings = [f"16 labelled instances found no match. {NEAR_MISS_BOXES}"]
    two.warnings = [f"10 labelled instances found no match. {NEAR_MISS_BOXES}"]
    three.warnings = []
    lines = warning_lines([one, two, three])
    assert lines == [
        "  ! angle 0,1,2: 16 labelled instances found no match.",
        "  ! tilt 0,1: 10 labelled instances found no match.",
        f"  ! angle 0,1,2, tilt 0,1: {NEAR_MISS_BOXES}",
    ]
    # raised by one measure only, the advice stays on its line
    two.warnings = []
    assert warning_lines([one, two]) == [f"  ! angle 0,1,2: {one.warnings[0]}"]


def test_a_warning_every_measure_raises_is_printed_once() -> None:
    results = [_result(m, n=12) for m in (angle(0, 1, 2), length(0, 1))]
    text = table(results)
    assert text.count("Only 12 readings: every figure here is fragile.") == 1
    assert "  ! every measure: Only 12 readings" in text


@needs_demo
def test_the_demos_near_miss_advice_comes_once(capsys) -> None:
    main(DEMO_ARGS + ["--tilt", "5,11", "--length", "5,7",
                      "--big-error", "tilt:5,length:10"])  # fmt: skip
    out = capsys.readouterr().out
    assert out.count("A prediction too far off") == 1
    assert out.count("found no match but overlap an unmatched prediction") == 2


# --- the figure in a notebook --------------------------------------------------


def test_plot_without_a_path_returns_a_figure_a_notebook_shows(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    r = _result(angle(0, 1, 2))
    fig = r.plot()
    assert not list(tmp_path.iterdir())
    png, size = fig._repr_png_()
    assert png.startswith(b"\x89PNG") and size == {"width": 700, "height": 420}
    # drawn at twice that, for a high-density screen
    assert int.from_bytes(png[16:20], "big") == 1400
    saved = r.plot(str(tmp_path / "p.svg"))
    assert (tmp_path / "p.svg").read_text(encoding="utf-8").startswith("<?xml")
    assert saved.axes[0].get_title(loc="left") == "predicted against truth"


def test_ipython_shows_the_figure_without_pyplot() -> None:
    pytest.importorskip("matplotlib")
    formatters = pytest.importorskip("IPython.core.formatters")
    fig = _result(angle(0, 1, 2)).plot()
    data = formatters.PNGFormatter()(fig)
    assert data is not None and data[0].startswith(b"\x89PNG")


def test_truth_and_prediction_share_their_ticks() -> None:
    """On lengths the shorter y-axis took 0, 50, 100, 150, 200 beside an
    x-axis of 0, 100, 200, though both run over the same limits."""
    pytest.importorskip("matplotlib")
    from poseaudit.plot import _figure

    truth, predicted = _sets(80)
    for measure in (length(0, 1), ratio(0, 1, 2, 3), tilt(0, 1)):
        r = audit(pair(truth, predicted), measure, 10, resamples=50,
                  jitter_repeats=0)  # fmt: skip
        fig = _figure(r)
        fig.draw_without_rendering()  # as saving does: ticks follow the layout
        left, right = fig.axes[:2]
        low, high = left.get_xlim()
        shown = [x for x in left.get_xticks() if low <= x <= high]
        assert len(shown) >= 3
        assert list(left.get_yticks()) == shown == list(right.get_xticks())
        assert left.get_ylim() == (low, high) == right.get_xlim()


def test_ticks_step_by_one_two_or_five() -> None:
    """Matplotlib's own choice put the trap figure's ticks at -8, 0, 8, 16."""
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from poseaudit.plot import _finish

    for low, high in ((-14.0, 19.0), (0.3, 2.9), (-60.0, 45.0)):
        ax = Figure().add_subplot()
        ax.set_ylim(low, high)
        _finish(ax, "px")
        step = float(np.diff(ax.get_yticks())[0])
        mantissa = step / 10 ** np.floor(np.log10(step))
        assert round(mantissa, 6) in (1, 2, 5)


# --- loader errors -------------------------------------------------------------


def _coco(tmp_path, **changes) -> Path:
    data = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 7, "image_id": 1, "bbox": [0, 0, 10, 10], "num_keypoints": 3,
             "keypoints": [1, 1, 2, 5, 5, 2, 9, 1, 2], "category_id": 1}
        ],
    }  # fmt: skip
    data.update(changes)
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_a_box_detectors_results_name_the_missing_keypoints(tmp_path) -> None:
    gt = _coco(tmp_path)
    boxes = tmp_path / "boxes.json"
    boxes.write_text(json.dumps([{"image_id": 1, "bbox": [0, 0, 9, 9],
                                  "score": 0.9, "category_id": 1}]))  # fmt: skip
    with pytest.raises(ValueError) as caught:
        pa.load_coco_results(boxes, gt)
    text = str(caught.value)
    assert text.startswith(f"{boxes}: result 0 has no 'keypoints'")
    assert "a box detector's results" in text and "KeyError" not in text


def test_annotation_fields_and_lists_are_named_when_missing(tmp_path) -> None:
    no_box = _coco(tmp_path, annotations=[{"id": 7, "image_id": 1,
                                           "keypoints": [1, 1, 2]}])  # fmt: skip
    with pytest.raises(ValueError, match=r"annotation 7 has no 'bbox'"):
        pa.load_coco(no_box)
    no_name = _coco(tmp_path, images=[{"id": 1}])
    with pytest.raises(ValueError, match=r"images\[0\] has no 'file_name'"):
        pa.load_coco(no_name)
    path = tmp_path / "images_only.json"
    path.write_text(json.dumps({"images": [{"id": 1, "file_name": "a.jpg"}]}))
    with pytest.raises(ValueError, match="no 'annotations' list"):
        pa.load_coco(path)
    # results need only the image list, so an image-info file names them
    results = tmp_path / "res.json"
    results.write_text(json.dumps([{"image_id": 1, "keypoints": [1, 1, 1] * 3,
                                    "score": 1.0}]))  # fmt: skip
    assert list(pa.load_coco_results(results, path, 0.5)) == ["a.jpg"]


def test_an_unknown_image_id_says_what_to_check(tmp_path) -> None:
    gt = _coco(tmp_path)
    results = tmp_path / "res.json"
    results.write_text(json.dumps([{"image_id": 99, "keypoints": [1, 1, 1] * 3}]))
    with pytest.raises(ValueError, match="another annotation file or split"):
        pa.load_coco_results(results, gt, 0.5)


def test_the_common_errors_in_the_docs_are_ones_the_code_raises() -> None:
    """docs/cli.md quotes the start of each message; the code must still say
    it, or the table would send a reader looking for an error that changed."""
    import re

    docs = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    section = docs.split("## When it stops", 1)[1].split("\n## ", 1)[0]
    quoted = re.findall(r"^\| `([^`]+)`", section, flags=re.M)
    assert len(quoted) >= 10
    source = "".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py")
    )
    flat = re.sub(r'"\s*\n\s*f?"', "", source)  # strings split over lines
    for message in quoted:
        words = message.replace("...", "\0").split("\0")
        assert all(part.strip() in flat for part in words if part.strip()), message
