"""Round 12: a figure that stays legible in one column of a paper, in
grayscale and to colour-blind readers; the report's headings and the section
it points to; the command's help."""

from pathlib import Path

import numpy as np
import pytest

from poseaudit import Instance, angle, audit, pair

ROOT = Path(__file__).parents[1]


def _result(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    def person(points):
        return [Instance.from_keypoints(points, np.ones(3, bool))]

    truth, predicted = {}, {}
    for k in range(n):
        points = rng.uniform(0, 100, (3, 2))
        truth[f"i{k}"] = person(points)
        predicted[f"i{k}"] = person(points + rng.normal(0, 3, (3, 2)))
    return audit(pair(truth, predicted), angle(0, 1, 2), 10, resamples=50,
                 jitter_repeats=0)  # fmt: skip


def _in_view(labels, limits, k: int) -> list:
    """Tick labels matplotlib draws: those of ticks inside the view."""
    low, high = sorted(limits)
    return [t for t in labels if low <= t.get_position()[k] <= high]


def _drawn(result):
    """The figure as `plot` lays it out, drawn, and the texts it shows."""
    matplotlib = pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from poseaudit import plot

    with matplotlib.rc_context(plot.STYLE):
        fig = plot._figure(result)
        FigureCanvasAgg(fig)
        fig.canvas.draw()
    texts = [*fig.texts, *(t for ax in fig.axes for t in ax.texts)]
    for ax in fig.axes:
        if ax.get_legend():
            texts += ax.get_legend().get_texts()
        if not ax.axison:  # the legend's own space
            continue
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += _in_view(ax.get_xticklabels(), ax.get_xlim(), 0)
        texts += _in_view(ax.get_yticklabels(), ax.get_ylim(), 1)
    if fig._suptitle is not None:
        texts.append(fig._suptitle)
    return fig, [t for t in texts if t.get_visible() and t.get_text().strip()]


@pytest.mark.parametrize("n", [1, 5, 300])
def test_every_text_stays_6_points_in_one_column_and_inside_the_figure(n) -> None:
    fig, texts = _drawn(_result(n))
    width = fig.get_size_inches()[0]
    renderer = fig.canvas.get_renderer()
    whole = fig.bbox
    assert len(texts) > 10
    for text in texts:
        assert text.get_fontsize() * 3.5 / width >= 6, text.get_text()
        box = text.get_window_extent(renderer)
        assert whole.x0 - 1 <= box.x0 and box.x1 <= whole.x1 + 1, text.get_text()
        assert whole.y0 - 1 <= box.y0 and box.y1 <= whole.y1 + 1, text.get_text()


def test_each_panels_lines_differ_in_dash_not_only_colour() -> None:
    """A grayscale print keeps the lines apart: no two lines of one panel
    share a dash pattern and a width."""
    fig, _ = _drawn(_result(300))
    for ax in fig.axes[:2]:
        styles = {
            (str(line.get_linestyle()), line.get_linewidth())
            for line in ax.get_lines()
            if line.get_label() and not line.get_label().startswith("_")
        }
        labelled = [x for x in ax.get_lines() if not x.get_label().startswith("_")]
        assert len(styles) == len(labelled) >= 2


def test_under_10_readings_the_plot_draws_no_percentile_limits() -> None:
    """The summary and the JSON give no percentile limits under 10 readings,
    where they are only the smallest and largest errors; the plot drew them."""
    result = _result(5)
    assert "limits       n/a" in result.summary()
    fig, texts = _drawn(result)
    right = fig.axes[1]
    heights = {round(float(line.get_ydata()[0]), 6) for line in right.get_lines()}
    for value in result.empirical_limits:
        assert round(float(value), 6) not in heights
    legend = " ".join(t.get_text() for t in fig.axes[2].get_legend().get_texts())
    assert "percentile" not in legend and "bias ± 1.96 SD" in legend


def test_the_bias_and_limits_carry_their_unit_and_value() -> None:
    result = _result(300)
    fig, texts = _drawn(result)
    shown = {t.get_text() for t in fig.axes[1].texts}
    bias = f"{result.bias:+.1f}°".replace("-", "−")
    assert bias in shown
    low, high = result.percentile_limits
    assert f"{low:+.1f}°".replace("-", "−") in shown
    assert f"{high:+.1f}°".replace("-", "−") in shown


def test_one_reading_is_not_called_readings() -> None:
    fig, _ = _drawn(_result(1))
    assert fig._suptitle.get_text().endswith(": 1 reading")


def test_plotting_leaves_the_users_matplotlib_settings_alone(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    before = dict(matplotlib.rcParams)
    _result(30).plot(str(tmp_path / "p.png"))
    after = dict(matplotlib.rcParams)
    assert {k for k in before if before[k] != after[k]} == set()


def test_value_labels_are_pushed_apart_only_as_far_as_they_need() -> None:
    from poseaudit.plot import _spread

    assert _spread([-50.0, 2.0, 80.0], 10.0) == [-50.0, 2.0, 80.0]
    low, high = _spread([-55.6, -50.9], 10.0)
    assert high - low == pytest.approx(10.0)
    assert (low + high) / 2 == pytest.approx((-55.6 - 50.9) / 2)
    placed = _spread([0.0, 0.1, 0.2, 9.0], 5.0)
    assert np.all(np.diff(placed) >= 5.0 - 1e-9)
    assert np.mean(placed) == pytest.approx(np.mean([0.0, 0.1, 0.2, 9.0]))


def test_the_report_has_headings_and_points_to_a_section_that_exists() -> None:
    text = _result(30).to_markdown()
    assert "\n## Summary\n" in text and "\n## How to read this\n" in text
    assert "README's" not in text
    link = "docs/statistics.md#which-way-you-sort-decides-the-story"
    assert link in text
    statistics = (ROOT / "docs" / "statistics.md").read_text(encoding="utf-8")
    assert "\n## Which way you sort decides the story\n" in statistics


def test_the_reports_numbers_are_right_aligned() -> None:
    text = _result(30).to_markdown()
    assert "| size | n | mean abs error | large errors | slope |\n|---|---:|" in text
    assert "| image | instance | size | truth | predicted | error |\n|---|---:|" in text


def test_the_help_reads_at_a_glance(capsys) -> None:
    from poseaudit.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    top = capsys.readouterr().out
    assert "usage: poseaudit [-h] [--version] COMMAND ..." in top
    assert "measuring" in top and "poseaudit audit --help" in top
    with pytest.raises(SystemExit):
        main(["audit", "--help"])
    out = capsys.readouterr().out
    assert "[--image-size W H]" in out and "W [H ...]" not in out
    assert "--big-error E" in out and "required: an error" in out
    assert "--plot FILE" in out and "PNG, SVG or PDF" in out
    assert "--pred small=res_s.json --pred large=res_l.json" in out
    assert max(len(line) for line in out.splitlines() if "poseaudit" not in line) <= 80


def test_a_lower_min_iou_is_advised_with_the_unlabelled_people_caveat() -> None:
    """On crowd-heavy data the predictions beside missed truths are often
    crowd members: a lower --min-iou pairs them with the wrong person."""
    from poseaudit import tilt

    def bar(degrees):
        rad = np.radians(degrees)
        ends = np.array([[0.0, 0.0], [200 * np.cos(rad), 200 * np.sin(rad)]])
        ends += [300, 300]
        box = np.concatenate([ends.min(axis=0) - 5, ends.max(axis=0) + 5])
        return Instance(box, ends, np.array([True, True]))

    truth = {f"i{k}": [bar(0)] for k in range(40)}
    predicted = {f"i{k}": [bar(0.5 if k < 30 else 20)] for k in range(40)}
    r = audit(pair(truth, predicted), tilt(0, 1), big_error=5)
    (note,) = [w for w in r.warnings if "overlap an unmatched" in w]
    assert "lower --min-iou" in note and "crowd regions" in note
    assert "pairs with the wrong person" in note


def test_a_crowd_warning_says_an_overlapping_prediction_is_paired(tmp_path) -> None:
    import json
    import warnings

    from poseaudit import load_coco

    data = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "iscrowd": 1, "bbox": [0, 0, 50, 50]},
            {"id": 2, "image_id": 1, "iscrowd": 0, "bbox": [0, 0, 10, 10],
             "keypoints": [1, 1, 2, 5, 5, 2], "num_keypoints": 2},
        ],
    }  # fmt: skip
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_coco(path)
    (text,) = [str(w.message) for w in caught]
    assert "1 crowd region (iscrowd 1) was left out and is in no count" in text
    assert "unless its box overlaps a labelled person by --min-iou" in text


@pytest.mark.parametrize("count", [0, None])
def test_a_person_with_null_keypoints_is_left_out_as_in_0_1_4(tmp_path, count):
    """`"keypoints": null` beside `num_keypoints` 0 loaded in 0.1.4 and
    stopped the load since the stale-count fix."""
    import json

    from poseaudit import load_coco

    person = {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
              "bbox": [0, 0, 10, 10], "keypoints": None,
              "num_keypoints": count}  # fmt: skip
    data = {"images": [{"id": 1, "file_name": "a.jpg"}], "annotations": [person]}
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    if count == 0:
        assert load_coco(path) == {"a.jpg": []}
    else:  # a count says there are points, yet there are none: name the file
        with pytest.raises((ValueError, TypeError)):
            load_coco(path)


def test_min_in_frame_under_2_is_refused(capsys) -> None:
    """A reading is read against the others in its image: with one reading
    there are none, and the run died with "SVD did not converge"."""
    from poseaudit import tilt

    r = _result(30)
    with pytest.raises(ValueError, match="min_in_frame must be 2 or more"):
        audit(pair({}, {}), tilt(0, 1), 5, relative_to="median", min_in_frame=1)
    assert r.n == 30


def test_tilts_near_horizontal_name_the_agreement_figures_they_inflate() -> None:
    """Shoulder lines split into two groups near +90 and -90: the spread of
    the truth is huge, so ICC, CCC and r look far too good."""
    from poseaudit import tilt

    rng = np.random.default_rng(3)

    def line(degrees):
        rad = np.radians(degrees)
        ends = np.array([[0.0, 0.0], [100 * np.sin(rad), 100 * np.cos(rad)]])
        return [Instance.from_keypoints(ends + 200, np.array([True, True]))]

    angles = rng.choice([-1, 1], 60) * rng.uniform(80, 89.9, 60)
    truth = {f"i{k}": line(a) for k, a in enumerate(angles)}
    predicted = {f"i{k}": line(a + rng.normal(0, 3)) for k, a in enumerate(angles)}
    r = audit(pair(truth, predicted), tilt(0, 1), 5, resamples=50, jitter_repeats=0)
    (note,) = [w for w in r.warnings if "of horizontal" in w]
    assert "ICC, CCC and r" in note and "errors" in note


def test_models_that_read_every_shared_reading_alike_are_flagged() -> None:
    from poseaudit import compare
    from poseaudit.compare import difference_warnings

    rng = np.random.default_rng(4)

    def person(points):
        return [Instance.from_keypoints(points, np.ones(3, bool))]

    truth, predicted = {}, {}
    for k in range(30):
        points = rng.uniform(0, 100, (3, 2))
        truth[f"i{k}"] = person(points)
        predicted[f"i{k}"] = person(points + rng.normal(0, 3, (3, 2)))
    same = compare({"a": predicted, "b": predicted}, truth, angle(0, 1, 2), 10,
                   resamples=50, jitter_repeats=0)  # fmt: skip
    (d,) = same.differences
    assert d.n_shared > 20 and d.n_differing == 0
    assert any(f"read all {d.n_shared} shared readings the same" in w
               for w in difference_warnings(same.differences))  # fmt: skip
    assert same.table()[0]["n_differing"] == 0
