"""Several measures in one run, and models compared on the readings they share."""

import json

import numpy as np
import pytest

from poseaudit.cli import main

SINGLE_KEYS = [
    "ba_slope", "ba_slope_ci", "band_by", "bands", "bias", "bias_ci", "big_error",
    "big_error_rate", "big_error_rate_ci", "ccc", "ccc_ci", "clusters", "deming",
    "deming_ci", "empirical_limits", "empirical_lower_ci", "empirical_upper_ci",
    "gain", "gain_ci", "gain_gap", "gain_gap_ci", "icc", "icc_ci", "jitter_gain",
    "jitter_gain_range", "jitter_p", "limits", "limits_coverage", "lower_limit_ci",
    "mean_abs_error", "mean_abs_error_ci", "measurable", "measure",
    "median_abs_error", "median_error", "n", "not_read", "offset", "p95_abs_error",
    "pearson", "percentile_limits", "poseaudit", "repeated_limits",
    "repeated_lower_ci", "repeated_upper_ci", "rmse", "rmse_ci", "robust_gain",
    "schema", "settings", "size_bands", "tail_shares", "thresholds",
    "upper_limit_ci", "warnings",
]  # fmt: skip


def _labels(tmp_path, images: int = 12):
    """Three-point skeletons in YOLO labels, the prediction a little off."""
    rng = np.random.default_rng(0)
    for folder in ("gt", "pred"):
        (tmp_path / folder).mkdir()
    for i in range(images):
        xy = rng.uniform(0.2, 0.8, (3, 2))
        moved = xy + rng.normal(0, 0.01, xy.shape)
        for folder, points in (("gt", xy), ("pred", moved)):
            values = " ".join(f"{x:.4f} {y:.4f} 2" for x, y in points)
            (tmp_path / folder / f"im{i}.txt").write_text(f"0 0.5 0.5 1 1 {values}\n")


def _run(tmp_path, *extra):
    main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
          "--pred", str(tmp_path / "pred"), "--image-size", "640", "480",
          "--keypoints", "3", "--big-error", "5", "--resamples", "50",
          "--jitter-repeats", "0", *extra])  # fmt: skip


def test_a_single_measure_keeps_its_json_shape(tmp_path) -> None:
    _labels(tmp_path)
    _run(tmp_path, "--angle", "0,1,2", "--json", str(tmp_path / "r.json"))
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert sorted(data) == SINGLE_KEYS
    assert data["measure"] == {"name": "angle", "points": [0, 1, 2], "unit": "deg"}


def test_several_measures_give_a_table_and_a_list_in_the_json(tmp_path, capsys) -> None:
    _labels(tmp_path)
    files = ["--json", str(tmp_path / "r.json"), "--csv", str(tmp_path / "r.csv")]
    _run(tmp_path, "--angle", "0,1,2", "--tilt", "0,1", "--angle", "1,2,0", *files)
    out = capsys.readouterr().out.splitlines()
    assert out[0].split()[:3] == ["measure", "n", "bias"]
    assert [line.split()[0:2] for line in out[1:4]] == [
        ["angle", "0,1,2"], ["tilt", "0,1"], ["angle", "1,2,0"],
    ]  # fmt: skip
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert sorted(data) == ["measures", "poseaudit", "schema", "settings"]
    assert [m["measure"]["name"] for m in data["measures"]] == [
        "angle",
        "tilt",
        "angle",
    ]
    assert sorted(data["measures"][0]) == SINGLE_KEYS
    rows = (tmp_path / "r.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("measure,image,") and rows[1].startswith(
        '"angle 0,1,2",im'
    )


def test_full_prints_each_measure_in_full(tmp_path, capsys) -> None:
    _labels(tmp_path)
    _run(tmp_path, "--angle", "0,1,2", "--tilt", "0,1", "--full")
    out = capsys.readouterr().out
    assert out.count("  agreement    CCC") == 2
    assert "angle (0, 1, 2): read" in out and "tilt (0, 1): read" in out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--angle", "0,1,2", "--angle", "0,1,2"], "twice"),
        (["--angle", "0,1,2", "--tilt", "0,1", "--plot", "p.png"], "--plot draws one"),
        ([], "at least one of"),
    ],
)
def test_several_measures_refuse_what_they_cannot_do(tmp_path, extra, message):
    _labels(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _run(tmp_path, *extra)
    assert message in str(stop.value)


def _sets(errors_by_model, missing=None, images=40, seed=0):
    """Truth tilts and, per model, the same tilts plus its errors; `missing`
    lists the images a model has no prediction for."""
    from test_audit import leaning

    rng = np.random.default_rng(seed)
    t = rng.uniform(-30, 30, images)
    truth = {f"im{i}": [leaning(t[i])] for i in range(images)}
    predictions = {}
    for name, errors in errors_by_model.items():
        skip = (missing or {}).get(name, ())
        predictions[name] = {
            f"im{i}": [leaning(t[i] + errors[i])]
            for i in range(images)
            if i not in skip
        }
    return truth, predictions


def test_a_known_difference_is_inside_its_paired_interval() -> None:
    import poseaudit as pa

    rng = np.random.default_rng(1)
    a = rng.uniform(0.5, 1.5, 40)
    b = a + 2.0  # every reading 2 degrees further off
    truth, predictions = _sets({"small": a, "large": b})
    c = pa.compare(predictions, truth, pa.tilt(0, 1), big_error=2.5, resamples=200,
                   jitter_repeats=0)  # fmt: skip
    (d,) = c.differences
    assert (d.a, d.b, d.n_shared, d.n_a, d.n_b) == ("small", "large", 40, 40, 40)
    assert d.mean_abs_error_diff == pytest.approx(-2.0)
    low, high = d.mean_abs_error_diff_ci
    assert low == pytest.approx(-2.0) and high == pytest.approx(-2.0)
    assert d.big_error_rate_diff == pytest.approx(-1.0)

    noisy = rng.normal(0, 3, 40)
    truth, predictions = _sets({"x": noisy, "y": noisy + rng.normal(0, 1, 40)})
    (e,) = pa.compare(predictions, truth, pa.tilt(0, 1), 5, resamples=400,
                      jitter_repeats=0).differences  # fmt: skip
    low, high = e.mean_abs_error_diff_ci
    assert low < e.mean_abs_error_diff < high and high - low < 2


def test_only_readings_both_models_made_are_compared() -> None:
    import poseaudit as pa

    errors = np.linspace(-2, 2, 40)
    truth, predictions = _sets({"all": errors, "some": errors * 2},
                               missing={"some": range(5)})  # fmt: skip
    c = pa.compare(predictions, truth, [pa.tilt(0, 1)], 3, resamples=100,
                   jitter_repeats=0)  # fmt: skip
    (d,) = c.differences
    assert (d.n_shared, d.n_a, d.n_b) == (35, 40, 35)
    shared = np.abs(errors[5:])
    assert d.mean_abs_error_diff == pytest.approx(shared.mean() - 2 * shared.mean())
    rows = c.table()
    assert rows[0]["n_shared"] == 35 and "mean_abs_error_diff_ci_low" in rows[0]
    assert c.to_csv().splitlines()[0].startswith("measure,a,b,n_shared")
    assert c.results["all"][0].n == 40


def test_one_image_gives_no_paired_interval() -> None:
    from test_audit import leaning

    import poseaudit as pa

    truth = {"one": [leaning(0), leaning(20)]}
    preds = {"a": {"one": [leaning(1), leaning(21)]},
             "b": {"one": [leaning(3), leaning(23)]}}  # fmt: skip
    (d,) = pa.compare(preds, truth, pa.tilt(0, 1), 5, resamples=50,
                      jitter_repeats=0).differences  # fmt: skip
    assert np.isnan(d.mean_abs_error_diff_ci).all()


def test_the_cli_compares_models_given_by_name(tmp_path, capsys) -> None:
    import shutil

    _labels(tmp_path)
    shutil.copytree(tmp_path / "gt", tmp_path / "exact")
    main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
          "--pred", f"off={tmp_path / 'pred'}", "--pred", f"exact={tmp_path / 'exact'}",
          "--image-size", "640", "480", "--keypoints", "3", "--big-error", "5",
          "--resamples", "50", "--jitter-repeats", "0", "--angle", "0,1,2",
          "--tilt", "0,1", "--json", str(tmp_path / "c.json"),
          "--csv", str(tmp_path / "c.csv")])  # fmt: skip
    out = capsys.readouterr().out
    assert "off:" in out and "exact:" in out and "off - exact" in out
    data = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert list(data["models"]) == ["off", "exact"]
    assert sorted(data["models"]["off"][0]) == SINGLE_KEYS
    assert [d["measure"] for d in data["differences"]] == ["angle 0,1,2", "tilt 0,1"]
    assert all(d["mean_abs_error_diff"] > 0 for d in data["differences"])
    assert data["models"]["exact"][0]["settings"]["pred"].endswith("exact")
    assert (tmp_path / "c.csv").read_text(encoding="utf-8").startswith("measure,a,b")


@pytest.mark.parametrize(
    ("preds", "message"),
    [(["a=x", "y"], "NAME=PATH"), (["a=x", "a=y"], "share a name")],
)
def test_compared_models_need_distinct_names(tmp_path, preds, message) -> None:
    _labels(tmp_path)
    args = [a for p in preds for a in ("--pred", p)]
    with pytest.raises(SystemExit) as stop:
        main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"), *args,
              "--image-size", "640", "480", "--keypoints", "3",
              "--big-error", "5", "--angle", "0,1,2"])  # fmt: skip
    assert message in str(stop.value)
