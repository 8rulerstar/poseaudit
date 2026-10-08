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
