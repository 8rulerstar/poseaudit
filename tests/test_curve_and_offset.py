"""The curve of each trial (peak, trough, range of motion) and the time
offset found between two .mot files."""

import json
import warnings

import numpy as np
import pytest

import poseaudit as pa
from poseaudit.cli import _pair_each, main
from poseaudit.io.mot import MAX_LAG_STEPS, Motion, estimate_offset, pair_mot

FAST = {"resamples": 200}


def _series(subjects=5, trials=2, frames=60, gain=0.9, shift=2.0, noise=0.0, seed=0):
    """Knee curves per subject and trial; the prediction is the reference
    scaled by `gain` and moved by `shift`, so its peak, trough and range
    are known."""
    rng = np.random.default_rng(seed)
    pred, ref, subject, trial, frame = [], [], [], [], []
    for s in range(subjects):
        for t in range(trials):
            r = 40 + 30 * np.sin(np.linspace(0, 2 * np.pi, frames) + rng.uniform(0, 1))
            pred += list(
                gain * r + shift + rng.normal(0, noise, frames)
                if noise
                else gain * r + shift
            )
            ref += list(r)
            subject += [f"s{s}"] * frames
            trial += [f"t{t}"] * frames
            frame += list(range(frames))
    return pred, ref, subject, trial, frame


def _audit(pred, ref, subject, trial, frame, **kw):
    return pa.audit_values(
        pred,
        ref,
        5,
        name="knee",
        unit="deg",
        subject=subject,
        trial=trial,
        frame=frame,
        **FAST,
        **kw,
    )


def test_peak_trough_and_range_match_the_known_curves():
    data = _series()
    result = _audit(*data)
    by = {c.feature: c for c in result.curve}
    assert [c.feature for c in result.curve] == ["peak", "trough", "range"]
    assert len(result.curve_trials) == 10 and by["peak"].n == 10
    for x in result.curve_trials:
        assert x.pred_peak == pytest.approx(0.9 * x.ref_peak + 2)
        assert x.pred_trough == pytest.approx(0.9 * x.ref_trough + 2)
        assert x.pred_range == pytest.approx(0.9 * x.ref_range)
    peaks = np.array([x.pred_peak - x.ref_peak for x in result.curve_trials])
    assert by["peak"].bias == pytest.approx(peaks.mean())
    assert by["range"].bias == pytest.approx(
        np.mean([-0.1 * x.ref_range for x in result.curve_trials])
    )
    assert by["peak"].bias_ci[0] <= by["peak"].bias <= by["peak"].bias_ci[1]


def test_each_side_takes_its_own_peak():
    """The predicted peak need not fall on the reference's frame."""
    ref = [0, 10, 20, 10, 0, 0]
    pred = [0, 0, 10, 0, 25, 0]
    result = _audit(pred, ref, ["a"] * 6, ["t"] * 6, list(range(6)))
    (x,) = result.curve_trials
    assert (x.pred_peak, x.ref_peak) == (25, 20)
    assert result.curve[0].bias == 5


def test_short_trials_and_no_series_are_left_out():
    pred, ref, subject, trial, frame = _series(subjects=2, trials=1, frames=2)
    result = _audit(pred, ref, subject, trial, frame)
    assert result.curve == [] and result.curve_trials == []
    one = pa.audit_values([1, 2], [1, 2], 5, unit="deg", subject=["a", "b"], **FAST)
    assert one.curve == []  # one value per subject: not a time series


def test_values_near_the_wrap_get_no_peak():
    pred, ref, subject, trial, frame = _series()
    ref = [v + 140 for v in ref]  # up to 210, past the +/-180 wrap
    pred = [v + 140 for v in pred]
    result = _audit(pred, ref, subject, trial, frame)
    assert result.curve == []
    assert any("wrap" in w and "peak" in w for w in result.warnings)


def test_one_subject_has_no_interval():
    result = _audit(*_series(subjects=1, trials=3))
    assert all(np.isnan(c.bias_ci[0]) for c in result.curve)


def test_curve_reaches_json_and_markdown():
    result = _audit(*_series())
    data = result.to_dict()
    assert [c["feature"] for c in data["curve"]] == ["peak", "trough", "range"]
    assert len(data["curve_trials"]) == 10
    text = pa.report.markdown(result)
    assert "## Peak, trough and range of motion per trial" in text
    assert "peak" in pa.report.summary(result)


def _sines(t, lag=0.0):
    x = t + lag
    return {
        "knee": 40
        + 30 * np.sin(2 * np.pi * 0.8 * x)
        + 10 * np.sin(2 * np.pi * 2.1 * x),
        "hip": 20 + 15 * np.sin(2 * np.pi * 0.8 * x + 1),
    }


@pytest.mark.parametrize("lag", [0.0, 0.137, -0.31, 0.05])
def test_offset_found_to_a_fraction_of_a_frame(lag):
    """The prediction shows at time t what the reference shows at t + lag."""
    t_ref = np.arange(0, 5, 0.01)
    t_pred = np.arange(0, 5, 1 / 30)
    ref = Motion(t_ref, _sines(t_ref))
    pred = Motion(t_pred, _sines(t_pred, lag))
    found = estimate_offset(pred, ref)
    assert found.offset == pytest.approx(lag, abs=0.004)
    assert not found.at_edge and found.r > 0.99


def test_offset_beyond_the_search_is_flagged():
    t = np.arange(0, 5, 0.01)
    found = estimate_offset(
        Motion(t, _sines(t, 0.6)), Motion(t, _sines(t)), max_lag=0.3
    )
    assert found.at_edge


def test_offset_survives_units_and_scale():
    """A column in radians, scaled and moved, lines up as the degrees did."""
    t = np.arange(0, 5, 0.01)
    ref = Motion(t, _sines(t))
    moved = {k: np.radians(0.8 * v + 5) for k, v in _sines(t, 0.2).items()}
    found = estimate_offset(Motion(t, moved, in_degrees=False), ref)
    assert found.offset == pytest.approx(0.2, abs=0.004)


def test_pair_mot_auto_uses_and_reports_the_offset():
    t = np.arange(0, 5, 0.01)
    with pytest.warns(UserWarning, match=r"time offset found: \+0\.150"):
        table = pair_mot(
            Motion(t, _sines(t, 0.15)), Motion(t, _sines(t)), time_offset="auto"
        )
    pred = np.array(table["pred"])
    ref = np.array(table["ref"])
    assert np.abs(pred - ref).max() < 0.5  # lined up, not 0.15 s apart
    with pytest.raises(ValueError, match="seconds or 'auto'"):
        pair_mot(Motion(t, _sines(t)), Motion(t, _sines(t)), time_offset="later")


def _write_mot(path, t, columns):
    names = list(columns)
    lines = ["demo", "inDegrees=yes", "endheader", "time\t" + "\t".join(names)]
    lines += [
        "\t".join([f"{t[i]:.4f}"] + [f"{columns[n][i]:.4f}" for n in names])
        for i in range(len(t))
    ]
    path.write_text("\n".join(lines) + "\n")


def test_cli_auto_offset_over_pairs(tmp_path, capsys):
    rows = ["subject,trial,pred,ref"]
    for s in range(3):
        t_ref = np.arange(0, 4, 0.01)
        t_pred = np.arange(0, 4, 1 / 30)
        _write_mot(tmp_path / f"r{s}.mot", t_ref, _sines(t_ref + s))
        _write_mot(tmp_path / f"p{s}.mot", t_pred, _sines(t_pred + s, 0.1))
        rows.append(f"s{s},t,p{s}.mot,r{s}.mot")
    (tmp_path / "pairs.csv").write_text("\n".join(rows) + "\n")
    out = tmp_path / "out.json"
    main(
        [
            "paired",
            "--mot-pairs",
            str(tmp_path / "pairs.csv"),
            "--time-offset",
            "auto",
            "--measure",
            "knee",
            "--big-error",
            "5",
            "--resamples",
            "200",
            "--json",
            str(out),
        ]
    )
    settings = json.loads(out.read_text())["settings"]
    assert settings["time_offset"] == "auto"
    applied = [p["applied"] for p in settings["time_offset_auto"]["per_pair"]]
    assert applied == pytest.approx([0.1] * 3, abs=0.004)
    assert "applied to each" in capsys.readouterr().err


def test_cli_auto_offset_refuses_a_table(tmp_path):
    (tmp_path / "t.csv").write_text("measure,pred,ref\nk,1,1\nk,2,2\n")
    with pytest.raises(SystemExit, match="already paired"):
        main(
            [
                "paired",
                "--table",
                str(tmp_path / "t.csv"),
                "--time-offset",
                "auto",
                "--big-error",
                "5",
            ]
        )


def test_offset_weighs_columns_alike():
    """A column of large unrelated values does not drown the one that lines
    up: each is standardised before they are pooled."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 5, 0.01)
    big_ref = rng.normal(0, 500, len(t))
    big_pred = rng.normal(0, 500, len(t))
    ref = Motion(t, {"knee": _sines(t)["knee"], "pelvis_tx": big_ref})
    pred = Motion(t, {"knee": _sines(t, 0.2)["knee"], "pelvis_tx": big_pred})
    assert estimate_offset(pred, ref).offset == pytest.approx(0.2, abs=0.03)


def _gait(x, cycle=1.1):
    return {
        "knee": 35
        + 30 * np.sin(2 * np.pi * x / cycle)
        + 5 * np.sin(4 * np.pi * x / cycle)
    }


@pytest.mark.parametrize(("lag", "max_lag"), [(-0.6, 1.5), (-0.5, 3.0)])
def test_a_cycle_inside_the_search_names_a_rival(lag, max_lag):
    """A repeating motion lines up again a cycle away: the lag found may be
    the wrong one, and the rival says so."""
    t = np.arange(0, 8, 0.01)
    found = estimate_offset(
        Motion(t, _gait(t + lag)), Motion(t, _gait(t)), max_lag=max_lag
    )
    assert found.rival is not None
    assert abs(found.rival - found.offset) == pytest.approx(1.1, abs=0.02) or abs(
        found.rival - found.offset
    ) == pytest.approx(2.2, abs=0.02)


def test_no_rival_within_half_a_cycle():
    t = np.arange(0, 8, 0.01)
    found = estimate_offset(
        Motion(t, _gait(t + 0.12)), Motion(t, _gait(t)), max_lag=0.5
    )
    assert found.rival is None and found.offset == pytest.approx(0.12, abs=0.004)


def test_rival_reaches_the_warning():
    t = np.arange(0, 8, 0.01)
    with pytest.warns(UserWarning, match="a whole cycle off"):
        pair_mot(
            Motion(t, _gait(t - 0.6)),
            Motion(t, _gait(t)),
            time_offset="auto",
            max_lag=1.5,
        )


def test_curve_figures_say_what_they_are():
    result = _audit(*_series())
    assert any(
        "of each trial" in w and "Filter both sides" in w for w in result.warnings
    )


# One step past the cap, not a huge number: with the cap removed (as a
# mutation test does) a huge one would build billions of lags and eat memory.
_JUST_OVER = (MAX_LAG_STEPS + 1) * 0.01 / 2


@pytest.mark.parametrize("max_lag", [0, -1, float("nan"), float("inf"), _JUST_OVER])
def test_max_lag_out_of_range_is_refused(max_lag):
    t = np.arange(0, 5, 0.01)
    with pytest.raises(ValueError, match="max_lag"):
        estimate_offset(Motion(t, _sines(t)), Motion(t, _sines(t)), max_lag=max_lag)


def _pairs_csv(tmp_path, lags, length=4.0):
    rows = ["subject,trial,pred,ref"]
    for s, lag in enumerate(lags):
        t_ref = np.arange(0, length, 0.01)
        t_pred = np.arange(0, length, 1 / 30)
        _write_mot(tmp_path / f"r{s}.mot", t_ref, _sines(t_ref + s))
        _write_mot(tmp_path / f"p{s}.mot", t_pred, _sines(t_pred + s, lag))
        rows.append(f"s{s},t,p{s}.mot,r{s}.mot")
    (tmp_path / "pairs.csv").write_text("\n".join(rows) + "\n")
    return str(tmp_path / "pairs.csv")


def _auto(pairs, *extra):
    main(
        [
            "paired",
            "--mot-pairs",
            pairs,
            "--time-offset",
            "auto",
            "--measure",
            "knee",
            "--big-error",
            "5",
            "--resamples",
            "200",
            *extra,
        ]
    )


def test_cli_applies_each_pair_its_own_offset(tmp_path):
    """Recordings started apart by different times: one median for all would
    misalign the odd one out by 0.2 s and read its error as the model's."""
    out = tmp_path / "out.json"
    _auto(_pairs_csv(tmp_path, [0.1, 0.1, 0.3]), "--json", str(out))
    result = json.loads(out.read_text())
    applied = [p["applied"] for p in result["settings"]["time_offset_auto"]["per_pair"]]
    assert applied == pytest.approx([0.1, 0.1, 0.3], abs=0.004)
    assert result["mean_abs_error"] < 0.5


def test_cli_warns_when_a_pair_sits_on_the_edge(tmp_path, capsys):
    _auto(_pairs_csv(tmp_path, [0.1, 0.1, 0.45]), "--max-lag", "0.3")
    err = capsys.readouterr().err
    assert "on the edge" in err
    assert "s2 t" in err and "s0 t" not in err


def test_cli_names_the_pair_that_failed(tmp_path):
    pairs = _pairs_csv(tmp_path, [0.1, 0.1])
    t = np.arange(0, 4, 0.01)
    _write_mot(
        tmp_path / "p1.mot",
        t,
        {"knee": np.full(len(t), 5.0), "hip": np.full(len(t), 1.0)},
    )
    with pytest.raises(SystemExit, match="s1"):
        _auto(pairs)


def test_a_tie_a_cycle_apart_takes_the_lag_nearer_zero():
    t = np.arange(0, 8, 0.01)
    sine = lambda x: {"k": np.sin(2 * np.pi * x)}  # noqa: E731
    found = estimate_offset(Motion(t, sine(t - 0.1)), Motion(t, sine(t)), max_lag=1.2)
    assert found.offset == pytest.approx(-0.1, abs=0.004)
    # +0.9 and -1.1 are both a cycle away and tie; which is named varies
    assert min(abs(found.rival - 0.9), abs(found.rival + 1.1)) < 0.02


def test_frames_left_out_are_added_up_over_the_pairs(tmp_path, capsys):
    _auto(_pairs_csv(tmp_path, [0.1, 0.1, 0.1]))
    err = capsys.readouterr().err
    assert "in 3 of 3 pairs" in err
    assert "of 120 predicted frames" not in err  # not one pair's count alone


def test_a_short_trial_near_the_wrap_does_not_void_the_others():
    pred, ref, subject, trial, frame = _series(subjects=3, trials=1)
    pred += [175.0, 176.0]  # two frames: left out, and near the +/-180 wrap
    ref += [175.0, 176.0]
    subject += ["s9", "s9"]
    trial += ["t9", "t9"]
    frame += [0, 1]
    result = _audit(pred, ref, subject, trial, frame)
    assert [c.feature for c in result.curve] == ["peak", "trough", "range"]


def test_pair_mot_takes_auto_in_any_case():
    t = np.arange(0, 5, 0.01)
    with pytest.warns(UserWarning, match="time offset found"):
        pair_mot(Motion(t, _sines(t, 0.1)), Motion(t, _sines(t)), time_offset="AUTO")


def test_a_near_tie_a_cycle_apart_still_takes_the_lag_nearer_zero():
    """Light noise lifts the peak a cycle away a hair above the true one
    (with this seed): within TIE it still counts as tied."""
    t = np.arange(0, 8, 0.01)
    noise = np.random.default_rng(8).normal(0, 0.02, len(t))
    pred = Motion(t, {"k": np.sin(2 * np.pi * (t - 0.1)) + noise})
    found = estimate_offset(pred, Motion(t, {"k": np.sin(2 * np.pi * t)}), max_lag=1.2)
    assert found.offset == pytest.approx(-0.1, abs=0.004)


def test_a_trough_near_the_wrap_gets_no_peak():
    pred, ref, subject, trial, frame = _series()
    ref = [v - 180 for v in ref]  # -170 to -110: only the trough nears the wrap
    pred = [v - 180 for v in pred]
    result = _audit(pred, ref, subject, trial, frame)
    assert result.curve == []


def test_a_trial_of_exactly_min_frames_is_kept():
    result = _audit(*_series(subjects=2, trials=1, frames=3))
    assert len(result.curve_trials) == 2


def test_frames_left_out_differ_by_pair_and_are_summed(tmp_path, capsys):
    """Pairs that leave out different counts: the warning gives their sum."""
    rows = ["subject,trial,pred,ref"]
    for s, end in enumerate([4.0, 4.1, 4.2]):
        t_ref = np.arange(0, 4, 0.01)
        t_pred = np.arange(0, end, 0.05)
        _write_mot(tmp_path / f"r{s}.mot", t_ref, _sines(t_ref))
        _write_mot(tmp_path / f"p{s}.mot", t_pred, _sines(t_pred))
        rows.append(f"s{s},t,p{s}.mot,r{s}.mot")
    (tmp_path / "pairs.csv").write_text("\n".join(rows) + "\n")
    main(
        ["paired", "--mot-pairs", str(tmp_path / "pairs.csv"), "--measure", "knee"]
        + ["--big-error", "5", "--resamples", "200"]
    )
    assert "6 of the predicted frames" in capsys.readouterr().err


def test_other_warnings_of_a_pair_still_show(tmp_path):
    def pair_mot(**_):
        warnings.warn("something else", stacklevel=1)
        return {}

    with pytest.warns(UserWarning, match="something else"):
        _pair_each([{}], pair_mot)


def test_a_pair_far_from_the_others_is_named(tmp_path, capsys):
    _auto(_pairs_csv(tmp_path, [0.1, 0.1, 0.1, 0.3]))
    err = capsys.readouterr().err
    assert "from the median" in err and "s3 t" in err and "s0 t" not in err


def test_pairs_that_agree_are_not_named(tmp_path, capsys):
    _auto(_pairs_csv(tmp_path, [0.1, 0.1, 0.1]))
    assert "from the median" not in capsys.readouterr().err


def test_a_rival_names_the_pair(tmp_path, capsys):
    rows = ["subject,trial,pred,ref"]
    t = np.arange(0, 8, 0.01)
    for s in range(2):
        _write_mot(tmp_path / f"r{s}.mot", t, {"knee": np.sin(2 * np.pi * t)})
        _write_mot(tmp_path / f"p{s}.mot", t, {"knee": np.sin(2 * np.pi * (t - 0.1))})
        rows.append(f"s{s},t,p{s}.mot,r{s}.mot")
    (tmp_path / "pairs.csv").write_text("\n".join(rows) + "\n")
    _auto(str(tmp_path / "pairs.csv"), "--max-lag", "1.2")
    err = capsys.readouterr().err
    assert "may be a whole cycle off" in err and "sync event" in err


def test_unit_is_refused_for_mot_files(tmp_path):
    with pytest.raises(SystemExit, match="carry their own"):
        _auto(_pairs_csv(tmp_path, [0.1]), "--unit", "knee=deg")


def test_a_measure_missing_from_a_file_is_named():
    t = np.arange(0, 5, 0.01)
    with pytest.raises(ValueError, match="no column 'right knee'"):
        estimate_offset(
            Motion(t, _sines(t)), Motion(t, _sines(t)), columns=["right knee"]
        )


def test_a_single_pair_is_lined_up_by_the_offset_found(tmp_path):
    t_ref = np.arange(0, 4, 0.01)
    t_pred = np.arange(0, 4, 1 / 30)
    _write_mot(tmp_path / "r.mot", t_ref, _sines(t_ref))
    _write_mot(tmp_path / "p.mot", t_pred, _sines(t_pred, 0.2))
    out = tmp_path / "o.json"
    main(
        ["paired", "--pred-mot", str(tmp_path / "p.mot"), "--ref-mot"]
        + [str(tmp_path / "r.mot"), "--time-offset", "auto", "--measure", "knee"]
        + ["--big-error", "5", "--resamples", "200", "--json", str(out)]
    )
    result = json.loads(out.read_text())
    assert result["settings"]["time_offset"] == pytest.approx(0.2, abs=0.004)
    assert result["mean_abs_error"] < 0.5
