"""Cycles: each trial cut at event times (or where the reference repeats),
and the peak, trough, range and stretched error of each cycle."""

import json

import numpy as np
import pytest

import poseaudit as pa
from poseaudit.cli import main
from poseaudit.cycles import found_starts, starts_for
from poseaudit.report import markdown, summary

FAST = {"resamples": 200}


def _gait(subjects=4, trials=2, length=6.0, gain=0.9, shift=2.0, noise=0.0, seed=0):
    """Knee-like curves of cycles 1.1 to 1.25 s; the prediction is the
    reference scaled by `gain` and moved by `shift`. Returns the columns
    and the true cycle starts by (subject, trial)."""
    rng = np.random.default_rng(seed)
    pred, ref, subject, trial, frame, events = [], [], [], [], [], {}
    for s in range(subjects):
        cycle = 1.1 + 0.05 * s
        for k in range(trials):
            t = np.arange(0, length, 0.01)
            phase = 2 * np.pi * t / cycle
            r = 40 - 30 * np.cos(phase) + 5 * np.sin(2 * phase)
            pred += list(gain * r + shift + rng.normal(0, noise, len(t)))
            ref += list(r)
            subject += [f"s{s}"] * len(t)
            trial += [f"t{k}"] * len(t)
            frame += list(t)
            events[(f"s{s}", f"t{k}")] = list(np.arange(0, length, cycle))
    return (pred, ref, subject, trial, frame), events


def _audit(columns, cycles, **kw):
    pred, ref, subject, trial, frame = columns
    return pa.audit_values(
        pred,
        ref,
        5,
        name="knee",
        unit="deg",
        subject=subject,
        trial=trial,
        frame=frame,
        cycles=cycles,
        **FAST,
        **kw,
    )


def test_each_cycle_gets_its_peak_trough_and_range():
    columns, events = _gait()
    result = _audit(columns, events)
    # whole cycles only: 6 s holds 4 or 5 of each length
    assert len(result.cycles) == sum(len(v) - 1 for v in events.values())
    c = result.cycles[0]
    assert (c.subject, c.trial, c.cycle, c.start) == ("s0", "t0", 1, 0.0)
    assert c.end == pytest.approx(1.1)
    assert c.pred_peak == pytest.approx(0.9 * c.ref_peak + 2, abs=1e-9)
    assert c.pred_range == pytest.approx(0.9 * c.ref_range, abs=1e-9)
    bias = {f.feature: f.bias for f in result.cycle}
    peak = np.mean([x.ref_peak for x in result.cycles])
    assert bias["peak"] == pytest.approx(-0.1 * peak + 2, abs=1e-9)
    assert np.isfinite(result.cycle[0].bias_ci).all()  # four subjects


def test_rmse_over_the_stretched_cycle():
    columns, events = _gait(gain=1.0, shift=3.0)
    result = _audit(columns, events)
    e = result.cycle_error
    assert e.rmse == pytest.approx(3.0, abs=1e-9)
    assert len(e.bias_curve) == 101 and len(e.sd_curve) == 101
    assert e.bias_curve == pytest.approx([3.0] * 101, abs=1e-9)


def test_bias_curve_follows_the_point_of_the_cycle():
    """An error that grows over the cycle shows in the curve, not the mean."""
    pred, ref, subject, trial, frame = _gait(gain=1.0, shift=0.0)[0]
    events = _gait()[1]
    for i, (s, k, t) in enumerate(zip(subject, trial, frame, strict=True)):
        starts = np.array(events[(s, k)])
        a = starts[starts <= t].max()
        b = starts[starts > t].min() if (starts > t).any() else a + 1
        pred[i] += 10 * (t - a) / (b - a)  # 0 at the start, 10 at the end
    result = _audit((pred, ref, subject, trial, frame), events)
    curve = result.cycle_error.bias_curve
    assert curve[0] == pytest.approx(0, abs=0.2)
    assert curve[50] == pytest.approx(5, abs=0.2)
    assert curve[100] == pytest.approx(10, abs=0.3)


def test_events_for_a_subject_or_for_all_apply_to_their_trials():
    events = {("s1", None): [0, 1], (None, None): [5, 6]}
    assert starts_for(events, ("s1", "t9")) == [0.0, 1.0]
    assert starts_for(events, ("s2", "t0")) == [5.0, 6.0]
    assert starts_for({("s1", "t0"): [2, 1]}, ("s1", "t0")) == [1.0, 2.0]
    assert starts_for({("s1", "t0"): [1, 2]}, ("s1", "t1")) is None


def test_a_missed_event_leaves_its_long_cycle_out():
    columns, events = _gait(subjects=2, trials=1)
    events[("s0", "t0")] = [x for i, x in enumerate(events[("s0", "t0")]) if i != 2]
    result = _audit(columns, events)
    assert not any(c.subject == "s0" and c.end - c.start > 2 for c in result.cycles)
    assert any("longer than 1.5" in w for w in result.warnings)


def test_trials_without_events_are_counted():
    columns, events = _gait(subjects=2, trials=1)
    del events[("s1", "t0")]
    result = _audit(columns, events)
    assert {c.subject for c in result.cycles} == {"s0"}
    assert any("1 of 2 trials have no cycle" in w for w in result.warnings)


@pytest.mark.parametrize("cycle", [0.8, 1.1, 1.6])
def test_cycles_found_where_the_reference_repeats(cycle):
    t = np.arange(0, 8, 0.01)
    phase = 2 * np.pi * t / cycle
    y = 40 - 30 * np.cos(phase) + 5 * np.sin(2 * phase)
    starts = found_starts(t, y)
    assert np.diff(starts) == pytest.approx(cycle, abs=0.02)
    # each a trough of the reference
    assert all(y[int(round(s * 100))] < y.min() + 1 for s in starts)


def test_a_curve_that_does_not_repeat_has_no_cycles():
    t = np.arange(0, 5, 0.01)
    assert found_starts(t, np.cumsum(np.random.default_rng(0).normal(size=500))) is None
    assert found_starts(t, np.full(500, 3.0)) is None


def test_auto_finds_about_the_true_cycles():
    columns, events = _gait(noise=1.0)
    auto = _audit(columns, "auto")
    given = _audit(columns, events)
    assert len(auto.cycles) >= len(given.cycles) - 8  # a part cycle a trial
    for a, g in zip(auto.cycle, given.cycle, strict=True):
        assert a.bias == pytest.approx(g.bias, abs=0.5)
    assert any("found from the reference curve" in w for w in auto.warnings)


def test_cycles_need_frames_and_a_known_kind():
    with pytest.raises(ValueError, match="frame or time"):
        pa.audit_values([1, 2, 3], [1, 2, 3], 5, cycles="auto", **FAST)
    columns, _ = _gait(subjects=2, trials=1)
    with pytest.raises(ValueError, match="start times or 'auto'"):
        _audit(columns, "heel")


def test_cycles_near_the_wrap_get_no_figures():
    (pred, ref, *rest), events = _gait(subjects=2, trials=1)
    result = _audit(([v + 120 for v in pred], [v + 120 for v in ref], *rest), events)
    assert result.cycle == [] and result.cycle_error is None
    assert any("no cycle figures" in w for w in result.warnings)


def test_cycles_reach_json_text_and_markdown(tmp_path):
    columns, events = _gait(subjects=2, trials=1)
    result = _audit(columns, events)
    d = result.to_dict()
    assert d["cycles"][0]["cycle"] == 1 and len(d["cycle_error"]["bias_curve"]) == 101
    json.dumps(d)
    assert "cycle RMSE" in summary(result)
    md = markdown(result)
    assert "## Peak, trough and range of motion per cycle" in md
    assert "## Each cycle" in md


def _write_mot(path, t, cols):
    names = list(cols)
    lines = [
        "x",
        "version=1",
        f"nRows={len(t)}",
        f"nColumns={len(names) + 1}",
        "inDegrees=yes",
        "endheader",
        "\t".join(["time", *names]),
    ]
    lines += [
        "\t".join(f"{v:.6f}" for v in [t[i], *(cols[n][i] for n in names)])
        for i in range(len(t))
    ]
    path.write_text("\n".join(lines) + "\n")


def test_cli_cycles_from_a_file_and_auto(tmp_path, capsys):
    t = np.arange(0, 6, 0.01)
    phase = 2 * np.pi * t / 1.2
    r = 40 - 30 * np.cos(phase) + 5 * np.sin(2 * phase)
    _write_mot(tmp_path / "r.mot", t, {"knee": r})
    _write_mot(tmp_path / "p.mot", t, {"knee": r + 2})
    (tmp_path / "ev.csv").write_text("time\n0\n1.2\n2.4\n3.6\n4.8\n")
    base = ["paired", "--pred-mot", str(tmp_path / "p.mot"), "--ref-mot"]
    base += [str(tmp_path / "r.mot"), "--big-error", "5", "--resamples", "200"]
    out = tmp_path / "o.json"
    main([*base, "--cycles", str(tmp_path / "ev.csv"), "--json", str(out)])
    result = json.loads(out.read_text())
    assert len(result["cycles"]) == 4
    assert result["cycle_error"]["rmse"] == pytest.approx(2.0, abs=1e-6)
    assert result["settings"]["cycles"] == str(tmp_path / "ev.csv")
    assert "cycle RMSE" in capsys.readouterr().out
    main([*base, "--cycles", "AUTO", "--json", str(out)])
    assert len(json.loads(out.read_text())["cycles"]) >= 3


def test_cli_cycles_file_needs_times(tmp_path):
    (tmp_path / "ev.csv").write_text("start\n0\n")
    (tmp_path / "bad.csv").write_text("time\n0\nsoon\n")
    t = np.arange(0, 2, 0.01)
    _write_mot(tmp_path / "r.mot", t, {"knee": np.sin(t)})
    base = ["paired", "--pred-mot", str(tmp_path / "r.mot"), "--ref-mot"]
    base += [str(tmp_path / "r.mot"), "--big-error", "5"]
    with pytest.raises(SystemExit, match="needs a column time"):
        main([*base, "--cycles", str(tmp_path / "ev.csv")])
    with pytest.raises(SystemExit, match="row 3"):
        main([*base, "--cycles", str(tmp_path / "bad.csv")])


def test_rmse_is_a_root_mean_square_and_has_an_interval():
    """An error rising from 0 to 10 over the cycle: RMSE 10/sqrt(3), not the
    mean error 5."""
    (pred, ref, subject, trial, frame), events = _gait(gain=1.0, shift=0.0)
    for i, (s, k, t) in enumerate(zip(subject, trial, frame, strict=True)):
        starts = np.array(events[(s, k)])
        a = starts[starts <= t].max()
        b = starts[starts > t].min() if (starts > t).any() else a + 1
        pred[i] += 10 * (t - a) / (b - a)
    e = _audit((pred, ref, subject, trial, frame), events).cycle_error
    assert e.rmse == pytest.approx(10 / np.sqrt(3), abs=0.1)
    assert np.isfinite(e.rmse_ci).all()


def test_a_recording_that_ends_before_its_trough_adds_no_cycle():
    t = np.arange(0, 3.8, 0.01)
    y = 40 - 30 * np.cos(2 * np.pi * t)  # troughs at 0, 1, 2, 3; cut short of 4
    assert found_starts(t, y) == pytest.approx([0, 1, 2, 3], abs=0.02)


def test_cycles_with_too_few_frames_are_left_out():
    (pred, ref, subject, trial, frame), events = _gait(subjects=2, trials=1)
    keep = [i for i in range(len(frame)) if i % 30 == 0]  # one frame each 0.3 s
    pick = [[c[i] for i in keep] for c in (pred, ref, subject, trial, frame)]
    result = _audit(tuple(pick), events)
    assert result.cycles == []
    assert any("under 5 frames" in w for w in result.warnings)
