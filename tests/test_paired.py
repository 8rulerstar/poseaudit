"""Paired values (a long table, arrays or OpenSim .mot files), repeated
measures by subject and trial, angle wrapping, and the statistics checked
against published values."""

import csv
import json
import math

import numpy as np
import pytest

import poseaudit as pa
from poseaudit import agreement as ag
from poseaudit.cli import main

FAST = {"resamples": 200}


def _long(subjects: int = 6, trials=("walk1", "walk2"), frames: int = 30, seed=0):
    """A long-format table: knee angles of each subject and trial, the
    prediction off by the subject's own offset plus noise."""
    rng = np.random.default_rng(seed)
    table: dict[str, list] = {k: [] for k in pa.paired.COLUMNS}
    for s in range(subjects):
        offset = rng.normal(0, 3)
        for trial in trials:
            ref = 40 + 30 * np.sin(np.linspace(0, 2 * np.pi, frames))
            pred = ref + offset + rng.normal(0, 2, frames)
            for k in range(frames):
                table["subject"].append(f"S{s}")
                table["trial"].append(trial)
                table["frame"].append(k)
                table["measure"].append("knee")
                table["pred"].append(float(pred[k]))
                table["ref"].append(float(ref[k]))
                table["unit"].append("deg")
    return table


# --- angle wrapping -----------------------------------------------------------


def test_179_against_minus_179_is_2_degrees_apart() -> None:
    knee = pa.paired.quantity("knee", "deg")
    assert knee.period == 360
    assert knee.difference(-179.0, 179.0) == pytest.approx(2.0)
    assert knee.difference(179.0, -179.0) == pytest.approx(-2.0)
    assert knee.difference(10.0, 350.0) == pytest.approx(20.0)
    assert abs(knee.middle(179.0, -179.0)) == pytest.approx(180.0)
    assert ag.wrap(358.0, 360.0) == pytest.approx(-2.0)


def test_wrapping_is_used_in_every_figure() -> None:
    ref = np.array([179.0, -179.0, 178.0, -178.0, 170.0, -170.0] * 5)
    pred = ref + np.tile([2.0, -2.0, 1.0, -1.0, 0.5, -0.5], 5)
    pred = (pred + 180) % 360 - 180  # 181 read as -179
    r = pa.audit_values(pred, ref, big_error=5, unit="deg", **FAST)
    assert max(abs(x.error) for x in r.readings) == pytest.approx(2.0)
    assert r.big_error_rate == 0 and r.mean_abs_error < 2
    # the slope and ICC are taken on truth plus error, not across the wrap
    assert r.gain == pytest.approx(1.0, abs=0.05) and r.icc > 0.99


def test_radians_wrap_at_pi_and_wrap_can_be_turned_off() -> None:
    r = pa.audit_values([-3.1], [3.1], big_error=0.5, unit="rad", resamples=50)
    assert r.readings[0].error == pytest.approx(2 * math.pi - 6.2)
    flat = pa.audit_values([-179.0], [179.0], 5, unit="deg", wrap=False)
    assert flat.readings[0].error == -358.0
    assert flat.measure.period is None


def test_a_length_does_not_wrap_and_a_missing_unit_is_warned_about() -> None:
    r = pa.audit_values([400.0, 1.0], [10.0, 2.0], 5, unit="mm", resamples=50)
    assert r.readings[0].error == 390.0 and r.measure.period is None
    none = pa.audit_values([1.0, 2.0], [1.5, 2.5], 1, resamples=50)
    assert any(w.startswith("No unit given for value") for w in none.warnings)


# --- arrays and tables --------------------------------------------------------


def test_subjects_are_the_clusters_with_summaries_by_subject_and_trial() -> None:
    table = _long()
    (r,) = pa.audit_paired(table, big_error=5, **FAST)
    assert r.n == 360 and r.clusters == 6
    assert r.settings["input"] == "paired" and r.settings["cluster"] == "subject"
    assert r.repeated_limits is not None and r.repeated_lower_ci is not None
    assert [g.subject for g in r.by_subject] == [f"S{s}" for s in range(6)]
    assert len(r.by_trial) == 12 and {g.trial for g in r.by_trial} == {
        "walk1",
        "walk2",
    }
    s0 = [x.error for x in r.readings if x.cluster == "S0"]
    assert r.by_subject[0].bias == pytest.approx(np.mean(s0))
    assert r.by_subject[0].n == 60
    # the subject offsets make the repeated limits wider than the pooled SD
    # would on its own rows; the exact intervals need independent readings
    assert np.isnan(r.lower_limit_exact_ci[0])
    assert r.size_bands == []


def test_the_bootstrap_resamples_subjects_not_frames() -> None:
    table = _long()
    (by_subject,) = pa.audit_paired(table, big_error=5, **FAST)
    loose = {k: v for k, v in table.items() if k not in ("subject", "trial")}
    (by_row,) = pa.audit_paired(loose, big_error=5, **FAST)
    width = by_subject.bias_ci[1] - by_subject.bias_ci[0]
    assert width > 3 * (by_row.bias_ci[1] - by_row.bias_ci[0])
    assert any(w.startswith("No subject given") for w in by_row.warnings)
    assert by_row.repeated_limits is None


def test_a_csv_a_mapping_and_a_dataframe_like_give_the_same(tmp_path) -> None:
    table = _long(subjects=3, frames=10)
    path = tmp_path / "t.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([k.upper() if k == "pred" else k for k in table])
        writer.writerows(zip(*table.values(), strict=True))

    class Frame:  # what audit_paired uses of a pandas DataFrame
        columns = list(table)

        def __getitem__(self, key):
            return np.asarray(table[key])

    results = [
        pa.audit_paired(x, big_error=5, **FAST)[0] for x in (table, path, Frame())
    ]
    assert results[0].to_dict() | {"settings": 0} == results[1].to_dict() | {
        "settings": 0
    }
    assert results[0].bias == results[2].bias


def test_a_real_dataframe_works_too() -> None:
    pd = pytest.importorskip("pandas")
    (r,) = pa.audit_paired(pd.DataFrame(_long(subjects=3)), big_error=5, **FAST)
    assert r.n == 180 and r.clusters == 3


def test_big_error_and_units_by_measure_name_or_unit() -> None:
    table = _long(subjects=3, frames=10)
    extra = {k: list(v) for k, v in table.items()}
    extra["measure"] = ["pelvis_tx"] * len(extra["measure"])
    extra["unit"] = ["m"] * len(extra["unit"])
    both = {k: table[k] + extra[k] for k in table}
    knee, pelvis = pa.audit_paired(both, big_error={"deg": 5, "pelvis_tx": 0.5}, **FAST)
    assert (knee.big_error, pelvis.big_error) == (5, 0.5)
    assert (knee.measure.unit, pelvis.measure.unit) == ("deg", "m")
    assert pelvis.measure.period is None
    with pytest.raises(ValueError, match="no value for pelvis_tx"):
        pa.audit_paired(both, big_error={"deg": 5})
    (only,) = pa.audit_paired(both, 5, measures=["knee"], units={"knee": "rad"})
    assert only.measure.unit == "rad"
    with pytest.raises(ValueError, match="no measure 'hip'"):
        pa.audit_paired(both, 5, measures=["hip"])


def test_missing_values_are_counted_not_scored() -> None:
    r = pa.audit_values(
        [1.0, np.nan, 3.0, 4.0, "", 6.0], [1.5, 2.0, np.nan, 4.5, 5.0, 6.5], 5
    )
    assert r.n == 3
    assert r.not_read.no_predicted_point == 2 and r.not_read.unlabelled == 1
    assert "no predicted value 2" in r.summary()


def test_tables_that_cannot_be_read_say_why(tmp_path) -> None:
    with pytest.raises(ValueError, match="needs columns pred and ref"):
        pa.audit_paired({"pred": [1.0], "reference": [1.0]}, 5)
    mixed = _long(subjects=2, frames=3)
    mixed["unit"][0] = "rad"
    with pytest.raises(ValueError, match="several units"):
        pa.audit_paired(mixed, 5)
    with pytest.raises(ValueError, match="not a number"):
        pa.audit_values(["a"], [1.0], 5)
    with pytest.raises(ValueError, match="trial needs subject"):
        pa.audit_values([1.0], [1.0], 5, trial=["t"])


def test_outputs_name_the_columns_a_biomechanist_expects(tmp_path) -> None:
    (r,) = pa.audit_paired(_long(subjects=3, frames=5), big_error=5, **FAST)
    rows = list(csv.reader(r.to_csv().splitlines()))
    assert rows[0] == ["subject", "trial", "frame", "ref", "pred", "error", "mean"]
    assert rows[1][:3] == ["S0", "walk1", "0.0"]
    text = r.to_markdown()
    assert "## Error by subject\n" in text and "## Error by subject and trial" in text
    assert "| subject | trial | frame | reference | predicted | error |" in text
    assert "off the reference" in text and "Error by size" not in text
    assert "keypoint jitter" not in text
    data = json.loads(json.dumps(r.to_dict(), default=str))
    assert len(data["by_subject"]) == 3 and len(data["by_trial"]) == 6
    assert repr(r).startswith("<AuditResult knee: read 30 of 30")


# --- OpenSim .mot files --------------------------------------------------------


def _mot(path, time, columns, degrees=True, header=True):
    lines = []
    if header:
        unit = "yes" if degrees else "no"
        lines += ["Coordinates", "version=1", f"nRows={len(time)}",
                  f"nColumns={len(columns) + 1}", f"inDegrees={unit}", "",
                  "endheader"]  # fmt: skip
    lines.append("\t".join(["time", *columns]))
    for k, t in enumerate(time):
        lines.append(
            "\t".join(f"{v:.6f}" for v in [t, *(c[k] for c in columns.values())])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_mot_file_is_read_with_its_units(tmp_path) -> None:
    t = np.arange(5) / 100
    path = _mot(tmp_path / "a.mot", t, {"knee_angle_r": t * 10, "pelvis_tx": t})
    m = pa.load_mot(path)
    assert m.time.tolist() == t.tolist() and list(m.columns) == [
        "knee_angle_r",
        "pelvis_tx",
    ]
    assert m.unit("knee_angle_r") == "deg" and m.unit("pelvis_tx") == "m"
    radians = pa.load_mot(_mot(tmp_path / "b.mot", t, {"hip": t}, degrees=False))
    assert radians.unit("hip") == "rad"
    bare = pa.load_mot(_mot(tmp_path / "c.mot", t, {"hip": t}, header=False))
    assert bare.columns["hip"].tolist() == t.tolist()
    (tmp_path / "bad.mot").write_text("endheader\ntime\thip\n0\t1\t2\n")
    with pytest.raises(ValueError, match="3 values under 2 column names"):
        pa.load_mot(tmp_path / "bad.mot")


def test_mot_files_are_aligned_by_time_and_column_name(tmp_path) -> None:
    ref_t = np.arange(0, 2.0001, 0.01)  # motion capture at 100 Hz
    pred_t = np.arange(0, 2.5, 1 / 30)  # video at 30 Hz, running past the end
    knee = lambda t: 40 + 30 * np.sin(2 * np.pi * t)  # noqa: E731
    ref = _mot(tmp_path / "ref.mot", ref_t, {"knee": knee(ref_t), "hip": ref_t})
    pred = _mot(tmp_path / "pred.mot", pred_t + 0.1,
                {"knee": knee(pred_t) + 1, "ankle": pred_t})  # fmt: skip
    with pytest.warns(UserWarning, match="14 of 75 predicted frames"):
        table = pa.pair_mot(pred, ref, subject="S1", trial="t1", time_offset=-0.1)
    assert set(table["measure"]) == {"knee"}  # the only column both hold
    assert max(table["frame"]) <= 2.0 + 1e-9 and len(table["frame"]) == 61
    (r,) = pa.audit_paired(table, big_error=5, resamples=50)
    # linear interpolation of a sine at 100 Hz is off by under 0.02 degrees
    assert r.bias == pytest.approx(1.0, abs=0.02)
    assert r.by_subject[0].subject == "S1" and r.by_trial[0].trial == "t1"
    with pytest.raises(ValueError, match="do not overlap in time"):
        pa.pair_mot(pred, ref, time_offset=10)
    with pytest.raises(ValueError, match="column 'ankle' is not in the reference"):
        pa.pair_mot(pred, ref, columns=["ankle"])
    other = _mot(tmp_path / "o.mot", ref_t, {"elbow": ref_t})
    with pytest.raises(ValueError, match="share no column name"):
        pa.pair_mot(pred, other)


def test_mot_interpolation_goes_the_short_way_round_and_keeps_gaps(tmp_path) -> None:
    ref_t = np.array([0.0, 0.1, 0.2, 0.3])
    ref = _mot(
        tmp_path / "r.mot", ref_t, {"pelvis_rotation": [170.0, -170.0, np.nan, 0]}
    )
    pred = _mot(tmp_path / "p.mot", np.array([0.05, 0.15, 0.3]),
                {"pelvis_rotation": [-179.0, 0.0, 1.0]})  # fmt: skip
    table = pa.pair_mot(pred, ref)
    # halfway from 170 to -170 is 180, not 0
    assert table["ref"][0] == pytest.approx(-180.0) or table["ref"][0] == pytest.approx(
        180.0
    )
    assert math.isnan(table["ref"][1])  # next to a missing reference value
    (r,) = pa.audit_paired(table, big_error=5, resamples=50)
    assert r.n == 2 and r.readings[0].error == pytest.approx(1.0)
    assert r.not_read.unlabelled == 1


# --- the command line ----------------------------------------------------------


def _write_table(path, table) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(table))
        writer.writerows(zip(*table.values(), strict=True))


def test_the_cli_audits_a_table(tmp_path, capsys) -> None:
    _write_table(tmp_path / "t.csv", _long())
    out = [str(tmp_path / f) for f in ("r.json", "r.csv", "r.md")]
    main(["paired", "--table", str(tmp_path / "t.csv"), "--big-error", "deg:5",
          "--resamples", "100", "--json", out[0], "--csv", out[1],
          "--report", out[2], "--full"])  # fmt: skip
    text = capsys.readouterr().out
    assert text.startswith("knee: read 360 of 360 reference values")
    assert "by subject   6 subjects" in text and "repeated" in text
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["settings"]["input"] == "paired" and data["clusters"] == 6
    assert (tmp_path / "r.csv").read_text().startswith("subject,trial,frame,ref")


def test_the_cli_audits_mot_files_and_lists_of_them(tmp_path, capsys) -> None:
    t = np.arange(0, 1, 0.02)
    rows = [["subject", "trial", "pred", "ref"]]
    rng = np.random.default_rng(1)
    for s in ("S1", "S2", "S3"):
        knee = 30 + 20 * np.cos(4 * t)
        _mot(tmp_path / f"{s}_ref.mot", t, {"knee": knee, "pelvis_tx": t})
        noisy = knee + rng.normal(1, 2, len(t))
        _mot(tmp_path / f"{s}_pred.mot", t, {"knee": noisy, "pelvis_tx": t + 0.01})
        rows.append([s, "walk", f"{s}_pred.mot", f"{s}_ref.mot"])
    with open(tmp_path / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    main(["paired", "--mot-pairs", str(tmp_path / "pairs.csv"), "--big-error",
          "deg:5,m:0.05", "--resamples", "100"])  # fmt: skip
    text = capsys.readouterr().out
    assert "knee " in text and "pelvis_tx" in text and "0.05 m" in text
    main(["paired", "--pred-mot", str(tmp_path / "S1_pred.mot"), "--ref-mot",
          str(tmp_path / "S1_ref.mot"), "--measure", "knee", "--big-error", "5",
          "--resamples", "50"])  # fmt: skip
    assert capsys.readouterr().out.startswith("knee: read 50 of 50")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "give one of --table"),
        (["--table", "a.csv", "--mot-pairs", "b.csv"], "got --table and --mot-pairs"),
        (["--pred-mot", "a.mot"], "--pred-mot and --ref-mot go together"),
        (["--table", "a.csv", "--subject", "S1"], "--subject and --trial name"),
        (["--table", "t.csv", "--big-error", "knee:x"], "--big-error takes"),
    ],
)
def test_the_cli_says_what_is_wrong(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.chdir(tmp_path)
    _write_table(tmp_path / "t.csv", _long(subjects=2, frames=3))
    big = [] if "--big-error" in extra else ["--big-error", "5"]
    with pytest.raises(SystemExit) as stop:
        main(["paired", *extra, *big])
    assert message in str(stop.value)


def test_the_cli_refuses_a_plot_of_several_measures(tmp_path) -> None:
    table = _long(subjects=2, frames=3)
    table["measure"] = ["knee", "hip"] * (len(table["measure"]) // 2)
    _write_table(tmp_path / "t.csv", table)
    with pytest.raises(SystemExit, match="--plot draws one measure"):
        main(["paired", "--table", str(tmp_path / "t.csv"), "--big-error", "5",
              "--plot", str(tmp_path / "p.png")])  # fmt: skip


# --- the statistics against published values ------------------------------------

# Shrout PE, Fleiss JL (1979). Intraclass correlations: uses in assessing rater
# reliability. Psychological Bulletin 86(2):420-428, Table 2: six targets rated
# by four judges. Their ICC(2,1), McGraw and Wong's ICC(A,1), is .29.
SHROUT_FLEISS = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                 [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]  # fmt: skip


def test_icc_a1_reproduces_shrout_and_fleiss() -> None:
    assert ag.icc_a1_table(np.array(SHROUT_FLEISS)) == pytest.approx(0.2898, abs=5e-5)
    # their mean squares: targets 11.24, judges 32.49, residual 1.02
    y = np.array(SHROUT_FLEISS, float)
    n, k = y.shape
    ms_r = k * ((y.mean(axis=1) - y.mean()) ** 2).sum() / (n - 1)
    assert ms_r == pytest.approx(11.24, abs=0.005)


def test_icc_a1_of_two_columns_is_the_table_of_two_raters() -> None:
    y = np.array(SHROUT_FLEISS, float)[:, :2]
    assert ag.icc_a1(y[:, 0], y[:, 1]) == ag.icc_a1_table(y)


def test_icc_a1_matches_pingouin_when_installed() -> None:
    pg = pytest.importorskip("pingouin")
    pd = pytest.importorskip("pandas")
    y = np.array(SHROUT_FLEISS, float)
    long = pd.DataFrame(
        [(i, j, y[i, j]) for i in range(6) for j in range(4)],
        columns=["target", "rater", "score"],
    )
    icc = pg.intraclass_corr(long, "target", "rater", "score").set_index("Type")
    assert ag.icc_a1_table(y) == pytest.approx(icc.loc["ICC2", "ICC"])


# Bland JM, Altman DG (1986). Lancet 327(8476):307-310, Table 1: peak
# expiratory flow rate (l/min) of 17 subjects, the first reading with the
# Wright meter and with the mini Wright meter. The paper gives a mean
# difference of -2.1 and an SD of 38.8, and limits of -79.7 to 75.5 with 2 SD.
WRIGHT = [494, 395, 516, 434, 476, 557, 413, 442, 650, 433, 417, 656, 267, 478,
          178, 423, 427]  # fmt: skip
MINI = [512, 430, 520, 428, 500, 600, 364, 380, 658, 445, 432, 626, 260, 477,
        259, 350, 451]  # fmt: skip


def test_bland_altman_reproduces_the_lancet_worked_example() -> None:
    r = pa.audit_values(WRIGHT, MINI, big_error=50, unit="l/min", resamples=200)
    assert r.bias == pytest.approx(-2.1, abs=0.05)
    sd = (r.limits[1] - r.limits[0]) / (2 * 1.959963984540054)
    assert sd == pytest.approx(38.8, abs=0.05)
    assert r.bias - 2 * sd == pytest.approx(-79.7, abs=0.1)
    assert r.bias + 2 * sd == pytest.approx(75.5, abs=0.1)
    # one reading per subject: the exact intervals of the limits are given
    assert np.isfinite(r.lower_limit_exact_ci).all()
    assert "exact CI (Carkeet)" in r.to_markdown()
    assert "(exact, Carkeet 2015)" in r.summary(full=True)


def test_the_exact_limit_intervals_use_the_noncentral_t() -> None:
    d = np.array(WRIGHT, float) - np.array(MINI, float)
    lower, upper = ag.limits_exact_ci(d)
    n, mean, sd = len(d), d.mean(), d.std(ddof=1)
    # symmetric about the bias, and wider than the limits' own spread
    assert lower[0] - mean == pytest.approx(-(upper[1] - mean))
    assert lower[0] < ag.limits(d)[0] < lower[1]
    # the 2.5th and 97.5th percentiles of scipy.stats.nct(16, 1.959964 * sqrt(17))
    assert upper[0] == pytest.approx(mean + sd * 5.422004997786 / math.sqrt(n))
    assert upper[1] == pytest.approx(mean + sd * 12.980655683840 / math.sqrt(n))


def test_the_noncentral_t_agrees_with_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    for n in (3, 10, 40, 500):
        nc = 1.959963984540054 * math.sqrt(n)
        for q in (0.025, 0.5, 0.975):
            assert ag.nct_ppf(q, n - 1, nc) == pytest.approx(
                stats.nct.ppf(q, n - 1, nc), rel=1e-9
            )
        assert ag.nct_cdf(nc, n - 1, nc) == pytest.approx(
            stats.nct.cdf(nc, n - 1, nc), abs=1e-9
        )


def test_the_exact_limit_intervals_cover_as_stated() -> None:
    """Simulated normal differences: each exact interval holds its limit in
    about 95% of samples."""
    rng = np.random.default_rng(3)
    z = 1.959963984540054
    held = []
    for _ in range(400):
        lower, upper = ag.limits_exact_ci(rng.normal(0, 1, 15))
        held.append((lower[0] <= -z <= lower[1], upper[0] <= z <= upper[1]))
    rate = np.mean(held, axis=0)
    assert ((rate > 0.92) & (rate < 0.98)).all()


# --- what is experimental --------------------------------------------------------


def test_the_jitter_reference_is_marked_experimental() -> None:
    from poseaudit.audit import EXPERIMENTAL

    rng = np.random.default_rng(0)
    truth, pred = {}, {}
    for i in range(40):
        xy = rng.uniform(100, 400, (3, 2))
        truth[f"im{i}"] = [pa.Instance.from_keypoints(xy, np.ones(3, bool))]
        moved = xy + rng.normal(0, 3, xy.shape)
        pred[f"im{i}"] = [pa.Instance.from_keypoints(moved, np.ones(3, bool))]
    r = pa.audit(pa.pair(truth, pred), pa.angle(0, 1, 2), 10, resamples=100,
                 jitter_repeats=50)  # fmt: skip
    assert r.to_dict()["experimental"] == list(EXPERIMENTAL)
    full = r.summary(full=True).splitlines()
    at = next(k for k, x in enumerate(full) if x.strip().startswith("experimental"))
    assert full[at + 1].strip().startswith("vs jitter")
    assert "(experimental" in r.to_markdown()


def test_named_clusters_of_keypoint_audits_get_a_summary_each() -> None:
    rng = np.random.default_rng(0)
    truth, pred = {}, {}
    for i in range(30):
        xy = rng.uniform(100, 400, (3, 2))
        truth[f"clip{i % 5}_{i}"] = [pa.Instance.from_keypoints(xy, np.ones(3, bool))]
        moved = xy + rng.normal(0, 3, xy.shape)
        pred[f"clip{i % 5}_{i}"] = [pa.Instance.from_keypoints(moved, np.ones(3, bool))]
    r = pa.audit(pa.pair(truth, pred), pa.angle(0, 1, 2), 10, resamples=100,
                 jitter_repeats=0, cluster=lambda n: n.split("_")[0])  # fmt: skip
    assert [g.subject for g in r.by_subject] == [f"clip{k}" for k in range(5)]
    assert all(g.trial is None and g.n == 6 for g in r.by_subject)
    assert "## Error by cluster\n" in r.to_markdown()
    plain = pa.audit(pa.pair(truth, pred), pa.angle(0, 1, 2), 10, resamples=100,
                     jitter_repeats=0)  # fmt: skip
    assert plain.by_subject == []


def test_the_paired_block_in_the_docs_is_what_the_example_prints(
    capsys, monkeypatch
) -> None:
    from pathlib import Path

    root = Path(__file__).parents[1]
    example = root / "examples" / "paired"
    if not (example / "angles.csv").exists():
        pytest.skip("the example data is not in this checkout")
    command = "$ poseaudit paired --table angles.csv --big-error 5\n"
    text = (root / "docs" / "cli.md").read_text(encoding="utf-8")
    shown = text.split(command)[1].split("```")[0]
    monkeypatch.chdir(example)
    main(command.split()[2:])
    assert capsys.readouterr().out == shown


# --- tables as spreadsheets save them, and columns named otherwise ------------


def test_a_semicolon_csv_with_decimal_commas_reads_as_numbers(tmp_path) -> None:
    path = tmp_path / "excel.csv"
    path.write_text(
        "subject;pred;ref;unit\nA;10,5;11,0;deg\nB;20,1;19,0;deg\nC;-3,25;-2;deg\n",
        encoding="utf-8",
    )
    table = pa.paired.read_table(path)
    assert table["pred"] == ["10.5", "20.1", "-3.25"]
    (r,) = pa.audit_paired(path, big_error=5, resamples=50)
    assert r.bias == pytest.approx((-0.5 + 1.1 - 1.25) / 3)
    tabs = tmp_path / "tabs.csv"
    tabs.write_text("pred\tref\n1,5\t2\n2\t2,5\n", encoding="utf-8")
    assert pa.paired.read_table(tabs)["ref"] == ["2", "2.5"]
    # with commas between cells a quoted 1,000 stays as it is, not 1.0
    commas = tmp_path / "commas.csv"
    commas.write_text('pred,ref\n"1,000",2\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a number"):
        pa.audit_paired(commas, big_error=5)


def test_a_csv_in_the_systems_own_encoding_is_read(tmp_path, monkeypatch) -> None:
    path = tmp_path / "ansi.csv"
    path.write_bytes("subject,pred,ref\nJosé,1,2\nZoë,2,2\n".encode("cp1252"))
    monkeypatch.setattr(pa.paired.locale, "getpreferredencoding", lambda _: "cp1252")
    assert pa.paired.read_table(path)["subject"] == ["José", "Zoë"]
    monkeypatch.setattr(pa.paired.locale, "getpreferredencoding", lambda _: "ascii")
    with pytest.raises(ValueError, match="save it as CSV UTF-8"):
        pa.paired.read_table(path)


def test_columns_named_otherwise_are_given_their_roles(tmp_path, capsys) -> None:
    data = {"Patient": ["A", "B", "C"], "Goniometer": [10, 20, 30],
            "App": [11, 19, 33], "pred": [0, 0, 0]}  # fmt: skip
    (r,) = pa.audit_paired(
        data,
        big_error=5,
        columns={"ref": "goniometer", "pred": "App", "subject": "Patient"},
        units={"value": "deg"},
        resamples=50,
    )
    assert [x.error for x in r.readings] == [1.0, -1.0, 3.0]  # not the pred column
    assert [g.subject for g in r.by_subject] == ["A", "B", "C"]
    with pytest.raises(ValueError, match="no column 'Nope' for pred"):
        pa.audit_paired(data, big_error=5, columns={"pred": "Nope"})
    with pytest.raises(ValueError, match="names a role 'patient'"):
        pa.audit_paired(data, big_error=5, columns={"patient": "Patient"})
    with pytest.raises(ValueError, match="--column pred=NAME"):
        pa.audit_paired({"a": [1], "b": [2]}, big_error=5)
    path = tmp_path / "wide.csv"
    path.write_text("Patient,Goniometer,App\nA,10,11\nB,20,19\nC,30,33\n")
    main(["paired", "--table", str(path), "--column", "pred=App", "--column",
          "ref=Goniometer", "--column", "subject=Patient", "--big-error", "5",
          "--resamples", "50"])  # fmt: skip
    assert "read 3 of 3" in capsys.readouterr().out
    for bad in (["--column", "pred"], ["--match", "a=b"]):
        with pytest.raises(SystemExit):
            main(["paired", "--table", str(path), *bad, "--big-error", "5"])


def test_mot_columns_named_otherwise_are_matched(tmp_path, capsys) -> None:
    t = np.arange(0, 1, 0.02)
    ref = _mot(tmp_path / "ref.mot", t, {"knee_angle_r": 40 + 10 * t, "hip": t})
    pred = _mot(tmp_path / "pred.mot", t,
                {"right knee": 41 + 10 * t, "knee_angle_r": t * 0})  # fmt: skip
    table = pa.pair_mot(pred, ref, match={"right knee": "knee_angle_r"})
    assert set(table["measure"]) == {"knee_angle_r"}
    (r,) = pa.audit_paired(table, big_error=5, resamples=50)
    assert r.bias == pytest.approx(1.0)  # the matched column, not the other one
    with pytest.raises(ValueError, match="no column 'left knee' in the prediction"):
        pa.pair_mot(pred, ref, match={"left knee": "knee_angle_r"})
    with pytest.raises(ValueError, match="no column 'ankle' in the reference"):
        pa.pair_mot(pred, ref, match={"right knee": "ankle"})
    main(["paired", "--pred-mot", str(pred), "--ref-mot", str(ref),
          "--match", "right knee=knee_angle_r", "--big-error", "5",
          "--resamples", "50"])  # fmt: skip
    assert "knee_angle_r: read 50 of 50" in capsys.readouterr().out


def test_frames_outside_the_reference_are_counted_in_a_warning(tmp_path) -> None:
    ref = _mot(tmp_path / "r.mot", np.arange(0, 1.001, 0.01), {"knee": np.ones(101)})
    pred = _mot(tmp_path / "p.mot", np.arange(0, 1.5, 0.1), {"knee": np.ones(15)})
    with pytest.warns(UserWarning, match="4 of 15 predicted frames lie outside"):
        table = pa.pair_mot(pred, ref)
    assert len(table["frame"]) == 11
