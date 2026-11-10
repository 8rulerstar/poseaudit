"""0.3.1: a .mot file with no inDegrees line, rows of a paired table that
repeat a key, and tests that pin down the unit conversion, the interpolation,
the repeated-readings limits and the count of empty rows."""

import math
import warnings

import numpy as np
import pytest

import poseaudit as pa
from poseaudit.agreement import Z95, repeated_limits
from poseaudit.io.mot import _at


def _write(path, header, names, rows):
    lines = [*header, "endheader", "\t".join(names)]
    lines += ["\t".join(f"{x:.9f}" for x in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


# --- inDegrees missing ------------------------------------------------------------


def test_a_mot_without_in_degrees_warns_and_hints_at_radians(tmp_path) -> None:
    t = np.arange(5) * 0.1
    rows = [[x, 0.3 * k, -1.2] for k, x in enumerate(t)]
    path = _write(tmp_path / "r.mot", ["Coordinates", "version=1"],
                  ["time", "knee_angle_r", "pelvis_tx"], rows)  # fmt: skip
    with pytest.warns(UserWarning, match="no inDegrees line") as caught:
        motion = pa.load_mot(path)
    assert "radians would" in str(caught[0].message)
    assert "inDegrees=no" in str(caught[0].message)
    # read as degrees all the same: nothing is converted behind the user's back
    assert motion.in_degrees
    assert motion.columns["knee_angle_r"].tolist() == pytest.approx(
        [0.3 * k for k in range(5)]
    )


def test_a_mot_without_in_degrees_in_degree_range_warns_more_softly(tmp_path) -> None:
    rows = [[k * 0.1, 40.0 + k] for k in range(4)]
    path = _write(tmp_path / "d.mot", ["Coordinates"], ["time", "hip"], rows)
    with pytest.warns(UserWarning, match="no inDegrees line") as caught:
        pa.load_mot(path)
    assert "radians would" not in str(caught[0].message)


def test_a_mot_that_states_its_unit_or_holds_only_translations_is_quiet(
    tmp_path,
) -> None:
    rows = [[k * 0.1, 0.1 * k] for k in range(4)]
    stated = _write(tmp_path / "s.mot", ["inDegrees=no"], ["time", "hip"], rows)
    moved = _write(tmp_path / "m.mot", ["Coordinates"], ["time", "pelvis_tx"], rows)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pa.load_mot(stated)
        pa.load_mot(moved)


def test_the_cli_shows_the_in_degrees_warning(tmp_path, capsys) -> None:
    rows = [[k * 0.1, 0.2 * k] for k in range(6)]
    pred = _write(tmp_path / "p.mot", ["inDegrees=yes"], ["time", "hip"],
                  [[x, math.degrees(v) + 1] for x, v in rows])  # fmt: skip
    ref = _write(tmp_path / "r.mot", ["Coordinates"], ["time", "hip"], rows)
    from poseaudit.cli import main

    main(["paired", "--pred-mot", str(pred), "--ref-mot", str(ref),
          "--big-error", "5", "--json", str(tmp_path / "o.json")])  # fmt: skip
    assert "no inDegrees line" in capsys.readouterr().err
    assert "no inDegrees line" in (tmp_path / "o.json").read_text()


# --- units and interpolation ------------------------------------------------------


def test_a_radian_prediction_against_a_degree_reference_is_compared_in_degrees(
    tmp_path,
) -> None:
    t = np.arange(5) * 0.1
    deg = [10.0, 20.0, 30.0, 40.0, 50.0]
    rows = [[x, math.radians(d + 2)] for x, d in zip(t, deg, strict=True)]
    pred = _write(tmp_path / "p.mot", ["inDegrees=no"], ["time", "hip"], rows)
    ref = _write(tmp_path / "r.mot", ["inDegrees=yes"], ["time", "hip"],
                 [[x, d] for x, d in zip(t, deg, strict=True)])  # fmt: skip
    table = pa.pair_mot(pred, ref)
    assert table["unit"] == ["deg"] * 5
    assert table["pred"] == pytest.approx([d + 2 for d in deg])
    assert table["ref"] == pytest.approx(deg)
    # and the other way round: the reference's unit wins
    table = pa.pair_mot(ref, pred)
    assert table["unit"] == ["rad"] * 5
    assert table["pred"] == pytest.approx([math.radians(d) for d in deg])


def test_interpolation_is_linear_between_the_two_samples_around() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 10.0, 0.0, 10.0])
    at = np.array([0.5, 1.5, 2.5, 2.0])
    for period in (None, 360.0):
        assert _at(times, values, at, period).tolist() == pytest.approx(
            [5.0, 5.0, 5.0, 0.0]
        )


# --- limits for repeated readings -------------------------------------------------


def test_repeated_limits_match_bland_and_altman_2007_by_hand() -> None:
    # unequal readings per subject, worked out with the formulas of Bland and
    # Altman (2007), section 5, written out here apart from the library
    data = {"a": [1.0, 2.0, 4.0], "b": [5.0, 6.0], "c": [-1.0, 0.0, 0.5, 2.5]}
    e = [x for xs in data.values() for x in xs]
    labels = [k for k, xs in data.items() for _ in xs]
    n, k = len(e), len(data)
    grand = sum(e) / n
    means = {s: sum(xs) / len(xs) for s, xs in data.items()}
    ss_between = sum(len(xs) * (means[s] - grand) ** 2 for s, xs in data.items())
    ss_within = sum((x - means[s]) ** 2 for s, xs in data.items() for x in xs)
    ms_b, ms_w = ss_between / (k - 1), ss_within / (n - k)
    divisor = (n**2 - sum(len(xs) ** 2 for xs in data.values())) / ((k - 1) * n)
    var = (ms_b - ms_w) / divisor + ms_w
    want = (grand - Z95 * math.sqrt(var), grand + Z95 * math.sqrt(var))
    got = repeated_limits(np.array(e), np.array(labels))
    assert got == pytest.approx(want, abs=1e-12)
    # the hand figure differs from treating every reading as independent
    assert got != pytest.approx(
        (grand - Z95 * np.std(e, ddof=1), grand + Z95 * np.std(e, ddof=1))
    )


# --- paired tables ----------------------------------------------------------------


def test_rows_empty_on_both_sides_are_not_counted_as_missed() -> None:
    nan = float("nan")
    table = {
        "pred": [1.0, nan, nan, 4.0, 5.0, 6.0],
        "ref": [1.5, 2.0, nan, nan, 5.5, 6.5],
    }
    (result,) = pa.audit_paired(table, big_error=5)
    assert result.n == 3
    assert result.not_read.no_predicted_point == 1  # pred empty, ref given
    assert result.not_read.unlabelled == 2  # ref empty, with or without pred


def _frames(subjects=2, frames=4):
    rows = {k: [] for k in ("subject", "trial", "frame", "measure", "pred", "ref")}
    for s in range(subjects):
        for f in range(frames):
            for m in ("knee", "hip"):
                rows["subject"].append(f"S{s}")
                rows["trial"].append("walk")
                rows["frame"].append(f)
                rows["measure"].append(m)
                rows["pred"].append(10.0 + f + s)
                rows["ref"].append(10.5 + f)
    return rows


def test_a_table_with_a_repeated_key_warns_once_per_measure() -> None:
    rows = _frames()
    for k in rows:  # S1's frame 2 and S0's frame 3 of the knee, twice
        rows[k] += [rows[k][2 * 4 + 2 * 2], rows[k][2 * 3]]
    knee, hip = pa.audit_paired(rows, big_error=5)
    (text,) = [w for w in knee.warnings if "repeat" in w]
    assert text.startswith("2 rows of knee repeat")
    assert "subject S0, trial walk, frame 3" in text  # the first in sorted order
    assert not [w for w in hip.warnings if "repeat" in w]


def test_a_table_without_repeats_or_frames_does_not_warn() -> None:
    rows = _frames()
    for result in pa.audit_paired(rows, big_error=5):
        assert not [w for w in result.warnings if "repeat" in w]
    del rows["frame"]  # repeated readings of a subject are expected here
    for result in pa.audit_paired(rows, big_error=5):
        assert not [w for w in result.warnings if "repeat" in w]


def test_mot_pairs_without_a_trial_column_are_separate_recordings(
    tmp_path, capsys
) -> None:
    """Two recordings of one subject with no trial column are not one table stacked
    twice; listing the same file twice still is."""
    import csv

    from test_paired import _mot

    from poseaudit.cli import main

    t = np.arange(0, 1, 0.02)
    for name in ("a", "b"):
        _mot(tmp_path / f"{name}_ref.mot", t, {"knee": 30 + 20 * np.cos(4 * t)})
        _mot(tmp_path / f"{name}_pred.mot", t, {"knee": 31 + 20 * np.cos(4 * t)})

    def run(rows):
        with open(tmp_path / "pairs.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["subject", "pred", "ref"], *rows])
        main(["paired", "--mot-pairs", str(tmp_path / "pairs.csv"),
              "--big-error", "5", "--resamples", "50"])  # fmt: skip
        return capsys.readouterr()

    two = run([["S1", "a_pred.mot", "a_ref.mot"], ["S1", "b_pred.mot", "b_ref.mot"]])
    assert "joined twice" not in two.out + two.err
    same = run([["S1", "a_pred.mot", "a_ref.mot"], ["S1", "a_pred.mot", "a_ref.mot"]])
    assert "joined twice" in same.out + same.err


def test_frames_written_as_1_and_1_0_and_01_are_one_key() -> None:
    rows = {
        "subject": ["A", "A", "A"],
        "frame": ["1", "1.0", "01"],
        "measure": ["k", "k", "k"],
        "pred": [10.0, 12.0, 11.0],
        "ref": [12.0, 12.0, 12.0],
    }
    (result,) = pa.audit_paired(rows, big_error=5)
    assert any(w.startswith("2 rows of k repeat") for w in result.warnings)


def test_rows_with_an_empty_frame_are_not_taken_for_repeats() -> None:
    rows = {
        "subject": ["A", "A", "A"],
        "frame": ["", "", float("nan")],
        "measure": ["k", "k", "k"],
        "pred": [10.0, 12.0, 11.0],
        "ref": [12.0, 12.0, 12.0],
    }
    (result,) = pa.audit_paired(rows, big_error=5)
    assert not [w for w in result.warnings if "repeat the subject" in w]
