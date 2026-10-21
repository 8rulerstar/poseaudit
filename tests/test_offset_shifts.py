"""--offset-shifts: the figures again with the time offset moved either way,
so the ones that rest on the offset found show apart from the ones that do
not."""

import json

import numpy as np
import pytest

from poseaudit.cli import main


def _sines(t, lag=0.0):
    x = t + lag
    return {
        "knee": 40
        + 30 * np.sin(2 * np.pi * 0.8 * x)
        + 10 * np.sin(2 * np.pi * 2.1 * x),
        "hip": 20 + 15 * np.sin(2 * np.pi * 0.8 * x + 1),
    }


def _write_mot(path, t, columns):
    names = list(columns)
    lines = ["demo", "inDegrees=yes", "endheader", "time\t" + "\t".join(names)]
    lines += [
        "\t".join([f"{t[i]:.4f}"] + [f"{columns[n][i]:.4f}" for n in names])
        for i in range(len(t))
    ]
    path.write_text("\n".join(lines) + "\n")


def _pair(tmp_path, lag=0.0, bias=3.0):
    """A reference at 100 Hz and a prediction at 30 Hz that reads it `bias`
    high and shows at time t what the reference shows at t + lag."""
    t_ref = np.arange(0, 8, 0.01)
    t_pred = np.arange(0, 8, 1 / 30)
    pred = {k: v + bias for k, v in _sines(t_pred, lag).items()}
    _write_mot(tmp_path / "ref.mot", t_ref, _sines(t_ref))
    _write_mot(tmp_path / "pred.mot", t_pred, pred)
    return str(tmp_path / "pred.mot"), str(tmp_path / "ref.mot")


def _run(tmp_path, *extra):
    pred, ref = _pair(tmp_path, *extra[:1])
    out = tmp_path / "out.json"
    md = tmp_path / "out.md"
    main(
        [
            "paired",
            "--pred-mot",
            pred,
            "--ref-mot",
            ref,
            "--measure",
            "knee",
            "--big-error",
            "5",
            "--resamples",
            "200",
            "--offset-shifts",
            "0.05,0.2",
            "--json",
            str(out),
            "--report",
            str(md),
            *extra[1:],
        ]
    )
    return json.loads(out.read_text()), md.read_text()


def test_shifts_leave_the_bias_and_move_the_rmse(tmp_path, capsys):
    data, md = _run(tmp_path)
    rows = {row["shift"]: row for row in data["offset_shifts"]}
    assert sorted(rows) == [-0.2, -0.05, 0.0, 0.05, 0.2]
    assert rows[0.0]["bias"] == pytest.approx(data["bias"])
    assert rows[0.0]["rmse"] == pytest.approx(data["rmse"])
    assert rows[0.0]["rmse"] == pytest.approx(3.0, abs=0.01)
    for shift in (-0.2, 0.2):
        # a constant error shifts with the curve; the curve's own swing does not
        assert rows[shift]["bias"] == pytest.approx(3.0, abs=0.6)
        assert rows[shift]["rmse"] > rows[0.0]["rmse"] + 5
    assert rows[0.2]["rmse"] > rows[0.05]["rmse"] > rows[0.0]["rmse"]
    printed = capsys.readouterr().out
    assert "offset +/-0.05 s moves bias" in printed
    assert "offset +/-0.2 s moves" in printed
    assert "## Moving the time offset" in md
    assert "| +0.2 |" in md


def test_shifts_are_measured_from_the_offset_found(tmp_path):
    """With --time-offset auto the row at 0 is the offset found, not 0 s."""
    data, _ = _run(tmp_path, 0.1, "--time-offset", "auto")
    rows = {row["shift"]: row for row in data["offset_shifts"]}
    assert data["settings"]["time_offset"] == pytest.approx(0.1, abs=0.004)
    assert rows[0.0]["rmse"] == pytest.approx(3.0, abs=0.05)
    assert min(rows.values(), key=lambda row: row["rmse"])["shift"] == 0.0


def test_shifts_refuse_a_table(tmp_path):
    (tmp_path / "t.csv").write_text("measure,pred,ref\nk,1,1\nk,2,2\n")
    with pytest.raises(SystemExit, match="already paired"):
        main(
            [
                "paired",
                "--table",
                str(tmp_path / "t.csv"),
                "--offset-shifts",
                "0.1",
                "--big-error",
                "5",
            ]
        )


@pytest.mark.parametrize("given", ["0", "-0.1", "a", "0.1,nan", ","])
def test_shifts_must_be_positive_seconds(given, tmp_path):
    pred, ref = _pair(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "paired",
                "--pred-mot",
                pred,
                "--ref-mot",
                ref,
                "--big-error",
                "5",
                f"--offset-shifts={given}",
            ]
        )


def test_a_shift_that_lowers_the_rmse_is_flagged(tmp_path, capsys):
    """The prediction runs 0.1 s ahead but the offset is left at 0: moving it
    lowers the RMSE, which says the offset used is off."""
    _run(tmp_path, 0.1)
    printed = capsys.readouterr().out
    assert "lowers the RMSE" in printed and "the offset used may be off" in printed


def test_no_flag_when_the_offset_is_right(tmp_path, capsys):
    _run(tmp_path)
    assert "lowers the RMSE" not in capsys.readouterr().out
