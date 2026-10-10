"""OpenSim motion files (.mot), as OpenSim's inverse kinematics, Pose2Sim and
Sports2D write them: a header ending in `endheader`, then a row of column
names led by `time`, then one row of numbers per frame.

Columns ending in _tx, _ty or _tz are translations in metres; the others are
angles, in degrees when the header says `inDegrees=yes` (or says nothing) and
in radians when it says `inDegrees=no`.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRANSLATION = re.compile(r"_t[xyz]$")


@dataclass(frozen=True)
class Motion:
    """A .mot file: the times, and the values of each column by name."""

    time: np.ndarray
    columns: dict[str, np.ndarray]
    in_degrees: bool = True

    def unit(self, name: str) -> str:
        if TRANSLATION.search(name):
            return "m"
        return "deg" if self.in_degrees else "rad"


def load_mot(path: str | Path) -> Motion:
    """Read a .mot (or .sto) file. A file with no `endheader` line is read
    from its first line that starts with `time`."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    end = next(
        (k for k, line in enumerate(lines) if line.strip().lower() == "endheader"),
        None,
    )
    header = lines[:end] if end is not None else []
    start = (end + 1) if end is not None else 0
    names_at = next(
        (k for k in range(start, len(lines)) if lines[k].strip()),
        None,
    )
    if names_at is None:
        raise ValueError(f"{path}: no column names after the header")
    if end is None:
        names_at = next(
            (k for k, x in enumerate(lines) if x.strip().lower().startswith("time")),
            None,
        )
        if names_at is None:
            raise ValueError(
                f"{path}: neither an endheader line nor a row of column names "
                "led by time; is it a .mot file?"
            )
    in_degrees = True
    for line in header:
        key, _, value = line.partition("=")
        if key.strip().lower() == "indegrees":
            in_degrees = value.strip().lower() not in ("no", "false", "0")
    names = _split(lines[names_at])
    if not names or names[0].lower() != "time":
        raise ValueError(
            f"{path}: the first column is {names[0] if names else 'missing'!r}, "
            "not time"
        )
    if len(set(names)) < len(names):
        raise ValueError(f"{path}: two columns share a name")
    rows = []
    for k in range(names_at + 1, len(lines)):
        cells = _split(lines[k])
        if not cells:
            continue
        if len(cells) != len(names):
            raise ValueError(
                f"{path}, line {k + 1}: {len(cells)} values under {len(names)} "
                "column names"
            )
        try:
            rows.append([float(c) for c in cells])
        except ValueError:
            raise ValueError(f"{path}, line {k + 1}: a value is not a number") from None
    if not rows:
        raise ValueError(f"{path}: no rows of values")
    values = np.array(rows, float)
    time = values[:, 0]
    if not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError(f"{path}: the times must be finite and increasing")
    columns = {name: values[:, j] for j, name in enumerate(names) if j}
    return Motion(time, columns, in_degrees)


def pair_mot(
    pred: "str | Path | Motion",
    ref: "str | Path | Motion",
    subject: str | None = None,
    trial: str | None = None,
    columns: Sequence[str] | None = None,
    time_offset: float = 0.0,
) -> dict[str, list]:
    """Line up a predicted and a reference .mot file by time and column name,
    as a long-format table for `audit_paired`: one row per frame and column
    both files hold.

    The values are compared at the prediction's times (plus `time_offset`,
    in seconds, to move the prediction onto the reference's clock), inside
    the span the reference covers; the reference is interpolated linearly
    there, angles unwrapped first so that 179 to -179 interpolates through
    180. A reference value next to a missing one (NaN) is missing too.
    `columns` picks the columns; by default every one both files hold.
    """
    p = pred if isinstance(pred, Motion) else load_mot(pred)
    r = ref if isinstance(ref, Motion) else load_mot(ref)
    shared = [c for c in r.columns if c in p.columns]
    if columns is not None:
        for c in columns:
            if c not in p.columns or c not in r.columns:
                side = "prediction" if c not in p.columns else "reference"
                raise ValueError(f"column {c!r} is not in the {side} file")
        chosen = list(columns)
    else:
        chosen = shared
    if not chosen:
        raise ValueError(
            "the two files share no column name: prediction has "
            f"{_few(p.columns)}, reference {_few(r.columns)}"
        )
    t = p.time + float(time_offset)
    inside = (t >= r.time[0] - 1e-9) & (t <= r.time[-1] + 1e-9)
    if not inside.any():
        raise ValueError(
            f"the files do not overlap in time: prediction {t[0]:g} to {t[-1]:g} s, "
            f"reference {r.time[0]:g} to {r.time[-1]:g} s; a time offset moves the "
            "prediction"
        )
    t = t[inside]
    table: dict[str, list] = {
        k: [] for k in ("frame", "measure", "pred", "ref", "unit")
    }
    for name in chosen:
        unit = r.unit(name)
        predicted = p.columns[name][inside]
        if p.unit(name) != unit:  # one file in degrees, the other in radians
            predicted = (
                np.degrees(predicted) if unit == "deg" else np.radians(predicted)
            )
        period = {"deg": 360.0, "rad": 2 * math.pi}.get(unit)
        reference = _at(r.time, r.columns[name], t, period)
        table["frame"] += t.tolist()
        table["measure"] += [name] * len(t)
        table["pred"] += predicted.tolist()
        table["ref"] += reference.tolist()
        table["unit"] += [unit] * len(t)
    n = len(table["frame"])
    if subject is not None:
        table["subject"] = [str(subject)] * n
    if trial is not None:
        table["trial"] = [str(trial)] * n
    return table


def _at(times: np.ndarray, values: np.ndarray, at: np.ndarray, period) -> np.ndarray:
    """`values` at the times `at`, linearly between the two samples around
    each; NaN where either of them is."""
    values = values.astype(float).copy()
    finite = np.isfinite(values)
    unwrapped = period is not None and finite.sum() > 1
    lowest = float(values[finite].min()) if finite.any() else 0.0
    if unwrapped:
        values[finite] = np.unwrap(values[finite], period=period)
    right = np.clip(np.searchsorted(times, at, side="left"), 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    exact = np.isclose(times[right], at, rtol=0, atol=1e-9)
    left = np.where(exact, right, left)
    span = times[right] - times[left]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(span > 0, (at - times[left]) / span, 0.0)
    out = values[left] * (1 - w) + values[right] * w
    out = np.where(exact, values[right], out)
    if unwrapped:  # back into the file's own range, e.g. -180 to 180
        out = lowest + (out - lowest) % period
    return out


def _split(line: str) -> list[str]:
    line = line.strip()
    return [c.strip() for c in line.split("\t")] if "\t" in line else line.split()


def _few(names) -> str:
    names = list(names)
    shown = ", ".join(names[:6])
    return shown + (f" and {len(names) - 6} more" if len(names) > 6 else "")
