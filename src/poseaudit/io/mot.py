"""OpenSim motion files (.mot), as OpenSim's inverse kinematics, Pose2Sim and
Sports2D write them: a header ending in `endheader`, then a row of column
names led by `time`, then one row of numbers per frame.

Columns ending in _tx, _ty or _tz are translations in metres; the others are
angles, in degrees when the header says `inDegrees=yes` (or says nothing,
with a warning) and in radians when it says `inDegrees=no`.
"""

import math
import re
import warnings
from collections.abc import Mapping, Sequence
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
    stated = False
    for line in header:
        key, _, value = line.partition("=")
        if key.strip().lower() == "indegrees":
            in_degrees = value.strip().lower() not in ("no", "false", "0")
            stated = True
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
    if not stated:
        _unstated(path, columns)
    return Motion(time, columns, in_degrees)


def _unstated(path: Path, columns: dict[str, np.ndarray]) -> None:
    """Warn that a file with angle columns and no inDegrees line is read as
    degrees, the more firmly when every angle lies where radians would."""
    angles = [v for name, v in columns.items() if not TRANSLATION.search(name)]
    if not angles:
        return
    finite = np.concatenate(angles)
    finite = finite[np.isfinite(finite)]
    text = f"{path}: no inDegrees line in the header, so the angles are read as degrees"
    if finite.size and (np.abs(finite) <= 2 * math.pi).all():
        text += (
            f"; but every angle lies within +/-2 pi ({np.abs(finite).max():.2f} "
            "at most), as radians would: if the file is in radians, add "
            "inDegrees=no to its header"
        )
    else:
        text += "; add inDegrees=yes or inDegrees=no to the header to say which"
    warnings.warn(text, stacklevel=3)


# the default search for `time_offset="auto"`, in seconds either way
MAX_LAG = 0.5
# lags tried per prediction frame interval: finer than one frame, then refined
LAG_STEPS_PER_FRAME = 2
# a lag must leave at least this share of the prediction's frames on the
# reference's span to be scored
MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class Offset:
    """The time offset that best lines a prediction up with its reference:
    `offset` seconds added to the prediction's times, the correlation `r` of
    the two there (pooled over the columns, each standardised), and whether
    the best lag lies on the edge of the search (`at_edge`), where the true
    one may lie beyond it."""

    offset: float
    r: float
    at_edge: bool


def estimate_offset(
    pred: "str | Path | Motion",
    ref: "str | Path | Motion",
    columns: Sequence[str] | None = None,
    max_lag: float = MAX_LAG,
    match: Mapping[str, str] | None = None,
) -> Offset:
    """The time offset, within +/-`max_lag` seconds, at which the predicted
    curves correlate best with the reference's: the lag where two systems
    recorded the same motion on clocks started apart. Lags are tried at half
    the prediction's frame interval and the best refined by a parabola
    through it and its neighbours. Each column is standardised, so the
    columns weigh alike and one in radians lines up with one in degrees;
    angles are unwrapped first. A lag that keeps fewer than half the
    predicted frames on the reference's span is not tried."""
    if not max_lag > 0:
        raise ValueError(f"max_lag must be above 0 seconds, got {max_lag}")
    p = pred if isinstance(pred, Motion) else load_mot(pred)
    r = ref if isinstance(ref, Motion) else load_mot(ref)
    if match:
        p = _matched(p, r, match)
    chosen = (
        list(columns)
        if columns is not None
        else [c for c in r.columns if c in p.columns]
    )
    chosen = [c for c in chosen if c in p.columns and c in r.columns]
    if not chosen or len(p.time) < 3:
        raise ValueError("estimating a time offset needs a shared column and 3 frames")
    step = float(np.median(np.diff(p.time))) / LAG_STEPS_PER_FRAME
    if not step > 0:
        raise ValueError("the prediction's times must increase")
    count = int(math.floor(max_lag / step + 1e-9))
    lags = np.arange(-count, count + 1) * step
    scores = np.array([_lag_score(p, r, chosen, float(lag)) for lag in lags])
    if not np.isfinite(scores).any():
        raise ValueError(
            f"no lag within +/-{max_lag:g} s leaves half the predicted frames on "
            "the reference's span with values to compare"
        )
    best = int(np.nanargmax(scores))
    offset = float(lags[best])
    if 0 < best < len(lags) - 1 and np.isfinite(scores[best - 1 : best + 2]).all():
        a, b, c = scores[best - 1 : best + 2]
        bend = a - 2 * b + c
        if bend < 0:  # a peak: the vertex of the parabola through the three
            offset += step * 0.5 * (a - c) / bend
    at_edge = best in (0, len(lags) - 1)
    return Offset(float(offset), float(scores[best]), bool(at_edge))


def _lag_score(p: Motion, r: Motion, chosen: Sequence[str], lag: float) -> float:
    """The correlation of prediction and reference with the prediction moved
    by `lag` seconds, every column standardised and pooled; NaN when too few
    frames overlap."""
    t = p.time + lag
    inside = (t >= r.time[0] - 1e-9) & (t <= r.time[-1] + 1e-9)
    if inside.sum() < max(3, MIN_OVERLAP * len(t)):
        return float("nan")
    xs, ys = [], []
    for name in chosen:
        period = {"deg": 360.0, "rad": 2 * math.pi}.get(r.unit(name))
        x = p.columns[name][inside].astype(float)
        y = _at(r.time, r.columns[name], t[inside], period)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        x, y = x[ok], y[ok]
        if period is not None:  # each side in its own unit, unwrapped
            x = np.unwrap(x, period={"deg": 360.0, "rad": 2 * math.pi}[p.unit(name)])
            y = np.unwrap(y, period=period)
        if x.std() == 0 or y.std() == 0:
            continue
        xs.append((x - x.mean()) / x.std())
        ys.append((y - y.mean()) / y.std())
    if not xs:
        return float("nan")
    return float(np.mean(np.concatenate(xs) * np.concatenate(ys)))


def pair_mot(
    pred: "str | Path | Motion",
    ref: "str | Path | Motion",
    subject: str | None = None,
    trial: str | None = None,
    columns: Sequence[str] | None = None,
    time_offset: "float | str" = 0.0,
    match: Mapping[str, str] | None = None,
    max_lag: float = MAX_LAG,
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
    `match` pairs columns named differently, prediction's name to the
    reference's, such as {"right knee": "knee_angle_r"}; the measure takes
    the reference's name. Predicted frames outside the reference's span are
    left out with a warning that counts them. `time_offset="auto"` takes
    the offset `estimate_offset` finds within +/-`max_lag` seconds and says
    which in a warning.
    """
    p = pred if isinstance(pred, Motion) else load_mot(pred)
    r = ref if isinstance(ref, Motion) else load_mot(ref)
    if match:
        p = _matched(p, r, match)
    if time_offset == "auto":
        found = estimate_offset(p, r, columns, max_lag)
        time_offset = found.offset
        warnings.warn(offset_text(found, max_lag), stacklevel=2)
    elif isinstance(time_offset, str):
        raise ValueError(f"time_offset takes seconds or 'auto', got {time_offset!r}")
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
    if not inside.all():
        warnings.warn(
            f"{int((~inside).sum())} of {len(inside)} predicted frames lie outside "
            f"the reference's {r.time[0]:g} to {r.time[-1]:g} s and are left out",
            stacklevel=2,
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


def offset_text(found: Offset, max_lag: float) -> str:
    """The warning that says which offset was taken, and when to doubt it."""
    text = (
        f"time offset found: {found.offset:+.3f} s added to the prediction's "
        f"times (correlation {found.r:.3f} there)"
    )
    if found.at_edge:
        text += (
            f"; it lies on the edge of the +/-{max_lag:g} s searched, so the true "
            "one may be further: widen the search"
        )
    return text


def _matched(p: Motion, r: Motion, match: Mapping[str, str]) -> Motion:
    """The prediction with its columns renamed to the reference's names."""
    for given, wanted in match.items():
        if given not in p.columns:
            raise ValueError(
                f"match: no column {given!r} in the prediction file; it has "
                f"{_few(p.columns)}"
            )
        if wanted not in r.columns:
            raise ValueError(
                f"match: no column {wanted!r} in the reference file; it has "
                f"{_few(r.columns)}"
            )
    targets = set(match.values())
    if len(targets) < len(match):
        raise ValueError("match pairs two prediction columns with one reference")
    renamed = {match[k]: v for k, v in p.columns.items() if k in match}
    renamed |= {
        k: v for k, v in p.columns.items() if k not in match and k not in targets
    }
    return Motion(p.time, renamed, p.in_degrees)


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
