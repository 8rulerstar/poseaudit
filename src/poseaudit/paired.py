"""Paired values given as they are, with no keypoints to match: a joint angle
from a pose pipeline against the same angle from motion capture or a
goniometer, frame by frame or once per trial.

The input is long-format: one row per value, with the measure it belongs to,
the prediction (`pred`) and the reference (`ref`), and optionally the
subject, trial, frame (or time) and unit. Readings of one subject are
resampled together, as images are for keypoints, and get limits of agreement
for repeated readings and a summary per subject and per trial.
"""

import csv
import io
import locale
import math
import numbers
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from poseaudit.audit import (
    AuditResult,
    BandBy,
    NotRead,
    Reading,
    _warnings,
    groups,
    score,
)
from poseaudit.measures import Quantity
from poseaudit.thresholds import Side

DEGREES = ("deg", "degree", "degrees", "°")
RADIANS = ("rad", "radian", "radians")
# what a table's columns may be called; `time` stands for `frame`
COLUMNS = ("subject", "trial", "frame", "measure", "pred", "ref", "unit")
Table = Mapping[str, Sequence[Any]]


def quantity(name: str, unit: str = "", wrap: bool | float | None = None) -> Quantity:
    """A measure given as values. Degrees wrap at +/-180 and radians at +/-pi
    unless `wrap` is False; a number sets the period itself."""
    unit = unit.strip()
    if unit.lower() in DEGREES:
        unit, period = "deg", 360.0
    elif unit.lower() in RADIANS:
        unit, period = "rad", 2 * math.pi
    else:
        period = None
    if wrap is False:
        period = None
    elif wrap is not None and wrap is not True:
        if isinstance(wrap, bool) or not float(wrap) > 0:
            raise ValueError(f"wrap takes True, False or a period above 0, got {wrap}")
        period = float(wrap)
    return Quantity(name, (), unit, period)


def audit_values(
    pred: Sequence[float] | np.ndarray,
    ref: Sequence[float] | np.ndarray,
    big_error: float,
    name: str = "value",
    unit: str = "",
    subject: Sequence[Any] | str | None = None,
    trial: Sequence[Any] | str | None = None,
    frame: Sequence[Any] | None = None,
    wrap: bool | float | None = None,
    bands: int | Sequence[float] = 4,
    band_by: BandBy = "truth",
    thresholds: Sequence[float] = (),
    threshold_side: Side | None = None,
    predicted_thresholds: Sequence[float] = (),
    noise_ratio: float | None = None,
    resamples: int = 2000,
    seed: int = 0,
    settings: Mapping | None = None,
) -> AuditResult:
    """Describe how far the predicted values `pred` are from the reference
    `ref`, two arrays of one measure, paired by position.

    subject: each value's subject (or one name for all). Values of one subject
        are resampled together for every interval and get limits of agreement
        for repeated readings. Without it every value counts as independent,
        which frames of one recording are not.
    trial: each value's trial within its subject, for a summary per trial.
    frame: each value's frame or time, kept for the CSV and the worst errors.
    unit: "deg" (or "degrees") and "rad" make differences wrap, so 179
        against -179 is 2 apart; `wrap=False` turns that off.
    A value missing (NaN) on the reference is not counted; one missing on the
    prediction only is counted as not read. The other options are `audit`'s.
    """
    measure = quantity(name, unit, wrap)
    p = _numbers(pred, "pred")
    t = _numbers(ref, "ref")
    if p.ndim != 1 or t.shape != p.shape:
        raise ValueError(
            f"pred and ref must be 1-D and of one length, got {p.shape} and {t.shape}"
        )
    n = len(p)
    subjects = _labels(subject, n, "subject")
    trials = _labels(trial, n, "trial")
    frames = None if frame is None else _numbers(frame, "frame")
    if frames is not None and frames.shape != (n,):
        raise ValueError(f"frame must hold one value per row: {n}, got {frames.shape}")
    if trials is not None and subjects is None:
        raise ValueError("trial needs subject: a trial is a trial of one subject")
    _check(big_error, band_by, threshold_side, noise_ratio, resamples)
    unlabelled = int((~np.isfinite(t)).sum())
    missing = int((np.isfinite(t) & ~np.isfinite(p)).sum())
    keep = np.flatnonzero(np.isfinite(t) & np.isfinite(p))
    readings = [
        Reading(
            image=subjects[i] if subjects is not None else f"row {i}",
            truth_index=int(i),
            predicted_index=int(i),
            class_id=None,
            cluster=subjects[i] if subjects is not None else f"row {i}",
            size=float("nan"),
            truth=float(t[i]),
            predicted=float(p[i]),
            error=measure.difference(float(p[i]), float(t[i])),
            mean=measure.middle(float(t[i]), float(p[i])),
            trial=trials[i] if trials is not None else None,
            frame=float(frames[i]) if frames is not None else None,
        )
        for i in keep
    ]
    not_read = NotRead(
        unlabelled=unlabelled,
        missed=0,
        no_predicted_point=missing,
        unmeasurable=0,
        too_few_in_frame=0,
        unmatched_predictions=0,
    )
    named = subjects is not None
    result = AuditResult(
        measure,
        float(big_error),
        band_by,
        readings,
        not_read,
        settings={
            **dict(settings or {}),
            "input": "paired",
            "unit": measure.unit,
            "wrap": measure.period,
            "bands": bands,
            "cluster": "subject" if named else "row",
            "resamples": resamples,
            "seed": seed,
            "noise_ratio": noise_ratio,
            "thresholds": list(thresholds),
            "threshold_side": threshold_side,
            "predicted_thresholds": list(predicted_thresholds),
        },
    )
    if not measure.unit:
        result.warnings.append(
            f"No unit given for {name}: differences are taken as they are, not "
            "wrapped. Give its unit (deg or rad make 179 against -179 2 apart)."
        )
    if not named and n > 1:
        result.warnings.append(
            "No subject given: every value is resampled on its own, as if "
            "independent. Frames of one person or one trial are not; give each "
            "row its subject, or the intervals come out far too narrow."
        )
    if not readings:
        result.warnings.append("No readable pair: nothing to report.")
        return result
    score(
        result,
        bands,
        None,
        noise_ratio,
        resamples,
        seed,
        None,
        thresholds,
        threshold_side or "above",
        predicted_thresholds,
    )
    if named:
        result.by_subject = groups(readings)
        if trials is not None:
            result.by_trial = groups(readings, by_trial=True)
    result.warnings += _warnings(result, bands, named, "subject" if named else "row")
    if frames is not None and _series(readings, named):
        result.settings["frames_pooled"] = True
        result.warnings.append(FRAMES_POOLED)
    return result


FRAMES_POOLED = (
    "ICC, CCC and r (marked *) pool the frames of each recording, so the range "
    "of motion alone pushes them towards 1: they do not show agreement frame by "
    "frame. Judge that by the bias, the limits and the mean |error|."
)


def _series(readings: Sequence[Reading], named: bool) -> bool:
    """Some subject and trial (or, with no subject, the whole input) holds
    several frames: a time series, not one value per trial."""
    if not named:
        return len(readings) > 1
    keys = [(x.cluster, x.trial) for x in readings]
    return len(set(keys)) < len(keys)


def audit_paired(
    data: "Table | str | Path | Sequence[Table] | Any",
    big_error: float | Mapping[str, float],
    measures: Sequence[str] | None = None,
    units: Mapping[str, str] | None = None,
    wrap: bool | float | None = None,
    columns: Mapping[str, str] | None = None,
    **options,
) -> list[AuditResult]:
    """Audit every measure in a long-format table, one result per measure in
    the order they first appear.

    data: a CSV file, a pandas DataFrame, a mapping of column name to values,
        or a list of such mappings (as `pair_mot` makes) to stack. Columns:
        `pred` and `ref` (required), `measure` (else every row is one measure,
        "value"), and optionally `subject`, `trial`, `frame` (or `time`) and
        `unit`.
    big_error: one value, or a mapping by measure name or by unit, such as
        {"knee_angle": 5, "deg": 10, "m": 0.02}; a name wins over a unit.
    measures: audit these only. units: a unit per measure name, over the
        `unit` column.
    columns: which of the data's columns holds each role, when they are named
        otherwise, such as {"ref": "Goniometer", "pred": "App",
        "subject": "Patient"}. The other options are `audit_values`'s.
    """
    table = read_table(data, columns)
    names = list(dict.fromkeys(table["measure"]))
    if measures is not None:
        unknown = [m for m in measures if m not in names]
        if unknown:
            raise ValueError(
                f"no measure {unknown[0]!r} in the table; it has "
                f"{', '.join(map(repr, names))}"
            )
        names = list(measures)
    if not names:
        raise ValueError("the table has no rows")
    results = []
    measure_column = np.asarray(table["measure"], dtype=object)
    for name in names:
        rows = np.flatnonzero(measure_column == name)
        unit = _unit_of(name, table, rows, units)
        pick = {k: [table[k][i] for i in rows] if k in table else None for k in COLUMNS}
        results.append(
            audit_values(
                [table["pred"][i] for i in rows],
                [table["ref"][i] for i in rows],
                big_error_of(name, unit, big_error),
                name=name,
                unit=unit,
                subject=pick["subject"],
                trial=pick["trial"],
                frame=pick["frame"],
                wrap=wrap,
                **options,
            )
        )
    return results


def big_error_of(name: str, unit: str, big_error: float | Mapping[str, float]) -> float:
    """`big_error` itself, or from a mapping the value for the measure's name,
    else for its unit."""
    if not isinstance(big_error, Mapping):
        value: Any = big_error
    else:
        short = quantity(name, unit).unit
        found = [big_error[k] for k in (name, unit, short) if k and k in big_error]
        if not found:
            raise ValueError(
                f"big_error gives no value for {name} ({unit or 'no unit'}): add "
                f"{name!r} or its unit to it"
            )
        value = found[0]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"big_error is a number or a mapping of numbers; got {value!r}")
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"big_error must be above 0, got {value}")
    return number


def read_table(data, columns: Mapping[str, str] | None = None) -> dict[str, list]:
    """A long-format table as a dict of equal-length lists, with the columns
    named as `COLUMNS` (lower case; `time` read as `frame`) and a `measure`
    column filled in when the table has none.

    columns: the data's own name for a role, such as {"pred": "App",
    "ref": "Goniometer"}; a column of that role's own name is then ignored.
    A CSV may be separated by commas, semicolons or tabs (as Excel saves it
    in many languages); with semicolons or tabs, 10,5 reads as 10.5."""
    if isinstance(data, (str, Path)):
        table = _read_csv(Path(data))
    elif isinstance(data, Mapping):
        table = {str(k): list(np.asarray(v, dtype=object)) for k, v in data.items()}
    elif hasattr(data, "columns") and hasattr(data, "__getitem__"):  # a DataFrame
        table = {str(k): list(np.asarray(data[k], dtype=object)) for k in data.columns}
    elif (
        isinstance(data, Sequence)
        and data
        and all(isinstance(x, Mapping) for x in data)
    ):
        return _stack([read_table(x, columns) for x in data])
    else:
        raise TypeError(
            "data is a CSV path, a DataFrame, a mapping of columns, or a list of "
            f"mappings; got {type(data).__name__}"
        )
    named: dict[str, list] = {}
    for key, values in _renamed(table, columns).items():
        name = key.strip().lower()
        name = "frame" if name == "time" else name
        if name in named:
            raise ValueError(f"the table has two columns named {name!r}")
        named[name] = values
    lengths = {len(v) for v in named.values()}
    if len(lengths) > 1:
        raise ValueError("the table's columns differ in length")
    missing = [c for c in ("pred", "ref") if c not in named]
    if missing:
        raise ValueError(
            f"the table needs columns pred and ref; it lacks {' and '.join(missing)} "
            f"(it has {', '.join(table) or 'none'}); name them with "
            "columns={'pred': ..., 'ref': ...} (--column pred=NAME)"
        )
    if "measure" not in named:
        named["measure"] = ["value"] * len(named["pred"])
    named["measure"] = [str(m) for m in named["measure"]]
    for key in ("subject", "trial"):
        if key in named:
            named[key] = [_text(x) for x in named[key]]
    return named


def _renamed(table: dict[str, list], columns: Mapping[str, str] | None) -> dict:
    """`table` with each column named in `columns` given its role's name."""
    if not columns:
        return table
    roles = {"time": "frame"}
    picked: dict[str, str] = {}
    for role, given in columns.items():
        key = roles.get(role.strip().lower(), role.strip().lower())
        if key not in COLUMNS:
            raise ValueError(
                f"columns names a role {role!r}; the roles are {', '.join(COLUMNS)}"
            )
        found = [c for c in table if c == given] or [
            c for c in table if c.strip().lower() == str(given).strip().lower()
        ]
        if not found:
            raise ValueError(
                f"no column {given!r} for {key}; the table has {', '.join(table)}"
            )
        picked[found[0]] = key
    out = {picked[c]: v for c, v in table.items() if c in picked}
    taken = set(out)
    for c, v in table.items():
        name = c.strip().lower()
        if c not in picked and roles.get(name, name) not in taken:
            out[c] = v
    return out


def _stack(tables: list[dict[str, list]]) -> dict[str, list]:
    keys = set(tables[0])
    if any(set(t) != keys for t in tables):
        raise ValueError(
            "tables to stack must have the same columns: give every one a subject "
            "(and a trial) or none"
        )
    return {k: [x for t in tables for x in t[k]] for k in tables[0]}


_DECIMAL_COMMA = re.compile(r"\s*[+-]?\d+,\d+\s*")


def _read_csv(path: Path) -> dict[str, list]:
    """A CSV's columns by name. Commas, semicolons or tabs, whichever the
    first line holds most of; UTF-8, else the system's own encoding (Excel's
    plain "CSV" on Windows)."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        own = locale.getpreferredencoding(False)
        try:
            text = raw.decode(own)
        except (UnicodeDecodeError, LookupError):
            raise ValueError(
                f"{path} is neither UTF-8 nor {own}: save it as CSV UTF-8"
            ) from None
    first = next((line for line in text.splitlines() if line.strip()), "")
    delimiter = max((",", ";", "\t"), key=first.count)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError(f"{path} is empty")
    if delimiter != ",":  # 10,5 is ten and a half where Excel uses semicolons
        rows = [rows[0]] + [
            [c.replace(",", ".") if _DECIMAL_COMMA.fullmatch(c) else c for c in row]
            for row in rows[1:]
        ]
    head, body = rows[0], rows[1:]
    for k, row in enumerate(body, start=2):
        if len(row) != len(head):
            raise ValueError(
                f"{path}, row {k}: {len(row)} cells under {len(head)} column names"
            )
    return {name: [row[j] for row in body] for j, name in enumerate(head)}


def _unit_of(name, table, rows, units) -> str:
    if units and name in units:
        return units[name]
    if "unit" not in table:
        return ""
    found = {str(table["unit"][i]).strip() for i in rows} - {"", "nan", "None"}
    if len(found) > 1:
        raise ValueError(f"{name} has rows in several units: {sorted(found)}")
    return found.pop() if found else ""


def _text(value) -> str:
    """A subject or trial name; 3.0 read from a number column is 3."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _numbers(values, what: str) -> np.ndarray:
    """Floats, with blanks and "nan" as NaN."""
    out = []
    for x in np.asarray(values, dtype=object).ravel():
        if x is None or (isinstance(x, str) and x.strip().lower() in ("", "nan", "na")):
            out.append(float("nan"))
            continue
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            raise ValueError(f"{what} holds {x!r}, which is not a number") from None
    return np.array(out, float).reshape(np.shape(values))


def _labels(values, n: int, what: str) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return [values] * n
    out = [_text(x) for x in values]
    if len(out) != n:
        raise ValueError(f"{what} must hold one value per row: {n}, got {len(out)}")
    return out


def _check(big_error, band_by, threshold_side, noise_ratio, resamples) -> None:
    if isinstance(big_error, bool) or not isinstance(big_error, numbers.Real):
        raise TypeError(f"big_error is a number; got {big_error!r}")
    if not (math.isfinite(float(big_error)) and float(big_error) > 0):
        raise ValueError(f"big_error must be above 0, got {big_error}")
    if band_by not in ("truth", "mean", "predicted"):
        raise ValueError(
            f"band_by must be 'truth', 'mean' or 'predicted', got {band_by!r}"
        )
    if threshold_side not in (None, "above", "below", "outside"):
        raise ValueError(
            "threshold_side must be 'above', 'below' or 'outside', got "
            f"{threshold_side!r}"
        )
    if noise_ratio is not None and not noise_ratio > 0:
        raise ValueError(f"noise_ratio must be above 0, got {noise_ratio}")
    if resamples < 1:
        raise ValueError(f"resamples must be 1 or more, got {resamples}")
