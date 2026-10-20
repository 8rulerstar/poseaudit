"""Each recording cut into cycles (gait strides, repetitions), and the cycles
compared: the peak, trough and range of each cycle, and the error over the
cycle once each is stretched to 0 to 100% of its length.

Gait and rehabilitation studies report figures per cycle, not per trial: a
trial's extremes over many cycles take the most extreme cycle, and noise in
any one of them. The cycles start at given event times (heel strikes, say) on
the reference's clock, or are found from the reference curve itself, one
trough a cycle apart.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from poseaudit.confidence import clustered_bootstrap
from poseaudit.curve import NAN, NEAR_WRAP, CurveFeature, Interval, features_of

# points a cycle is stretched to: 0, 1, ..., 100% of its length
POINTS = 101
# fewer frames than this to a cycle is not a curve
MIN_FRAMES = 5
# a cycle this much shorter or longer than its trial's median is a missed or
# extra event, not a cycle
SHORT, LONG = 0.5, 1.5
# found cycles: a repeat of the reference this well correlated sets the length
MIN_REPEAT = 0.5
# and the next trough is looked for this far either side of one length on
SEARCH = 0.3
FOUND = (
    "Cycles were found from the reference curve, one trough a cycle apart: "
    "that is where its own curve repeats, not a gait event such as a heel "
    "strike. Give event times for cycles that start where the study's do."
)

Key = tuple[str | None, str | None]


@dataclass(frozen=True)
class CurveCycle:
    """One cycle of a trial, from `start` to `end` (the reference's time):
    the peak, trough and range of each side over its `frames`, and `rmse`,
    the root mean square error over the cycle stretched to 0 to 100%."""

    subject: str | None
    trial: str | None
    cycle: int
    start: float
    end: float
    frames: int
    pred_peak: float
    ref_peak: float
    pred_trough: float
    ref_trough: float
    pred_range: float
    ref_range: float
    rmse: float


@dataclass(frozen=True)
class CycleError:
    """The error over the stretched cycles: `rmse` the mean of the cycles'
    RMSE (interval resampling whole subjects), and at each of the 101 points
    from 0 to 100% the mean error (`bias_curve`) and its SD across cycles
    (`sd_curve`)."""

    n: int
    rmse: float
    rmse_ci: Interval
    bias_curve: list[float]
    sd_curve: list[float]


def cycle_summary(
    readings: Sequence,
    named: bool,
    period: float | None,
    events: "Mapping[Key, Sequence[float]] | str",
    resamples: int,
    seed: int,
) -> tuple[list[CurveFeature], list[CurveCycle], CycleError | None, list[str]]:
    """Cut every trial into cycles at `events` (start times by subject and
    trial, see `starts_for`) or, with "auto", where the reference repeats.
    Returns the features across cycles, the cycles, the error over the
    stretched cycles and any warnings."""
    if isinstance(events, str) and events != "auto":
        raise ValueError(f"cycles must be start times or 'auto', got {events!r}")
    series: dict[Key, list] = {}
    for r in readings:
        key = (r.cluster, r.trial) if named else (None, None)
        series.setdefault(key, []).append(r)
    cycles: list[CurveCycle] = []
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    no_events = odd = thin = 0
    for key, members in series.items():
        members = sorted(members, key=lambda r: r.frame)
        t = np.array([r.frame for r in members], dtype=float)
        p = np.array([r.predicted for r in members])
        y = np.array([r.truth for r in members])
        starts = (
            found_starts(t, y) if isinstance(events, str) else starts_for(events, key)
        )
        if starts is None or len(starts) < 2:
            no_events += 1
            continue
        bounds = list(zip(starts[:-1], starts[1:], strict=True))
        length = float(np.median([b - a for a, b in bounds]))
        for i, (a, b) in enumerate(bounds):
            if not SHORT * length <= b - a <= LONG * length:
                odd += 1
                continue
            inside = (t >= a) & (t < b)
            if inside.sum() < MIN_FRAMES:
                thin += 1
                continue
            share = (t[inside] - a) / (b - a)
            grid = np.linspace(0, 1, POINTS)
            pp, yy = (
                np.interp(grid, share, p[inside]),
                np.interp(grid, share, y[inside]),
            )
            curves.append((pp, yy))
            pc, yc = p[inside], y[inside]
            cycles.append(
                CurveCycle(
                    key[0],
                    key[1],
                    i + 1,
                    float(a),
                    float(b),
                    int(inside.sum()),
                    float(pc.max()),
                    float(yc.max()),
                    float(pc.min()),
                    float(yc.min()),
                    float(np.ptp(pc)),
                    float(np.ptp(yc)),
                    float(np.sqrt(np.mean((pp - yy) ** 2))),
                )
            )
    warnings = []
    if events == "auto" and series:
        warnings.append(FOUND)
    if no_events:
        warnings.append(
            f"{no_events} of {len(series)} trials have no cycle "
            + (
                "found (the reference does not repeat clearly)"
                if events == "auto"
                else "(fewer than two event times given for them)"
            )
            + " and are left out of the cycle figures."
        )
    if odd:
        warnings.append(
            f"{odd} cycles are shorter than {SHORT:g} or longer than {LONG:g} "
            "times their trial's median and are left out: a missed or an extra "
            "event."
        )
    if thin:
        warnings.append(
            f"{thin} cycles have under {MIN_FRAMES} frames read on both sides "
            "and are left out."
        )
    if period is not None and cycles:
        edge = period / 2 * (1 - 2 * NEAR_WRAP)
        values = [
            v
            for c in cycles
            for v in (c.pred_peak, c.ref_peak, c.pred_trough, c.ref_trough)
        ]
        if max(abs(v) for v in values) > edge:
            warnings.append(
                f"Values come within {NEAR_WRAP:.0%} of the period of the wrap "
                f"(+/-{period / 2:g}), where a peak is not defined: no cycle "
                "figures."
            )
            return [], cycles, None, warnings
    if not cycles:
        return [], [], None, warnings
    return (
        features_of(cycles, resamples, seed),
        cycles,
        _error(cycles, curves, resamples, seed),
        warnings,
    )


def _error(cycles, curves, resamples: int, seed: int) -> CycleError:
    """The cycles' RMSE and the error at each point of the stretched cycle."""
    rmse = np.array([c.rmse for c in cycles])
    diff = np.array([pp - yy for pp, yy in curves])
    subjects = np.array([str(c.subject) for c in cycles])
    ci = NAN
    if len(set(subjects)) > 1:
        low, high = clustered_bootstrap(
            subjects, lambda idx: np.array([rmse[idx].mean()]), resamples, seed=seed
        )
        ci = (float(low[0]), float(high[0]))
    sd = diff.std(axis=0, ddof=1) if len(diff) > 1 else np.full(POINTS, np.nan)
    return CycleError(
        len(cycles),
        float(rmse.mean()),
        ci,
        [float(v) for v in diff.mean(axis=0)],
        [float(v) for v in sd],
    )


def starts_for(events: Mapping[Key, Sequence[float]], key: Key) -> list[float] | None:
    """The event times of one trial: given for its subject and trial, else
    for its subject (every trial), else for all (no subject or trial)."""
    for k in (key, (key[0], None), (None, None)):
        if k in events:
            return sorted(float(x) for x in events[k])
    return None


def found_starts(t: np.ndarray, y: np.ndarray) -> list[float] | None:
    """Where the reference `y` repeats: its cycle length is the first lag at
    which it correlates with itself by `MIN_REPEAT` or more after first
    falling below it, and each cycle starts at a trough about one length
    after the last. None when it does not repeat."""
    n = len(y)
    if n < 2 * MIN_FRAMES or np.ptp(y) == 0:
        return None
    z = y - y.mean()
    corr = np.array(
        [
            np.corrcoef(z[:-k], z[k:])[0, 1] if np.std(z[k:]) > 0 else 0.0
            for k in range(1, n // 2)
        ]
    )
    fell = np.flatnonzero(corr < MIN_REPEAT)
    if not len(fell):
        return None
    later = np.flatnonzero((np.arange(len(corr)) > fell[0]) & (corr >= MIN_REPEAT))
    if not len(later):
        return None
    k = int(later[0])
    # the top of that rise is the length, in frames
    while k + 1 < len(corr) and corr[k + 1] > corr[k]:
        k += 1
    length = k + 1
    i = int(np.argmin(y[:length]))
    starts = [i]
    while True:
        lo = starts[-1] + int(round((1 - SEARCH) * length))
        hi = starts[-1] + int(round((1 + SEARCH) * length)) + 1
        hi = min(hi, n)
        if hi - lo < 2:
            break
        at = lo + int(np.argmin(y[lo:hi]))
        if at == n - 1:  # the recording ends before the trough does
            break
        starts.append(at)
    return [float(t[i]) for i in starts] if len(starts) > 1 else None
