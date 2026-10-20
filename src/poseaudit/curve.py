"""The curve of each recording, summarised: the peak, the trough and the
range of motion of the predicted and of the reference values, compared
trial by trial.

Frame by frame a time series pools every frame, so a wide range of motion
alone makes ICC and r look good. Gait and rehabilitation studies report the
peak angle and the range of motion of each trial instead; these compare those
figures, each side taking its own peak (the two need not fall on one frame).
Only frames read on both sides are used.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from poseaudit import agreement as ag
from poseaudit.confidence import clustered_bootstrap

FEATURES = ("peak", "trough", "range")
# fewer frames than this to a trial is not a curve
MIN_FRAMES = 3
# said whenever the figures are given: what they are, and what noise does
EXTREMES = (
    "Peak, trough and range are each trial's extremes over all its frames as "
    "given, every cycle together, not per cycle: noise in a side pushes its "
    "peak up and its trough down and widens its range (frame noise of SD 2 "
    "widened ranges by 3 to 6 in simulation, more for smaller motions, enough "
    "to turn an underread range into an overread one). Filter both sides as "
    "for the study first."
)
# added unless cycles were asked for
PER_CYCLE = " Give --cycles (cycles= in Python) for per-cycle values."
# a periodic measure whose values come this close to its wrap (as a share of
# the period from the edge) has no peak to take: 179 and -179 are 2 apart
NEAR_WRAP = 0.1

Interval = tuple[float, float]
NAN: Interval = (float("nan"), float("nan"))


@dataclass(frozen=True)
class CurveTrial:
    """One trial's curve: the peak (largest value), trough (smallest) and
    range (peak minus trough) of each side, over the `frames` read on both."""

    subject: str | None
    trial: str | None
    frames: int
    pred_peak: float
    ref_peak: float
    pred_trough: float
    ref_trough: float
    pred_range: float
    ref_range: float


@dataclass(frozen=True)
class CurveFeature:
    """How far one feature of the curve (`peak`, `trough` or `range`) is
    off across the `n` trials: predicted minus reference. `limits` are bias
    +/- 1.96 SD, for repeated trials of a subject when there are; the
    intervals resample whole subjects and are NaN with fewer than two."""

    feature: str
    n: int
    bias: float
    bias_ci: Interval
    sd: float
    limits: Interval
    mean_abs_error: float
    mean_abs_error_ci: Interval


def curve_summary(
    readings: Sequence,
    named: bool,
    period: float | None,
    resamples: int,
    seed: int,
    cycles: bool = False,
) -> tuple[list[CurveFeature], list[CurveTrial], list[str]]:
    """The features of every trial with `MIN_FRAMES` frames or more, and
    their errors across trials. Without subjects the readings are one trial.
    Returns the features, the trials and any warnings."""
    series: dict[tuple[str | None, str | None], list] = {}
    for r in readings:
        key = (r.cluster, r.trial) if named else (None, None)
        series.setdefault(key, []).append(r)
    trials = []
    short = 0
    for (subject, trial), members in series.items():
        if len(members) < MIN_FRAMES:
            short += 1
            continue
        p = np.array([r.predicted for r in members])
        t = np.array([r.truth for r in members])
        trials.append(
            CurveTrial(
                subject,
                trial,
                len(members),
                float(p.max()),
                float(t.max()),
                float(p.min()),
                float(t.min()),
                float(p.max() - p.min()),
                float(t.max() - t.min()),
            )
        )
    warnings = []
    if short:
        warnings.append(
            f"{short} of {len(series)} trials have under {MIN_FRAMES} frames read "
            "on both sides and are left out of the peak and range figures."
        )
    if period is not None and trials:
        edge = period / 2 * (1 - 2 * NEAR_WRAP)
        # only the trials kept: a short one left out must not void the rest
        values = [
            v
            for x in trials
            for v in (x.pred_peak, x.ref_peak, x.pred_trough, x.ref_trough)
        ]
        if max(abs(v) for v in values) > edge:
            warnings.append(
                f"Values come within {NEAR_WRAP:.0%} of the period of the wrap "
                f"(+/-{period / 2:g}), where a peak is not defined: no peak or "
                "range figures. Express the angle so its motion stays clear of "
                "the wrap, or give wrap=False if it never crosses."
            )
            return [], trials, warnings
    if not trials:
        return [], [], warnings
    warnings.append(EXTREMES + ("" if cycles else PER_CYCLE))
    return features_of(trials, resamples, seed), trials, warnings


def subject_means(values: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """The mean of `values` (rows) for each subject, in order of first
    appearance: so a subject with more trials or cycles weighs no more."""
    labels = list(dict.fromkeys(subjects))
    return np.array([values[subjects == s].mean(axis=0) for s in labels])


def features_of(trials: Sequence, resamples: int, seed: int) -> list[CurveFeature]:
    """The error of the peak, trough and range across `trials` (anything with
    the `CurveTrial` fields): the bias and mean |error| average each
    subject's mean, so each subject counts once, with intervals resampling
    subjects; the SD and limits take every trial."""
    errors = np.array(
        [
            [
                x.pred_peak - x.ref_peak,
                x.pred_trough - x.ref_trough,
                x.pred_range - x.ref_range,
            ]
            for x in trials
        ]
    )
    subjects = np.array([str(x.subject) for x in trials])
    per = subject_means(np.hstack([errors, np.abs(errors)]), subjects)
    bias_ci = mae_ci = [NAN] * len(FEATURES)
    if len(per) > 1:
        low, high = clustered_bootstrap(
            np.arange(len(per)).astype(str),
            lambda idx: per[idx].mean(axis=0),
            resamples=resamples,
            seed=seed,
        )
        k = len(FEATURES)
        bias_ci = [(float(low[i]), float(high[i])) for i in range(k)]
        mae_ci = [(float(low[k + i]), float(high[k + i])) for i in range(k)]
    features = []
    for i, name in enumerate(FEATURES):
        e = errors[:, i]
        features.append(
            CurveFeature(
                name,
                len(e),
                float(per[:, i].mean()),
                bias_ci[i],
                float(e.std(ddof=1)) if len(e) > 1 else float("nan"),
                ag.repeated_limits(e, subjects),
                float(per[:, len(FEATURES) + i].mean()),
                mae_ci[i],
            )
        )
    return features
