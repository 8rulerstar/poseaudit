"""Read one measure off every matched pair and describe how far off it is."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import numpy as np

from poseaudit import agreement as ag
from poseaudit.confidence import clustered_bootstrap, wilson
from poseaudit.measures import Measure, Ratio, Tilt
from poseaudit.pairing import Pairing
from poseaudit.reference import draw, jitter_gain, local_shifts, prepare, strata, unwrap
from poseaudit.thresholds import Side, ThresholdAgreement, matched_threshold
from poseaudit.thresholds import agreement as threshold_agreement

BandBy = Literal["truth", "mean", "predicted"]
Interval = tuple[float, float]
NAN: Interval = (float("nan"), float("nan"))
Cluster = Callable[[str], str] | Mapping[str, str] | None
Baseline = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class Reading:
    image: str
    truth_index: int
    predicted_index: int
    class_id: int | None
    cluster: str
    size: float  # mean length of the measure's segments in the truth, px
    truth: float
    predicted: float
    error: float  # signed, predicted minus truth
    mean: float  # halfway between truth and predicted


@dataclass(frozen=True)
class Band:
    low: float
    high: float
    n: int
    mean_abs_error: float
    bias: float
    bias_ci: Interval


@dataclass(frozen=True)
class SizeBand:
    """How error depends on how large the measured part is in the image."""

    low: float
    high: float
    n: int
    mean_abs_error: float
    big_error_rate: float
    big_error_rate_ci: Interval
    gain: float


@dataclass(frozen=True)
class NotRead:
    """Why instances did not become readings."""

    unlabelled: int  # the truth lacks a point of the measure: not the model's doing
    missed: int  # labelled, but no prediction matched it
    no_predicted_point: int  # matched, but the prediction lacks a point
    unmeasurable: int  # geometry that cannot be measured, e.g. two points on one pixel
    too_few_in_frame: int  # relative mode: too few readings in the image for a baseline
    unmatched_predictions: int  # predictions no truth matched


@dataclass
class AuditResult:
    measure: Measure
    big_error: float
    band_by: BandBy
    readings: list[Reading]
    not_read: NotRead
    mean_abs_error: float = float("nan")
    mean_abs_error_ci: Interval = NAN
    median_abs_error: float = float("nan")
    p95_abs_error: float = float("nan")
    rmse: float = float("nan")
    rmse_ci: Interval = NAN
    big_error_rate: float = float("nan")
    big_error_rate_ci: Interval = NAN
    bias: float = float("nan")
    bias_ci: Interval = NAN
    median_error: float = float("nan")
    limits: Interval = NAN
    lower_limit_ci: Interval = NAN
    upper_limit_ci: Interval = NAN
    limits_coverage: float = float("nan")  # share of errors inside `limits`
    tail_shares: Interval = NAN  # share of errors below and above `limits`
    empirical_limits: Interval = NAN
    empirical_lower_ci: Interval = NAN
    empirical_upper_ci: Interval = NAN
    repeated_limits: Interval | None = None
    repeated_lower_ci: Interval | None = None
    repeated_upper_ci: Interval | None = None
    gain: float = float("nan")
    gain_ci: Interval = NAN
    offset: float = float("nan")
    jitter_gain: float | None = None  # what keypoint jitter alone gives
    jitter_gain_range: Interval | None = None
    jitter_p: float | None = None  # share of jitter-only rebuilds at or below gain
    gain_gap: float | None = None  # gain minus jitter_gain
    gain_gap_ci: Interval | None = None
    robust_gain: float = float("nan")  # Theil-Sen: barely moved by gross failures
    ba_slope: float = float("nan")
    ba_slope_ci: Interval = NAN
    deming: float | None = None
    deming_ci: Interval | None = None
    icc: float = float("nan")
    icc_ci: Interval = NAN
    ccc: float = float("nan")
    ccc_ci: Interval = NAN
    pearson: float = float("nan")
    clusters: int = 0
    bands: list[Band] = field(default_factory=list)
    size_bands: list[SizeBand] = field(default_factory=list)
    thresholds: list[ThresholdAgreement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.readings)

    @property
    def measurable(self) -> int:
        """Instances whose truth has every point of the measure."""
        r = self.not_read
        return (
            self.n
            + r.missed
            + r.no_predicted_point
            + r.unmeasurable
            + r.too_few_in_frame
        )

    def worst(self, count: int = 10) -> list[Reading]:
        return sorted(self.readings, key=lambda r: -abs(r.error))[:count]

    def summary(self, full: bool = False) -> str:
        """The headline figures; `full` adds every agreement statistic."""
        from poseaudit.report import summary

        return summary(self, full)

    def to_markdown(self, path: str | None = None) -> str:
        from poseaudit.report import markdown

        return _written(markdown(self), path)

    def to_csv(self, path: str | None = None) -> str:
        """One row per reading, for another tool to recompute anything here."""
        from poseaudit.report import csv_rows

        return _written(csv_rows(self), path)

    def plot(self, path: str) -> None:
        """Truth against prediction, and a Bland-Altman plot. Needs matplotlib."""
        from poseaudit.plot import plot

        plot(self, path)

    def to_dict(self) -> dict:
        """Every figure. `limits` are the normal ones (bias ± 1.96 SD);
        `percentile_limits` (also under their old name, `empirical_limits`)
        are the 2.5th and 97.5th percentiles of the errors."""
        from poseaudit._version import __version__

        skip = ("measure", "readings")
        out = {"poseaudit": __version__, "schema": 1}
        out |= {k: v for k, v in asdict(self).items() if k not in skip}
        out["measure"] = {
            "name": self.measure.name,
            "points": list(self.measure.points),
            "unit": self.measure.unit,
        }
        out["percentile_limits"] = tuple(self.empirical_limits)
        out["n"] = self.n
        out["measurable"] = self.measurable
        return out


def _written(text: str, path: str | None) -> str:
    if path is not None:
        with open(path, "w", encoding="utf-8", newline="") as file:
            file.write(text)
    return text


def audit(
    pairing: Pairing,
    measure: Measure,
    big_error: float,
    bands: int | Sequence[float] = 4,
    band_by: BandBy = "truth",
    size_bands: int | Sequence[float] = 3,
    cluster: Cluster = None,
    thresholds: Sequence[float] = (),
    threshold_side: Side | None = None,
    predicted_thresholds: Sequence[float] = (),
    relative_to: str | Baseline | None = None,
    relative_abs: bool = False,
    min_in_frame: int = 3,
    noise_ratio: float | None = None,
    mixed_classes: bool = False,
    jitter_repeats: int = 500,
    resamples: int = 2000,
    seed: int = 0,
    settings: Mapping | None = None,
) -> AuditResult:
    """Describe how far `measure` read off predictions is from the truth.

    bands: a number of equal-count bands, or the edges to cut at.
    band_by: what to sort readings by. The truth suits labels much less noisy
        than the model; the mean of both suits two equally noisy readings.
    size_bands: a number of bands, or pixel edges, by the size of the measured
        part (the mean length of its segments in the truth).
    cluster: readings sharing a cluster are resampled together, and limits of
        agreement for repeated readings are added. Default: the image. Pass a
        function of the image name (e.g. its subject or video) or a mapping.
    thresholds: decision thresholds applied to the truth; threshold_side is
        "above", "below" or "outside" (|value|), by default "outside" for a
        tilt and "above" otherwise. The prediction is held to the same
        threshold, to each of `predicted_thresholds`, and to the one that flags
        as many instances as the truth does.
    relative_to: read each value against the other readings in its image,
        each side separately, e.g. to remove a camera roll from tilts: "median",
        a percentile such as "p15", or a function of those readings. With
        `relative_abs`, sizes are compared: |value| minus the baseline of the
        others' |values|. Each reading's baseline leaves the reading itself out,
        so no reading is its own zero. Images with fewer than `min_in_frame`
        readings are left out.
    noise_ratio: var(prediction noise) / var(truth noise), when known, adds a
        Deming slope.
    mixed_classes: allow readings of several classes in one audit. By default
        that is an error: parts that differ in kind hide each other's errors.
    jitter_repeats: rebuilds for the gain that keypoint jitter alone gives.
    settings: anything else to record in the report, e.g. input paths.
    """
    if threshold_side not in (None, "above", "below", "outside"):
        raise ValueError(
            "threshold_side must be 'above', 'below' or 'outside', got "
            f"{threshold_side!r}"
        )
    if band_by not in ("truth", "mean", "predicted"):
        raise ValueError(
            f"band_by must be 'truth', 'mean' or 'predicted', got {band_by!r}"
        )
    if not big_error > 0:
        raise ValueError(f"big_error must be above 0, got {big_error}")
    if noise_ratio is not None and not noise_ratio > 0:
        raise ValueError(f"noise_ratio must be above 0, got {noise_ratio}")
    if relative_abs and relative_to is None:
        raise ValueError("relative_abs needs relative_to")
    cluster_of = _cluster_function(cluster)
    _check_indices(pairing, measure)
    raw, counts = _read_pairs(pairing, measure)
    relative = _baseline(relative_to)
    raw, too_few = _relative(raw, measure, relative, relative_abs, min_in_frame)
    not_read = NotRead(
        unlabelled=counts["unlabelled"] + _unlabelled(pairing.missed, measure),
        missed=len(pairing.missed) - _unlabelled(pairing.missed, measure),
        no_predicted_point=counts["no_predicted_point"],
        unmeasurable=counts["unmeasurable"],
        too_few_in_frame=too_few,
        unmatched_predictions=len(pairing.extra),
    )
    difference = (lambda v, t: v - t) if relative_abs else measure.difference
    readings = [
        Reading(
            p.image,
            p.truth_index,
            p.predicted_index,
            p.truth.class_id,
            cluster_of(p.image),
            _size(p.truth.keypoints, measure),
            t,
            v,
            difference(v, t),
            t + difference(v, t) / 2 if relative_abs else measure.middle(t, v),
        )
        for p, t, v in raw
    ]
    side = threshold_side or (
        "outside" if isinstance(measure, Tilt) and not relative_abs else "above"
    )
    result = AuditResult(
        measure,
        big_error,
        band_by,
        readings,
        not_read,
        settings={
            **dict(settings or {}),  # the caller's notes; the audit's own win
            "bands": bands,
            "size_bands": size_bands,
            "cluster": "image" if cluster is None else "custom",
            "relative_to": relative_to
            if isinstance(relative_to, str) or relative_to is None
            else "custom",
            "relative_abs": relative_abs,
            "resamples": resamples,
            "seed": seed,
            "noise_ratio": noise_ratio,
            "jitter_repeats": jitter_repeats,
            "min_in_frame": min_in_frame,
            "mixed_classes": mixed_classes,
            "thresholds": list(thresholds),
            "threshold_side": threshold_side,
            "predicted_thresholds": list(predicted_thresholds),
        },
    )
    result.warnings = list(pairing.warnings)
    beside, boxed = _missed_beside_extra(pairing, measure)
    classes = sorted({r.class_id for r in readings if r.class_id is not None})
    if len(classes) > 1 and not mixed_classes:
        raise ValueError(
            f"readings mix classes {classes}: audit one class at a time (classes=, "
            "--classes) or allow it with mixed_classes=True (--mixed-classes)"
        )
    if relative is not None and not isinstance(measure, Tilt):
        result.warnings.append(
            f"Relative readings of {type(measure).__name__.lower()}s: a camera "
            "roll moves tilts, not angles or lengths, so this rarely means much."
        )
    units = _units_warning(pairing)
    if units:
        result.warnings.append(units)
    if not readings:
        result.warnings.append("No readable pair: nothing to report.")
        return result
    jitter = (
        _jitter_inputs(raw, measure) if relative is None and jitter_repeats else None
    )
    if relative is None and jitter_repeats and jitter is None and len(raw) >= 3:
        result.warnings.append(
            "No jitter reference: some truths have zero length, so their "
            "displacements cannot be scaled to other objects."
        )
    if jitter is not None and len(raw) < 60:
        result.warnings.append(
            f"Only {_count(len(raw), 'reading')} for the jitter reference. With so "
            "few it "
            "follows noise that varies along the range only coarsely: its p can "
            "flag an honest model as squashed more often than stated, and it can "
            "miss mild squashing. A large p here does not show that the model "
            "does not squash."
        )
    _fill(result, bands, size_bands, noise_ratio, resamples, seed, jitter)
    # images (by default) or named clusters holding several readings are
    # resampled whole: Wilson would treat every reading as independent
    grouped = result.clusters < result.n
    _decide(result, thresholds, side, predicted_thresholds, grouped)
    if result.clusters < 2:  # nothing to resample, and Wilson would treat the
        result.big_error_rate_ci = NAN  # readings as independent
    elif grouped:
        large = np.array([abs(r.error) >= result.big_error for r in readings])
        low, high = clustered_bootstrap(
            [r.cluster for r in readings],
            lambda idx: np.array([large[idx].mean()]),
            resamples=resamples,
            seed=seed,
        )
        result.big_error_rate_ci = _wider(
            result.big_error_rate_ci, (float(low[0]), float(high[0]))
        )
    big = round(result.big_error_rate * result.n)
    if beside >= 3 and (beside >= 0.1 * big or beside >= 0.05 * result.measurable):
        result.warnings.append(
            f"{beside} labelled instances found no match but overlap an unmatched "
            f"prediction, compared with {big} large errors among the readings. A "
            "prediction too far off to reach the matching threshold counts as "
            "missed, not as an error, which flatters the figures; "
            + (
                "for thin parts try a lower --min-iou (min_iou)."
                if boxed
                else "without boxes, try a lower --min-similarity "
                "(min_keypoint_similarity), a factor of ten at a time: the "
                "similarity falls off fast with distance."
            )
        )
    lost = result.not_read.missed + result.not_read.no_predicted_point
    if thresholds and lost:
        result.warnings.append(
            f"Decisions count readings only: {lost} labelled instances were not "
            "read, and a defect among them is neither caught nor missed there."
        )
    if cluster is None:
        result.repeated_limits = None  # an image's readings are different objects
        result.repeated_lower_ci = result.repeated_upper_ci = None
    if jitter is not None and np.isfinite(result.gain):
        truth_points, units, sizes = jitter
        mid, spread, below = jitter_gain(
            measure,
            truth_points,
            np.array(
                [p.predicted.keypoints[list(measure.points)] for p, _t, _v in raw]
            ),
            sizes,
            np.array([r.truth for r in readings]),
            result.gain,
            jitter_repeats,
            seed,
        )
        result.jitter_gain, result.jitter_gain_range, result.jitter_p = (
            mid,
            spread,
            below,
        )
        result.gain_gap = result.gain - mid
        if cluster is not None:
            # the rebuild draws objects one by one; errors a subject carries
            # across its readings would make p far too small (17-27% false
            # alarms at 5% in simulation). The interval resamples clusters.
            result.jitter_p = None
    result.warnings += _warnings(result, bands, cluster is not None)
    return result


def _cluster_function(cluster: Cluster) -> Callable[[str], str]:
    if cluster is None:
        return lambda image: image
    if callable(cluster):
        return lambda image: str(cluster(image))
    return lambda image: str(cluster[image])


def _baseline(relative_to) -> Baseline | None:
    if relative_to is None or callable(relative_to):
        return relative_to
    if relative_to == "median":
        return np.median
    found = re.fullmatch(r"p(\d+(?:\.\d+)?)", str(relative_to))
    if found and 0 <= float(found.group(1)) <= 100:
        q = float(found.group(1))
        return lambda values: float(np.percentile(values, q))
    raise ValueError(
        "relative_to must be 'median', a percentile such as 'p15', or a "
        f"function; got {relative_to!r}"
    )


def _size(keypoints: np.ndarray, measure: Measure) -> float:
    """Mean length of the measure's segments: a-b, b-c for an angle; a-b and
    c-d for a ratio, not the gap between them."""
    if isinstance(measure, Ratio):
        a, b, c, d = (keypoints[i] for i in measure.points)
        return float((np.linalg.norm(b - a) + np.linalg.norm(d - c)) / 2)
    ends = [keypoints[i] for i in measure.points]
    return float(
        np.mean([np.linalg.norm(b - a) for a, b in zip(ends, ends[1:], strict=False)])
    )


def _missed_beside_extra(pairing, measure: Measure) -> tuple[int, bool]:
    """Measurable truths left unmatched that overlap a prediction left unmatched
    too: possibly one badly placed prediction, dropped instead of scored. Also
    whether those truths have boxes, which decides the knob to turn."""
    needed = list(measure.points)
    found = [t for _image, t in pairing.near_misses if t.visible[needed].all()]
    return len(found), any(t.boxed for t in found)


def _unlabelled(missed, measure: Measure) -> int:
    needed = list(measure.points)
    return sum(1 for _image, t in missed if not t.visible[needed].all())


def _instances(pairing: Pairing):
    """Every instance in a pairing, with its image and side."""
    return [
        *((p.image, "truth", p.truth) for p in pairing.pairs),
        *((p.image, "prediction", p.predicted) for p in pairing.pairs),
        *((image, "truth", t) for image, t in pairing.missed),
        *((image, "prediction", x) for image, x in pairing.extra),
    ]


def _check_indices(pairing: Pairing, measure: Measure) -> None:
    """Unpaired instances too: an index past the end would otherwise only
    show up as instances that were never read."""
    highest = max(measure.points)
    for image, side, instance in _instances(pairing):
        k = len(instance.keypoints)
        if highest >= k:
            raise ValueError(
                f"{measure.name} uses keypoint {highest}, but a {side} instance in "
                f"{image} has {k} keypoints (indices 0-{k - 1}); indices count "
                "from 0"
            )


def _normalised(instances) -> bool | None:
    """Every visible coordinate within about [0, 1]: fractions of the image
    size, it seems, not pixels. A little slack, as in the YOLO loader: MediaPipe
    lets points run past the frame. None with no visible point to go by."""
    seen = [i.keypoints[i.visible] for i in instances]
    xy = np.concatenate([np.zeros((0, 2)), *seen])
    if not len(xy):
        return None
    return bool(((xy >= -0.05) & (xy <= 1.05)).all())


def _units_warning(pairing: Pairing) -> str | None:
    """Coordinates that look normalised, on both sides or on one only. Judged
    on every instance, paired or not: with one side normalised nothing pairs."""
    looks = {
        name: _normalised([i for _image, s, i in _instances(pairing) if s == side])
        for name, side in (("ground-truth", "truth"), ("predicted", "prediction"))
    }
    normalised = [name for name, v in looks.items() if v]
    pixels = [name for name, v in looks.items() if v is False]
    if not normalised:
        return None
    if not pixels:
        return (
            "Every visible coordinate lies within 0 to 1 (give or take 0.05): "
            "normalised coordinates? They squeeze an image's longer side, so "
            "angles, tilts and lengths come out wrong. Multiply x by the image "
            "width and y by its height."
        )
    return (
        f"The {normalised[0]} coordinates look normalised (every visible one lies "
        f"within 0 to 1) but the {pixels[0]} ones look like pixels: the two "
        "sides are in different units, so they cannot pair, which is why little "
        f"or nothing was read. Multiply the {normalised[0]} x by the image width "
        "and y by its height."
    )


def _read_pairs(pairing: Pairing, measure: Measure):
    needed = list(measure.points)
    counts = {"unlabelled": 0, "no_predicted_point": 0, "unmeasurable": 0}
    raw = []
    for p in pairing.pairs:
        if not p.truth.visible[needed].all():
            counts["unlabelled"] += 1
            continue
        if not p.predicted.visible[needed].all():
            counts["no_predicted_point"] += 1
            continue
        truth = measure.read(p.truth.keypoints)
        predicted = measure.read(p.predicted.keypoints)
        if not (np.isfinite(truth) and np.isfinite(predicted)):
            counts["unmeasurable"] += 1
            continue
        raw.append((p, truth, predicted))
    return raw, counts


def _relative(raw, measure: Measure, baseline, use_abs: bool, min_in_frame: int):
    if baseline is None:
        return raw, 0
    by_image: dict[str, list] = {}
    for row in raw:
        by_image.setdefault(row[0].image, []).append(row)
    out, too_few = [], 0
    for rows in by_image.values():
        if len(rows) < min_in_frame:
            too_few += len(rows)
            continue
        t = np.array([r[1] for r in rows])
        v = np.array([r[2] for r in rows])
        if use_abs:
            t, v = np.abs(t), np.abs(v)
        for j, (p, _t, _v) in enumerate(rows):
            others = np.arange(len(rows)) != j
            base_t = _base(baseline, t[others], measure, use_abs)
            base_p = _base(baseline, v[others], measure, use_abs)
            if use_abs:
                out.append((p, t[j] - base_t, v[j] - base_p))
            else:
                out.append(
                    (p, measure.shift(t[j], base_t), measure.shift(v[j], base_p))
                )
    return out, too_few


def _base(baseline, values: np.ndarray, measure: Measure, use_abs: bool) -> float:
    """The baseline of a frame's other readings. Tilts are axial: +89 and -89
    are 2 degrees apart, so they are measured from the frame's own main
    direction first and the baseline is taken there."""
    if use_abs or not isinstance(measure, Tilt):
        return float(baseline(values))
    doubled = np.radians(2 * values)
    centre = np.degrees(np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())) / 2
    around = np.array([measure.shift(x, centre) for x in values])
    return measure.shift(float(baseline(around)), -centre)


def _fill(result, bands, size_bands, noise_ratio, resamples, seed, jitter) -> None:
    rs = result.readings
    t = np.array([r.truth for r in rs])
    e = np.array([r.error for r in rs])
    p = t + e  # unwrapped for a tilt, so a fit is not cut at +/-90
    sizes = np.array([r.size for r in rs])
    level = {"truth": t, "mean": t + e / 2, "predicted": p}[result.band_by]
    clusters = np.array([r.cluster for r in rs])
    result.clusters = len(np.unique(clusters))
    absolute = np.abs(e)
    big = int((absolute >= result.big_error).sum())

    result.mean_abs_error = float(absolute.mean())
    result.median_abs_error = float(np.median(absolute))
    result.p95_abs_error = float(np.percentile(absolute, 95))
    result.rmse = float(np.sqrt((e**2).mean()))
    result.big_error_rate = big / len(e)
    result.big_error_rate_ci = wilson(big, len(e))
    result.bias = float(e.mean())
    result.median_error = float(np.median(e))
    result.limits = ag.limits(e)
    low, high = result.limits
    result.limits_coverage = float(((e >= low) & (e <= high)).mean())
    result.tail_shares = (float((e < low).mean()), float((e > high).mean()))
    result.empirical_limits = ag.empirical_limits(e)
    repeated = len(np.unique(clusters)) < len(clusters)
    if repeated:
        result.repeated_limits = ag.repeated_limits(e, clusters)
    result.gain, result.offset = ag.gain(t, p)
    result.robust_gain = ag.robust_gain(t, p)
    result.ba_slope = ag.ba_slope(t, p)
    result.icc = ag.icc_a1(t, p)
    result.ccc = ag.ccc(t, p)
    result.pearson = (
        float(np.corrcoef(t, p)[0, 1])
        if not (ag._flat(t) or ag._flat(p))
        else float("nan")
    )
    if noise_ratio is not None:
        result.deming = ag.deming(t, p, noise_ratio)

    band_index = _band_index(level, bands)
    count = int(band_index.max()) + 1

    def statistics(idx: np.ndarray, draw: np.ndarray) -> np.ndarray:
        ti, pi, ei, bi = t[idx], p[idx], e[idx], band_index[idx]
        per_band = [
            ei[bi == k].mean() if (bi == k).any() else np.nan for k in range(count)
        ]
        lo, hi = ag.limits(ei)
        elo, ehi = ag.empirical_limits(ei)
        # by draw, not by cluster: a cluster drawn twice is two subjects
        rlo, rhi = ag.repeated_limits(ei, draw) if repeated else (np.nan, np.nan)
        deming = ag.deming(ti, pi, noise_ratio) if noise_ratio is not None else np.nan
        slope = ag.gain(ti, pi)[0]
        gap = (
            slope - _rebuilt_gain(result.measure, jitter, idx, ti, rng)
            if jitter
            else np.nan
        )
        return np.array(
            [
                ei.mean(), lo, hi, np.abs(ei).mean(), np.sqrt((ei**2).mean()),
                slope, ag.ba_slope(ti, pi), ag.icc_a1(ti, pi),
                ag.ccc(ti, pi), deming, elo, ehi, rlo, rhi, gap, *per_band,
            ]
        )  # fmt: skip

    rng = np.random.default_rng(seed + 1)

    if result.clusters < 2:  # every resample would hold the same readings
        ci = [NAN] * (15 + count)
    else:
        lows, highs = clustered_bootstrap(
            clusters, statistics, resamples=resamples, seed=seed, copies=True
        )
        ci = [(float(a), float(b)) for a, b in zip(lows, highs, strict=True)]
    result.bias_ci, result.lower_limit_ci, result.upper_limit_ci = ci[0], ci[1], ci[2]
    result.mean_abs_error_ci, result.rmse_ci = ci[3], ci[4]
    result.gain_ci, result.ba_slope_ci = ci[5], ci[6]
    result.icc_ci, result.ccc_ci = ci[7], ci[8]
    if noise_ratio is not None:
        result.deming_ci = ci[9]
    result.empirical_lower_ci, result.empirical_upper_ci = ci[10], ci[11]
    if repeated:
        result.repeated_lower_ci, result.repeated_upper_ci = ci[12], ci[13]
    if jitter:
        result.gain_gap_ci = ci[14]
    edges = None if isinstance(bands, int) else sorted(bands)
    result.bands = _bands(level, e, band_index, count, ci[15:], edges)
    result.size_bands = _size_bands(sizes, t, p, e, size_bands, result.big_error)
    if result.clusters < 2:  # Wilson's would treat the readings as independent
        result.size_bands = [
            replace(b, big_error_rate_ci=NAN) for b in result.size_bands
        ]


def _decide(result, thresholds, side, predicted_thresholds, grouped=False) -> None:
    t = np.array([r.truth for r in result.readings])
    p = np.array([r.predicted for r in result.readings])  # as read, not unwrapped
    rows = []
    for th in thresholds:
        rows.append(threshold_agreement(t, p, th, side))
        for pth in predicted_thresholds:
            rows.append(threshold_agreement(t, p, th, side, pth))
        matched = matched_threshold(t, p, th, side)
        if np.isfinite(matched):
            rows.append(threshold_agreement(t, p, th, side, matched, matched=True))
            fitted = result.gain * th + (0 if side == "outside" else result.offset)
            if side == "outside":
                fitted = abs(fitted)
            moved = abs(matched - th) >= 0.02 * max(abs(th), 1.0)
            if moved and (matched - th) * (fitted - th) < 0:
                result.warnings.append(
                    f"The same-count threshold for {th:g} moves to {matched:.2f}, the "
                    f"opposite way from the fitted line, which maps {th:g} to "
                    f"{fitted:.2f}: a few large errors among the extreme "
                    "predictions set the count, not the model's scale. Try "
                    "thresholds of your own with --pred-threshold."
                )
    if result.clusters < 2:
        rows = [replace(row, sensitivity_ci=NAN, precision_ci=NAN) for row in rows]
    elif grouped and rows:
        rows = _clustered_rates(result, t, p, rows)
    result.thresholds = rows


def _clustered_rates(result, t, p, rows):
    """With several readings to an image or a named cluster, the catch and
    precision intervals resample whole clusters: Wilson intervals treat every
    reading as independent."""
    from poseaudit.thresholds import flagged

    groups = [r.cluster for r in result.readings]
    out = []
    for row in rows:
        truth = flagged(t, row.threshold, row.side)
        pred = flagged(p, row.predicted_threshold, row.side)

        def rates(idx, truth=truth, pred=pred):
            tp = float((truth[idx] & pred[idx]).sum())
            return np.array(
                [
                    tp / truth[idx].sum() if truth[idx].any() else np.nan,
                    tp / pred[idx].sum() if pred[idx].any() else np.nan,
                ]
            )

        low, high = clustered_bootstrap(
            groups,
            rates,
            resamples=result.settings["resamples"],
            seed=result.settings["seed"],
        )
        out.append(
            replace(
                row,
                sensitivity_ci=_wider(row.sensitivity_ci, (low[0], high[0])),
                precision_ci=_wider(row.precision_ci, (low[1], high[1])),
            )
        )
    return out


def _wider(wilson_ci, resampled) -> tuple[float, float]:
    """The cluster interval, but never narrower than Wilson's: with no event or
    only events, every resample gives the same rate and the bootstrap interval
    shrinks to a point."""
    lo, hi = (float(x) for x in resampled)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return wilson_ci
    return (min(lo, wilson_ci[0]), max(hi, wilson_ci[1]))


def _jitter_inputs(raw, measure):
    """The measure's points on the truth, each object's displacements divided by
    its size, and the sizes, aligned with the readings."""
    points = list(measure.points)
    truth_points = np.array([p.truth.keypoints[points] for p, _t, _v in raw])
    predicted_points = np.array([p.predicted.keypoints[points] for p, _t, _v in raw])
    sizes = np.array([_size(p.truth.keypoints, measure) for p, _t, _v in raw])
    if len(raw) < 3 or not (sizes > 0).all():
        return None
    return (
        truth_points,
        local_shifts(measure, truth_points, predicted_points, sizes),
        sizes,
    )


_PER_RESAMPLE = 10  # rebuilds averaged per bootstrap resample


def _rebuilt_gain(measure, jitter, idx, truth_values, rng, repeats=_PER_RESAMPLE):
    """The jitter-only gain on a resample, donors drawn from the same resample:
    the mean over a few rebuilds, so the interval carries the resampling and
    little of the rebuild's own randomness (one rebuild made it far too wide)."""
    truth_points, units, sizes = jitter
    points, own = truth_points[idx], sizes[idx]
    prepared = prepare(measure, points, units[idx], strata(truth_values))
    gains = []
    for _ in range(repeats):
        values = draw(measure, points, own, prepared, rng)
        keep = np.isfinite(values)
        if keep.sum() >= 3:
            gains.append(
                ag.gain(
                    truth_values[keep],
                    unwrap(measure, values[keep], truth_values[keep]),
                )[0]
            )
    return float(np.mean(gains)) if gains else np.nan


def _band_index(level: np.ndarray, bands: int | Sequence[float]) -> np.ndarray:
    """Band of each reading; -1 for one outside explicitly given edges."""
    if isinstance(bands, int):
        edges = np.unique(np.quantile(level, np.linspace(0, 1, bands + 1)))
    else:
        edges = np.asarray(sorted(bands), float)
    if len(edges) < 2:
        return np.zeros(len(level), int)
    index = np.searchsorted(edges[1:-1], level, side="right")
    outside = (level < edges[0]) | (level > edges[-1])
    return np.where(outside, -1, index)


def _bands(level, errors, band_index, count, cis, edges=None) -> list[Band]:
    """Labelled by the edges when they were given, else by the values inside."""
    out = []
    for k in range(count):
        inside = band_index == k
        if inside.any():
            chosen = errors[inside]
            low = edges[k] if edges else float(level[inside].min())
            high = edges[k + 1] if edges else float(level[inside].max())
            out.append(
                Band(
                    float(low),
                    float(high),
                    int(inside.sum()),
                    float(np.abs(chosen).mean()),
                    float(chosen.mean()),
                    cis[k],
                )
            )
    return out


def _size_bands(sizes, t, p, e, bands, big_error) -> list[SizeBand]:
    if isinstance(bands, int):
        # to a hundredth of a pixel: sizes that differ only by rounding would
        # otherwise make bands that all read 150-150 px
        edges = np.unique(np.round(np.quantile(sizes, np.linspace(0, 1, bands + 1)), 2))
        if len(edges) == 1:
            edges = np.repeat(edges, 2)
        sizes = np.round(sizes, 2)
    else:
        edges = np.asarray([0, *sorted(bands), np.inf], float)
    out = []
    for k, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = k == len(edges) - 2
        inside = (sizes >= low) & ((sizes <= high) if last else (sizes < high))
        n = int(inside.sum())
        if n:
            big = int((np.abs(e[inside]) >= big_error).sum())
            out.append(
                SizeBand(
                    float(low),
                    float(high),
                    n,
                    float(np.abs(e[inside]).mean()),
                    big / n,
                    wilson(big, n),
                    ag.gain(t[inside], p[inside])[0],
                )
            )
    return out


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _warnings(result: AuditResult, bands, named_clusters: bool) -> list[str]:
    notes = []
    rs = result.readings
    if result.n < 30:
        notes.append(
            f"Only {_count(result.n, 'reading')}: every figure here is fragile."
        )
    if 2 <= result.n <= 40:
        notes.append(
            f"The percentile limits rest on {result.n} readings: with 40 or fewer "
            "the 2.5th and 97.5th percentiles fall between the two smallest and the "
            "two largest errors, so they are little more than the extremes."
        )
    big = round(result.big_error_rate * result.n)
    if big == 0:
        from poseaudit.report import _amount

        high = result.big_error_rate_ci[1]
        notes.append(
            f"No large errors ({_amount(result.big_error, result.measure.unit)} "
            "or more)"
            + (
                f"; the rate could still be up to {high:.1%}."
                if np.isfinite(high)
                else "."
            )
        )
    elif big < 10:
        notes.append(
            f"Only {_count(big, 'large error')}: judge the rate by its interval."
        )
    kind = "cluster" if named_clusters else "image"
    if result.clusters < 2:
        notes.append(
            f"Only one {kind}: resampling it gives the same readings every time, "
            "so there are no intervals."
        )
    elif result.clusters < 20:
        notes.append(
            f"Only {_count(result.clusters, kind)}: bootstrap intervals are too "
            "narrow with this few."
        )
    r = result.not_read
    lost = r.missed + r.no_predicted_point + r.unmeasurable
    if result.measurable and lost / result.measurable >= 0.3:
        notes.append(
            f"{lost} of {result.measurable} labelled instances were not read: the "
            "figures describe the ones that were, which are usually easier."
        )
    classes = {x.class_id for x in rs if x.class_id is not None}
    if len(classes) > 1:
        notes.append(
            f"Readings mix classes {sorted(classes)}: parts that differ in kind can "
            "hide each other's errors. Audit one class at a time."
        )
    if not isinstance(bands, int):
        outside = result.n - sum(b.n for b in result.bands)
        if outside:
            notes.append(f"{outside} readings fall outside the band edges given.")
    if isinstance(result.measure, Tilt) and not result.settings.get("relative_abs"):
        near = np.mean([abs(x.mean) > 75 for x in rs])
        if near >= 0.1:
            notes.append(
                f"{near:.0%} of tilts are within 15 degrees of horizontal, where the "
                "axis wraps from +90 to -90: bands and slopes there are unreliable."
            )
    below, above = result.tail_shares
    if result.n >= 30 and max(below, above) > 0.04:
        notes.append(
            f"{below:.1%} of errors fall below the normal limits and {above:.1%} "
            "above them, against 2.5% each for a normal error: use the percentile "
            "limits."
        )
    return notes
