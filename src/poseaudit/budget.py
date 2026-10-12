"""How large must a limb be for its angle to agree with the labels often enough?

The audit says how far off the angles were on the images given. This turns
the keypoint errors behind them into a size, for that model on images like
those: each point's
error is fitted as a fixed part plus a part that grows with the limb,

    spread(L) = hypot(fixed_px, share * L)        (per axis, in pixels)

and the angle's error is simulated from it. The part of an error that all
three points share is left out of the fit, since moving the whole limb does
not change its angle: on straight limbs the three points move together, and
fitting their raw errors would overstate the angle's error there.

Straight and bent limbs are fitted apart, as their errors differ (bent limbs
are read worse and their points move less together). Each fit is checked
before it is used: every reading is predicted by a fit made without it (two
halves predicting each other, over five random splits), and the predicted
large-error rate must match the rate read, overall and on the shorter and
the longer limbs, within two standard errors and never closer than 6 percentage points.
That is a check for agreement within a tolerance, not a test that the model
of the errors is right: a fit off by a few points passes. A group that fails
gets no size.

With COCO val2017, four YOLO11 pose models, a large error of 15 degrees and
straight from 150 degrees, the check passed all 16 elbow groups and 3 of 16
knee groups (validation/error_budget.out.md); at other settings the groups
that pass change. In simulation (validation/budget_power.py) the check
passed a right fit in 36 of 40 data sets of 100 readings and in all 40 of
300 and of 1,000; the sizes given ranged from 13 to 98 px at 100 readings,
35 to 68 at 300 and 44 to 59 at 1,000, around 48. The verdict rarely turns on
the seed; it can turn on the data drawn.

The figures are agreement with the labels: the labels' own spread is in them.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from poseaudit.measures import Angle
from poseaudit.pairing import Pairing

# median of a 2-D normal's radius, in units of its per-axis spread
_RAYLEIGH_MEDIAN = float(np.sqrt(2 * np.log(2)))
# removing the shared shift of three points leaves 2/3 of each one's variance
_UNSHARED = float(np.sqrt(3 / 2))
_MIN_GROUP = 30  # readings a group needs before it is fitted
_ENOUGH = 1000  # readings a group needs before its check can be trusted
_MIN_PART = 20  # held-out readings each of the shorter and longer limbs needs
_DRAWS = 400  # simulated readings per limb in the check
_SPLITS = 5  # random two-way splits the check averages over
_SHAPES = 300  # of the group's limbs, at most, the size is judged on
_SEARCH_DRAWS = 1000  # simulated readings of each of them
_FIXED = np.linspace(0, 20, 401)  # px
_SHARE = np.linspace(0, 0.5, 251)  # of the limb's length
_LIMBS = np.unique(np.round(np.geomspace(5, 2000, 200)))  # px, searched for the size


@dataclass
class BudgetGroup:
    """One group of a budget: the straight or the bent limbs."""

    name: str
    n: int
    typical_angle: float = float("nan")
    fixed_px: float = float("nan")
    share: float = float("nan")
    read_rate: float = float("nan")  # large-error rate read
    predicted_rate: float = float("nan")  # what the other half's fit predicted
    tolerance: float = float("nan")
    passed: bool = False
    floor_rate: float = float("nan")  # large-error rate on a limb of any size
    smallest_limb_px: float | None = None
    note: str = ""
    few: bool = False  # fewer than _ENOUGH readings (or clusters): weak check
    clusters: int | None = None  # named clusters among the readings
    redraw_passed: int | None = None  # checks passed on the data drawn again
    redraw_sizes: list[float] = field(default_factory=list)  # sizes they gave
    below_search: bool = False  # the size is the search's smallest, 5 px
    outside_data: bool = False  # the size lies outside the limbs read
    near_floor: bool = False  # the floor is near the target: the size is touchy
    # the check: (part, predicted rate, rate read, passed)
    checks: list[tuple[str, float, float, bool]] = field(default_factory=list)


Group = BudgetGroup  # its name in 0.4.0


@dataclass
class BudgetResult:
    measure: str
    big_error: float
    target: float
    straight_from: float
    clustered: bool = False
    redraws: int = 0
    groups: list[BudgetGroup] = field(default_factory=list)

    def summary(self) -> str:
        deg = "\N{DEGREE SIGN}"
        lines = [
            f"budget for {self.measure}: an error of {self.big_error:g}{deg} or "
            f"more counts as large; target: {self.target:.0%} large or fewer",
            "",
        ]
        for g in self.groups:
            head = (
                f"{g.name} (from {self.straight_from:g}{deg})"
                if (g.name == "straight")
                else f"{g.name} (below {self.straight_from:g}{deg})"
            )
            count = f"{g.n} reading{'' if g.n == 1 else 's'}"
            if g.clusters is not None:
                count += f" in {g.clusters} cluster{'' if g.clusters == 1 else 's'}"
            lines.append(f"{head}: {count}")
            if np.isnan(g.fixed_px):
                lines.append(f"  {g.note}")
                lines.append("")
                continue
            if g.few:
                lines.append(
                    f"  few {'clusters' if g.clusters is not None else 'readings'}"
                    f": under {_ENOUGH:,}, a right fit fails the check about one "
                    "time in ten, and the size can be off by a third or more"
                )
            lines += [
                f"  point error: {g.fixed_px:.1f} px + {g.share:.1%} of the limb "
                "(per axis, the part the three points do not share)",
                "  check, each reading predicted by a fit that did not see it, "
                "predicted vs read large: "
                + ", ".join(
                    f"{part} {guess:.0%} vs {read:.0%}{'' if ok else ' (off)'}"
                    for part, guess, read, ok in g.checks
                )
                + (" - passed" if g.passed else " - failed"),
                f"  floor, however large the limb: {g.floor_rate:.0%} large",
            ]
            if g.smallest_limb_px is not None:
                size = (
                    f"{g.smallest_limb_px:.0f} px or less"
                    if g.below_search
                    else f"{g.smallest_limb_px:.0f} px"
                )
                lines.append(
                    f"  smallest limb for {self.target:.0%} or fewer: {size}, the "
                    "mean of its two segments, on limbs bent like this group's "
                    f"(median {g.typical_angle:.0f}{deg}): the group as a whole "
                    "meets the target, straighter or more bent limbs of it may not"
                )
                if g.outside_data:
                    lines.append("  outside the limb lengths read: an extrapolation")
                if g.near_floor:
                    lines.append(
                        f"  the floor ({g.floor_rate:.0%}) is near the target: "
                        "a little more or less error moves this size a lot"
                    )
            else:
                lines.append(f"  smallest limb: none. {g.note}")
            if g.redraw_passed is not None:
                again = (
                    f"  data drawn again {self.redraws} times: the check passed "
                    f"{g.redraw_passed}"
                )
                if len(g.redraw_sizes) == 1:
                    again += f"; 1 gave a size, {g.redraw_sizes[0]:.0f} px"
                elif g.redraw_sizes:
                    low, middle, high = np.percentile(g.redraw_sizes, [0, 50, 100])
                    again += (
                        f"; {len(g.redraw_sizes)} gave a size, from {low:.0f} to "
                        f"{high:.0f} px (median {middle:.0f})"
                    )
                else:
                    again += "; none gave a size"
                lines.append(again)
            lines.append("")
        lines.append(
            (
                "Readings of one cluster were kept together. "
                if self.clustered
                else "Every reading counted as independent; frames of one video or "
                "person need --cluster. "
            )
            + "Agreement with the labels, for this model on images like these: the "
            "labels' own spread is in these figures, and the check passes a fit "
            "within its tolerance (6 percentage points or more), so a size can "
            "miss the target by that much."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "measure": self.measure,
            "big_error": self.big_error,
            "target": self.target,
            "straight_from": self.straight_from,
            "clustered": self.clustered,
            "redraws": self.redraws,
            "groups": [
                {
                    **{k: _finite(v) for k, v in g.__dict__.items() if k != "checks"},
                    "checks": [
                        {
                            "part": part,
                            "predicted_rate": _finite(guess),
                            "read_rate": _finite(read),
                            "passed": ok,
                        }
                        for part, guess, read, ok in g.checks
                    ],
                }
                for g in self.groups
            ],
        }


def _finite(value):
    """NaN and infinity as None, which JSON can hold."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def budget(
    pairing: Pairing,
    measure: Angle,
    big_error: float,
    target: float = 0.2,
    straight_from: float = 150.0,
    seed: int = 0,
    cluster: Callable[[str], str] | None = None,
    redraw: int = 0,
) -> BudgetResult:
    """Fit the keypoint error of `measure`'s three points on the pairs, check
    the fit on readings it did not see, and find the smallest limb whose angle
    is off by `big_error` or more in at most `target` of readings.

    Readings of one image are never split between a fit and its check. Give
    `cluster`, a function from image name to subject or clip, when frames of
    one video or one person repeat: readings of a cluster then stay together,
    and the clusters, not the readings, are counted for the check and for
    the few-readings warning.

    `redraw` repeats the whole budget on the images (or clusters) drawn
    again with replacement that many times, and reports for each group how
    often the check passed and the spread of the sizes given: how much the
    answer depends on the data at hand."""
    if not isinstance(measure, Angle):
        raise ValueError("budget reads an angle: give three keypoints")
    if not 0 < target < 1:
        raise ValueError(f"target must be between 0 and 1, got {target}")
    if not 0 < big_error < 180:
        raise ValueError(f"big error must be above 0 and below 180, got {big_error}")
    if not 0 < straight_from < 180:
        raise ValueError(
            f"straight_from must be between 0 and 180 degrees, got {straight_from}"
        )
    if redraw < 0:
        raise ValueError(f"redraw must be 0 or more, got {redraw}")
    if not pairing.pairs:
        raise ValueError("no prediction was paired with a label: nothing to fit")
    longest = max(
        min(len(p.truth.keypoints), len(p.predicted.keypoints)) for p in pairing.pairs
    )
    if max(measure.points) >= longest:
        raise ValueError(
            f"{measure.title()}: keypoint {max(measure.points)} is past the "
            f"{longest} keypoints (counted from 0) the data has"
        )
    truth, predicted, images = _readings(pairing, measure)
    keys = np.array([cluster(i) for i in images] if cluster else images, dtype=object)
    if not len(truth):
        raise ValueError(
            f"{measure.title()}: no pair shows all three keypoints on both sides"
        )
    named = cluster is not None
    result = BudgetResult(measure.title(), big_error, target, straight_from, named)
    settings = (named, big_error, target, straight_from)
    result.groups = _groups(truth, predicted, keys, *settings, seed)
    if redraw:
        result.redraws = redraw
        units = np.unique(keys)
        rng = np.random.default_rng(seed)
        found: list[list[BudgetGroup]] = []
        for k in range(redraw):
            # each unit drawn keeps its readings; a unit drawn twice counts
            # as two, so its copies are named apart
            drawn = rng.choice(len(units), len(units))
            rows = [np.flatnonzero(keys == units[i]) for i in drawn]
            index = np.concatenate(rows)
            copies = np.concatenate(
                [
                    [f"{n}#{units[i]}"] * len(r)
                    for n, (i, r) in enumerate(zip(drawn, rows, strict=True))
                ]
            ).astype(object)
            found.append(
                _groups(truth[index], predicted[index], copies, *settings, seed + 1 + k)
            )
        for g, again in zip(result.groups, zip(*found, strict=True), strict=True):
            g.redraw_passed = sum(other.passed for other in again)
            g.redraw_sizes = [
                other.smallest_limb_px
                for other in again
                if other.smallest_limb_px is not None
            ]
    return result


def _groups(truth, predicted, keys, named, big_error, target, straight_from, seed):
    """The straight and the bent group, each fitted, checked and sized."""
    three = Angle("angle", (0, 1, 2), "deg")
    angles = three.read_many(truth)
    rng = np.random.default_rng(seed)
    return [
        _group(
            name,
            truth[chosen],
            predicted[chosen],
            keys[chosen],
            named,
            three,
            big_error,
            target,
            rng,
        )
        for name, chosen in (
            ("straight", angles >= straight_from),
            ("bent", angles < straight_from),
        )
    ]


def _readings(pairing: Pairing, measure: Angle):
    """(n, 3, 2) truth and prediction, and the image names, for pairs that
    show all three points on both sides, neither segment under a pixel."""
    points = list(measure.points)
    truth, predicted, images = [], [], []
    for p in pairing.pairs:
        if max(points) >= min(len(p.truth.keypoints), len(p.predicted.keypoints)):
            continue
        if not (p.truth.visible[points].all() and p.predicted.visible[points].all()):
            continue
        g = np.asarray(p.truth.keypoints, float)[points]
        if min(np.hypot(*(g[0] - g[1])), np.hypot(*(g[2] - g[1]))) < 1:
            continue
        truth.append(g)
        predicted.append(np.asarray(p.predicted.keypoints, float)[points])
        images.append(p.image)
    return (
        np.array(truth).reshape(-1, 3, 2),
        np.array(predicted).reshape(-1, 3, 2),
        images,
    )


def _lengths(truth: np.ndarray) -> np.ndarray:
    """(n, 3): the limb length each point is judged by; the middle point by
    the mean of the two segments."""
    first = np.hypot(*(truth[:, 0] - truth[:, 1]).T)
    second = np.hypot(*(truth[:, 2] - truth[:, 1]).T)
    return np.stack([first, (first + second) / 2, second], axis=1)


def fit_spread(truth: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """(fixed_px, share) of spread(L) = hypot(fixed_px, share * L), matched to
    the median unshared error in bins of limb length."""
    error = predicted - truth
    unshared = (error - error.mean(axis=1, keepdims=True)) * _UNSHARED
    size = _lengths(truth).ravel()
    radius = np.hypot(unshared[..., 0], unshared[..., 1]).ravel()
    bins = max(2, min(6, len(size) // 30))
    edges = np.quantile(size, np.linspace(0, 1, bins + 1))
    sizes, medians = [], []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (size >= low) & (size <= high)
        if inside.sum() and np.median(radius[inside]) > 0:
            sizes.append(np.median(size[inside]))
            medians.append(np.median(radius[inside]) / _RAYLEIGH_MEDIAN)
    sizes_, medians_ = np.array(sizes), np.array(medians)
    # every (fixed, share) on the grid at once: (fixed, share, bin)
    model = np.hypot(
        _FIXED[:, None, None], _SHARE[None, :, None] * sizes_[None, None, :]
    )
    with np.errstate(divide="ignore"):
        misfit = ((np.log(model) - np.log(medians_)) ** 2).sum(axis=2)
    i, j = np.unravel_index(np.argmin(misfit), misfit.shape)
    return float(_FIXED[i]), float(_SHARE[j])


def large_rate(
    truth: np.ndarray,
    fixed_px: float,
    share: float,
    big_error: float,
    rng: np.random.Generator,
    measure: Angle | None = None,
    draws: int = _DRAWS,
) -> np.ndarray:
    """Per limb, the share of simulated readings off by `big_error` or more."""
    measure = measure or Angle("angle", (0, 1, 2), "deg")
    spread = np.hypot(fixed_px, share * _lengths(truth))  # (n, 3)
    rates = np.empty(len(truth))
    for k, (points, s) in enumerate(zip(truth, spread, strict=True)):
        drawn = points + rng.normal(size=(draws, 3, 2)) * s[None, :, None]
        off = np.abs(measure.read_many(drawn) - measure.read(points))
        rates[k] = np.mean(off >= big_error)
    return rates


def _tolerance(large: np.ndarray, n: int | None = None) -> float:
    """How far a predicted rate may be from the rate read: two standard
    errors of the rate read over `n` independent units (the readings by
    default), and never under 6 percentage points."""
    rate = large.mean()
    n = len(large) if n is None else n
    return float(max(0.06, 2 * np.sqrt(rate * (1 - rate) / n)))


def _group(
    name, truth, predicted, keys, named, measure, big_error, target, rng
) -> BudgetGroup:
    group = BudgetGroup(name, len(truth))
    units = np.unique(keys) if len(keys) else np.empty(0, object)
    if named:
        group.clusters = len(units)
    if len(truth) < _MIN_GROUP:
        group.note = f"too few to fit ({_MIN_GROUP} needed)"
        return group
    group.few = (len(units) if named else len(truth)) < _ENOUGH

    def tolerance(inside: np.ndarray) -> float:
        # named clusters count once each: their readings are not independent
        n = len(np.unique(keys[inside])) if named else int(inside.sum())
        return _tolerance(large[inside], n)

    group.typical_angle = float(np.median(measure.read_many(truth)))
    off = np.abs(measure.read_many(predicted) - measure.read_many(truth))
    large = off >= big_error
    # every reading is predicted by a fit that did not see it: two halves,
    # each fitted and predicting the other, over several random splits. One
    # split alone decided some groups by the luck of the split.
    expected = np.zeros(len(truth))
    for _ in range(_SPLITS):
        # split images (or named clusters), never the readings of one of them
        order = rng.permutation(len(units))
        first = np.isin(keys, units[order[: len(order) // 2]])
        halves = np.flatnonzero(first), np.flatnonzero(~first)
        if not (len(halves[0]) and len(halves[1])):
            # one image or cluster only: split its readings instead
            shuffled = rng.permutation(len(truth))
            halves = shuffled[: len(truth) // 2], shuffled[len(truth) // 2 :]
        for fit_on, check_on in (halves, halves[::-1]):
            fixed, share = fit_spread(truth[fit_on], predicted[fit_on])
            expected[check_on] += large_rate(
                truth[check_on], fixed, share, big_error, rng, measure
            )
    expected /= _SPLITS
    group.read_rate = float(large.mean())
    group.predicted_rate = float(expected.mean())
    group.tolerance = tolerance(np.ones(len(truth), bool))
    # all readings, then the shorter and longer limbs: a fit can match the
    # whole by luck, too high on one side and too low on the other
    size = _lengths(truth)[:, 1]
    parts = [("all", np.ones(len(size), bool))]
    middle = np.median(size)
    shorter, longer = size < middle, size >= middle
    if min(shorter.sum(), longer.sum()) >= _MIN_PART:
        parts += [
            (f"under {middle:.0f} px", shorter),
            (f"from {middle:.0f} px", longer),
        ]
    else:
        group.note = "too few limbs of different sizes to check by size; "
    for part, inside in parts:
        guess, read = float(expected[inside].mean()), float(large[inside].mean())
        group.checks.append(
            (part, guess, read, bool(abs(guess - read) <= tolerance(inside)))
        )
    group.passed = all(check[3] for check in group.checks)
    group.fixed_px, group.share = fit_spread(truth, predicted)
    # the group's own limbs, their angles and the ratio of their segments
    # kept, each scaled to a mean segment of one: a size is judged on limbs
    # shaped like the group's, since the rate is far from even across angles
    # (one median angle gave sizes that missed the target on straight limbs)
    shapes = truth - truth[:, 1:2]
    segments = _lengths(shapes)[:, [0, 2]].mean(axis=1)
    shapes = shapes / segments[:, None, None]
    if len(shapes) > _SHAPES:
        shapes = shapes[rng.choice(len(shapes), _SHAPES, replace=False)]
    # the same draws for every size, so the rate falls smoothly as limbs grow
    draws = rng.normal(size=(len(shapes), _SEARCH_DRAWS, 3, 2))
    group.floor_rate = _group_rate(
        shapes, 1e5, 0.0, group.share, big_error, draws, measure
    )
    if not group.passed:
        group.note += "the fit failed its check on these readings"
        return group
    if group.floor_rate > target:
        group.note += (
            f"even very large limbs are off by that much in "
            f"{group.floor_rate:.0%} of readings"
        )
        return group

    def meets(length: float) -> bool:
        rate = _group_rate(
            shapes, length, group.fixed_px, group.share, big_error, draws, measure
        )
        return rate <= target

    if not meets(_LIMBS[-1]):
        group.note += f"not reached by {_LIMBS[-1]:.0f} px"
        return group
    # the rate falls as the limbs grow: halve the range until one size is left
    low, high = -1, len(_LIMBS) - 1
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if meets(_LIMBS[middle]) else (middle, high)
    group.smallest_limb_px = float(_LIMBS[high])
    if high == 0:
        group.below_search = True
    seen = _lengths(truth)[:, [0, 2]]
    if not seen.min() <= group.smallest_limb_px <= seen.max():
        group.outside_data = True
    if group.floor_rate > 0.75 * target:
        group.near_floor = True
    return group


def _group_rate(shapes, length, fixed_px, share, big_error, draws, measure) -> float:
    """The large-error rate of the group's limb shapes scaled to a mean
    segment of `length` px, from given standard normal draws."""
    limbs = shapes * length
    spread = np.hypot(fixed_px, share * _lengths(limbs))  # (k, 3)
    drawn = limbs[:, None] + draws * spread[:, None, :, None]
    read = measure.read_many(drawn.reshape(-1, 3, 2)).reshape(len(limbs), -1)
    return float(np.mean(np.abs(read - measure.read_many(limbs)[:, None]) >= big_error))
