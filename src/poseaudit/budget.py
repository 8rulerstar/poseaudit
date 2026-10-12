"""How large must a limb be in the image for its angle to be usable?

The audit says how far off the angles were on the images given. This turns
the keypoint errors behind them into a rule that carries over: each point's
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
the longer limbs, within two standard errors and never closer than 6 points.
That is a check for agreement within a tolerance, not a test that the model
of the errors is right: a fit off by a few points passes. A group that fails
gets no size.

With COCO val2017, four YOLO11 pose models, a large error of 15 degrees and
straight from 150 degrees, the check passed most elbow groups and refused
most knee groups (validation/error_budget.out.md); at other settings the
groups that pass change. In simulation the check passed a right fit 85 to
88% of the time with 100 to 300 readings a group, and every time with 1,000.

The figures are agreement with the labels: the labels' own spread is in them.
"""

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
_SEARCH_DRAWS = 20000  # for the one limb of the size search and the floor
_FIXED = np.linspace(0, 20, 401)  # px
_SHARE = np.linspace(0, 0.5, 251)  # of the limb's length
_LIMBS = np.unique(np.round(np.geomspace(5, 2000, 200)))  # px, searched for the size


@dataclass
class Group:
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
    few: bool = False  # fewer than _ENOUGH readings: the check is weak
    below_search: bool = False  # the size is the search's smallest, 5 px
    outside_data: bool = False  # the size lies outside the limbs read
    near_floor: bool = False  # the floor is near the target: the size is touchy
    # the check: (part, predicted rate, rate read, passed)
    checks: list[tuple[str, float, float, bool]] = field(default_factory=list)


@dataclass
class BudgetResult:
    measure: str
    big_error: float
    target: float
    straight_from: float
    groups: list[Group] = field(default_factory=list)

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
            lines.append(f"{head}: {g.n} readings")
            if np.isnan(g.fixed_px):
                lines.append(f"  {g.note}")
                lines.append("")
                continue
            if g.few:
                lines.append(
                    f"  few readings: under {_ENOUGH:,}, the check fails a right "
                    "fit about one time in seven (simulated)"
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
                    f"  smallest limb for {self.target:.0%} or fewer: {size} a "
                    f"segment, at the group's median angle, {g.typical_angle:.0f}"
                    f"{deg}; more bent limbs of the group need less, straighter "
                    "ones more"
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
            lines.append("")
        lines.append(
            "Agreement with the labels, for this model on images like these: the "
            "labels' own spread is in these figures, and the check passes a fit "
            "within its tolerance (6 points or more), so a size can miss the "
            "target by that much."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "measure": self.measure,
            "big_error": self.big_error,
            "target": self.target,
            "straight_from": self.straight_from,
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
) -> BudgetResult:
    """Fit the keypoint error of `measure`'s three points on the pairs, check
    the fit on readings it did not see, and find the smallest limb whose angle is off
    by `big_error` or more in at most `target` of readings."""
    if not isinstance(measure, Angle):
        raise ValueError("budget reads an angle: give three keypoints")
    if not 0 < target < 1:
        raise ValueError(f"target must be between 0 and 1, got {target}")
    if not big_error > 0:
        raise ValueError(f"big error must be above 0, got {big_error}")
    if not 0 < straight_from < 180:
        raise ValueError(
            f"straight_from must be between 0 and 180 degrees, got {straight_from}"
        )
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
    truth, predicted = _readings(pairing, measure)
    result = BudgetResult(measure.title(), big_error, target, straight_from)
    three = Angle("angle", (0, 1, 2), "deg")
    angles = three.read_many(truth) if len(truth) else np.empty(0)
    rng = np.random.default_rng(seed)
    for name, chosen in (
        ("straight", angles >= straight_from),
        ("bent", angles < straight_from),
    ):
        result.groups.append(
            _group(
                name, truth[chosen], predicted[chosen], three, big_error, target, rng
            )
        )
    return result


def _readings(pairing: Pairing, measure: Angle):
    """(n, 3, 2) truth and prediction for pairs that show all three points on
    both sides, with neither segment shorter than a pixel."""
    points = list(measure.points)
    truth, predicted = [], []
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
    return (np.array(truth).reshape(-1, 3, 2), np.array(predicted).reshape(-1, 3, 2))


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


def _tolerance(large: np.ndarray) -> float:
    """How far a predicted rate may be from the rate read: two standard
    errors of the rate read, and never under 6 points."""
    rate = large.mean()
    return float(max(0.06, 2 * np.sqrt(rate * (1 - rate) / len(large))))


def _limb(length: float, degrees: float) -> np.ndarray:
    turn = np.radians(degrees)
    return np.array(
        [[length, 0.0], [0.0, 0.0], [length * np.cos(turn), length * np.sin(turn)]]
    )


def _group(name, truth, predicted, measure, big_error, target, rng) -> Group:
    group = Group(name, len(truth))
    if len(truth) < _MIN_GROUP:
        group.note = f"too few to fit ({_MIN_GROUP} needed)"
        return group
    group.few = len(truth) < _ENOUGH
    group.typical_angle = float(np.median(measure.read_many(truth)))
    off = np.abs(measure.read_many(predicted) - measure.read_many(truth))
    large = off >= big_error
    # every reading is predicted by a fit that did not see it: two halves,
    # each fitted and predicting the other, over several random splits. One
    # split alone decided some groups by the luck of the split.
    expected = np.zeros(len(truth))
    for _ in range(_SPLITS):
        order = rng.permutation(len(truth))
        halves = order[: len(order) // 2], order[len(order) // 2 :]
        for fit_on, check_on in (halves, halves[::-1]):
            fixed, share = fit_spread(truth[fit_on], predicted[fit_on])
            expected[check_on] += large_rate(
                truth[check_on], fixed, share, big_error, rng, measure
            )
    expected /= _SPLITS
    group.read_rate = float(large.mean())
    group.predicted_rate = float(expected.mean())
    group.tolerance = _tolerance(large)
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
            (part, guess, read, bool(abs(guess - read) <= _tolerance(large[inside])))
        )
    group.passed = all(check[3] for check in group.checks)
    group.fixed_px, group.share = fit_spread(truth, predicted)
    # the same draws for every size, so the rate falls smoothly as the limb grows
    draws = rng.normal(size=(_SEARCH_DRAWS, 3, 2))
    group.floor_rate = _rate(
        _limb(1e5, group.typical_angle), 0.0, group.share, big_error, draws, measure
    )
    if not group.passed:
        group.note += "the fit failed its check on these readings"
        return group
    if group.floor_rate > target:
        group.note += (
            f"even a very large limb is off by that much in "
            f"{group.floor_rate:.0%} of readings"
        )
        return group
    for length in _LIMBS:
        rate = _rate(
            _limb(length, group.typical_angle),
            group.fixed_px,
            group.share,
            big_error,
            draws,
            measure,
        )
        if rate <= target:
            group.smallest_limb_px = float(length)
            break
    else:
        group.note += f"not reached by {_LIMBS[-1]:.0f} px"
        return group
    if group.smallest_limb_px <= _LIMBS[0]:
        group.below_search = True
    seen = _lengths(truth)[:, [0, 2]]
    if not seen.min() <= group.smallest_limb_px <= seen.max():
        group.outside_data = True
    if group.floor_rate > 0.75 * target:
        group.near_floor = True
    return group


def _rate(limb, fixed_px, share, big_error, draws, measure) -> float:
    """The large-error rate of one limb, from given standard normal draws."""
    spread = np.hypot(fixed_px, share * _lengths(limb[None])[0])
    drawn = limb + draws * spread[None, :, None]
    return float(
        np.mean(np.abs(measure.read_many(drawn) - measure.read(limb)) >= big_error)
    )
