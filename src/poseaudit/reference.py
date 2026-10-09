"""What would a model with no tendency at all score?

Keypoint jitter alone bends the gain: an angle near 180 degrees can only be
read smaller, one near 0 only larger, so random scatter of the points pulls
both ends toward the middle. To tell a model's own tendency from that, rebuild
the predictions from the truth plus this model's own keypoint displacements,
and fit the gain again.

Each rebuilt object takes the displacements of all its points from one
object (possibly itself), together, scaled to its own size. A model's errors
are correlated within an object (a whole limb shifts at once, which barely
changes its angle); drawing each point from a different object would break
that, inflate the angle noise and bias the reference low.

The displacements are carried in the frame of each point's own segment,
mirrored for angles that bend the other way, and laid onto the other
object's frames: errors along a limb stay along it, errors across it stay
across, and errors toward the inside of the bend stay on the inside.

Donors come from objects with a similar true value (one of up to six
equal-count groups): a model may be noisier on straight arms than on folded
ones. Either kind of noise, moved elsewhere, would make an honest model look
squashed. Within each group the mean shift is taken out first: a shift that a
whole group shares is the tendency under test, not jitter.

What remains assumed: that the objects are independent, and that apart from
that shared shift an object's errors depend only on its true value and its
own frames. Gross failures tied to the true value (a straight arm read as
folded) and noise in the truth also land below the reference; the test
cannot tell them from squashing.
"""

from dataclasses import replace

import numpy as np

from poseaudit.agreement import gain
from poseaudit.measures import Angle, Measure, Ratio, Tilt


def rebuild(
    measure: Measure,
    truth_points: np.ndarray,  # (n, P, 2) the measure's points, truth
    unit_shifts: np.ndarray,  # (n, P, 2) displacements / size, in own frames
    sizes: np.ndarray,  # (n,) size of each object, px
    rng: np.random.Generator,
    strata: np.ndarray | None = None,  # (n,) donors only within the same group
) -> np.ndarray:
    """One jitter-only reading per object: truth plus a whole donor's shifts,
    laid onto this object's frames. The donors are the same n objects, the
    object itself included."""
    return draw(
        measure,
        truth_points,
        sizes,
        prepare(measure, truth_points, unit_shifts, strata),
        rng,
    )


def frames(measure: Measure, truth_points: np.ndarray):
    """The frames to lay shifts onto: the cosine and sine of each point's
    segment heading and which way each object bends. They depend on each
    object alone, so a resample can take its rows instead of recomputing."""
    headings = _headings(measure, truth_points)
    return np.cos(headings), np.sin(headings), _hands(measure, truth_points)


def prepare(measure, truth_points, unit_shifts, strata=None, framed=None):
    """What every rebuild of these objects shares: the groups, each group's
    scatter about its mean shift, and the frames to lay shifts onto (`framed`,
    from `frames`, when they are already known)."""
    n = len(truth_points)
    groups = np.zeros(n, int) if strata is None else strata
    members = [np.flatnonzero(groups == k) for k in np.unique(groups)]
    scatter = np.empty_like(unit_shifts)
    for m in members:
        # a tendency is a shift shared by a group; keep only the scatter about it
        scatter[m] = unit_shifts[m] - unit_shifts[m].mean(axis=0)
    return (
        members,
        scatter,
        frames(measure, truth_points) if framed is None else framed,
    )


def draw(measure, truth_points, sizes, prepared, rng) -> np.ndarray:
    members, scatter, (cos, sin, hands) = prepared
    donors = np.empty(len(truth_points), int)
    for m in members:
        donors[m] = m[rng.integers(0, len(m), size=len(m))]
    shifts = _mirrored(scatter[donors], hands)
    rebuilt = truth_points + _rotated(shifts, cos, sin) * sizes[:, None, None]
    return _local(measure).read_many(rebuilt)


def _local(measure: Measure) -> Measure:
    """The measure read off its own points in its own order, so a rebuild
    need not be laid out at the keypoint indices first. A keypoint listed
    twice (the shared elbow of a ratio) is read from its last listing, as
    laying the points out would leave it."""
    last = {index: slot for slot, index in enumerate(measure.points)}
    return replace(measure, points=tuple(last[i] for i in measure.points))


def jitter_gain(
    measure: Measure,
    truth_points: np.ndarray,
    predicted_points: np.ndarray,
    sizes: np.ndarray,
    truth_values: np.ndarray,
    observed_gain: float,
    repeats: int = 500,
    seed: int = 0,
) -> tuple[float, tuple[float, float], float]:
    """Median jitter-only gain, its 2.5-97.5% range, and (rebuilds at or below
    the observed gain + 1) / (rebuilds + 1): a one-sided p for "reads
    differences smaller than its jitter explains", by squashing or by gross
    failures alike."""
    nan = float("nan")
    if len(truth_values) < 3:
        return nan, (nan, nan), nan
    rng = np.random.default_rng(seed)
    units = local_shifts(measure, truth_points, predicted_points, sizes)
    prepared = prepare(measure, truth_points, units, strata(truth_values))
    gains: list[float] = []
    for _ in range(repeats):
        values = draw(measure, truth_points, sizes, prepared, rng)
        keep = np.isfinite(values)
        if keep.sum() >= 3:
            gains.append(
                gain(
                    truth_values[keep],
                    unwrap(measure, values[keep], truth_values[keep]),
                )[0]
            )
    if not gains:
        return nan, (nan, nan), nan
    spread = np.array(gains)
    low, mid, high = np.percentile(spread, [2.5, 50, 97.5])
    below = (np.sum(spread <= observed_gain) + 1) / (len(spread) + 1)
    return float(mid), (float(low), float(high)), float(below)


def local_shifts(
    measure: Measure,
    truth_points: np.ndarray,
    predicted_points: np.ndarray,
    sizes: np.ndarray,
) -> np.ndarray:
    """Displacements divided by size, each point's in the frame of the
    segment it belongs to."""
    units = (predicted_points - truth_points) / sizes[:, None, None]
    local = _turned(units, -_headings(measure, truth_points))
    return _mirrored(local, _hands(measure, truth_points))


def _hands(measure: Measure, points: np.ndarray) -> np.ndarray:
    """(n,) +1 or -1: which way an angle bends. A left arm bent one way is a
    mirror image of a right arm bent the other; in frames that only turn,
    their errors across the limb would point opposite ways."""
    if not isinstance(measure, Angle):
        return np.ones(len(points))
    u, v = points[:, 0] - points[:, 1], points[:, 2] - points[:, 1]
    return np.where(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0] < 0, -1.0, 1.0)


def _mirrored(vectors: np.ndarray, hands: np.ndarray) -> np.ndarray:
    """Local (along, across) vectors with the across part flipped for -1."""
    out = vectors.copy()
    out[..., 1] *= hands[:, None]
    return out


def _headings(measure: Measure, points: np.ndarray) -> np.ndarray:
    """(n, P) direction, radians, of the segment each point belongs to: a-b
    for a and b, then b-c for c; for a ratio c and d take c-d."""
    count = points.shape[1]
    segment = [max(k - 1, 0) for k in range(count)]
    if isinstance(measure, Ratio):
        segment = [0, 0, 2, 2]
    d = np.stack([points[:, s + 1] - points[:, s] for s in segment], axis=1)
    return np.arctan2(d[..., 1], d[..., 0])


def _turned(vectors: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """(n, P, 2) vectors turned by angle (n, P)."""
    return _rotated(vectors, np.cos(angle), np.sin(angle))


def _rotated(vectors: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """(n, P, 2) vectors turned by an angle given by its cosine and sine."""
    x, y = vectors[..., 0], vectors[..., 1]
    return np.stack([cos * x - sin * y, sin * x + cos * y], axis=-1)


def strata(truth_values: np.ndarray, most: int = 6) -> np.ndarray:
    """Equal-count group of true values each object falls in: up to `most`,
    with at least 10 objects per group. Fewer, larger groups follow noise that
    changes along the range too coarsely; validation/jitter_reference.py
    measures the result."""
    k = max(1, min(most, len(truth_values) // 10))
    edges = np.quantile(truth_values, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, truth_values, side="right")


def unwrap(measure: Measure, values: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Readings moved next to the truth, so a tilt fit is not cut at +/-90."""
    if isinstance(measure, Tilt):
        return truth + ((values - truth + 90.0) % 180.0 - 90.0)
    return np.asarray(values, float)
