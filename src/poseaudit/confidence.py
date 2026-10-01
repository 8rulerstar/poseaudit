"""Intervals. Readings from one image are not independent (one camera, one
scene), so resampling is by image, not by reading."""

import warnings
from collections.abc import Callable, Sequence

import numpy as np

Z95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """For a rate. A percentile bootstrap of a rare event collapses to [0, 0]
    when none occurred and covers far less than 95% when few did."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def clustered_bootstrap(
    groups: Sequence[str] | np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile intervals, resampling whole groups. `statistic` receives the
    indices of the readings drawn and returns one value per quantity, so every
    interval in a report comes from the same resamples. Seeded, so a report is
    reproducible."""
    labels, inverse = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(labels))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(resamples):
        drawn = rng.integers(0, len(members), size=len(members))
        stats.append(statistic(np.concatenate([members[g] for g in drawn])))
    tail = (1.0 - level) / 2.0 * 100.0
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a band absent from a resample
        low, high = np.nanpercentile(np.array(stats), [tail, 100.0 - tail], axis=0)
    return low, high
