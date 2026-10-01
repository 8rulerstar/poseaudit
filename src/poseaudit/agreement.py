"""Agreement statistics between two readings of the same quantity.

`t` is the reference (ground truth), `p` the prediction, `e = p - t`.
"""

import numpy as np

from poseaudit.confidence import Z95


def gain(t: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of predicted on truth by least squares.

    A slope of 0.6 means the model reads a 10 degree change as 6. Unbiased
    when the truth is much less noisy than the prediction, the usual case for
    human labels or motion capture against a model.
    """
    if len(t) < 3 or _flat(t):
        return float("nan"), float("nan")
    dt = t - t.mean()
    slope = float(dt @ (p - p.mean()) / (dt @ dt))
    return slope, float(p.mean() - slope * t.mean())


def robust_gain(
    t: np.ndarray, p: np.ndarray, pairs: int = 500_000, seed: int = 0
) -> float:
    """Theil-Sen slope of predicted on truth: the median of the slopes between
    pairs of readings, all of them up to about 1000 readings and a random
    `pairs` of them above. A few gross failures move a least-squares gain a lot
    and this one little; so does noise that grows with the value, so a gap
    between the two is not by itself proof of gross failures."""
    n = len(t)
    if n < 3 or _flat(t):
        return float("nan")
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i, j = rng.integers(0, n, pairs), rng.integers(0, n, pairs)
    dt = t[j] - t[i]
    keep = dt != 0
    return float(np.median((p[j] - p[i])[keep] / dt[keep]))


def ba_slope(t: np.ndarray, p: np.ndarray) -> float:
    """Slope of the error on the mean of both readings (Bland-Altman).

    Equal to (var p - var t) / (2 var mean): a test of equal spread, which reads
    as proportional bias only when both readings are about equally noisy. A
    noisier prediction pushes it upward and can hide real compression.
    """
    m = (t + p) / 2
    if len(t) < 3 or np.ptp(m) == 0:
        return float("nan")
    return float(np.polyfit(m, p - t, 1)[0])


def deming(t: np.ndarray, p: np.ndarray, noise_ratio: float) -> float:
    """Slope of predicted on truth when both carry noise, `noise_ratio` being
    var(prediction noise) / var(truth noise). Infinity gives `gain`."""
    if len(t) < 3:
        return float("nan")
    sxx, syy = np.var(t, ddof=1), np.var(p, ddof=1)
    sxy = np.cov(t, p, ddof=1)[0, 1]
    if sxy == 0:
        return float("nan")
    if np.isinf(noise_ratio):
        return float(sxy / sxx)
    a = syy - noise_ratio * sxx
    return float((a + np.sqrt(a * a + 4 * noise_ratio * sxy * sxy)) / (2 * sxy))


def icc_a1(t: np.ndarray, p: np.ndarray) -> float:
    """ICC(A,1): two-way, absolute agreement, single measurement
    (McGraw and Wong 1996)."""
    y = np.column_stack([t, p])
    n, k = y.shape
    if n < 2:
        return float("nan")
    grand = y.mean()
    ss_rows = k * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_error = ((y - grand) ** 2).sum() - ss_rows - ss_cols
    ms_r = ss_rows / (n - 1)
    ms_c = ss_cols / (k - 1)
    ms_e = ss_error / ((n - 1) * (k - 1))
    denominator = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denominator) if denominator else float("nan")


def ccc(t: np.ndarray, p: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    if len(t) < 2:
        return float("nan")
    cov = np.mean((t - t.mean()) * (p - p.mean()))
    denominator = t.var() + p.var() + (t.mean() - p.mean()) ** 2
    return float(2 * cov / denominator) if denominator else float("nan")


def limits(e: np.ndarray) -> tuple[float, float]:
    """Bias +/- 1.96 SD: 95% of errors, if they are roughly normal."""
    if len(e) < 2:
        return float("nan"), float("nan")
    half = Z95 * e.std(ddof=1)
    return float(e.mean() - half), float(e.mean() + half)


def empirical_limits(e: np.ndarray) -> tuple[float, float]:
    """The 2.5th and 97.5th percentiles: no assumption about the shape, which
    matters when a few gross failures make the tails heavy."""
    if len(e) < 2:
        return float("nan"), float("nan")
    low, high = np.percentile(e, [2.5, 97.5])
    return float(low), float(high)


def repeated_limits(e: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Limits for several readings per subject, the true value varying within
    a subject (Bland and Altman 2007): between- and within-subject variance
    are added rather than every reading counted as independent."""
    labels, inverse = np.unique(clusters, return_inverse=True)
    counts = np.bincount(inverse)
    n, groups = len(e), len(labels)
    if groups < 2 or groups == n:
        return limits(e)
    means = np.bincount(inverse, weights=e) / counts
    grand = e.mean()
    ms_between = (counts * (means - grand) ** 2).sum() / (groups - 1)
    ms_within = ((e - means[inverse]) ** 2).sum() / (n - groups)
    divisor = (n * n - (counts**2).sum()) / ((groups - 1) * n)
    between = max(0.0, (ms_between - ms_within) / divisor)
    half = Z95 * np.sqrt(between + ms_within)
    return float(grand - half), float(grand + half)


def _flat(t: np.ndarray) -> bool:
    """No spread to fit a slope on: a range that is only rounding noise."""
    return bool(np.ptp(t) <= 1e-9 * max(1.0, float(np.abs(t).max())))
