"""Agreement statistics between two readings of the same quantity.

`t` is the reference (ground truth), `p` the prediction, `e = p - t`.
"""

import math

import numpy as np

from poseaudit.confidence import Z95


def wrap(difference, period: float):
    """A difference taken the short way round a circle of `period` (360 for
    degrees): 179 against -179 is 2 apart, not 358. In [-period/2, period/2)."""
    half = period / 2
    return (np.asarray(difference, float) + half) % period - half


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
    if len(t) < 3 or _flat(t) or np.ptp(m) == 0:
        return float("nan")  # a flat truth would give exactly 2 whatever p is
    return float(np.polyfit(m, p - t, 1)[0])


def deming(t: np.ndarray, p: np.ndarray, noise_ratio: float) -> float:
    """Slope of predicted on truth when both carry noise, `noise_ratio` being
    var(prediction noise) / var(truth noise). Infinity gives `gain`."""
    if len(t) < 3 or _flat(t):
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
    (McGraw and Wong 1996). NaN when the truth does not vary: there is nothing
    to agree on, not an agreement of 0."""
    if len(t) < 2 or _flat(t):
        return float("nan")
    return icc_a1_table(np.column_stack([t, p]))


def icc_a1_table(y: np.ndarray) -> float:
    """ICC(A,1) of an n targets by k raters table: Shrout and Fleiss's ICC(2,1),
    McGraw and Wong's ICC(A,1). `icc_a1` is the case of two raters."""
    y = np.asarray(y, float)
    n, k = y.shape
    if n < 2 or k < 2:
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
    """Lin's concordance correlation coefficient; NaN when the truth does not
    vary."""
    if len(t) < 2 or _flat(t):
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


def limits_exact_ci(
    e: np.ndarray, level: float = 0.95, z: float = Z95
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Exact parametric intervals for the lower and upper normal limits,
    bias -/+ z SD (Carkeet 2015). The upper limit estimates mu + z sigma, and
    (mu + z sigma - mean) / (SD / sqrt(n)) follows a noncentral t with n - 1
    degrees of freedom and noncentrality z sqrt(n), so its quantiles give the
    interval. It assumes independent, normally distributed differences: one
    reading per subject. NaN under 3 readings or with no spread."""
    nan = (float("nan"), float("nan"))
    n = len(e)
    if n < 3:
        return nan, nan
    mean, sd = float(np.mean(e)), float(np.std(e, ddof=1))
    if not sd > 0:
        return nan, nan
    tail = (1.0 - level) / 2.0
    root = math.sqrt(n)
    low = nct_ppf(tail, n - 1, z * root) / root
    high = nct_ppf(1.0 - tail, n - 1, z * root) / root
    upper = (mean + sd * low, mean + sd * high)
    lower = (mean - sd * high, mean - sd * low)
    return lower, upper


_ERF = np.frompyfunc(math.erf, 1, 1)


def _chi_grid(df: float, points: int = 4001) -> tuple[np.ndarray, np.ndarray]:
    """sqrt(W / df) for W chi-squared with `df` degrees of freedom, on a grid
    in log W wide enough to hold all but about e^-35 of the mass, with
    trapezoid weights that sum to 1."""
    centre = math.log(df)
    spread = math.sqrt(2.0 / df)
    y = np.linspace(
        centre - 80.0 / df - 12.0 * spread,
        centre + math.log(1.0 + 12.0 * spread + 80.0 / df),
        points,
    )
    log_density = (
        (df / 2) * y - np.exp(y) / 2 - (df / 2) * math.log(2.0) - math.lgamma(df / 2)
    )
    weights = np.exp(log_density - log_density.max())
    weights[[0, -1]] /= 2
    return np.exp(y / 2) / math.sqrt(df), weights / weights.sum()


def nct_cdf(x: float, df: float, nc: float) -> float:
    """P(T <= x) for T noncentral t: (Z + nc) / sqrt(W / df), integrated over
    W on a grid. NumPy alone; agrees with scipy.stats.nct to about 1e-9."""
    s, w = _chi_grid(df)
    return _mixed_normal(x * s - nc, w)


def _mixed_normal(z: np.ndarray, weights: np.ndarray) -> float:
    """The normal CDF at each of `z`, averaged with `weights`."""
    erf = np.asarray(_ERF(z / math.sqrt(2.0)), float)
    return float((0.5 * (1.0 + erf)) @ weights)


def nct_ppf(q: float, df: float, nc: float) -> float:
    """The `q` quantile of the noncentral t, by bisection on `nct_cdf`."""
    s, w = _chi_grid(df)

    def cdf(x: float) -> float:
        return _mixed_normal(x * s - nc, w)

    low, high = nc - 10.0, nc + 10.0
    while cdf(low) > q:
        low = nc - 2 * (nc - low)
    while cdf(high) < q:
        high = nc + 2 * (high - nc)
    for _ in range(200):
        middle = (low + high) / 2
        if cdf(middle) < q:
            low = middle
        else:
            high = middle
        if high - low <= 1e-12 * max(1.0, abs(middle)):
            break
    return (low + high) / 2


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
    """No spread to fit a slope on: a range that is only rounding noise, a few
    thousand times the spacing of doubles at the values' size."""
    return bool(np.ptp(t) <= 1e-12 * max(1.0, float(np.abs(t).max())))
