import csv
import io
import re
from typing import TYPE_CHECKING

import numpy as np

from poseaudit._version import __version__
from poseaudit.measures import Tilt

if TYPE_CHECKING:
    from poseaudit.audit import AuditResult


def _unit(result: "AuditResult") -> str:
    unit = result.measure.unit
    return {"deg": "°", "px": " px"}.get(unit, f" {unit}" if unit else "")


def paired(result: "AuditResult") -> bool:
    """Values given as pairs (a table or .mot files), not read off keypoints."""
    return result.settings.get("input") == "paired"


def _digits(result: "AuditResult") -> int:
    """Decimals for values in the measure's unit: 2 for degrees and pixels, and
    enough for a unitless ratio that small errors do not all print as 0.00."""
    if result.measure.unit or result.n == 0:
        return 2
    scale = result.p95_abs_error
    if not np.isfinite(scale) or scale <= 0:
        return 2
    return int(min(6, max(2, 1 - np.floor(np.log10(scale)))))


def _f(value: float, fmt: str) -> str:
    """`value` formatted, but never as -0.000: a value that rounds to zero
    has no sign worth printing. NaN (a figure not defined here) is n/a."""
    if value != value:
        return "n/a"
    text = f"{value:{fmt}}"
    if text.startswith("-") and not text.strip("-0.%"):
        text = ("+" if fmt.startswith("+") else "") + text[1:]
    return text


def _range(low: float, high: float, fmt: str, u: str) -> str:
    """Limits, or n/a when too few readings give none."""
    if np.isnan(low) or np.isnan(high):
        return "n/a"
    return f"{_f(low, fmt)}{u} to {_f(high, fmt)}{u}"


def _ci(interval, fmt: str) -> str:
    low, high = interval
    if not (np.isfinite(low) and np.isfinite(high)):  # e.g. one cluster only
        return "[no interval]"
    return f"[{_f(low, fmt)} to {_f(high, fmt)}]"


def _p(p: float, repeats) -> str:
    """No rebuild at or below the slope gives the smallest p the rebuilds can
    show, 1/(repeats+1); the true p may be lower still."""
    if repeats and p <= 1 / (repeats + 1) + 1e-12:
        return f"p(slope <= jitter) <= {1 / (repeats + 1) + 5e-4:.3f}*"
    return f"p(slope <= jitter) {p:.3f}*"


def _what(result: "AuditResult") -> str:
    m = result.measure
    rel = result.settings.get("relative_to")
    if not rel:
        return m.title()
    kind = "|value| minus the others' " if result.settings.get("relative_abs") else ""
    return f"{m.title()}, relative ({kind}{rel} of the rest of each image)"


def _decision(th, u: str, d: int = 2) -> str:
    rule = f"{th.side} {th.threshold:g}{u}"
    if th.matched:
        rule += (
            f", predicted {th.predicted_threshold:.{d}f}{u} (same count, fitted here)"
        )
        # chosen on these very readings: optimistic until checked on other data
    elif th.predicted_threshold != th.threshold:
        rule += f", predicted {th.predicted_threshold:g}{u}"
    return (
        f"  {rule}: caught {th.true_positive} of "
        f"{th.true_positive + th.false_negative} "
        f"({th.sensitivity:.0%} {_ci(th.sensitivity_ci, '.0%')}), false alarms "
        f"{th.false_positive} of {th.false_positive + th.true_negative}"
    )


NORMALISED = "Every visible coordinate lies within 0 to 1"


def _first(result: "AuditResult") -> list[str]:
    """Warnings to read before any figure: with normalised coordinates every
    figure is in the wrong units."""
    return [w for w in result.warnings if w.startswith(NORMALISED)]


def _size_unit(result: "AuditResult") -> str:
    """Sizes are in pixels, unless the coordinates look normalised."""
    return "" if _first(result) else " px"


def _normal_fits(result: "AuditResult") -> bool:
    """The normal limits are worth showing beside the percentile ones: enough
    readings to give them (3), and no more errors past either than chance
    allows (the same test as the warning)."""
    from poseaudit.agreement import tails_off

    return result.n >= 3 and not tails_off(result.n, result.tail_shares)


def summary(result: "AuditResult", full: bool = False) -> str:
    text = _summary(result, full)
    return _as_reference(text) if paired(result) else text


_REFERENCE = (
    (re.compile(r"\bthe truth's\b"), "the reference's"),
    (re.compile(r"\btrue(\s+)values\b"), r"reference\1values"),
    (re.compile(r"\b(the|on|by) truth\b"), r"\1 reference"),
    (re.compile(r"\btruth and prediction\b"), "reference and prediction"),
    (re.compile(r"\bpred on truth\b"), "pred on reference"),
)


def _as_reference(text: str) -> str:
    """Paired values are compared with a reference (motion capture, a
    goniometer), not a ground truth: the prose says so."""
    for pattern, word in _REFERENCE:
        text = pattern.sub(word, text)
    return text


def _summary(result: "AuditResult", full: bool = False) -> str:
    u = _unit(result)
    d = _digits(result)
    r = result.not_read
    first = _first(result)
    if paired(result):
        lines = [
            f"{_what(result)}: read {result.n} of {result.measurable} reference values",
            f"  not read     no predicted value {r.no_predicted_point}",
            f"  not counted  {r.unlabelled} with no reference value",
        ]
    else:
        lines = [
            f"{_what(result)}: read {result.n} of {result.measurable} labelled "
            "instances",
            *(f"  ! {w}" for w in first),
            f"  not read     no matching prediction {r.missed}, prediction lacked a "
            f"point {r.no_predicted_point}, unmeasurable {r.unmeasurable}"
            + (
                f", too few in frame {r.too_few_in_frame}" if r.too_few_in_frame else ""
            ),
            f"  not counted  {r.unlabelled} with a point unlabelled in the truth, "
            f"{r.unmatched_predictions} unmatched predictions",
        ]
    rest = [w for w in result.warnings if w not in first]
    if result.n == 0:
        return "\n".join(lines + [f"  ! {w}" for w in rest])
    elo, ehi = _percentile_limits(result)
    lo, hi = result.limits
    f2 = f"+.{d}f"
    lines += [
        f"  bias         {_f(result.bias, f2)}{u} {_ci(result.bias_ci, f2)}, "
        f"median {_f(result.median_error, f2)}{u}",
        f"  limits       {_range(elo, ehi, f2, u)} "
        "(percentile, 2.5th to 97.5th; JSON percentile_limits)",
    ]
    if full or _normal_fits(result):
        lines.append(
            f"  normal       {_range(lo, hi, f2, u)} (bias +/- 1.96 SD; JSON limits)"
        )
    if full and np.isfinite(lo):
        lines.append(
            f"  limit CIs    lower {_ci(result.lower_limit_ci, f2)}, upper "
            f"{_ci(result.upper_limit_ci, f2)} (cluster bootstrap)"
        )
        if np.isfinite(result.lower_limit_exact_ci[0]):
            lines.append(
                f"               lower {_ci(result.lower_limit_exact_ci, f2)}, upper "
                f"{_ci(result.upper_limit_exact_ci, f2)} (exact, Carkeet 2015)"
            )
    if result.repeated_limits is not None:  # only with named clusters
        rlo, rhi = result.repeated_limits
        who = "subjects" if result.settings.get("cluster") == "subject" else "clusters"
        lines.append(f"  repeated     {_range(rlo, rhi, f2, u)} ({who})")
    if len(result.by_subject) > 1 and _repeated(result):
        lines.append(_by_subject_line(result, f2, u, d))
    lines += [
        f"  |error|      mean {result.mean_abs_error:.{d}f}{u} "
        f"{_ci(result.mean_abs_error_ci, f'.{d}f')}, median "
        f"{result.median_abs_error:.{d}f}{u}, 95th pct {result.p95_abs_error:.{d}f}{u}",
        f"  RMSE         {result.rmse:.{d}f}{u} {_ci(result.rmse_ci, f'.{d}f')}",
        f"  >= {result.big_error:g}{u}".ljust(15)
        + f"{result.big_error_rate:.1%} {_ci(result.big_error_rate_ci, '.1%')}",
    ]
    if len(result.size_bands) > 1:
        lines.append(
            "  by size      "
            + ", ".join(
                f"{label} {b.big_error_rate:.0%} (n {b.n})"
                for label, b in zip(
                    _size_labels(result.size_bands, unit=_size_unit(result)),
                    result.size_bands,
                    strict=True,
                )
            )
        )
    lines += [
        f"  slope        {_f(result.gain, '.3f')} {_ci(result.gain_ci, '.3f')} "
        f"(pred on truth, 1 is ideal); Theil-Sen {_f(result.robust_gain, '.3f')}",
        f"  ICC(A,1)     {_f(result.icc, '.3f')} {_ci(result.icc_ci, '.3f')}",
    ]
    if full:
        lines += _full(result)
    lines += [_decision(th, u, d) for th in result.thresholds]
    lines += [f"  ! {w}" for w in rest]
    return "\n".join(lines)


def _repeated(result: "AuditResult") -> bool:
    """Some subject (or cluster) holds several readings: with one each, a
    summary per subject only repeats the readings."""
    return any(g.n > 1 for g in result.by_subject)


def _by_subject_line(result: "AuditResult", f2: str, u: str, d: int) -> str:
    """How far the subjects (or named clusters) differ from one another."""
    groups = result.by_subject
    who = "subjects" if result.settings.get("cluster") == "subject" else "clusters"
    bias = [g.bias for g in groups]
    mae = [g.mean_abs_error for g in groups]
    return (
        f"  by subject   {len(groups)} {who}: bias {_f(min(bias), f2)}{u} to "
        f"{_f(max(bias), f2)}{u}, mean |error| {min(mae):.{d}f}{u} to "
        f"{max(mae):.{d}f}{u}"
    )


def _full(result: "AuditResult") -> list[str]:
    """The other agreement statistics, then the experimental slope against the
    jitter reference, set apart: `summary(full=True)` and `--full`."""
    lines = [
        f"  BA slope     {_f(result.ba_slope, '+.3f')} "
        f"{_ci(result.ba_slope_ci, '+.3f')}"
    ]
    if result.deming is not None:
        lines.append(
            f"  Deming       {_f(result.deming, '.3f')} {_ci(result.deming_ci, '.3f')} "
            f"(noise ratio {result.settings.get('noise_ratio')})"
        )
    lines.append(
        f"  agreement    CCC {_f(result.ccc, '.3f')} {_ci(result.ccc_ci, '.3f')}, "
        f"r {_f(result.pearson, '.3f')}"
    )
    if result.jitter_gain is not None:
        lines.append("  experimental, still being validated:")
        lines.append(
            f"  vs jitter    {_f(result.jitter_gain, '.3f')} from keypoint jitter "
            "alone; gap "
            f"{result.gain_gap:+.3f} {_ci(result.gain_gap_ci, '+.3f')}, "
            + (
                _p(result.jitter_p, result.settings.get("jitter_repeats"))
                if result.jitter_p is not None
                else "no p with named clusters"
            )
        )
        if result.jitter_p is not None:
            lines.append(
                "               * swaps and gross failures lower the slope too, "
                "and noisy"
            )
            lines.append(
                "                 labels make p small for an honest model: read the gap"
            )
    elif result.settings.get("relative_to") is not None:
        lines.append(
            "  vs jitter    not computed: a relative reading also moves with the "
            "other parts in the frame"
        )
    return lines


SORTING = (
    "https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md"
    "#which-way-you-sort-decides-the-story"
)

GUIDE = """\
## How to read this

- **slope** (pred on truth): the least-squares slope of predicted on true
  values, the proportional bias; 1 is ideal (`gain` in the JSON). It is not
  the model's tendency alone:
  - keypoint jitter bends it, since near the ends of a range an error can only
    go one way.
  - **vs jitter** (experimental: still being validated, and reported apart
    from the core figures) is the slope on predictions rebuilt from the truth
    plus this
    model's displacements from the labels: each object takes all its points'
    shifts from one object with a similar true value (possibly itself), minus
    the shift that such a group shares, carried in each segment's own frame
    and mirrored for angles that bend the other way.
  - **gap** is the slope minus that reference; its interval averages a few
    rebuilds in every resample. **p(slope <= jitter)** is (1 + rebuilds at or
    below this slope) / (1 + rebuilds), one-sided. It assumes independent
    objects, so with named clusters it is left out.
  - A gap below 0 with a small p means the model reads differences as smaller
    than its displacements from the labels alone would make them. Possible
    causes: squashing, gross failures tied to the true value (a straight part
    read as bent), left and right swapped on one side, or noise in the truth.
    Gross failures in random directions are part of the rebuilds and do not
    widen the gap.
    With labels nearly as noisy as the model the p is often small for an
    honest model; read the size of the gap then, not the p.
  - noise in the truth pulls the slope toward 0; with a known ratio of noise
    variances, read the Deming slope, which removes that but not the jitter
    at the ends.
  - a few gross failures (a point on the wrong part) can move it a lot; the
    **Theil-Sen** slope (`robust_gain` in the JSON) barely moves with them.
    Noise that grows with the value or with smaller objects also separates
    the two, so their difference does not by itself measure gross failures.
- **bias by band**: sorting by the truth puts the truth's extremes in the end
  bands, where jitter and regression to the mean read as bias; see
  [Which way you sort decides the story]({sorting}).
- **BA slope**: the slope of the error on the mean of both readings. It compares
  the two spreads, so it reads as compression only when truth and prediction are
  about equally noisy; a noisier prediction pushes it up.
- **limits**: where 95% of errors fall. The percentile limits assume nothing;
  the normal ones assume a bell-shaped error, which a few gross failures break.
  Their intervals are cluster bootstraps; when every reading is its own
  {cluster}, exact parametric intervals for the normal limits (Carkeet 2015)
  are given too.
- **ICC(A,1), CCC, r**: agreement between the two readings, 1 at best.
- Intervals are percentile bootstraps over whole {cluster}s ({resamples}
  resamples, seed {seed}). The overall rate of large errors and the decision
  rates resample whole {cluster}s too when some hold several readings (never
  narrower than Wilson's); with one reading to each they use Wilson intervals.
  Rates per size band stay Wilson.
"""


def _size_labels(bands, joint: str = "-", unit: str = " px") -> list[str]:
    """Band edges in whole pixels, or with as many decimals as keep each band's
    two edges apart: the same number for every band."""
    spans = [(b.low, b.high) for b in bands if np.isfinite(b.high) and b.low != b.high]
    digits = next(
        (k for k in (0, 1) if all(f"{a:.{k}f}" != f"{b:.{k}f}" for a, b in spans)), 2
    )
    return [
        f"{b.low:.{digits}f}{joint}{b.high:.{digits}f}{unit}"
        if np.isfinite(b.high)
        else f"{b.low:.{digits}f}+{unit}"
        for b in bands
    ]


MIN_PERCENTILE = 10  # below this the 2.5th and 97.5th are just the extremes
NO_INTERVAL = (float("nan"), float("nan"))


def _percentile_limits(result: "AuditResult") -> tuple[float, float]:
    """The percentile limits, or NaN (printed n/a) under MIN_PERCENTILE
    readings, where they are only the smallest and largest errors."""
    if result.n < MIN_PERCENTILE:
        return NO_INTERVAL
    return result.empirical_limits


def _cluster_word(result: "AuditResult") -> str:
    """What the bootstrap resamples whole: images, named clusters, subjects
    or rows."""
    given = result.settings.get("cluster")
    return given if given in ("image", "subject", "row") else "cluster"


def _limit_row(name, low, high, low_ci, high_ci, u, d=2) -> str:
    if np.isnan(low) or np.isnan(high):
        return f"| {name} | n/a | | n/a | |"
    return (
        f"| {name} | {_f(low, f'+.{d}f')}{u} | {_ci(low_ci, f'+.{d}f')} "
        f"| {_f(high, f'+.{d}f')}{u} | {_ci(high_ci, f'+.{d}f')} |"
    )


def _amount(value: float, unit: str, fmt: str = "g") -> str:
    """A value and its unit in words: 1 degree, 15 degrees, 2.5 px."""
    text = f"{value:{fmt}}"
    word = {"deg": "degree" if text == "1" else "degrees", "px": "px"}.get(
        unit, unit or None
    )
    return f"{text} {word}" if word else text


def _share(rate: float) -> str:
    """Whole percent, but a rare event never shows as 0% nor a near-certain
    one as 100%."""
    text = f"{rate:.0%}"
    if text == "0%" and rate > 0:
        return "<1%"
    if text == "100%" and rate < 1:
        return ">99%"
    return text


def _subject(result: "AuditResult") -> str:
    m = result.measure
    rel = result.settings.get("relative_to")
    if not rel:
        return f"the {m.name}"
    base = "a baseline" if rel == "custom" else f"the {rel}"
    if result.settings.get("relative_abs"):
        return f"|{m.name}| minus {base} of the others' |{m.name}| in its image"
    return f"the {m.name} relative to {base} of the rest of its image"


def _sign(result: "AuditResult") -> str:
    """Which way a positive error points. A tilt's error is folded into
    [-90, 90): a truth of +88 read as -88 is +4, the axis turned 4 degrees
    clockwise, not -176."""
    relative = result.settings.get("relative_to") is not None
    against = ", each measured against the rest of its image" if relative else ""
    if result.settings.get("relative_abs"):
        return (
            "An error is predicted minus truth: a positive error means the "
            "prediction's |value|, less its image's baseline, is larger."
        )
    period = getattr(result.measure, "period", None)
    if period:
        return (
            "An error is predicted minus reference, taken the short way round "
            f"(differences wrap at +/-{period / 2:g}): a positive error means the "
            "prediction is larger."
        )
    if isinstance(result.measure, Tilt):
        return (
            "An error is predicted minus truth, taken the short way round: a "
            "positive error means the predicted axis is turned clockwise from the "
            f"true one as seen in the image{against}."
        )
    return (
        "An error is predicted minus truth: a positive error means the prediction "
        f"is larger{against}."
    )


LABELS = (
    "These figures are agreement with the labels, not error against the true "
    "value: noise in the labels is part of them."
)
REFERENCE = (
    "These figures are agreement with the reference, not error against the "
    "true value: noise in the reference is part of them."
)


def _lead(result: "AuditResult") -> str:
    """The headline in words: what it is measured against and on how many,
    so a sentence lifted from it does not read as error against the world,
    and which way an error points."""
    m = result.measure
    d = 1 if m.unit else _digits(result)
    if paired(result):
        return (
            f"On average {_subject(result)} is "
            f"{_amount(result.mean_abs_error, m.unit, f'.{d}f')} off the "
            f"reference; {_share(result.big_error_rate)} of readings are off by "
            f"{_amount(result.big_error, m.unit)} or more ({result.n} of "
            f"{result.measurable} reference values read). {REFERENCE} "
            f"{_sign(result)}"
        )
    return (
        f"On average {_subject(result)} is "
        f"{_amount(result.mean_abs_error, m.unit, f'.{d}f')} off the labels; "
        f"{_share(result.big_error_rate)} of readings are off by "
        f"{_amount(result.big_error, m.unit)} or more ({result.n} of "
        f"{result.measurable} labelled instances read). {LABELS} {_sign(result)}"
    )


def _cell(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value) or "none"
    return str(value).replace("|", "\\|")  # a regex may hold one


def _settings(result: "AuditResult") -> list[str]:
    return ["## Settings", "", "| setting | value |", "|---|---|"] + [
        f"| {key} | {_cell(value)} |" for key, value in result.settings.items()
    ]


def markdown(result: "AuditResult", worst: int = 10) -> str:
    text = _markdown(result, worst)
    return _as_reference(text) if paired(result) else text


def _markdown(result: "AuditResult", worst: int = 10) -> str:
    u = _unit(result)
    d = _digits(result)
    out = [f"# poseaudit {__version__}: {_what(result)}", ""]
    out += [f"**{w}**\n" for w in _first(result)]
    if result.n:
        out += [_lead(result), ""]
    out += [
        f"An error of {_amount(result.big_error, result.measure.unit)} or more "
        "counts as large (`--big-error`, `big_error` in the JSON); bands are cut "
        f"on the {result.band_by}; the settings are listed at the end.",
        "",
        "## Summary",
        "",
        "```",
        summary(result, full=True),
        "```",
        "",
    ]
    if result.n == 0:
        return "\n".join(out + _settings(result)) + "\n"
    cluster = _cluster_word(result)
    guide = GUIDE.format(
        u=u,
        cluster=cluster,
        resamples=result.settings.get("resamples"),
        seed=result.settings.get("seed"),
        sorting=SORTING,
    )
    if paired(result):  # no keypoints: nothing jitters, and nothing to rebuild
        start = guide.index("  - keypoint jitter bends it")
        guide = guide[:start] + guide[guide.index("  - noise in the truth") :]
        guide = guide.replace("\n  Rates per size band stay Wilson.", "")
    out.append(guide)
    lo, hi = result.limits
    elo, ehi = _percentile_limits(result)
    below, above = result.tail_shares
    out += [
        "## Limits of agreement",
        "",
        "| limits | lower | 95% CI | upper | 95% CI |",
        "|---|---:|---:|---:|---:|",
        _limit_row(
            "percentile 2.5-97.5",
            elo,
            ehi,
            *(
                (result.empirical_lower_ci, result.empirical_upper_ci)
                if result.n >= MIN_PERCENTILE
                else (NO_INTERVAL, NO_INTERVAL)
            ),
            u,
            d,
        ),
        _limit_row(
            "normal, bias ± 1.96 SD",
            lo,
            hi,
            result.lower_limit_ci,
            result.upper_limit_ci,
            u,
            d,
        ),
    ]
    if np.isfinite(result.lower_limit_exact_ci[0]):
        out.append(
            _limit_row(
                "normal, exact CI (Carkeet)",
                lo,
                hi,
                result.lower_limit_exact_ci,
                result.upper_limit_exact_ci,
                u,
                d,
            )
        )
    if result.repeated_limits is not None:
        rlo, rhi = result.repeated_limits
        out.append(
            _limit_row(
                "repeated readings",
                rlo,
                rhi,
                result.repeated_lower_ci,
                result.repeated_upper_ci,
                u,
                d,
            )
        )
    how = f"The intervals of the limits are percentile bootstraps over whole {cluster}s"
    if np.isfinite(result.lower_limit_exact_ci[0]):
        how += (
            "; the exact ones for the normal limits use the noncentral t "
            "(Carkeet 2015) and assume independent, normal errors."
        )
    else:
        how += (
            ". Exact parametric intervals are left out: they assume one reading "
            f"per {cluster}."
            if result.clusters < result.n
            else "."
        )
    out += ["", how]
    if np.isfinite(below):
        out += [
            "",
            f"{below:.1%} of errors fall below the normal limits and {above:.1%} "
            "above them (2.5% each for a normal error).",
        ]
    out += _group_tables(result, u, d)
    if result.size_bands:
        out += [
            "",
            "## Error by size of the measured part",
            "",
            "| size | n | mean abs error | large errors | slope |",
            "|---|---:|---:|---:|---:|",
        ]
        spans = _size_labels(result.size_bands, " to ", _size_unit(result))
        for span, sb in zip(spans, result.size_bands, strict=True):
            out.append(
                f"| {span} | {sb.n} | {sb.mean_abs_error:.{d}f}{u} | "
                f"{sb.big_error_rate:.1%} {_ci(sb.big_error_rate_ci, '.1%')} | "
                f"{_f(sb.gain, '.3f')} |"
            )
    out += [
        "",
        f"## Error by {result.band_by} value",
        "",
        "| value | n | mean abs error | bias | bias 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    e = 1 if result.measure.unit else d  # band edges of a ratio need its decimals
    for b in result.bands:
        out.append(
            f"| {b.low:.{e}f} to {b.high:.{e}f}{u} | {b.n} | "
            f"{b.mean_abs_error:.{d}f}{u} "
            f"| {b.bias:+.{d}f}{u} | {_ci(b.bias_ci, f'+.{d}f')} |"
        )
    if result.thresholds:
        out += ["", "## Decisions at a threshold", ""]
        out += [
            "| truth | prediction | caught | missed | false alarms | agreed clear | "
            "sensitivity | precision |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for th in result.thresholds:
            pred = f"{th.predicted_threshold:.{d}f}{u}" + (
                " (same count, fitted on this data: check it on other data)"
                if th.matched
                else ""
            )
            out.append(
                f"| {th.side} {th.threshold:g}{u} | {pred} | {th.true_positive} | "
                f"{th.false_negative} | {th.false_positive} | {th.true_negative} | "
                f"{th.sensitivity:.1%} {_ci(th.sensitivity_ci, '.1%')} | "
                f"{th.precision:.1%} {_ci(th.precision_ci, '.1%')} |"
            )
    if paired(result):
        out += [
            "",
            "## Largest errors",
            "",
            "| subject | trial | frame | reference | predicted | error |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in result.worst(worst):
            frame = "" if r.frame is None else f"{r.frame:g}"
            out.append(
                f"| {_cell(r.cluster)} | {_cell(r.trial)} | {frame} | "
                f"{r.truth:.{d}f}{u} | {r.predicted:.{d}f}{u} | {r.error:+.{d}f}{u} |"
            )
        out += ["", *_settings(result)]
        return "\n".join(out) + "\n"
    out += [
        "",
        "## Largest errors",
        "",
        "| image | instance | size | truth | predicted | error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    px = _size_unit(result)
    sd = 0 if px else 3  # fractions of the image would all round to 0
    for r in result.worst(worst):
        out.append(
            f"| {r.image} | {r.truth_index} | {r.size:.{sd}f}{px} | {r.truth:.{d}f}{u} "
            f"| {r.predicted:.{d}f}{u} | {r.error:+.{d}f}{u} |"
        )
    out += ["", *_settings(result)]
    return "\n".join(out) + "\n"


def _group_tables(result: "AuditResult", u: str, d: int) -> list[str]:
    """The errors per subject, and per trial of a subject when trials are
    given: a subject far off the others shows here, not in the pooled
    figures."""
    out: list[str] = []
    who = "subject" if result.settings.get("cluster") == "subject" else "cluster"
    for title, rows, trials in (
        (f"## Error by {who}", result.by_subject, False),
        (f"## Error by {who} and trial", result.by_trial, True),
    ):
        if not rows or not _repeated(result):
            continue
        head = f"| {who} |" + (" trial |" if trials else "")
        rule = "|---|" + ("---|" if trials else "")
        out += [
            "",
            title,
            "",
            head + " n | bias | SD | mean abs error | RMSE |",
            rule + "---:|---:|---:|---:|---:|",
        ]
        for g in rows:
            lead = f"| {_cell(g.subject)} |" + (
                f" {_cell(g.trial)} |" if trials else ""
            )
            sd = _f(g.sd, f".{d}f") + (u if np.isfinite(g.sd) else "")
            out.append(
                f"{lead} {g.n} | {_f(g.bias, f'+.{d}f')}{u} | {sd} | "
                f"{g.mean_abs_error:.{d}f}{u} | {g.rmse:.{d}f}{u} |"
            )
    return out


CSV_COLUMNS = [
    "image",
    "truth_index",
    "predicted_index",
    "class_id",
    "cluster",
    "size",
    "truth",
    "predicted",
    "error",
    "mean",
]


# paired values: a reading's cluster is its subject, truth its reference
PAIRED_COLUMNS = {
    "subject": "cluster",
    "trial": "trial",
    "frame": "frame",
    "ref": "truth",
    "pred": "predicted",
    "error": "error",
    "mean": "mean",
}


def _columns(result: "AuditResult") -> dict[str, str]:
    """CSV header: the reading's attribute it holds."""
    if paired(result):
        return PAIRED_COLUMNS
    return {c: c for c in CSV_COLUMNS}


def _values(r, columns: dict[str, str]) -> list:
    return ["" if getattr(r, a) is None else getattr(r, a) for a in columns.values()]


def csv_rows(result: "AuditResult") -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    columns = _columns(result)
    writer.writerow(list(columns))
    for r in result.readings:
        writer.writerow(_values(r, columns))
    return buffer.getvalue()


def label(result: "AuditResult") -> str:
    """The measure as given on the command line: angle 5,7,9."""
    m = result.measure
    if not m.points:
        return m.name
    return f"{m.name} {','.join(str(p) for p in m.points)}"


def row(result: "AuditResult") -> dict:
    """The headline figures as one flat row, for tables of several measures or
    models. Floats as computed, NaN where a figure is not defined."""
    lo, hi = _percentile_limits(result)
    return {
        "measure": label(result),
        "unit": result.measure.unit,
        "n": result.n,
        "bias": result.bias,
        "bias_ci_low": result.bias_ci[0],
        "bias_ci_high": result.bias_ci[1],
        "limits_low": lo,
        "limits_high": hi,
        "mean_abs_error": result.mean_abs_error,
        "mean_abs_error_ci_low": result.mean_abs_error_ci[0],
        "mean_abs_error_ci_high": result.mean_abs_error_ci[1],
        "rmse": result.rmse,
        "big_error": result.big_error,
        "big_error_rate": result.big_error_rate,
        "big_error_rate_ci_low": result.big_error_rate_ci[0],
        "big_error_rate_ci_high": result.big_error_rate_ci[1],
        "slope": result.gain,
        "icc": result.icc,
    }


TABLE_COLUMNS = ("measure", "n", "bias", "limits", "mean |error|", "RMSE")


def _threshold(result: "AuditResult") -> str:
    return f"{result.big_error:g}{_unit(result)}"


def _cells(result: "AuditResult", threshold: bool = False) -> list[str]:
    """A measure's row; with `threshold`, its own large-error threshold
    before the rate, for a table whose measures differ in it."""
    u = _unit(result)
    d = _digits(result)
    lo, hi = _percentile_limits(result)
    own = [f">= {_threshold(result)}"] if threshold else []
    if result.n == 0:
        return [label(result), "0"] + ["n/a"] * 4 + own + ["n/a"] * 3
    return [
        label(result),
        str(result.n),
        f"{_f(result.bias, f'+.{d}f')}{u} {_ci(result.bias_ci, f'+.{d}f')}",
        _range(lo, hi, f"+.{d}f", u),
        f"{result.mean_abs_error:.{d}f}{u} {_ci(result.mean_abs_error_ci, f'.{d}f')}",
        f"{result.rmse:.{d}f}{u}",
        *own,
        f"{result.big_error_rate:.1%} {_ci(result.big_error_rate_ci, '.1%')}",
        _f(result.gain, ".3f"),
        _f(result.icc, ".3f"),
    ]


_WHOLE = re.compile(r"[+-]?\d+")


def _on_the_point(cells: list[str]) -> list[str]:
    """Figures padded so their decimal points line up (" 7.61°" under
    "29.57°"), whatever unit follows; a count comes out right-aligned. A cell
    with no figure, such as n/a, stays at the left."""
    whole = [len(m.group(0)) if (m := _WHOLE.match(c)) else None for c in cells]
    lead = max((w for w in whole if w is not None), default=0)
    return [
        c if w is None else " " * (lead - w) + c
        for c, w in zip(cells, whole, strict=True)
    ]


def layout(
    rows: list[list[str]],
    figures: frozenset[int] = frozenset(),
    keep: int = 1,
    width: int | None = None,
) -> list[str]:
    """A header row and data rows as aligned text. In the columns in
    `figures` the decimal points line up, and the header of a column of
    counts is right-aligned over it. When `width` is given and the table is
    wider, it is cut into blocks of columns that fit, one under the other,
    each led by the first `keep` columns, instead of every row wrapping onto
    the next line of a narrow terminal."""
    rows = [list(row) for row in rows]
    right = set()
    for k in figures:
        cells = _on_the_point([row[k] for row in rows[1:]])
        for row, cell in zip(rows[1:], cells, strict=True):
            row[k] = cell
        if all(c.strip().isdigit() for c in cells):
            right.add(k)
    widths = [max(len(row[k]) for row in rows) for k in range(len(rows[0]))]
    blocks: list[list[int]] = [list(range(len(widths)))]
    if width is not None:
        lead = sum(widths[:keep]) + 2 * keep
        blocks = [[]]
        used = lead
        for k in range(keep, len(widths)):
            # a line filling the last column wraps on Windows: stop one short
            if blocks[-1] and used + widths[k] >= width:
                blocks.append([])
                used = lead
            blocks[-1].append(k)
            used += widths[k] + 2
        blocks = [list(range(keep)) + b for b in blocks]
    lines: list[str] = []
    for block in blocks:
        if lines:
            lines.append("")
        lines += [
            "  ".join(
                row[k].rjust(widths[k]) if k in right else row[k].ljust(widths[k])
                for k in block
            ).rstrip()
            for row in rows
        ]
    return lines


def table(
    results: "list[AuditResult]", degree: str = "°", width: int | None = None
) -> str:
    """One row per measure: n, bias, percentile limits, mean |error| with its
    interval, RMSE, the large-error rate with its interval, slope and ICC(A,1).
    The rate's header names the threshold when every measure shares it, and
    a column before it gives each measure's own when they differ. The
    warnings follow (`warning_lines`). With `width`, a table wider than that
    is cut into blocks of columns that fit."""
    head, one = _head(results)
    cells = [_cells(r, threshold=not one) for r in results]
    rows = [[c.replace("°", degree) for c in row] for row in [head, *cells]]
    slope = len(head) - 2
    figures = frozenset({1, 2, 5, slope, slope + 1})  # n, bias, RMSE, slope, ICC
    lines = layout(rows, figures, keep=1, width=width)
    return "\n".join(lines + warning_lines(results))


def _head(results: "list[AuditResult]") -> tuple[list[str], bool]:
    """The table's header, and whether every measure shares one large-error
    threshold: then the rate's header names it, else a column before the
    rate gives each measure's own."""
    thresholds = list(dict.fromkeys(map(_threshold, results)))
    one = len(thresholds) <= 1
    rate = (
        [f">= {thresholds[0]}"] if one and thresholds else ["large if", "large errors"]
    )
    return [*TABLE_COLUMNS, *rate, "slope", "ICC(A,1)"], one


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _names(names: list[str]) -> str:
    return " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)


# left-aligned columns of a Markdown table; the others hold figures
_TEXT_COLUMNS = ("measure", "model", "limits", "large if", "a - b")


def _markdown_head(head: list[str]) -> list[str]:
    return [
        _markdown_row(head),
        "|" + "|".join("---" if h in _TEXT_COLUMNS else "---:" for h in head) + "|",
    ]


def comparison_markdown(comparison, notes: "list[str] | None" = None) -> str:
    """`Comparison.to_markdown`: every model's figures in one table, the
    differences on shared readings in another, then the warnings (`notes`
    first: those raised while loading) and the settings."""
    from poseaudit.compare import difference_cells, difference_warnings

    names = list(comparison.results)
    per_model = list(comparison.results.values())
    first = per_model[0] if per_model else []
    settings = comparison.settings()
    head, one = _head(first)
    head = ["mean abs error" if h == "mean |error|" else h for h in head]
    head.insert(1, "model")
    if one and first:
        large = (
            f"An error of {_amount(first[0].big_error, first[0].measure.unit)} "
            "or more counts as large."
        )
    else:
        large = "What counts as a large error differs by measure: see large if."
    cluster = "image" if settings.get("cluster") == "image" else "cluster"
    resamples = settings.get("resamples")
    measures = f"{len(first)} {'measure' if len(first) == 1 else 'measures'}"
    out = [
        f"# poseaudit {__version__}: {_names(names)} compared",
        "",
        f"Each model read against the same labels, on {measures}. {large} {LABELS}",
        "",
        "## Each model",
        "",
        *_markdown_head(head),
    ]
    for k in range(len(first)):
        for name, results in zip(names, per_model, strict=True):
            cells = _cells(results[k], threshold=not one)
            out.append(_markdown_row([cells[0], name, *cells[1:]]))
    how = (
        "Each row covers the readings that model made, which differ from model "
        "to model when one skips people another reads: rank models on the "
        "differences below, which use only the readings both made. n counts "
        "readings; limits are the 2.5th to 97.5th percentiles of the errors; "
        "slope is that of predicted on true values, 1 at best."
    )
    if isinstance(resamples, int):
        how += (
            f" Intervals are 95% percentile bootstraps over whole {cluster}s "
            f"({resamples:,} resamples, seed {settings.get('seed')})."
        )
    out += ["", how]
    if comparison.differences:
        out += [
            "",
            "## Differences on shared readings",
            "",
            "Model a minus model b on the labelled instances both read, so below "
            "0 a is the closer to the truth. The paired intervals resample whole "
            f"{cluster}s.",
            "",
            *_markdown_head(
                [
                    "measure",
                    "a - b",
                    "shared",
                    "n a",
                    "n b",
                    "mean abs error a - b",
                    "large-error rate a - b",
                ]  # fmt: skip
            ),
        ]
        out += [
            _markdown_row(cells)
            for cells in difference_cells(comparison.differences, comparison.results)
        ]
    warned = [f"- loading: {n}" for n in notes or []]
    for name, results in zip(names, per_model, strict=True):
        warned += [f"- {name}, {line[4:]}" for line in warning_lines(results)]
    warned += [f"- {w}" for w in difference_warnings(comparison.differences)]
    if warned:
        out += ["", "## Warnings", "", *warned]
    rows = [f"| {key} | {_cell(value)} |" for key, value in settings.items()]
    paths = [
        f"{name}: {results[0].settings['pred']}"
        for name, results in zip(names, per_model, strict=True)
        if results and "pred" in results[0].settings
    ]
    if paths:
        rows.append(f"| pred | {_cell('; '.join(paths))} |")
    out += ["", "## Settings", "", "| setting | value |", "|---|---|", *rows]
    return "\n".join(out) + "\n"


def warning_lines(results: "list[AuditResult]") -> list[str]:
    """The warnings of several measures, each once. A measure's own come
    first, under its name; a warning several measures raise alike comes after
    them, once, naming them ("every measure" for all). Advice that ends the
    warnings of several measures, such as what to do about near misses, is
    split from the counts that differ between them and given once too."""
    from poseaudit.audit import SHARED_ADVICE

    names = [label(r) for r in results]
    raised = [w for r in results for w in dict.fromkeys(r.warnings)]
    split = [a for a in SHARED_ADVICE if sum(w.endswith(" " + a) for w in raised) > 1]
    who: dict[str, list[str]] = {}
    for name, result in zip(names, results, strict=True):
        for w in result.warnings:
            advice = next((a for a in split if w.endswith(" " + a)), None)
            parts = [w[: -len(advice)].rstrip(), advice] if advice else [w]
            for part in parts:
                if name not in who.setdefault(part, []):
                    who[part].append(name)
    own = [f"  ! {n[0]}: {text}" for text, n in who.items() if len(n) == 1]
    shared = [
        f"  ! {'every measure' if len(n) == len(names) else ', '.join(n)}: {text}"
        for text, n in who.items()
        if len(n) > 1
    ]
    return own + shared


def csv_rows_many(results: "list[AuditResult]") -> str:
    """The readings of several measures, each row led by its measure."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    columns = _columns(results[0]) if results else {}
    writer.writerow(["measure", *columns])
    for result in results:
        for r in result.readings:
            writer.writerow([label(result), *_values(r, columns)])
    return buffer.getvalue()
