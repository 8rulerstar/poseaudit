import csv
import io
from typing import TYPE_CHECKING

import numpy as np

from poseaudit._version import __version__
from poseaudit.measures import Tilt

if TYPE_CHECKING:
    from poseaudit.audit import AuditResult


def _unit(result: "AuditResult") -> str:
    return {"deg": "°", "px": " px"}.get(result.measure.unit, "")


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
        return f"{m.name} {m.points}"
    kind = "|value| minus the others' " if result.settings.get("relative_abs") else ""
    return f"{m.name} {m.points}, relative ({kind}{rel} of the rest of each image)"


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
    readings, and tails close to 2.5% each (the same test as the warning)."""
    return result.n >= 30 and max(result.tail_shares) <= 0.04


def summary(result: "AuditResult", full: bool = False) -> str:
    u = _unit(result)
    d = _digits(result)
    r = result.not_read
    first = _first(result)
    lines = [
        f"{_what(result)}: read {result.n} of {result.measurable} labelled instances",
        *(f"  ! {w}" for w in first),
        f"  not read     no matching prediction {r.missed}, prediction lacked a "
        f"point {r.no_predicted_point}, unmeasurable {r.unmeasurable}"
        + (f", too few in frame {r.too_few_in_frame}" if r.too_few_in_frame else ""),
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
    if result.repeated_limits is not None:  # only with named clusters
        rlo, rhi = result.repeated_limits
        lines.append(f"  repeated     {_range(rlo, rhi, f2, u)} (clusters)")
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


def _full(result: "AuditResult") -> list[str]:
    """The slope against the jitter reference and the other agreement
    statistics: `summary(full=True)` and `--full`."""
    lines = []
    if result.jitter_gain is not None:
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
    lines.append(
        f"  BA slope     {_f(result.ba_slope, '+.3f')} "
        f"{_ci(result.ba_slope_ci, '+.3f')}"
    )
    if result.deming is not None:
        lines.append(
            f"  Deming       {_f(result.deming, '.3f')} {_ci(result.deming_ci, '.3f')} "
            f"(noise ratio {result.settings.get('noise_ratio')})"
        )
    lines.append(
        f"  agreement    CCC {_f(result.ccc, '.3f')} {_ci(result.ccc_ci, '.3f')}, "
        f"r {_f(result.pearson, '.3f')}"
    )
    return lines


GUIDE = """\
How to read this:

- **slope** (pred on truth): the least-squares slope of predicted on true
  values, the proportional bias; 1 is ideal (`gain` in the JSON). It is not
  the model's tendency alone:
  - keypoint jitter bends it, since near the ends of a range an error can only
    go one way.
  - **vs jitter** is the slope on predictions rebuilt from the truth plus this
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
  bands, where jitter and regression to the mean read as bias; see the
  README's "Which way you sort decides the story".
- **BA slope**: the slope of the error on the mean of both readings. It compares
  the two spreads, so it reads as compression only when truth and prediction are
  about equally noisy; a noisier prediction pushes it up.
- **limits**: where 95% of errors fall. The percentile limits assume nothing;
  the normal ones assume a bell-shaped error, which a few gross failures break.
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
    word = {"deg": "degree" if text == "1" else "degrees", "px": "px"}.get(unit)
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


def _lead(result: "AuditResult") -> str:
    """The headline in words, and which way an error points."""
    m = result.measure
    d = 1 if m.unit else _digits(result)
    return (
        f"On average {_subject(result)} is "
        f"{_amount(result.mean_abs_error, m.unit, f'.{d}f')} off; "
        f"{_share(result.big_error_rate)} of readings are off by "
        f"{_amount(result.big_error, m.unit)} or more. {_sign(result)}"
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
    u = _unit(result)
    d = _digits(result)
    out = [f"# poseaudit {__version__}: {_what(result)}", ""]
    out += [f"**{w}**\n" for w in _first(result)]
    if result.n:
        out += [_lead(result), ""]
    out += [
        f"big error >= {result.big_error:g}{u}, bands cut on the {result.band_by}; "
        "the settings are listed at the end.",
        "",
        "```",
        summary(result, full=True),
        "```",
        "",
    ]
    if result.n == 0:
        return "\n".join(out + _settings(result)) + "\n"
    cluster = "image" if result.settings.get("cluster") == "image" else "cluster"
    out.append(
        GUIDE.format(
            u=u,
            cluster=cluster,
            resamples=result.settings.get("resamples"),
            seed=result.settings.get("seed"),
        )
    )
    lo, hi = result.limits
    elo, ehi = _percentile_limits(result)
    below, above = result.tail_shares
    out += [
        "## Limits of agreement",
        "",
        "| limits | lower | 95% CI | upper | 95% CI |",
        "|---|---|---|---|---|",
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
    out += [
        "",
        f"{below:.1%} of errors fall below the normal limits and {above:.1%} above "
        "them (2.5% each for a normal error).",
        "",
        "## Error by size of the measured part",
        "",
        "| size | n | mean abs error | large errors | slope |",
        "|---|---|---|---|---|",
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
        "|---|---|---|---|---|",
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
            "|---|---|---|---|---|---|---|---|",
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
    out += [
        "",
        "## Largest errors",
        "",
        "| image | instance | size | truth | predicted | error |",
        "|---|---|---|---|---|---|",
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


def csv_rows(result: "AuditResult") -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for r in result.readings:
        writer.writerow(
            ["" if getattr(r, c) is None else getattr(r, c) for c in CSV_COLUMNS]
        )
    return buffer.getvalue()
