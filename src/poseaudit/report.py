import csv
import io
from typing import TYPE_CHECKING

import numpy as np

from poseaudit._version import __version__

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


def _ci(interval, fmt: str) -> str:
    low, high = interval
    return f"[{low:{fmt}} to {high:{fmt}}]"


def _p(p: float, repeats) -> str:
    """No rebuild at or below the gain gives the smallest p the rebuilds can
    show, 1/(repeats+1); the true p may be lower still."""
    if repeats and p <= 1 / (repeats + 1) + 1e-12:
        return f"p(squash) <= {1 / (repeats + 1) + 5e-4:.3f}*"
    return f"p(squash) {p:.3f}*"


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


def summary(result: "AuditResult", full: bool = False) -> str:
    u = _unit(result)
    d = _digits(result)
    r = result.not_read
    lines = [
        f"{_what(result)}: read {result.n} of {result.measurable} labelled instances",
        f"  not read     no matching prediction {r.missed}, prediction lacked a "
        f"point {r.no_predicted_point}, unmeasurable {r.unmeasurable}"
        + (f", too few in frame {r.too_few_in_frame}" if r.too_few_in_frame else ""),
        f"  not counted  {r.unlabelled} with a point unlabelled in the truth, "
        f"{r.unmatched_predictions} unmatched predictions",
    ]
    if result.n == 0:
        return "\n".join(lines + [f"  ! {w}" for w in result.warnings])
    elo, ehi = result.empirical_limits
    lines += [
        f"  |error|      mean {result.mean_abs_error:.{d}f}{u} "
        f"{_ci(result.mean_abs_error_ci, f'.{d}f')}, median "
        f"{result.median_abs_error:.{d}f}{u}, 95th pct {result.p95_abs_error:.{d}f}{u}",
        f"  >= {result.big_error:g}{u}".ljust(15)
        + f"{result.big_error_rate:.1%} {_ci(result.big_error_rate_ci, '.1%')}",
    ]
    if len(result.size_bands) > 1:
        lines.append(
            "  by size      "
            + ", ".join(
                f"{_px(b.low, b.high)} px {b.big_error_rate:.0%} (n {b.n})"
                if np.isfinite(b.high)
                else f"{b.low:.0f}+ px {b.big_error_rate:.0%} (n {b.n})"
                for b in result.size_bands
            )
        )
    lines += [
        f"  bias         {result.bias:+.{d}f}{u} {_ci(result.bias_ci, f'+.{d}f')}, "
        f"median {result.median_error:+.{d}f}{u}",
        f"  limits       {elo:+.{d}f}{u} to {ehi:+.{d}f}{u} "
        "(2.5th to 97.5th percentile of errors)",
        f"  gain         {result.gain:.3f} {_ci(result.gain_ci, '.3f')}; robust "
        f"{result.robust_gain:.3f}",
    ]
    if result.jitter_gain is not None:
        lines.append(
            f"  vs jitter    {result.jitter_gain:.3f} from keypoint jitter alone; gap "
            f"{result.gain_gap:+.3f} {_ci(result.gain_gap_ci, '+.3f')}, "
            + (
                _p(result.jitter_p, result.settings.get("jitter_repeats"))
                if result.jitter_p is not None
                else "no p with named clusters"
            )
        )
        if result.jitter_p is not None:
            lines.append(
                "               * swaps and gross failures lower the gain too, "
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
    if full:
        lo, hi = result.limits
        lines.append(
            f"  normal       {lo:+.{d}f}{u} to {hi:+.{d}f}{u}, "
            f"RMSE {result.rmse:.{d}f}{u}"
        )
        if result.repeated_limits is not None:
            rlo, rhi = result.repeated_limits
            lines.append(
                f"  repeated     {rlo:+.{d}f}{u} to {rhi:+.{d}f}{u} (clusters)"
            )
        lines.append(
            f"  BA slope     {result.ba_slope:+.3f} {_ci(result.ba_slope_ci, '+.3f')}"
        )
        if result.deming is not None:
            lines.append(
                f"  Deming       {result.deming:.3f} {_ci(result.deming_ci, '.3f')} "
                f"(noise ratio {result.settings.get('noise_ratio')})"
            )
        lines.append(
            f"  agreement    ICC(A,1) {result.icc:.3f} {_ci(result.icc_ci, '.3f')}, "
            f"CCC {result.ccc:.3f} {_ci(result.ccc_ci, '.3f')}, r {result.pearson:.3f}"
        )
    lines += [_decision(th, u, d) for th in result.thresholds]
    lines += [f"  ! {w}" for w in result.warnings]
    return "\n".join(lines)


GUIDE = """\
How to read this:

- **gain**: the least-squares slope of predicted on true values; 1 is ideal.
  It is not the model's tendency alone:
  - keypoint jitter bends it, since near the ends of a range an error can only
    go one way.
  - **vs jitter** is the gain on predictions rebuilt from the truth plus this
    model's own displacements: each object takes all its points' shifts from
    one object with a similar true value (possibly itself), minus the shift
    that such a group shares, carried in each segment's own frame and
    mirrored for angles that bend the other way.
  - **gap** is the gain minus that reference; its interval averages a few
    rebuilds in every resample. **p(squash)** is (1 + rebuilds at or below
    this gain) / (1 + rebuilds), one-sided. It assumes independent objects,
    so with named clusters it is left out.
  - A gap below 0 with a small p means the model reads differences as smaller
    than its jitter explains. Possible causes: squashing, gross failures tied
    to the true value (a straight part read as bent), left and right swapped
    on one side, or noise in the truth. Gross failures in random directions
    are part of the rebuilds and do not widen the gap.
    With labels nearly as noisy as the model the p is often small for an
    honest model; read the size of the gap then, not the p.
  - noise in the truth pulls the gain toward 0; with a known ratio of noise
    variances, read the Deming slope, which removes that but not the jitter
    at the ends.
  - a few gross failures (a point on the wrong part) can move it a lot; the
    **robust gain** (Theil-Sen) barely moves with them. Noise that grows with
    the value or with smaller objects also separates the two, so their
    difference does not by itself measure gross failures.
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
  resamples, seed {seed}). Rates of large errors and of decisions use Wilson
  intervals, which treat readings as independent, unless clusters are named;
  then the overall rate and the decision rates resample whole clusters too
  (never narrower than Wilson's). Rates per size band stay Wilson.
"""


def _px(low: float, high: float, joint: str = "-") -> str:
    """Band edges in whole pixels, or with a decimal when those would coincide."""
    digits = 0 if round(low) != round(high) else 1
    return f"{low:.{digits}f}{joint}{high:.{digits}f}"


def _limit_row(name, low, high, low_ci, high_ci, u, d=2) -> str:
    return (
        f"| {name} | {low:+.{d}f}{u} | {_ci(low_ci, f'+.{d}f')} "
        f"| {high:+.{d}f}{u} | {_ci(high_ci, f'+.{d}f')} |"
    )


def markdown(result: "AuditResult", worst: int = 10) -> str:
    u = _unit(result)
    d = _digits(result)
    out = [
        f"# poseaudit {__version__}: {_what(result)}",
        "",
        f"big error >= {result.big_error:g}{u}, bands cut on the {result.band_by}, "
        f"settings {result.settings}",
        "",
        "```",
        summary(result, full=True),
        "```",
        "",
    ]
    if result.n == 0:
        return "\n".join(out)
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
    elo, ehi = result.empirical_limits
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
            result.empirical_lower_ci,
            result.empirical_upper_ci,
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
        "| size | n | mean abs error | large errors | gain |",
        "|---|---|---|---|---|",
    ]
    for sb in result.size_bands:
        span = (
            f"{_px(sb.low, sb.high, ' to ')} px"
            if np.isfinite(sb.high)
            else f"{sb.low:.0f}+ px"
        )
        out.append(
            f"| {span} | {sb.n} | {sb.mean_abs_error:.{d}f}{u} | "
            f"{sb.big_error_rate:.1%} {_ci(sb.big_error_rate_ci, '.1%')} | "
            f"{sb.gain:.3f} |"
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
    for r in result.worst(worst):
        out.append(
            f"| {r.image} | {r.truth_index} | {r.size:.0f} px | {r.truth:.{d}f}{u} "
            f"| {r.predicted:.{d}f}{u} | {r.error:+.{d}f}{u} |"
        )
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
