"""Two panels a reviewer asks for first. Needs `pip install "poseaudit[plot]"`.

The figure is drawn for print. It is 7 inches wide, a two-column paper's full
text width, with 12 point text, so shrunk to one 3.5 inch column its smallest
text is still 6 points. The colours are from Okabe and Ito's palette, which
readers with red-green colour blindness tell apart, and the lines in each
legend also differ in dash pattern, so a grayscale copy still matches each
line to its entry.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from poseaudit.audit import AuditResult

WIDTH = 7.0  # inches: a two-column paper's full text width
HEIGHT = 4.2
TEXT = 12.0  # points: 6 or more when printed one 3.5 inch column wide
DPI = 300

POINTS = "#0072B2"  # Okabe and Ito's blue
FIT = "#D55E00"  # their vermillion
INK = "#222222"
MUTED = "#707070"
GRID = "#E6E6E6"

FEW = 10  # under this many readings the points are drawn over the lines

DASHED = (0, (5, 3))
DOTTED = (0, (1, 2))

STYLE = {
    "font.size": TEXT,
    "axes.titlesize": TEXT,
    "axes.labelsize": TEXT,
    "xtick.labelsize": TEXT,
    "ytick.labelsize": TEXT,
    "legend.fontsize": TEXT,
    "figure.titlesize": TEXT,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,  # TrueType text in a PDF, which journals ask for
    "ps.fonttype": 42,
}


def needs_matplotlib() -> None:
    try:
        import matplotlib.figure  # noqa: F401
    except ImportError as error:
        raise ImportError(
            'plotting needs matplotlib: pip install "poseaudit[plot]"'
        ) from error


def plot(result: "AuditResult", path: str) -> None:
    needs_matplotlib()
    import matplotlib

    if result.n == 0:
        raise ValueError("no readings to plot")
    from poseaudit._files import write_with

    # the temporary file's name ends .tmp, so the format comes from the path
    form = Path(path).suffix.lstrip(".").lower() or "png"
    # the style holds for this figure only; the user's settings are left alone
    with matplotlib.rc_context(cast(Any, STYLE)):
        fig = _figure(result)
        write_with(path, lambda temporary: fig.savefig(temporary, dpi=DPI, format=form))


def _figure(result: "AuditResult") -> "Figure":
    """The two panels, unsaved. `plot` draws them inside
    `matplotlib.rc_context(STYLE)`, which sets the sizes and colours."""
    from matplotlib.figure import Figure  # no pyplot: leaves the user's backend alone

    from poseaudit.report import _ci, _digits, _percentile_limits

    t = np.array([r.truth for r in result.readings])
    e = np.array([r.error for r in result.readings])
    p = t + e
    unit = {"deg": "°", "px": " px"}.get(result.measure.unit, "")
    in_unit = {"deg": " (°)", "px": " (px)"}.get(result.measure.unit, "")
    d = 1 if result.measure.unit else _digits(result)
    name = result.measure.name

    fig = Figure(figsize=(WIDTH, HEIGHT), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.2])
    left, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    readings = "reading" if result.n == 1 else "readings"
    fig.suptitle(
        f"{name} {result.measure.points}: {result.n:,} {readings}", fontweight="bold"
    )
    low, high = _span(np.concatenate([t, p]))
    dots = _dots(result.n)

    left.set_title("predicted against truth")
    left.plot(
        [low, high], [low, high], color=INK, lw=1.2, ls=DASHED, zorder=2,
        label="predicted = truth",
    )  # fmt: skip
    left.scatter(t, p, **dots)
    if np.isfinite(result.gain):
        x = np.array([t.min(), t.max()])
        left.plot(
            x, result.gain * x + result.offset, color=FIT, lw=2.4, zorder=3,
            path_effects=_edge(),
            label=f"fit: slope {_number(result.gain, '.2f')} "
            f"{_number(_ci(result.gain_ci, '.2f'))}",
        )  # fmt: skip
    left.set(
        xlim=(low, high),
        ylim=(low, high),
        xlabel=f"true {name}{in_unit}",
        ylabel=f"predicted {name}{in_unit}",
    )

    right.set_title("Bland-Altman")
    right.scatter(t + e / 2, e, **dots)
    right.axhline(
        result.bias, color=FIT, lw=2.4, zorder=3, path_effects=_edge(), label="bias"
    )
    shown = [result.bias]
    limits = (
        (_percentile_limits(result), INK, DASHED, 1.2, "2.5th and 97.5th percentiles"),
        (result.limits, MUTED, DOTTED, 1.6, "bias ± 1.96 SD"),
    )
    for (lo, hi), color, style, width, what in limits:
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue  # too few readings: the summary has none either
        right.axhline(lo, color=color, ls=style, lw=width, zorder=2, label=what)
        right.axhline(hi, color=color, ls=style, lw=width, zorder=2)
        shown += [lo, hi]
    right.set(
        xlim=(low, high),
        xlabel=f"mean of truth and prediction{in_unit}",
        ylabel=f"predicted minus truth{in_unit}",
    )
    for ax in (left, right):
        _finish(ax, result.measure.unit)
    _key(fig.add_subplot(grid[1, :]), left, right)
    _values_at_right(fig, right, shown, f"+.{d}f", unit)
    return fig


def _key(key: "Axes", left: "Axes", right: "Axes") -> None:
    """One legend under both panels: the left panel's lines in the first
    column and the right panel's in the second."""
    from matplotlib.lines import Line2D

    handles, labels = left.get_legend_handles_labels()
    more, words = right.get_legend_handles_labels()
    blank = Line2D([], [], linestyle="none")
    while len(handles) < len(more):
        handles, labels = [*handles, blank], [*labels, ""]
    key.axis("off")
    key.legend(
        handles + more,
        labels + words,
        ncol=2,
        loc="upper center",
        borderaxespad=0,
        handlelength=2.2,
        labelspacing=0.3,
        columnspacing=2.0,
    )


def _edge() -> list:
    """A thin white edge that parts a line from the points under it."""
    from matplotlib.patheffects import withStroke

    return [withStroke(linewidth=4.4, foreground="white")]


def _number(value, fmt: str = "") -> str:
    """A value formatted as the summary formats it, with the minus sign the
    axes use."""
    from poseaudit.report import _f

    text = value if isinstance(value, str) else _f(value, fmt)
    return text.replace("-", "−")


def _values_at_right(
    fig: "Figure", ax: "Axes", values: list[float], fmt: str, unit: str
) -> None:
    """Each line's value just right of the axes, in the order of the lines;
    labels that would overlap are pushed apart as little as they need. Lines
    that print the same value, such as every line at 0 when all errors are
    0, share one label rather than stacking copies of it."""
    texts: dict[str, list[float]] = {}
    for value in sorted(values):
        texts.setdefault(f"{_number(value, fmt)}{unit}", []).append(value)
    order = [float(np.mean(group)) for group in texts.values()]
    labels = [
        ax.text(
            1.02,
            value,
            text,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            clip_on=False,
        )  # fmt: skip
        for text, value in zip(texts, order, strict=True)
    ]
    fig.draw_without_rendering()  # lays the figure out, labels included
    low, high = ax.get_ylim()
    ax.set_ylim(low, high)  # moving the labels must not rescale the axis
    height = ax.get_window_extent().height * 72 / fig.dpi  # points
    gap = 1.2 * TEXT * (high - low) / height
    for label, y in zip(labels, _spread(order, gap), strict=True):
        label.set_y(y)


def _spread(values: list[float], gap: float) -> list[float]:
    """Positions for labels of sorted `values`, at least `gap` apart: labels
    that would collide form a group, spaced by `gap` and centred on the mean
    of their values, and groups that then collide merge."""
    groups: list[list[float]] = []
    for value in values:
        groups.append([value])
        while len(groups) > 1:
            below, above = groups[-2], groups[-1]
            top = np.mean(below) + (len(below) - 1) * gap / 2
            bottom = np.mean(above) - (len(above) - 1) * gap / 2
            if bottom - top >= gap:
                break
            groups[-2:] = [below + above]
    return [
        float(np.mean(group) + (k - (len(group) - 1) / 2) * gap)
        for group in groups
        for k in range(len(group))
    ]


def _span(values: np.ndarray) -> tuple[float, float]:
    """Shared limits for truth and prediction, with a small margin."""
    low, high = float(values.min()), float(values.max())
    pad = 0.04 * (high - low) or 1.0
    return low - pad, high + pad


def _dots(n: int) -> dict:
    """Points that stay readable from a few hundred readings to many
    thousands: fainter and smaller as they crowd, and drawn as an image in a
    PDF or SVG once there are so many that each would be a vector shape."""
    return {
        "s": 14 if n <= 2000 else 5,
        "color": POINTS,
        "alpha": float(np.clip(10 / np.sqrt(n), 0.08, 0.5)),
        "linewidths": 0,
        # a handful of points over the lines, or a lone reading on the bias
        # line disappears under it
        "zorder": 4 if n < FEW else 1,
        "rasterized": n > 5000,
    }


def _finish(ax: "Axes", unit: str) -> None:
    """Ticks and a light grid. Tick labels carry their whole value: an offset
    or a power of ten in the corner (lengths near 512 px read as "+5.12e2",
    millions as "1e6") lands on the axis label and is easily missed. Values
    of 100,000 or more take an SI prefix (1.6M) so the labels stay short."""
    from matplotlib.ticker import EngFormatter, MaxNLocator, ScalarFormatter

    for axis, (low, high) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        # degrees across a joint's range: steps of 45 rather than 50; over a
        # few degrees, whole steps rather than 4.5 and 9.0
        wide = unit == "deg" and abs(high - low) >= 90
        steps = [1, 1.5, 2, 3, 4.5, 5, 10] if wide else None
        big = max(abs(low), abs(high)) >= 1e5
        # "1.6M" is wider than "45": fewer ticks keep the labels apart
        axis.set_major_locator(MaxNLocator(nbins=3 if big else 5, steps=steps))
        if big:
            axis.set_major_formatter(EngFormatter(sep=""))
        else:
            plain = ScalarFormatter(useOffset=False)
            plain.set_scientific(False)
            axis.set_major_formatter(plain)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
