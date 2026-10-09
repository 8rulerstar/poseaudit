"""Two panels a reviewer asks for first. Needs `pip install "poseaudit[plot]"`."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from poseaudit.audit import AuditResult


def needs_matplotlib() -> None:
    try:
        import matplotlib.figure  # noqa: F401
    except ImportError as error:
        raise ImportError(
            'plotting needs matplotlib: pip install "poseaudit[plot]"'
        ) from error


def plot(result: "AuditResult", path: str) -> None:
    needs_matplotlib()
    from matplotlib.figure import Figure  # no pyplot: leaves the user's backend alone

    if result.n == 0:
        raise ValueError("no readings to plot")

    t = np.array([r.truth for r in result.readings])
    e = np.array([r.error for r in result.readings])
    p = t + e
    unit = {"deg": " (°)", "px": " (px)"}.get(result.measure.unit, "")
    name = result.measure.name
    fig = Figure(figsize=(10, 4.4))
    left, right = fig.subplots(1, 2)

    left.scatter(t, p, s=8, alpha=0.5)
    span = np.array([min(t.min(), p.min()), max(t.max(), p.max())])
    left.plot(span, span, color="grey", lw=1, label="predicted = truth")
    left.plot(
        span,
        result.gain * span + result.offset,
        lw=1.5,
        label=f"slope {result.gain:.2f} {tuple(round(v, 2) for v in result.gain_ci)}",
    )
    left.set(xlabel=f"true {name}{unit}", ylabel=f"predicted {name}{unit}")
    left.legend(loc="upper left", fontsize=8)

    m = t + e / 2
    right.scatter(m, e, s=8, alpha=0.5)
    right.axhline(result.bias, lw=1.5, label=f"bias {result.bias:+.2f}")
    for limit, style in (
        (result.empirical_limits, "--"),
        (result.limits, ":"),
    ):
        for value in limit:
            right.axhline(value, ls=style, color="grey", lw=1)
    right.plot([], [], ls="--", color="grey", label="2.5-97.5% limits")
    right.plot([], [], ls=":", color="grey", label="normal limits")
    right.set(xlabel=f"mean of truth and prediction{unit}", ylabel=f"error{unit}")
    right.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"{name} {result.measure.points}: {result.n} readings")
    fig.tight_layout()
    from poseaudit._files import write_with

    # the temporary file's name ends .tmp, so the format comes from the path
    form = Path(path).suffix.lstrip(".").lower() or "png"
    write_with(path, lambda temporary: fig.savefig(temporary, dpi=150, format=form))
