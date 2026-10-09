"""The README figures, from the committed demo data.

python figures.py ../../docs

panels.png is what `--plot` draws. trap.png and the social preview (a
1280 x 640 card for the repository's settings, as SVG and PNG) use the same
type sizes and colours.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

import poseaudit as pa  # noqa: E402
from poseaudit import plot  # noqa: E402

HERE = Path(__file__).parent

# Okabe and Ito's blue, vermillion and bluish green, each with its own marker
# and dash pattern so that the lines part in grayscale too
SORTS = {
    "truth": ("sorted by the true angle", plot.POINTS, "o", "-"),
    "predicted": ("sorted by the predicted angle", plot.FIT, "s", (0, (5, 3))),
    "mean": ("sorted by the mean of both", "#009E73", "^", (0, (1, 1.5))),
}


def main(out: str) -> None:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    truth = pa.load_coco(HERE / "gt_200.json")
    predicted = pa.load_coco_results(HERE / "pred_yolo11n.json", HERE / "gt_200.json")
    pairing = pa.pair(truth, predicted)
    elbow = pa.angle(5, 7, 9, name="elbow angle")

    results = {by: pa.audit(pairing, elbow, big_error=15, band_by=by) for by in SORTS}
    results["truth"].plot(str(folder / "panels.png"))
    with matplotlib.rc_context(plot.STYLE):
        trap(results).savefig(folder / "trap.png", dpi=plot.DPI)
        card = preview(results["truth"])
        card.savefig(folder / "social-preview.png", dpi=100)
        with matplotlib.rc_context({"svg.hashsalt": "poseaudit"}):  # same ids
            card.savefig(folder / "social-preview.svg", metadata={"Date": None})
    for by, result in results.items():
        print(by, [f"{b.bias:+.1f}" for b in result.bands])


def trap(results: dict) -> Figure:
    fig = Figure(figsize=(plot.WIDTH, 4.6), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.22])
    ax, key = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    for by, result in results.items():
        label, color, marker, style = SORTS[by]
        middles = [(b.low + b.high) / 2 for b in result.bands]
        ax.plot(
            middles, [b.bias for b in result.bands], color=color, marker=marker,
            ls=style, lw=2, ms=7, label=label,
        )  # fmt: skip
    ax.axhline(0, color=plot.MUTED, lw=0.8, zorder=0)
    ax.set(xlabel="elbow angle of the band (°)", ylabel="mean error in the band (°)")
    ax.set_title(f"The same {results['truth'].n} readings, sorted three ways")
    ax.grid(True, color=plot.GRID, lw=0.8)
    ax.set_axisbelow(True)
    key.axis("off")
    key.legend(
        *ax.get_legend_handles_labels(), loc="upper left", borderaxespad=0,
        handlelength=3, labelspacing=0.3,
    )  # fmt: skip
    return fig


def preview(result) -> Figure:
    """A plain card: the name, what it does, and the demo's Bland-Altman."""
    import numpy as np

    from poseaudit.report import _percentile_limits

    fig = Figure(figsize=(12.8, 6.4))
    fig.text(0.06, 0.74, "poseaudit", fontsize=64, fontweight="bold")
    lines = [
        "How far off are the joint angles,",
        "tilts and lengths you read",
        "from a pose model?",
    ]
    for k, line in enumerate(lines):
        fig.text(0.06, 0.58 - 0.085 * k, line, fontsize=28, color=plot.INK)
    fig.text(0.06, 0.12, "pip install poseaudit", fontsize=22, color=plot.MUTED)
    ax = fig.add_axes((0.64, 0.14, 0.32, 0.72))
    t = np.array([r.truth for r in result.readings])
    e = np.array([r.error for r in result.readings])
    ax.scatter(t + e / 2, e, s=22, color=plot.POINTS, alpha=0.5, linewidths=0)
    ax.axhline(result.bias, color=plot.FIT, lw=3.5)
    for value in _percentile_limits(result):
        ax.axhline(value, color=plot.INK, lw=1.6, ls=plot.DASHED)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    main(*sys.argv[1:])
