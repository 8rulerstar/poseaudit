"""docs/cycles.gif, the README's GIF of `paired --cycles`, from angles.csv.

python cycles_gif.py ../../docs

Each trial of the knee holds one stride (from 0 to 1.1 s), given here as
event times. The cycles are stretched to 0 to 100% as poseaudit stretches
them; the mean error and its SD at each point are poseaudit's own
(`cycle_error.bias_curve`, `sd_curve`). The data are synthetic
(make_data.py). Needs Matplotlib and Pillow: pip install "poseaudit[plot,images]".
"""

import csv
import io
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

import poseaudit as pa  # noqa: E402
from poseaudit import plot  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "coco_elbow"))
from gifs import WIDTH, _rgb, _save  # noqa: E402

MEASURE = "knee_angle_r"
START, END = 0.0, 1.1  # seconds: one stride in each trial


def main(out: str) -> None:
    with open(HERE / "angles.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["measure"] == MEASURE]

    def column(name):
        return [r[name] for r in rows]

    subject, trial = column("subject"), column("trial")
    time = np.array(column("time"), float)
    pred = np.array(column("pred"), float)
    ref = np.array(column("ref"), float)
    result = pa.audit_values(
        pred, ref, big_error=5, name=MEASURE, unit="deg",
        subject=subject, trial=trial, frame=time,
        cycles={(None, None): [START, END]},
    )  # fmt: skip
    error = result.cycle_error
    x = np.linspace(0, 100, 101)

    curves = []  # each cycle stretched to 0 to 100%
    for key in dict.fromkeys(zip(subject, trial, strict=True)):
        mine = np.array([(s, t) == key for s, t in zip(subject, trial, strict=True)])
        inside = mine & (time >= START - 1e-9) & (time <= END + 1e-9)
        t = time[inside]
        phase = 100 * (t - START) / (END - START)
        curves.append(
            (np.interp(x, phase, pred[inside]), np.interp(x, phase, ref[inside]))
        )

    bias = np.array(error.bias_curve)
    sd = np.array(error.sd_curve)
    lo, hi = error.rmse_ci
    n = len(curves)

    def frame(shown: int, stage: int) -> Image.Image:
        fig = Figure(figsize=(WIDTH / 100, 4.6), dpi=100)
        with matplotlib.rc_context({**plot.STYLE, "font.size": 12}):
            if stage == 0:
                head = f"Right knee, {shown} of {n} cycles stretched to 0 to 100%"
                line = "prediction (blue) against motion capture (grey), 6 subjects"
            else:
                head = f"Mean error over the cycle, with its SD across {n} cycles"
                line = (
                    f"RMSE {error.rmse:.2f}° [95%: {lo:.2f} to {hi:.2f}]; the peak "
                    f"is read low: {bias.min():+.1f}° at "
                    f"{x[bias.argmin()]:.0f}% of the cycle"
                )
            fig.text(0.03, 0.935, head, fontsize=14, fontweight="bold")
            fig.text(0.03, 0.865, line, fontsize=12, color=plot.MUTED)
            top = fig.add_axes((0.09, 0.47, 0.88, 0.33))
            for p, r in curves[:shown]:
                top.plot(x, r, color=plot.MUTED, lw=1, alpha=0.6)
                top.plot(x, p, color=plot.POINTS, lw=1, alpha=0.6)
            top.set_xlim(0, 100)
            top.set_ylim(-5, 80)
            top.set_ylabel("angle (°)")
            top.tick_params(labelbottom=False)
            bottom = fig.add_axes((0.09, 0.13, 0.88, 0.28))
            bottom.axhline(0, color=plot.INK, lw=0.8)
            if stage >= 1:
                for p, r in curves:
                    bottom.plot(x, p - r, color=plot.POINTS, lw=0.8, alpha=0.25)
                bottom.fill_between(
                    x, bias - sd, bias + sd, color=plot.FIT, alpha=0.2, lw=0
                )
                bottom.plot(x, bias, color=plot.FIT, lw=2.5, label="mean error ± SD")
                bottom.legend(loc="upper center", frameon=False)
            bottom.set_xlim(0, 100)
            bottom.set_ylim(-12, 12)
            bottom.set_ylabel("error (°)")
            bottom.set_xlabel("cycle (%)")
            for ax in (top, bottom):
                ax.grid(color=plot.GRID, lw=0.8)
                ax.set_axisbelow(True)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        return Image.open(buffer).convert("RGB")

    frames = [(frame(k, 0), 450) for k in range(1, n + 1)]
    frames[-1] = (frames[-1][0], 1200)
    frames.append((frame(n, 1), 4000))
    colors = ["#ffffff", plot.INK, plot.MUTED, plot.POINTS, plot.FIT, plot.GRID]
    path = Path(out) / "cycles.gif"
    _save(frames, path, [_rgb(c) for c in colors])
    print(f"cycles.gif: {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main(*sys.argv[1:])
