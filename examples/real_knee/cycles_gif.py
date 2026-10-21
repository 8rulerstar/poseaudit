"""docs/cycles.gif, the README's GIF of `paired --cycles`, from the output of
build.py and the JSON of the audit run on it:

poseaudit paired --mot-pairs out/pairs.csv --cycles out/events.csv \
  --big-error 5 --offset-shifts 0.1 --json out/report.json
python cycles_gif.py out ../../docs

The top panel draws the first CYCLES strides of each subject's first trial,
stretched to 0 to 100% as poseaudit stretches them; the mean error and its
SD at each point are poseaudit's own (`cycle_error.bias_curve`, `sd_curve`,
over every cycle). Needs Matplotlib and Pillow:
pip install "poseaudit[plot,images]"."""

import io
import json
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

CYCLES = 4  # per subject, drawn in the top panel
SOURCE = "Data: Zenodo 10.5281/zenodo.19720803, CC BY 4.0 (see examples/real_knee)"


def main(folder: str, out: str) -> None:
    base = Path(folder)
    report = json.loads((base / "report.json").read_text())
    pairs = {(row["subject"], row["trial"]): row for row in _rows(base / "pairs.csv")}
    x = np.linspace(0, 100, 101)
    curves: dict[str, list] = {}
    for c in report["cycles"]:
        mine = curves.setdefault(c["subject"], [])
        first = next(t for s, t in pairs if s == c["subject"])
        if c["trial"] != first or len(mine) == CYCLES:
            continue
        row = pairs[(c["subject"], c["trial"])]
        pred, ref = (pa.load_mot(base / row[k]) for k in ("pred", "ref"))
        mine.append(tuple(_stretch(m, c["start"], c["end"], x) for m in (pred, ref)))

    peaks: dict[str, list] = {}
    for c in report["cycles"]:
        peaks.setdefault(c["subject"], []).append(c["pred_peak"] - c["ref_peak"])
    low = sum(np.mean(v) < 0 for v in peaks.values())
    error = report["cycle_error"]
    bias, sd = np.array(error["bias_curve"]), np.array(error["sd_curve"])
    peak = next(c for c in report["cycle"] if c["feature"] == "peak")
    shifts = {row["shift"]: row for row in report["offset_shifts"]}
    moved = max(abs(shifts[s]["rmse"] - shifts[0.0]["rmse"]) for s in (-0.1, 0.1))
    moved_peak = max(
        abs(shifts[s]["cycle_peak_bias"] - shifts[0.0]["cycle_peak_bias"])
        for s in (-0.1, 0.1)
    )
    subjects = list(curves)

    def frame(shown: int, stage: int) -> Image.Image:
        fig = Figure(figsize=(WIDTH / 100, 4.8), dpi=100)
        with matplotlib.rc_context({**plot.STYLE, "font.size": 12}):
            if stage == 0:
                head = f"Knee flexion, {shown} of {len(subjects)} subjects"
                line = "side-view YOLOv8, 2D (blue), against 3D markers (grey)"
            else:
                lo, hi = peak["bias_ci"]
                head = (
                    f"YOLOv8 2D vs markers 3D: peak per stride {peak['bias']:+.1f}° "
                    f"[{lo:+.1f} to {hi:+.1f}]"
                )
                line = (
                    f"low in {low}/{len(peaks)} subjects; mixes model, 2D vs 3D, "
                    f"camera, hip; offset ±0.1 s: RMSE {moved:.1f}°, "
                    f"peak {moved_peak:.2f}°"
                )
            fig.text(0.03, 0.94, head, fontsize=14, fontweight="bold")
            fig.text(0.03, 0.875, line, fontsize=11, color=plot.MUTED)
            fig.text(0.03, 0.015, SOURCE, fontsize=9, color=plot.MUTED)
            top = fig.add_axes((0.09, 0.50, 0.88, 0.32))
            for s in subjects[:shown]:
                for p, r in curves[s]:
                    top.plot(x, r, color=plot.MUTED, lw=0.8, alpha=0.5)
                    top.plot(x, p, color=plot.POINTS, lw=0.8, alpha=0.5)
            top.plot([], [], color=plot.POINTS, lw=2, label="YOLOv8, 2D")
            top.plot([], [], color=plot.MUTED, lw=2, label="markers, 3D")
            top.legend(loc="upper left", frameon=False, ncol=2)
            top.set_xlim(0, 100)
            top.set_ylim(-10, 100)
            top.set_ylabel("angle (°)")
            top.tick_params(labelbottom=False)
            bottom = fig.add_axes((0.09, 0.15, 0.88, 0.29))
            bottom.axhline(0, color=plot.INK, lw=0.8)
            if stage >= 1:
                bottom.fill_between(
                    x, bias - sd, bias + sd, color=plot.FIT, alpha=0.2, lw=0
                )
                bottom.plot(x, bias, color=plot.FIT, lw=2.5, label="mean error ± SD")
                bottom.legend(loc="lower left", frameon=False)
            bottom.set_xlim(0, 100)
            bottom.set_ylim(-45, 15)
            bottom.set_ylabel("error (°)")
            bottom.set_xlabel("stride (%), from the lowest point of the heel marker")
            for ax in (top, bottom):
                ax.grid(color=plot.GRID, lw=0.8)
                ax.set_axisbelow(True)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        return Image.open(buffer).convert("RGB")

    frames = [(frame(k, 0), 350) for k in range(1, len(subjects) + 1)]
    frames[-1] = (frames[-1][0], 1200)
    frames.append((frame(len(subjects), 1), 5000))
    colors = ["#ffffff", plot.INK, plot.MUTED, plot.POINTS, plot.FIT, plot.GRID]
    path = Path(out) / "cycles.gif"
    _save(frames, path, [_rgb(c) for c in colors])
    print(f"cycles.gif: {path.stat().st_size / 1024:.0f} KB")


def _rows(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    head = lines[0].split(",")
    return [dict(zip(head, line.split(","), strict=True)) for line in lines[1:]]


def _stretch(motion, start: float, end: float, x: np.ndarray) -> np.ndarray:
    t = np.asarray(motion.time)
    v = np.asarray(motion.columns["knee_flexion"])
    inside = (t >= start) & (t <= end)
    return np.interp(x, 100 * (t[inside] - start) / (end - start), v[inside])


if __name__ == "__main__":
    main(*sys.argv[1:])
