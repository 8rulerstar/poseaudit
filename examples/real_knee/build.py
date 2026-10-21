"""Knee flexion from a side-view YOLOv8 against marker motion capture, from
"A Synchronized Dual-System Marker-based and Markerless Motion Capture
Dataset for Gait and Fundamental Fitness Activities" (Capital University of
Physical Education and Sports, 2026, 10.5281/zenodo.19720803, CC BY 4.0).

python build.py data out

`data` holds the two folders `download.py` unpacks. Every walking trial
with both a marker file and a side-camera file is used (31 trials of 17
subjects). How the two sides are made comparable, fixed in this file:

- marker knee: the 3D angle at the knee marker between the hip joint centre
  (Harrington et al. 2007, from the pelvis markers) and the ankle marker;
- YOLO knee: the 2D angle at the knee keypoint between the hip and ankle
  keypoints of the side camera, points under 0.3 confidence left out;
- leg: the one the side camera sees more confidently, as YOLO names it
  (the right in every trial);
- time offset: the lag, within 1.5 s either way, where the fore-aft gap of
  the two ankles (right ahead of left) correlates best between the sides.
  The knee angles are not used for it. Lining up by the knee angles was
  dropped as circular, and by foot strikes because with a symmetric gait a
  swap of legs and half a stride look alike.
- YOLO frame rate: 60 when the frames outnumber the marker duration by 45
  a second, else 30.

Writes, in `out`: one .mot pair per trial in `mot/` with the prediction on
the marker clock and `pairs.csv` listing them; `events.csv`, the marker heel
strikes of that leg; `trials.csv`, what was decided per trial.
Needs pandas and scipy."""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

TRIALS = ("Walk_2km", "Walk_5km", "Walk_8km")
MARKER_RATE = 120
MIN_CONFIDENCE = 0.3
MAX_LAG = 1.5  # seconds either way
COCO = {"L": (11, 13, 15), "R": (12, 14, 16)}  # hip, knee, ankle


def read_trc(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lines = path.read_text().splitlines()
    names = [n for n in lines[3].split("\t")[2:] if n.strip()]
    width = 2 + 3 * len(names)
    rows = [
        [float(v) if v.strip() else np.nan for v in line.split("\t")[:width]]
        for line in lines[5:]
        if line.strip()
    ]
    a = np.array(rows)
    return a[:, 1], {n: a[:, 2 + 3 * i : 5 + 3 * i] for i, n in enumerate(names)}


def flexion(hip, knee, ankle) -> np.ndarray:
    """180 minus the angle at the knee: 0 with the leg straight."""
    u, v = hip - knee, ankle - knee
    cos = (u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
    return 180 - np.degrees(np.arccos(np.clip(cos, -1, 1)))


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def hip_centres(m) -> dict[str, np.ndarray]:
    """Harrington et al. 2007, in metres: back 0.24 PD + 9.9 mm, down
    0.30 PW + 10.9 mm and out 0.33 PW + 7.3 mm from the middle of the ASIS."""
    front = (m["LASI"] + m["RASI"]) / 2
    back = (m["LPSI"] + m["RPSI"]) / 2
    right = unit(m["RASI"] - m["LASI"])
    forward = front - back
    forward = unit(forward - (forward * right).sum(-1, keepdims=True) * right)
    up = np.cross(right, forward)
    width = np.linalg.norm(m["RASI"] - m["LASI"], axis=-1, keepdims=True)
    depth = np.linalg.norm(front - back, axis=-1, keepdims=True)
    centre = front - (0.24 * depth + 0.0099) * forward - (0.30 * width + 0.0109) * up
    out = (0.33 * width + 0.0073) * right
    return {"R": centre + out, "L": centre - out}


def heel_strikes(time, heel_height) -> np.ndarray:
    h = pd.Series(heel_height).interpolate(limit_direction="both").to_numpy()
    lows, _ = find_peaks(
        -h, distance=int(0.45 * MARKER_RATE), prominence=np.nanstd(h) * 0.5
    )
    return time[lows]


def best_lag(t_pred, gap_pred, t_ref, gap_ref) -> tuple[float, float]:
    """The lag added to the prediction's times where the gaps correlate best,
    and that correlation."""
    best = (-np.inf, 0.0)
    for lag in np.arange(-MAX_LAG, MAX_LAG + 1e-9, 1 / 240):
        ref = np.interp(t_pred + lag, t_ref, gap_ref, left=np.nan, right=np.nan)
        ok = ~np.isnan(ref) & ~np.isnan(gap_pred)
        best = max(best, (np.corrcoef(ref[ok], gap_pred[ok])[0, 1], lag))
    return best[1], best[0]


def write_mot(path: Path, time, knee) -> None:
    ok = ~np.isnan(knee)
    head = [path.stem, "version=1", f"nRows={ok.sum()}", "nColumns=2"]
    head += ["inDegrees=yes", "endheader", "time\tknee_flexion"]
    body = [f"{t:.5f}\t{k:.4f}" for t, k in zip(time[ok], knee[ok], strict=True)]
    path.write_text("\n".join(head + body) + "\n")


def trial(data: Path, subject: str, name: str, out: Path):
    time, m = read_trc(data / "02_markerbased_data" / subject / f"{name}.trc")
    hips = hip_centres(m)
    k = pd.read_csv(data / "03_markerless_data" / subject / "side" / f"{name}.csv")
    k = k[k.person == 0]
    frames = k.frame.max() + 1
    fps = 60 if frames / time[-1] > 45 else 30
    points = {}
    for i in range(17):
        p = k[k.keypoint == i].set_index("frame").reindex(range(frames))
        p.loc[p.confidence < MIN_CONFIDENCE, ["x", "y"]] = np.nan
        points[i] = p
    t_pred = np.arange(frames) / fps

    forward = (m["LASI"] + m["RASI"]) / 2 - (m["LPSI"] + m["RPSI"]) / 2
    gap_ref = ((m["RANK"] - m["LANK"]) * unit(forward)).sum(-1)
    hip_x = (points[11].x + points[12].x) / 2
    facing = np.sign(np.nanmedian(points[0].x - hip_x))
    gap_pred = facing * (points[16].x - points[15].x).to_numpy()
    lag, gap_r = best_lag(t_pred, gap_pred, time, gap_ref)

    left, right = points[13].confidence.mean(), points[14].confidence.mean()
    leg = "L" if left > right else "R"
    hip, knee, ankle = (points[i][["x", "y"]].to_numpy() for i in COCO[leg])
    pred = flexion(hip, knee, ankle)
    ref = flexion(hips[leg], m[f"{leg}KNE"], m[f"{leg}ANK"])

    stem = f"{subject}_{name}"
    write_mot(out / "mot" / f"{stem}_yolo.mot", t_pred + lag, pred)
    write_mot(out / "mot" / f"{stem}_marker.mot", time, ref)
    strikes = heel_strikes(time, m[f"{leg}HEE"][:, 1])  # Y is up in these files
    return stem, leg, fps, lag, gap_r, strikes


def main(data: str, out: str) -> None:
    data_path, out_path = Path(data), Path(out)
    (out_path / "mot").mkdir(parents=True, exist_ok=True)
    pairs, events, trials = [], [], []
    subjects = sorted(p.name for p in (data_path / "02_markerbased_data").iterdir())
    for subject in subjects:
        for name in TRIALS:
            marker = data_path / "02_markerbased_data" / subject / f"{name}.trc"
            side = data_path / "03_markerless_data" / subject / "side" / f"{name}.csv"
            if not (marker.exists() and side.exists()):
                continue
            stem, leg, fps, lag, gap_r, strikes = trial(
                data_path, subject, name, out_path
            )
            s = f"S{subject}"
            pairs.append([s, name, f"mot/{stem}_yolo.mot", f"mot/{stem}_marker.mot"])
            events += [[s, name, f"{t:.4f}"] for t in strikes]
            trials.append([s, name, leg, fps, f"{lag:.4f}", f"{gap_r:.3f}"])
            print(*trials[-1], flush=True)
    for name, head, rows in (
        ("pairs.csv", ["subject", "trial", "pred", "ref"], pairs),
        ("events.csv", ["subject", "trial", "time"], events),
        ("trials.csv", ["subject", "trial", "leg", "fps", "lag_s", "gap_r"], trials),
    ):
        with open(out_path / name, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(head)
            writer.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:])
