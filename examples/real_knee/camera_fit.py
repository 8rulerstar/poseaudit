"""Split the side-view knee peak gap into a camera part and a keypoint part.

python camera_fit.py data out

`data` and `out` as for build.py (it reads out/trials.csv and out/mot/). For
each trial it fits one camera for the whole recording: a least-squares affine
map from the 3D markers to the side video's keypoints (shoulders, hip joint
centres, knees, ankles, both sides, every frame where the keypoint is read),
with one refit after dropping points over 3 times the median residual. The
nearest scaled orthographic camera (the affine map's rows made orthonormal)
gives the viewing angle. Then, with the 95th percentile as the peak:

- camera part: the knee flexion of the markers drawn through that camera,
  minus the 3D flexion of the markers;
- keypoint part: the video's flexion minus the markers drawn through it.

The two add up to the whole gap. It also measures, at the frames of highest
marker flexion, how far each right-leg keypoint sits from its marker drawn
through the camera, across the thigh line, in thigh lengths. Writes
out/camera_fit.csv.

Limits: the camera is fitted to the same keypoints whose offsets it is meant to
separate, so a systematic offset shared by many points can leak into it; one
camera per trial, so a camera that moved during a trial is not caught; the
knee marker is lateral and the hip joint centre is a regression estimate, so
"the marker" is itself an approximation of the joint. Needs pandas and scipy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from build import MIN_CONFIDENCE, flexion, hip_centres, read_trc, unit
from scipy.stats import spearmanr

PAIRS = {5: "LSHO", 6: "RSHO", 11: "HL", 12: "HR", 13: "LKNE", 14: "RKNE"}
PAIRS |= {15: "LANK", 16: "RANK"}


def fit_affine(x: np.ndarray, u: np.ndarray):
    """u = x @ a.T + b by least squares, refitted once without outliers."""
    keep = np.ones(len(x), bool)
    for _ in range(2):
        xh = np.c_[x[keep], np.ones(keep.sum())]
        sol, *_ = np.linalg.lstsq(xh, u[keep], rcond=None)
        res = np.linalg.norm(np.c_[x, np.ones(len(x))] @ sol - u, axis=1)
        keep = res < 3 * np.median(res) + 1e-9
    return sol[:3].T, sol[3]


def angle_2d(h, k, a) -> np.ndarray:
    u, v = h - k, a - k
    c = (u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
    return 180 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def keypoint_track(k: pd.DataFrame, i: int, time, fps: int, offset: float):
    p = k[(k.keypoint == i) & (k.confidence >= MIN_CONFIDENCE)].sort_values("frame")
    tp = p.frame.to_numpy() / fps + offset
    x = np.interp(time, tp, p.x, left=np.nan, right=np.nan)
    y = np.interp(time, tp, p.y, left=np.nan, right=np.nan)
    near = np.zeros(len(time), bool)
    if len(tp):
        j = np.clip(np.searchsorted(tp, time), 1, len(tp) - 1)
        gap = np.minimum(np.abs(time - tp[j - 1]), np.abs(time - tp[j]))
        near = gap < 0.5 / fps
    uv = np.c_[x, y]
    uv[~near] = np.nan
    return uv


def read_mot(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    i = lines.index("endheader") + 2
    return np.array([float(line.split()[1]) for line in lines[i:] if line.strip()])


def trial(data: Path, out: Path, t, pairs: dict = PAIRS) -> dict:
    sid, name = t.subject, t.trial
    time, m = read_trc(data / "02_markerbased_data" / sid / f"{name}.trc")
    hc = hip_centres(m)
    m = dict(m, HL=hc["L"], HR=hc["R"])
    side = data / "03_markerless_data" / sid / "side" / f"{name}.csv"
    k = pd.read_csv(side)
    k = k[k.person == 0]
    tracks = {i: keypoint_track(k, i, time, t.fps, t.offset_s) for i in PAIRS}
    x, u = [], []
    for i, mk in pairs.items():
        ok = ~np.isnan(tracks[i]).any(1) & ~np.isnan(m[mk]).any(1)
        x.append(m[mk][ok])
        u.append(tracks[i][ok])
    a, b = fit_affine(np.vstack(x), np.vstack(u))
    us, _, vt = np.linalg.svd(a, full_matrices=False)
    rot = us @ vt  # rows of the nearest scaled orthographic camera

    view = unit(np.cross(rot[0], rot[1]))
    lat = unit(np.nanmean(unit(m["RASI"] - m["LASI"]), 0))
    front = (m["LASI"] + m["RASI"]) / 2 - (m["LPSI"] + m["RPSI"]) / 2
    fwd = np.nanmean(unit(front), 0)
    fwd = unit(fwd - (fwd @ lat) * lat)
    up = np.cross(lat, fwd)
    view = view if view @ lat >= 0 else -view

    h, kn, an = m["HR"], m["RKNE"], m["RANK"]
    f3 = flexion(h, kn, an)
    drawn = angle_2d(h @ rot.T, kn @ rot.T, an @ rot.T)
    video = read_mot(out / "mot" / f"S{sid}_{name}_yolo.mot")

    def peak(v):
        return np.nanpercentile(v, 95)

    row = dict(
        subject=f"S{sid}",
        trial=name,
        yaw=np.degrees(np.arctan2(view @ fwd, view @ lat)),
        pitch=np.degrees(np.arcsin(np.clip(view @ up, -1, 1))),
        gap=peak(video) - peak(f3),
        camera=peak(drawn) - peak(f3),
        keypoint=peak(video) - peak(drawn),
    )
    top = f3 >= np.nanpercentile(f3, 95)
    thigh = (kn - h) @ a.T
    length2 = (thigh**2).sum(1)
    for i, mk in ((12, "hip"), (14, "knee"), (16, "ankle")):
        d = tracks[i] - (m[PAIRS[i]] @ a.T + b)
        across = (d[:, 0] * thigh[:, 1] - d[:, 1] * thigh[:, 0]) / length2
        row[f"{mk}_across"] = np.nanmean(across[top])
    return row


def main(data: Path, out: Path) -> None:
    trials = pd.read_csv(out / "trials.csv", dtype={"subject": str})
    t = pd.DataFrame([trial(data, out, r) for r in trials.itertuples()])
    t.to_csv(out / "camera_fit.csv", index=False)
    no_knee = {i: mk for i, mk in PAIRS.items() if i not in (13, 14)}
    t2 = pd.DataFrame([trial(data, out, r, no_knee) for r in trials.itertuples()])
    s2 = t2.groupby("subject").mean(numeric_only=True)
    s = t.groupby("subject").mean(numeric_only=True)
    print(s.round(2).to_string())
    print(
        f"over {len(s)} subjects: gap {s.gap.mean():.1f}, camera part "
        f"{s.camera.mean():.1f}, keypoint part {s.keypoint.mean():.1f}; "
        f"|yaw| up to {s.yaw.abs().max():.0f}, |pitch| up to {s.pitch.abs().max():.0f}"
    )
    print(
        f"camera fitted without the knees: camera part {s2.camera.mean():.1f}, "
        f"keypoint part {s2.keypoint.mean():.1f}, variance "
        f"{s2.camera.var() / s2.gap.var():.2f} and "
        f"{s2.keypoint.var() / s2.gap.var():.2f} of the gap's"
    )
    for c in ("camera", "keypoint"):
        print(f"{c} part: variance {s[c].var() / s.gap.var():.2f} of the gap's")
    for c in ("hip_across", "knee_across", "ankle_across"):
        rho = spearmanr(s[c], s.keypoint)
        print(
            f"{c} vs keypoint part: Spearman {rho.statistic:+.2f} (p={rho.pvalue:.3f})"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
