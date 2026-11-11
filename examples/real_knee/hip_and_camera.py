"""Two follow-up checks on why the side-view knee peak is read low, per subject.

python hip_and_camera.py data out

`data` and `out` as for build.py. For each trial (right leg) it measures, from
the markers at the frames of highest knee flexion (top 5 percent), the hip
ab/adduction of the femur in the pelvis frame and how far the hip-knee-ankle
plane tilts out of the pelvis's sagittal plane, and, from the video, the
shank-to-thigh length ratio divided by the same ratio from the markers. Each is
compared with the trial's peak gap (95th percentile of the video angle minus
95th percentile of the marker angle), by Spearman correlation over the subject
means. Writes out/hip_and_camera.csv.

Limits: there is only a lateral knee marker, so the femur axis leans outward
and the absolute adduction is biased; rotation enters only through the plane
tilt; the length ratio uses the same hip, knee and ankle keypoints as the video
angle, so a keypoint error moves both. Needs pandas, scipy and openpyxl."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from build import COCO, MIN_CONFIDENCE, flexion, hip_centres, read_trc, unit


def mot(path: Path):
    lines = path.read_text().splitlines()
    i = lines.index("endheader") + 2
    a = np.array([[float(x) for x in line.split()] for line in lines[i:] if line.strip()])
    return a[:, 0], a[:, 1]


def leg_lengths_2d(csv: Path) -> tuple[float, float]:
    k = pd.read_csv(csv)
    k = k[k.person == 0]
    hip, knee, ankle = (k[k.keypoint == i].set_index("frame") for i in COCO["R"])
    f = hip.index.intersection(knee.index).intersection(ankle.index)
    ok = ((hip.loc[f].confidence > MIN_CONFIDENCE) & (knee.loc[f].confidence > MIN_CONFIDENCE)
          & (ankle.loc[f].confidence > MIN_CONFIDENCE)).to_numpy()
    f = f[ok]
    xy = lambda p: p.loc[f, ["x", "y"]].to_numpy()
    return (np.median(np.linalg.norm(xy(hip) - xy(knee), axis=1)),
            np.median(np.linalg.norm(xy(knee) - xy(ankle), axis=1)))


def main(data: Path, out: Path) -> None:
    rows = []
    for _, r in pd.read_csv(out / "pairs.csv").iterrows():
        _, video = mot(out / r.pred)
        _, marker = mot(out / r.ref)
        sid = r.subject[1:]
        _, m = read_trc(data / "02_markerbased_data" / sid / f"{r.trial}.trc")
        hip, knee, ankle = hip_centres(m)["R"], m["RKNE"], m["RANK"]
        right = unit(m["RASI"] - m["LASI"])
        femur = unit(knee - hip)
        adduction = np.degrees(np.arcsin(np.clip(-(femur * right).sum(-1), -1, 1)))
        normal = unit(np.cross(hip - knee, ankle - knee))
        tilt = np.degrees(np.arccos(np.clip(np.abs((normal * right).sum(-1)), 0, 1)))
        f3 = flexion(hip, knee, ankle)
        top = f3 >= np.nanpercentile(f3, 95)
        thigh2, shank2 = leg_lengths_2d(data / "03_markerless_data" / sid / "side" / f"{r.trial}.csv")
        thigh3 = np.nanmedian(np.linalg.norm(hip - knee, axis=-1))
        shank3 = np.nanmedian(np.linalg.norm(knee - ankle, axis=-1))
        rows.append(dict(subject=r.subject, trial=r.trial,
                         peak_gap=np.nanpercentile(video, 95) - np.nanpercentile(marker, 95),
                         adduction=np.nanmean(adduction[top]), plane_tilt=np.nanmean(tilt[top]),
                         length_ratio=(shank2 / thigh2) / (shank3 / thigh3)))
    t = pd.DataFrame(rows)
    t.to_csv(out / "hip_and_camera.csv", index=False)
    s = t.groupby("subject").mean(numeric_only=True)
    for c in ("adduction", "plane_tilt", "length_ratio"):
        rho = spearmanr(s[c], s.peak_gap)
        print(f"{c:12s} Spearman {rho.statistic:+.2f} (p={rho.pvalue:.3f}) over {len(s)} subjects")
    print(s.round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
