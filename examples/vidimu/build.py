"""Joint angles of four monocular video 3D pose models against IMU inverse
kinematics, from the VIDIMU benchmark: "Paving the Way Towards Kinematic
Assessment Using Monocular Video" (10.5281/zenodo.15088423, CC BY 4.0;
dataset: VIDIMU, 16 subjects, 13 daily activities).

python build.py data out

`data` holds what `download.py` unpacks. The comparison is the authors':
for each activity the one joint they chose, the first trial (T01 to T05)
that every model and the IMU hold, and the video angle as the 3D angle
between two keypoint vectors:

- knee: hip minus knee against knee minus ankle (A01 to A04);
- elbow: shoulder minus elbow against elbow minus wrist (A05, A06, A09);
- shoulder: shoulder minus elbow against neck minus torso (A07, A08, A10 to A13);

against the IMU's OpenSim coordinate of the same name (knee_angle_l/r,
elbow_flex_l/r, arm_flex_l/r). The two definitions differ, so a constant gap
between them is expected; it is the same for all four models.

Writes, per model, `out/<model>/mot/` (one prediction and one reference
.mot per subject and activity, the column named "<activity> <joint>") and
`out/<model>/pairs.csv`. The video runs at 30 Hz from time 0; the IMU at
50 Hz, its all-zero first row dropped. Needs pandas."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["bodytrack", "mmpose", "motionagformer", "motionbert"]
# activity: (IMU coordinate, the four keypoints a, b, c, d: angle of a-b against c-d)
JOINTS = {
    **{
        a: ("knee_angle_l", "left_hip left_knee left_knee left_ankle")
        for a in ("A01", "A03")
    },
    **{
        a: ("knee_angle_r", "right_hip right_knee right_knee right_ankle")
        for a in ("A02", "A04")
    },
    **{
        a: ("elbow_flex_r", "right_shoulder right_elbow right_elbow right_wrist")
        for a in ("A05", "A09")
    },
    "A06": ("elbow_flex_l", "left_shoulder left_elbow left_elbow left_wrist"),
    **{
        a: ("arm_flex_r", "right_shoulder right_elbow neck torso")
        for a in ("A07", "A10", "A11", "A13")
    },
    **{
        a: ("arm_flex_l", "left_shoulder left_elbow neck torso") for a in ("A08", "A12")
    },
}


def angle(frame: pd.DataFrame, points: str) -> np.ndarray:
    a, b, c, d = (
        frame[[f"{p}_{x}" for x in "xyz"]].to_numpy(float) for p in points.split()
    )
    u, v = a - b, c - d
    with np.errstate(
        invalid="ignore"
    ):  # frames with a missing keypoint: NaN, filled below
        cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def write_mot(path: Path, time: np.ndarray, name: str, values: np.ndarray) -> None:
    lines = [path.stem, "inDegrees=yes", "endheader", f"time\t{name}"]
    lines += [f"{t:.4f}\t{v:.4f}" for t, v in zip(time, values, strict=True)]
    path.write_text("\n".join(lines) + "\n")


def main(data: str, out: str) -> None:
    root = Path(data) / "jointangles"
    imu = root / "jointangles_imus"
    rows: dict[str, list[str]] = {m: [] for m in MODELS}
    for activity, (joint, points) in JOINTS.items():
        for subject in sorted(p.name for p in imu.iterdir() if p.is_dir()):
            trial = next(
                (
                    t
                    for t in ("T01", "T02", "T03", "T04", "T05")
                    if (imu / subject / f"ik_{subject}_{activity}_{t}.mot").exists()
                    and all(
                        (
                            root
                            / f"jointangles_{m}"
                            / subject
                            / f"{subject}_{activity}_{t}.csv"
                        ).exists()
                        for m in MODELS
                    )
                ),
                None,
            )
            if trial is None:
                continue
            ref = pd.read_csv(
                imu / subject / f"ik_{subject}_{activity}_{trial}.mot",
                sep="\t",
                skiprows=6,
            ).iloc[1:]
            name = f"{activity} {joint}"
            for model in MODELS:
                frame = pd.read_csv(
                    root
                    / f"jointangles_{model}"
                    / subject
                    / f"{subject}_{activity}_{trial}.csv"
                )
                values = (
                    pd.Series(angle(frame, points))
                    .interpolate(limit_direction="both")
                    .to_numpy()
                )
                folder = Path(out) / model / "mot"
                folder.mkdir(parents=True, exist_ok=True)
                stem = f"{subject}_{activity}"
                write_mot(
                    folder / f"{stem}_pred.mot",
                    np.arange(len(values)) / 30.0,
                    name,
                    values,
                )
                write_mot(
                    folder / f"{stem}_ref.mot",
                    ref["time"].to_numpy(),
                    name,
                    ref[joint].to_numpy(),
                )
                rows[model].append(
                    f"{subject},{activity},mot/{stem}_pred.mot,mot/{stem}_ref.mot"
                )
    for model, lines in rows.items():
        (Path(out) / model / "pairs.csv").write_text(
            "subject,trial,pred,ref\n" + "\n".join(lines) + "\n"
        )
        print(f"{model}: {len(lines)} pairs")


if __name__ == "__main__":
    main(*sys.argv[1:])
