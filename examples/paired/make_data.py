"""Synthetic paired data for the examples in docs/cli.md: knee and hip angles
from a made-up pose pipeline against a made-up motion capture, for 6 subjects
walking twice, and one trial written as OpenSim .mot files. Nothing here is
measured; it only shows the input formats. Run it from this folder."""

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RATE = 50  # frames per second
FRAMES = 60


def walk(rng, subject_offset):
    """One trial: a reference angle curve and a prediction with a subject's
    own offset, noise, and a pipeline that reads 10% of the knee's swing
    less than it is."""
    t = np.arange(FRAMES) / RATE
    phase = 2 * np.pi * t / 1.1
    knee = 35 - 30 * np.cos(phase) + rng.normal(0, 0.5)
    hip = 10 + 25 * np.sin(phase) + rng.normal(0, 0.5)
    knee_pred = 35 - 27 * np.cos(phase) + subject_offset + rng.normal(0, 3, FRAMES)
    hip_pred = hip + 0.5 * subject_offset + rng.normal(0, 2.5, FRAMES)
    return t, {"knee_angle_r": (knee_pred, knee), "hip_flexion_r": (hip_pred, hip)}


def write_mot(path, t, columns):
    names = ["time", *columns]
    lines = [
        "Coordinates",
        "version=1",
        f"nRows={len(t)}",
        f"nColumns={len(names)}",
        "inDegrees=yes",
        "endheader",
        "\t".join(names),
    ]
    for k, time in enumerate(t):
        lines.append(
            "\t".join(f"{v:.6f}" for v in [time, *(c[k] for c in columns.values())])
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rng = np.random.default_rng(7)
    rows = []
    for s in range(1, 7):
        offset = rng.normal(0, 2)
        for trial in ("walk1", "walk2"):
            t, measures = walk(rng, offset)
            for name, (pred, ref) in measures.items():
                for k in range(FRAMES):
                    rows.append(
                        [f"S{s:02d}", trial, f"{t[k]:.2f}", name,
                         f"{pred[k]:.3f}", f"{ref[k]:.3f}", "deg"]
                    )  # fmt: skip
            if s == 1 and trial == "walk1":
                write_mot(HERE / "S01_walk1_pose.mot", t,
                          {n: p for n, (p, _r) in measures.items()})  # fmt: skip
                write_mot(HERE / "S01_walk1_mocap.mot", t,
                          {n: r for n, (_p, r) in measures.items()})  # fmt: skip
    with open(HERE / "angles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["subject", "trial", "time", "measure", "pred", "ref", "unit"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
