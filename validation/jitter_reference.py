"""Calibration and power of the jitter reference, by simulation.

Synthetic elbows with known behaviour go through `poseaudit.audit`; for each
case the script reports how often the one-sided p(squash) is 0.05 or less and
how often the gap's interval lies wholly below 0. A model with no tendency
should land near 5% and 2.5%; a squashing one near 100%.

    uv run python validation/jitter_reference.py               # 600 runs, 200/200
    uv run python validation/jitter_reference.py 150 2000 500 "isotropic;label noise"

Arguments: runs per case, bootstrap resamples, jitter rebuilds, and cases to
run separated by semicolons (all by default). Fewer
resamples than the tool's 2,000 make the gap's interval slightly too narrow,
so its false alarm rates here run a little high. Every run is seeded, so the
table is the same on every machine with the same numpy.
"""

import sys
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from poseaudit import angle, audit
from poseaudit.pairing import Pair, Pairing
from poseaudit.types import Instance

CASES = {
    # name: (arms, description)
    "isotropic": (300, "no tendency; every point scatters 8% of the arm"),
    "across forearm": (300, "no tendency; the wrist scatters 20% across the forearm"),
    "straight noisier": (300, "no tendency; three times the noise above 150 degrees"),
    "bend bisector": (300, "no tendency; the elbow scatters along the bend's bisector"),
    "random failures": (300, "no tendency; 10% of wrists thrown in a random direction"),
    "label noise": (300, "no tendency; the labels scatter as much as the model"),
    "isotropic, 40 arms": (40, "as isotropic, with 40 arms"),
    "straight noisier, 40 arms": (40, "as straight noisier, with 40 arms"),
    "squash 7%": (300, "reads angles 7% closer to 110 degrees, plus isotropic noise"),
    "squash 15%": (300, "reads angles 15% closer to 110 degrees, plus isotropic noise"),
}


def arm(degrees: float, size: float) -> np.ndarray:
    elbow = np.array([300.0, 300.0])
    a = np.radians(degrees)
    wrist = elbow + size * np.array([np.sin(a), -np.cos(a)])
    return np.array([elbow + [0.0, -size], elbow, wrist])


def run(job: tuple[str, int, int, int]) -> tuple[str, bool, bool]:
    case, seed, resamples, repeats = job
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(seed)
    n = CASES[case][0]
    squash = {"squash 7%": 0.93, "squash 15%": 0.85}.get(case, 1.0)
    pairs = []
    for i in range(n):
        t, size = rng.uniform(40, 178), rng.uniform(40, 160)
        truth, pred = arm(t, size), arm(110 + squash * (t - 110), size)
        if rng.random() < 0.5:  # half the arms bend the other way
            truth[:, 0], pred[:, 0] = 600 - truth[:, 0], 600 - pred[:, 0]
        noise = rng.normal(0, 0.08 * size, (3, 2))
        if case == "across forearm":
            u = (pred[2] - pred[1]) / np.linalg.norm(pred[2] - pred[1])
            across = np.array([-u[1], u[0]])
            noise[2] = u * rng.normal(0, 0.03 * size) + across * rng.normal(
                0, 0.2 * size
            )
        if case.startswith("straight noisier") and t > 150:
            noise *= 3
        if case == "bend bisector":
            u = (truth[0] - truth[1]) / np.linalg.norm(truth[0] - truth[1])
            v = (truth[2] - truth[1]) / np.linalg.norm(truth[2] - truth[1])
            bisector = (u + v) / np.linalg.norm(u + v)
            noise = rng.normal(0, 0.03 * size, (3, 2))
            noise[1] += bisector * rng.normal(0, 0.25 * size)
        if case == "random failures" and rng.random() < 0.1:
            noise[2] = rng.normal(0, 0.8 * size, 2)
        if case == "label noise":
            truth = truth + rng.normal(0, 0.08 * size, (3, 2))
        turn = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        box, seen = np.array([-900.0, -900, 900, 900]), np.ones(3, bool)
        pairs.append(
            Pair(
                f"i{i}",
                Instance(box, truth @ rot.T, seen),
                Instance(box, (pred + noise) @ rot.T, seen),
            )
        )
    r = audit(
        Pairing(pairs=pairs),
        angle(0, 1, 2),
        15,
        resamples=resamples,
        jitter_repeats=repeats,
        seed=seed,
    )
    return case, bool(r.jitter_p <= 0.05), bool(r.gain_gap_ci[1] < 0)


def main(runs: int, resamples: int, repeats: int, cases: list[str]) -> None:
    # seeds follow the case's name, so adding a case changes no other row
    jobs = [
        (case, zlib.crc32(case.encode()) % 1_000_000 * 1_000 + j, resamples, repeats)
        for case in CASES
        if case in cases
        for j in range(runs)
    ]
    with ProcessPoolExecutor() as pool:
        out = list(pool.map(run, jobs, chunksize=8))
    print(
        f"{runs} runs per case; {resamples} resamples and {repeats} rebuilds per run\n"
    )
    print("| case | p(squash) <= 0.05 | interval wholly below 0 | setup |")
    print("|---|---|---|---|")
    for case, (_n, description) in CASES.items():
        rows = [o for o in out if o[0] == case]
        if not rows:
            continue
        p = np.mean([o[1] for o in rows])
        below = np.mean([o[2] for o in rows])
        print(f"| {case} | {p:.1%} | {below:.1%} | {description} |")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:4]]
    runs, resamples, repeats = args + [600, 200, 200][len(args) :]
    main(
        runs,
        resamples,
        repeats,
        sys.argv[4].split(";") if len(sys.argv) > 4 else list(CASES),
    )
