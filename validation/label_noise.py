"""How much of the demo's disagreement could COCO's own labels account for?

COCO publishes, per keypoint, the spread of repeated human labels relative to
the object's scale (its OKS sigmas; shoulder 0.079, elbow 0.072, wrist 0.062,
scale = sqrt(area)). This script moves the demo's labelled arms by that much
and measures two things:

- the mean change in the elbow angle: what label noise alone would add to the
  disagreement;
- the slope of a perfect model (one that reads the labelled angle exactly) on
  the noisy labels: how far label noise alone pulls the gain below 1.

The sigmas can be read as a per-axis spread of sigma*s/2 (if they measure the
distance between two labellers) or sigma*s/sqrt(2) (if they measure one label
against a consensus); both are shown. Each label's points are treated as
independent, which is a rough assumption.

A second table asks whether label noise could explain the demo's gap below
the jitter reference: honest models, read off the demo's arms with noise on
the labels and on the predictions, go through `poseaudit.audit`.

    uv run python validation/label_noise.py
"""

import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from poseaudit import agreement as ag
from poseaudit import angle, audit, load_coco, load_coco_results, pair
from poseaudit.pairing import Pair, Pairing
from poseaudit.types import Instance

DEMO = Path(__file__).parents[1] / "examples" / "coco_elbow"
SIGMA = np.array([0.079, 0.072, 0.062])  # COCO: left shoulder, elbow, wrist
REPEATS = 300
HONEST_RUNS = 30


def elbow(points: np.ndarray) -> np.ndarray:
    u = points[..., 0, :] - points[..., 1, :]
    v = points[..., 2, :] - points[..., 1, :]
    cross = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    return np.degrees(np.arctan2(cross, (u * v).sum(-1)))


def honest(job):
    """One honest model on the demo's arms: labels and predictions both
    scattered around the labelled points taken as the world."""
    points, scale, label, model, seed = job
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(seed)
    box, seen = np.array([-1e4, -1e4, 1e4, 1e4]), np.ones(3, bool)
    pairs = []
    for i, (p, s) in enumerate(zip(points, scale, strict=True)):
        spread = SIGMA[:, None] * s
        truth = p + rng.normal(0, 1, (3, 2)) * spread * label
        pred = p + rng.normal(0, 1, (3, 2)) * spread * model
        pairs.append(
            Pair(f"i{i}", Instance(box, truth, seen), Instance(box, pred, seen))
        )
    r = audit(
        Pairing(pairs=pairs),
        angle(0, 1, 2),
        15,
        resamples=50,
        jitter_repeats=200,
        seed=seed,
    )
    t = np.array([x.truth for x in r.readings])
    p = np.array([x.predicted for x in r.readings])
    return (
        label,
        model,
        r.gain - r.jitter_gain,
        r.jitter_p <= 0.05,
        ag.deming(t, p, 1.0),
    )


def main() -> None:
    warnings.simplefilter("ignore")
    gt = DEMO / "gt_200.json"
    result = audit(
        pair(load_coco(gt), load_coco_results(DEMO / "pred_yolo11n.json", gt, 0.5)),
        angle(5, 7, 9),
        15,
        jitter_repeats=0,
        resamples=10,
    )
    data = json.loads(gt.read_text())
    names = {image["id"]: image["file_name"] for image in data["images"]}
    by_image: dict[str, list] = {}
    for a in data["annotations"]:
        if not a.get("iscrowd") and a.get("num_keypoints", 1):
            by_image.setdefault(names[a["image_id"]], []).append(a)
    chosen = [by_image[r.image][r.truth_index] for r in result.readings]
    points = np.array(
        [np.reshape(a["keypoints"], (-1, 3))[[5, 7, 9], :2] for a in chosen], float
    )
    scale = np.sqrt([a["area"] for a in chosen])
    true = elbow(points)
    rng = np.random.default_rng(0)
    print(f"{len(true)} arms of the demo, {REPEATS} draws\n")
    print(
        "| per-axis spread | mean angle change | off by 15° or more "
        "| gain of a perfect model |"
    )
    print("|---|---|---|---|")
    for label, factor in (("sigma*s/2", 0.5), ("sigma*s/sqrt(2)", 2**-0.5)):
        spread = SIGMA[None, :, None] * scale[:, None, None] * factor
        changes, large, slopes = [], [], []
        for _ in range(REPEATS):
            noisy = elbow(points + rng.normal(0, 1, points.shape) * spread)
            changes.append(np.abs(noisy - true).mean())
            large.append((np.abs(noisy - true) >= 15).mean())
            d = noisy - noisy.mean()
            slopes.append(d @ (true - true.mean()) / (d @ d))
        print(
            f"| {label} | {np.mean(changes):.1f}° | {np.mean(large):.1%} "
            f"| {np.mean(slopes):.3f} |"
        )
    print(
        f"\nobserved: mean {result.mean_abs_error:.2f}°, "
        f"{result.big_error_rate:.1%} off by 15° or more, gain {result.gain:.3f}"
    )
    settings = [(0.0, 0.71), (0.5, 0.5), (0.71, 0.71), (0.5, 1.0)]
    jobs = [
        (points, scale, a, b, 100 * k + j)
        for k, (a, b) in enumerate(settings)
        for j in range(HONEST_RUNS)
    ]
    with ProcessPoolExecutor() as pool:
        out = list(pool.map(honest, jobs))
    print(
        f"\nhonest models on the same arms, {HONEST_RUNS} runs each "
        "(spreads in units of sigma*s per axis)\n"
    )
    print(
        "| label spread | model spread | mean gap | p(squash) <= 0.05 "
        "| Deming, ratio 1 |"
    )
    print("|---|---|---|---|---|")
    for a, b in settings:
        rows = [o for o in out if o[0] == a and o[1] == b]
        print(
            f"| {a} | {b} | {np.mean([o[2] for o in rows]):+.3f} "
            f"| {np.mean([o[3] for o in rows]):.0%} "
            f"| {np.mean([o[4] for o in rows]):.3f} |"
        )


if __name__ == "__main__":
    main()
