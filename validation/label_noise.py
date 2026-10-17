"""How much of the demo's disagreement could COCO's own labels account for?

COCO publishes, per keypoint, the spread of repeated human labels relative to
the object's scale (its OKS sigmas; shoulder 0.079, elbow 0.072, wrist 0.062,
scale = sqrt(area)). This script moves the demo's labelled arms by that much
and measures two things:

- the mean change in the elbow angle: what label noise of that size alone
  would add to the disagreement;
- the slope of a perfect model (one that reads the labelled angle exactly) on
  the noisy labels: how far label noise alone pulls the gain below 1.

The sigmas can be read as a per-axis spread of sigma*s/2 (if they measure the
distance between two labellers) or sigma*s/sqrt(2) (if they measure one label
against a consensus); both are shown. Each label's points are treated as
independent, with isotropic normal noise: the sigmas give the size of the
spread, not its shape, so this is one assumption among several.

A second table splits the first by arm size, with the same pixel edges as
the README's `--size-bands 30,60` (the mean of the two labelled segments).
Short segments are partly foreshortened arms, whose labels are no less
spread in pixels, so the same label noise turns into larger angle changes
there; the table shows what that noise alone would give in each band.

A third table holds these rates against the repo's own val2017 runs. It
splits the demo's arms into straight (a labelled angle of 150° or more) and
bent, in the bins of limb length (the mean of the two segments) that
`validation/error_budget.out.md` reads `yolo11x-pose`'s left elbow in, and
puts that model's read rates beside them. Under this noise a model that
reads every arm exactly still disagrees with the labels at the "labels
alone" rate, and models with independent point noise of their own, 0.5 or 1
times the label spread, disagree more often. A real model that disagrees
less often than that most likely means that the labels' noise is not shaped
this way, or that the model's errors follow the labels' (a model trained on
COCO learns its annotators' habits); there may be other causes. Either way
these rates do not bound the labels' share. Both sides are COCO val2017
people, but not the same arms: the model's rates are every left elbow it
read on val2017. That run keeps every detection, so where the model emits
duplicates, pairing may take the closest of them and lower its rate; how
much is not measured here.

A fourth table asks whether label noise could explain the demo's gap below
the jitter reference: honest models, read off the demo's arms with noise on
the labels and on the predictions, go through `poseaudit.audit`.

    uv run python validation/label_noise.py
"""

import json
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from poseaudit import agreement as ag
from poseaudit import angle, audit, load_coco, load_coco_results, pair
from poseaudit.pairing import Pair, Pairing
from poseaudit.types import Instance

DEMO = Path(__file__).parents[1] / "examples" / "coco_elbow"
BUDGET = Path(__file__).with_name("error_budget.out.md")
SIGMA = np.array([0.079, 0.072, 0.062])  # COCO: left shoulder, elbow, wrist
SIZE_EDGES = (30.0, 60.0)  # the README's --size-bands 30,60
SPREADS = (("sigma*s/2", 0.5), ("sigma*s/sqrt(2)", 2**-0.5))
MODEL_NOISE = (0.5, 1.0)  # a model's own point noise, in label spreads
STRAIGHT = 150.0  # error_budget.py's straight limbs, in degrees
REPEATS = 300
HONEST_RUNS = 30


def elbow(points: np.ndarray) -> np.ndarray:
    u = points[..., 0, :] - points[..., 1, :]
    v = points[..., 2, :] - points[..., 1, :]
    cross = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    return np.degrees(np.arctan2(cross, (u * v).sum(-1)))


def yolo11x() -> dict[str, tuple[float, list[tuple[float, float, float]]]]:
    """What validation/error_budget.out.md reads for yolo11x-pose's left
    elbow on val2017, straight and bent: the rate over the group, and each of
    its five bins of limb length as (low px, high px, rate)."""
    text = BUDGET.read_text(encoding="utf-8").split("## pred_yolo11x\n")[1]
    table = text.split("\n## ")[0].splitlines()
    out = {}
    for group in ("straight", "bent"):
        (row,) = [x for x in table if x.startswith(f"| left elbow | {group} |")]
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        overall = float(cells[4].split(",")[0].split("/")[1])
        bins = re.findall(r"(\d+)-(\d+): [\d.]+/([\d.]+)", cells[8])
        out[group] = (overall, [(float(a), float(b), float(r)) for a, b, r in bins])
    return out


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
        size_bands=list(SIZE_EDGES),
    )
    data = json.loads(gt.read_text(encoding="utf-8"))
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
    # The size the README's bands sort by: the mean labelled segment length.
    size = (
        np.linalg.norm(points[:, 0] - points[:, 1], axis=1)
        + np.linalg.norm(points[:, 2] - points[:, 1], axis=1)
    ) / 2
    band = np.digitize(size, SIZE_EDGES)
    by_band, alone = {}, {}
    for label, factor in SPREADS:
        spread = SIGMA[None, :, None] * scale[:, None, None] * factor
        changes, large, slopes, per_arm = [], [], [], np.zeros(len(true))
        for _ in range(REPEATS):
            noisy = elbow(points + rng.normal(0, 1, points.shape) * spread)
            off = np.abs(noisy - true) >= 15
            changes.append(np.abs(noisy - true).mean())
            large.append(off.mean())
            per_arm += off
            d = noisy - noisy.mean()
            slopes.append(d @ (true - true.mean()) / (d @ d))
        by_band[label] = [per_arm[band == b].mean() / REPEATS for b in range(3)]
        alone[label] = per_arm / REPEATS
        print(
            f"| {label} | {np.mean(changes):.1f}° | {np.mean(large):.1%} "
            f"| {np.mean(slopes):.3f} |"
        )
    print(
        f"\nobserved: mean {result.mean_abs_error:.2f}°, "
        f"{result.big_error_rate:.1%} off by 15° or more, gain {result.gain:.3f}"
    )
    print("\nby arm size (mean labelled segment), off by 15° or more\n")
    print(
        "| arm segments | arms | observed | labels alone, sigma*s/2 "
        "| labels alone, sigma*s/sqrt(2) |"
    )
    print("|---|---|---|---|---|")
    names = ("0-30 px", "30-60 px", "60+ px")
    for b, observed in enumerate(result.size_bands):
        assert observed.n == int((band == b).sum()), "bands must match the audit"
        cells = "".join(f" | {by_band[label][b]:.0%}" for label, _ in SPREADS)
        print(f"| {names[b]} | {observed.n} | {observed.big_error_rate:.0%}{cells} |")
    # a model with independent noise of its own, at the lower reading
    spread = SIGMA[None, :, None] * scale[:, None, None] * SPREADS[0][1]
    own = {k: np.zeros(len(true)) for k in MODEL_NOISE}
    for _ in range(REPEATS):
        labelled = elbow(points + rng.normal(0, 1, points.shape) * spread)
        for k in MODEL_NOISE:
            model = elbow(points + rng.normal(0, 1, points.shape) * spread * k)
            own[k] += np.abs(model - labelled) >= 15
    print(
        "\nagainst val2017: straight (labelled 150° or more) and bent arms in "
        "the bins of limb length of validation/error_budget.out.md, off by 15° "
        "or more. Label noise alone (SE over the demo's arms), then with a "
        "model's own independent point noise of "
        f"{' and '.join(f'{k:g}' for k in MODEL_NOISE)} times the label spread "
        "(sigma*s/2), beside what yolo11x-pose reads on val2017's left elbows\n"
    )
    first, second = (label for label, _ in SPREADS)
    print(
        f"| arms | segments | demo arms | labels alone, {first} "
        f"| model noise {', '.join(f'{k:g}' for k in MODEL_NOISE)} "
        f"| labels alone, {second} | yolo11x read, val2017 |"
    )
    print("|---|---|---|---|---|---|---|")
    for group, (overall, bins) in yolo11x().items():
        kept = true >= STRAIGHT if group == "straight" else true < STRAIGHT
        inner = [high for _low, high, _rate in bins[:-1]]
        where = np.digitize(size, inner)
        spans = [
            f"under {inner[0]:.0f} px",
            *(f"{a:.0f}-{b:.0f} px" for a, b in zip(inner, inner[1:], strict=False)),
            f"{inner[-1]:.0f}+ px",
        ]
        rows = [("all", kept, overall)] + [
            (span, kept & (where == b), rate)
            for b, (span, (_low, _high, rate)) in enumerate(
                zip(spans, bins, strict=True)
            )
        ]
        for span, inside, rate in rows:
            floor = alone[first][inside]
            se = floor.std(ddof=1) / np.sqrt(len(floor))
            noisy = ", ".join(f"{own[k][inside].mean() / REPEATS:.0%}" for k in own)
            print(
                f"| {group} | {span} | {len(floor)} | {floor.mean():.0%} "
                f"(SE {se * 100:.1f}) | {noisy} | {alone[second][inside].mean():.0%} "
                f"| {rate:.0%} |"
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
        "| label spread | model spread | mean gap | p(slope <= jitter) <= 0.05 "
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
