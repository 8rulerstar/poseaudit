"""Does `poseaudit budget`'s fit hold? COCO val2017, four YOLO11 pose models.

Every val2017 image with a labelled person (2,346 images), elbows and knees.
For each model, joint and group (straight from 150 degrees, bent below), the
script runs `budget` and then, on all readings, the large-error rate the fit
predicts against the rate read, in five bins of limb length.

Preparing the data (about 390 MB of images; the models download on first use):

    # person_keypoints_val2017.json from the COCO 2017 annotations
    uv run python validation/error_budget.py prepare person_keypoints_val2017.json data/
    cd data/images && xargs -n 20 curl -sf --remote-name-all < ../urls.txt && cd ../..
    for m in n s m x; do
      uv run --with ultralytics python examples/coco_elbow/predict.py \\
        data/gt.json data/images data/pred_$m.json yolo11$m-pose.pt
    done

Then:

    uv run python validation/error_budget.py run data/gt.json data/pred_n.json \\
      data/pred_s.json data/pred_m.json data/pred_x.json \\
      > validation/error_budget.out.md

A large error is 15 degrees or more; the target is 20%. Then three more
tables: how many groups pass on seeds 0 to 9, how many pass when the large
error or the straight limit changes, and how the last model's left elbow
fares when the images are drawn again with replacement. Seeded, so the
tables are the same on every machine with the same numpy. Predictions for
the published table came from ultralytics 8.4.150; other versions may read
differently.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

from poseaudit import angle, load_coco, load_coco_results, pair
from poseaudit.budget import _lengths, _readings, budget, fit_spread, large_rate

JOINTS = {
    "left elbow": (5, 7, 9),
    "right elbow": (6, 8, 10),
    "left knee": (11, 13, 15),
    "right knee": (12, 14, 16),
}
BIG_ERROR = 15.0
TARGET = 0.2


def prepare(annotations: str, out: str) -> None:
    data = json.loads(Path(annotations).read_text(encoding="utf-8"))
    ids = {
        a["image_id"]
        for a in data["annotations"]
        if not a["iscrowd"] and a["num_keypoints"] > 0
    }
    images = [i for i in data["images"] if i["id"] in ids]
    keep = ("id", "image_id", "category_id", "bbox", "keypoints", "num_keypoints",
            "iscrowd", "area")  # fmt: skip
    folder = Path(out)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    subset = {
        "info": {"description": "COCO val2017 images with a labelled person"},
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": [
            {k: a[k] for k in keep} for a in data["annotations"] if a["image_id"] in ids
        ],
        "categories": data["categories"],
    }
    (folder / "gt.json").write_text(json.dumps(subset), encoding="utf-8")
    # the COCO image host serves plain HTTP only
    (folder / "urls.txt").write_text(
        "\n".join(i["coco_url"] for i in images) + "\n", encoding="utf-8"
    )
    print(f"{len(images)} images -> {folder / 'gt.json'}, {folder / 'urls.txt'}")


def run(gt: str, predictions: list[str]) -> None:
    warnings.simplefilter("ignore")
    truth = load_coco(gt)
    three = angle(0, 1, 2)
    images = len(json.loads(Path(gt).read_text(encoding="utf-8"))["images"])
    print(
        f"# budget on COCO val2017 ({images:,} images): {BIG_ERROR:g} degrees, "
        f"target {TARGET:.0%}\n"
    )
    print(
        "predicted / read large-error rate, each reading predicted by a fit "
        "made without it (all, shorter, "
        "longer limbs; * outside the tolerance), and on all readings in five "
        "bins of limb length\n"
    )
    for path in predictions:
        name = Path(path).stem
        pairing = pair(truth, load_coco_results(path, gt))
        print(f"## {name}\n")
        print(
            "| joint | group | n | point error | check | passed | smallest limb "
            "| floor | by limb length (px: predicted/read) |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for joint, points in JOINTS.items():
            result = budget(pairing, angle(*points), BIG_ERROR, TARGET)
            bins = _bins(pairing, angle(*points), three)
            for g in result.groups:
                check = ", ".join(
                    f"{guess:.2f}/{read:.2f}{'' if ok else '*'}"
                    for _, guess, read, ok in g.checks
                )
                smallest = g.smallest_limb_px
                size = "-" if smallest is None else f"{smallest:.0f}"
                print(
                    f"| {joint} | {g.name} | {g.n} | {g.fixed_px:.1f} px + "
                    f"{g.share:.1%} | {check} | {'yes' if g.passed else 'no'} | "
                    f"{size} | {g.floor_rate:.2f} | {bins[g.name]} |"
                )
        print()
    _stability(truth, gt, predictions)


def _pairings(truth, gt, predictions):
    return {Path(p).stem: pair(truth, load_coco_results(p, gt)) for p in predictions}


def _count(pairings, big_error, straight_from, seed=0) -> tuple[int, int]:
    """(elbow groups passed, knee groups passed)."""
    passed = {"elbow": 0, "knee": 0}
    for pairing in pairings.values():
        for joint, points in JOINTS.items():
            result = budget(
                pairing, angle(*points), big_error, TARGET, straight_from, seed
            )
            passed[joint.split()[1]] += sum(g.passed for g in result.groups)
    return passed["elbow"], passed["knee"]


def _stability(truth, gt, predictions) -> None:
    pairings = _pairings(truth, gt, predictions)
    groups = 4 * len(pairings)
    print(f"## groups passed, of {groups} elbow and {groups} knee groups\n")
    print("| seed | elbow | knee |")
    print("|---|---|---|")
    for seed in range(10):
        elbow, knee = _count(pairings, BIG_ERROR, 150.0, seed)
        print(f"| {seed} | {elbow} | {knee} |")
    print("\n| large error | straight from | elbow | knee |")
    print("|---|---|---|---|")
    for big_error, straight_from in [(10, 150), (15, 150), (20, 150), (15, 140),
                                     (15, 160)]:  # fmt: skip
        elbow, knee = _count(pairings, big_error, straight_from)
        print(f"| {big_error}° | {straight_from}° | {elbow} | {knee} |")
    print()
    _redrawn(pairings[sorted(pairings)[-1]], sorted(pairings)[-1])


def _redrawn(pairing, name: str, draws: int = 20) -> None:
    """The left elbow of one model with `redraw`: does the verdict hold when
    the data, not the seed, change?"""
    result = budget(pairing, angle(5, 7, 9), BIG_ERROR, TARGET, redraw=draws)
    print(f"## {name}, left elbow, images drawn again {draws} times\n")
    print("| group | readings | check passed | sizes |")
    print("|---|---|---|---|")
    for g in result.groups:
        sizes = g.redraw_sizes
        span = f"{min(sizes):.0f} to {max(sizes):.0f} px" if sizes else "-"
        print(f"| {g.name} | {g.n} | {g.redraw_passed}/{draws} | {span} |")
    print()


def _bins(pairing, measure, three) -> dict[str, str]:
    t, p, _ = _readings(pairing, measure)
    angles = three.read_many(t)
    out = {}
    for name, chosen in (("straight", angles >= 150), ("bent", angles < 150)):
        tt, pp = t[chosen], p[chosen]
        fixed, share = fit_spread(tt, pp)
        size = _lengths(tt)[:, 1]
        read = np.abs(three.read_many(pp) - three.read_many(tt)) >= BIG_ERROR
        guess = large_rate(tt, fixed, share, BIG_ERROR, np.random.default_rng(0))
        edges = np.quantile(size, np.linspace(0, 1, 6))
        out[name] = " ".join(
            f"{lo:.0f}-{hi:.0f}: {guess[inside].mean():.2f}/{read[inside].mean():.2f}"
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
            for inside in [(size >= lo) & (size <= hi)]
        )
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["prepare"]:
        prepare(*sys.argv[2:4])
    elif sys.argv[1:2] == ["run"]:
        run(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
