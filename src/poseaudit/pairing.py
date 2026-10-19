"""Match each ground-truth instance to at most one prediction.

Pairs are built by image name and similarity, never by list position: two
lists built by different code paths can drift out of order and still have the
same length. Within an image the most similar pair is taken first (greedy),
and ties are broken by keypoint similarity, then box IoU, so the result does
not depend on the order instances were listed in. Only candidates that agree
on every one of these scores fall back to list position, and a warning says
when such a tie decided a pair between predictions with different keypoints.
Instances of different classes are never paired.

The similarity mixes box IoU with keypoint similarity (OKS), so among several
predictions overlapping one truth the one whose keypoints sit closest to it
tends to win: with duplicate detections of one person, the best of several
tries is scored. A warning counts the truths whose pair OKS changed while the
prediction box overlap alone would have paired them with was left unpaired;
predictions OKS moved from one truth to another are not counted.

With `matching="optimal"` the same qualifying pairs are assigned to maximise
the summed rank over the image instead (the Hungarian method), which can pair
differently in crowds where taking the best pair first leaves its neighbours
worse off.
"""

from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

from poseaudit.types import Dataset, Instance


@dataclass(frozen=True)
class Pair:
    image: str
    truth: Instance
    predicted: Instance
    truth_index: int = 0  # position in the image's truth list, for tracing back
    predicted_index: int = 0


@dataclass
class Pairing:
    pairs: list[Pair] = field(default_factory=list)
    missed: list[tuple[str, Instance]] = field(default_factory=list)
    extra: list[tuple[str, Instance]] = field(default_factory=list)  # no truth
    # unmatched truths that overlap an unmatched prediction below the threshold
    near_misses: list[tuple[str, Instance]] = field(default_factory=list)
    images_without_prediction: list[str] = field(default_factory=list)
    images_without_truth: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    matching: str = "greedy"
    # truths whose partner (or lack of one) differs from the greedy pairing
    changed_from_greedy: int = 0

    def __repr__(self) -> str:
        """Counts only: the dataclass's own repr prints every keypoint array."""
        notes = len(self.warnings)
        return (
            f"<Pairing: {len(self.pairs)} pairs, {len(self.missed)} truths and "
            f"{len(self.extra)} predictions unmatched, "
            f"{notes} {'warning' if notes == 1 else 'warnings'}>"
        )


def pair(
    truth: Dataset,
    predicted: Dataset,
    min_iou: float = 0.3,
    min_keypoint_similarity: float = 0.5,
    min_shared_points: int = 2,
    matching: str = "greedy",
) -> Pairing:
    """Boxes are compared by IoU. A pair qualifies when its IoU reaches
    `min_iou`; qualifying pairs are then ranked by the mean of IoU and OKS
    (object keypoint similarity), so two people whose boxes nearly coincide
    (one in front of the other, a hug) are told apart by their keypoints. A
    prediction whose keypoints are all wrong still pairs on its box: its
    error is what the audit is for. When the two share no visible point, the
    rank is the IoU alone.

    OKS uses COCO's per-keypoint sigmas when the skeleton has 17 points and
    one sigma for every point otherwise (their mean, 0.0627), with the truth
    box's area as the object's scale.

    When either side has no real box (a results file without `bbox`, or
    `sv.KeyPoints` without detections), the keypoints themselves are compared,
    since a box around two points on a line is empty.

    Image names are compared as given. When none match but they do once the
    file extension is dropped (`x.jpg` against `x`), extensions are dropped and
    a warning says so.

    matching: "greedy" takes the best-ranked pair first; "optimal" picks the
    pairs whose ranks sum highest. Both use only the pairs that qualify
    above, and `changed_from_greedy` counts the truths optimal pairs
    differently.
    """
    if matching not in ("greedy", "optimal"):
        raise ValueError(f"matching must be 'greedy' or 'optimal', got {matching!r}")
    result = Pairing(matching=matching)
    tied_images: list[str] = []
    switched: list[str] = []  # one image per truth whose pair OKS chose over a
    # prediction left unpaired
    _same_skeleton(truth, predicted)
    truth, predicted = _aligned_names(truth, predicted, result.warnings)
    use_classes = _comparable_classes(truth, predicted, result.warnings)
    result.images_without_truth = sorted(set(predicted) - set(truth))
    for image in result.images_without_truth:
        result.extra.extend((image, p) for p in predicted[image])
    for image in sorted(truth):
        truths, preds = truth[image], predicted.get(image)
        if preds is None:
            result.images_without_prediction.append(image)
            result.missed.extend((image, t) for t in truths)
            continue
        candidates, near, by_box = [], [], []
        for i, t in enumerate(truths):
            for j, p in enumerate(preds):
                if use_classes and t.class_id != p.class_id:
                    continue
                if t.boxed and p.boxed:
                    iou, floor = _iou(t.bbox, p.bbox), min_iou
                    oks = _oks(t, p)
                    score = iou
                    rank = iou if np.isnan(oks) else (iou + oks) / 2
                elif (t.visible & p.visible).sum() < min_shared_points:
                    continue
                else:
                    score, floor = _keypoint_similarity(t, p), min_keypoint_similarity
                    iou, oks, rank = 0.0, score, score
                if score >= floor:
                    oks = -1.0 if np.isnan(oks) else oks
                    candidates.append((rank, oks, iou, i, j))
                    by_box.append((score, i, j))  # the rank without OKS
                elif score > 0:
                    near.append((i, j))
        candidates.sort(key=lambda c: (-c[0], -c[1], -c[2], c[3], c[4]))
        ties = tied_images if matching == "greedy" else []  # greedy's ties only
        chosen = _greedy(candidates, truths, preds, image, ties)
        if matching == "optimal":
            best = _optimal(candidates, len(truths), len(preds))
            was, now = dict(chosen), dict(best)
            result.changed_from_greedy += sum(
                int(was.get(i) != now.get(i)) for i in range(len(truths))
            )
            chosen = best
        used_t = {i for i, _ in chosen}
        used_p = {j for _, j in chosen}
        result.pairs.extend(Pair(image, truths[i], preds[j], i, j) for i, j in chosen)
        switched += [image] * _switched_by_keypoints(dict(chosen), by_box)
        result.missed.extend(
            (image, t) for i, t in enumerate(truths) if i not in used_t
        )
        result.extra.extend((image, p) for j, p in enumerate(preds) if j not in used_p)
        result.near_misses.extend(
            (image, truths[i])
            for i in sorted({i for i, j in near if i not in used_t and j not in used_p})
        )
    if tied_images:
        result.warnings.append(
            f"{len(tied_images)} pairs were decided between predictions that "
            "score exactly alike but differ in keypoints, by list position "
            f"(e.g. {tied_images[0]!r}); the pairing there is arbitrary."
        )
    if switched:
        result.warnings.append(_switched_warning(switched))
    _few_matched(truth, predicted, result)
    folders = _folders_differ(
        result.images_without_prediction, result.images_without_truth
    )
    if truth and predicted and not set(truth) & set(predicted):
        result.warnings.append(
            "No image name appears in both truth and predictions: check that "
            "both sides name images the same way." + (f" {folders}" if folders else "")
        )
    elif folders:
        result.warnings.append(folders)
    return result


def _greedy(candidates, truths, preds, image, tied_images) -> list[tuple[int, int]]:
    """The best-ranked open pair first, from candidates sorted best first."""
    used_t: set[int] = set()
    used_p: set[int] = set()
    chosen = []
    for k, (*_scores, i, j) in enumerate(candidates):
        if i in used_t or j in used_p:
            continue
        if _tied(candidates, k, used_t, used_p, truths, preds):
            tied_images.append(image)
        used_t.add(i)
        used_p.add(j)
        chosen.append((i, j))
    return chosen


def _optimal(candidates, rows: int, cols: int) -> list[tuple[int, int]]:
    """The qualifying pairs whose ranks sum highest, in truth order. Pairs
    that do not qualify weigh 0, so they are never worth taking; ranks are
    above 0, so every pair an assignment takes from them is a real one."""
    weight = np.zeros((rows, cols))
    for rank, *_scores, i, j in candidates:
        weight[i, j] = rank
    assigned = linear_sum_assignment(weight, maximize=True)
    pairs = zip(assigned[0].tolist(), assigned[1].tolist(), strict=True)
    return [(i, j) for i, j in pairs if weight[i, j] > 0]


def linear_sum_assignment(
    cost: np.ndarray, maximize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The rows and columns of a rectangular cost matrix that pair every row
    (or every column, whichever are fewer) at the least total cost, like
    scipy.optimize.linear_sum_assignment. Shortest augmenting paths with
    potentials (Jonker-Volgenant), O(n^2 m); rows are added in order and the
    first column of equal reduced cost wins, so equal inputs give equal
    results."""
    cost = np.asarray(cost, dtype=float)
    if not np.isfinite(cost).all():
        raise ValueError("cost matrix must be finite")
    if maximize:
        cost = -cost
    flip = cost.shape[0] > cost.shape[1]
    if flip:
        cost = cost.T
    n, m = cost.shape
    u = np.zeros(n + 1)  # row potentials, 1-based with 0 a free slot
    v = np.zeros(m + 1)
    owner = np.zeros(m + 1, dtype=int)  # row holding each column, 0 for none
    way = np.zeros(m + 1, dtype=int)
    for row in range(1, n + 1):
        owner[0] = row
        col = 0
        least = np.full(m + 1, np.inf)
        done = np.zeros(m + 1, dtype=bool)
        while owner[col] != 0:
            done[col] = True
            r = owner[col]
            reduced = cost[r - 1] - u[r] - v[1:]
            open_ = ~done[1:]
            better = open_ & (reduced < least[1:])
            least[1:][better] = reduced[better]
            way[1:][better] = col
            gaps = np.where(open_, least[1:], np.inf)
            nxt = int(np.argmin(gaps)) + 1
            delta = gaps[nxt - 1]
            u[owner[done]] += delta
            v[done] -= delta
            least[1:][open_] -= delta
            col = nxt
        while col:
            prev = way[col]
            owner[col] = owner[prev]
            col = prev
    cols = np.flatnonzero(owner[1:])
    rows = owner[1:][cols] - 1
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    if flip:
        rows, cols = cols, rows
        order = np.argsort(rows)
        rows, cols = rows[order], cols[order]
    return rows, cols


def _folders_differ(unmatched: list[str], unused: list[str]) -> str:
    """A note when unmatched truth images share their last name with a
    prediction's image (`imgs/a.jpg` against `a`): only folders differ, and
    folders are kept so that `cam1/a.jpg` and `cam2/a.jpg` stay apart."""
    others = {_basename(n) for n in unused}
    alike = sorted(n for n in unmatched if _basename(n) in others)
    if not alike:
        return ""
    return (
        f"{len(alike)} truth images without a prediction match a predicted image "
        f"by their last name alone (e.g. {alike[0]!r}): folders differ, so they "
        "were not paired; use the same names on both sides (e.g. Path(p).name)."
    )


def _basename(name: str) -> str:
    """The name after its last folder, `/` or `\\`, without an image extension."""
    return _drop_ext(name.replace("\\", "/").rsplit("/", 1)[-1])


def _tied(candidates, k, used_t, used_p, truths, preds) -> bool:
    """True when another open candidate for the same truth or prediction
    scores exactly like candidate k but pairs different keypoints."""
    key, i, j = candidates[k][:3], candidates[k][3], candidates[k][4]
    for other in candidates[k + 1 :]:
        if other[:3] != key:
            break
        a, b = other[3], other[4]
        if a in used_t or b in used_p:
            continue
        if a == i and not _same_points(preds[b], preds[j]):
            return True
        if b == j and not _same_points(truths[a], truths[i]):
            return True
    return False


def _switched_by_keypoints(chosen: dict[int, int], by_box) -> int:
    """Truths with two or more qualifying predictions whose pair would change
    if the rank left OKS out (greedy pairing again, by IoU alone; keypoint
    similarity where there are no boxes, which has no OKS to drop), and whose
    pair by IoU alone was left unpaired: the case of duplicates. A prediction
    that went to another truth instead is a person the keypoints told apart,
    not a try left over."""
    rivals: dict[int, int] = {}
    for _score, i, _j in by_box:
        rivals[i] = rivals.get(i, 0) + 1
    alone: dict[int, int] = {}
    taken: set[int] = set()
    for _score, i, j in sorted(by_box, key=lambda c: (-c[0], c[1], c[2])):
        if i not in alone and j not in taken:
            alone[i] = j
            taken.add(j)
    paired = set(chosen.values())
    return sum(
        1
        for i, count in rivals.items()
        if count >= 2
        and i in alone
        and chosen.get(i) != alone[i]
        and alone[i] not in paired
    )


def _switched_warning(images: list[str]) -> str:
    """Among predictions that overlap one truth, the one whose keypoints sit
    closest to it wins: the best of several tries when they are duplicates."""
    n = len(images)
    which = (
        f"{n} labelled instances had" if n > 1 else "1 labelled instance had"
    ) + " two or more predictions overlapping enough to pair"
    return (
        f"{which}, and keypoint similarity (OKS), not box overlap alone, chose "
        "which one: the prediction box overlap alone would have taken was left "
        f"unpaired (e.g. {images[0]!r}). With duplicate detections of one "
        "person that scores the model on its best try, which flatters it: drop "
        "duplicates first (--min-score, min_score= in the loaders). Where the "
        "one left unpaired is another, unlabelled person, the keypoints rightly "
        "tell them apart."
    )


def _same_points(a: Instance, b: Instance) -> bool:
    return np.array_equal(a.visible, b.visible) and np.array_equal(
        a.keypoints[a.visible], b.keypoints[b.visible]
    )


def _few_matched(truth: Dataset, predicted: Dataset, result: Pairing) -> None:
    """Warn when names overlap yet (nearly) nothing pairs: ids or names that
    point at different images, or coordinates on different scales."""
    labelled = [name for name, instances in truth.items() if instances]
    shared = [name for name in labelled if predicted.get(name)]
    if not labelled or not shared:
        return  # no overlap at all has its own warning
    if not result.pairs:
        blind = [
            side
            for side, data in (("truth", truth), ("predictions", predicted))
            if not any(i.visible.any() for name in shared for i in data[name])
        ]
        if blind:
            result.warnings.append(
                "No truth instance matched a prediction: no point is visible in "
                f"the {' or the '.join(blind)} of the images on both sides. Check the "
                "visibility flags or, for predictions, the confidence cut "
                "(min_confidence, --min-conf)."
            )
            return
        result.warnings.append(
            f"No truth instance matched a prediction, although {len(shared)} "
            "images appear on both sides: check that image ids or names refer "
            "to the same images and that both use pixel coordinates."
        )
    elif len(labelled) >= 10 and len(shared) <= 0.1 * len(labelled):
        result.warnings.append(
            f"Only {len(shared)} of {len(labelled)} labelled images have any "
            "prediction: check that image ids or names refer to the same images."
        )


def _comparable_classes(truth: Dataset, predicted: Dataset, warnings) -> bool:
    """Classes gate matching only when both sides number them the same way.
    COCO calls a person 1 and Ultralytics 0: sets that share no value are two
    numbering schemes, not two sets of objects."""
    ids = {
        side: {i.class_id for instances in data.values() for i in instances}
        for side, data in (("truth", truth), ("predicted", predicted))
    }
    if None in ids["truth"] | ids["predicted"]:
        return False
    if ids["truth"] and ids["predicted"] and not ids["truth"] & ids["predicted"]:
        warnings.append(
            f"Class ids share no value (truth {sorted(ids['truth'], key=str)}, "
            f"predictions {sorted(ids['predicted'], key=str)}): matched ignoring "
            "classes. Filter both sides to one class if they hold several."
        )
        return False
    return True


def _same_skeleton(truth: Dataset, predicted: Dataset) -> None:
    sizes = {
        side: {len(i.keypoints) for instances in data.values() for i in instances}
        for side, data in (("truth", truth), ("predictions", predicted))
    }
    if len(sizes["truth"] | sizes["predictions"]) > 1:
        counts = {k: " or ".join(map(str, sorted(v))) for k, v in sizes.items()}
        raise ValueError(
            f"skeletons differ: truth has {counts['truth']} keypoints, "
            f"predictions {counts['predictions']}"
        )


def _aligned_names(
    truth: Dataset, predicted: Dataset, warnings: list[str]
) -> tuple[Dataset, Dataset]:
    """Drop image extensions when that matches more images than the names as
    given: all of them (`x.jpg` against `x`) or only some (`a.jpg`, `b.jpg`
    against `a.jpg`, `b`)."""
    stem_t = _by_stem(truth)
    stem_p = _by_stem(predicted)
    if stem_t is None or stem_p is None:
        return truth, predicted
    if len(set(stem_t) & set(stem_p)) <= len(set(truth) & set(predicted)):
        return truth, predicted
    odd = next(
        (n for n in sorted(predicted) if n not in truth and _drop_ext(n) in stem_t),
        None,
    ) or min(predicted)
    warnings.append(
        f"Image names matched only without their extensions (e.g. {odd!r}); "
        "extensions dropped."
    )
    return stem_t, stem_p


def _by_stem(dataset: Dataset) -> Dataset | None:
    """None when dropping extensions would merge two images."""
    out: Dataset = {}
    for name, instances in dataset.items():
        stem = _drop_ext(name)
        if stem in out:
            return None
        out[stem] = instances
    return out


def _drop_ext(name: str) -> str:
    """The name without an image extension, folders kept, so cam1/a.jpg and
    cam2/a.jpg stay apart."""
    suffix = PurePosixPath(name).suffix  # the name is kept as given, on any OS
    return name[: -len(suffix)] if suffix.lower() in _IMAGE else name


_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# COCO's per-keypoint OKS sigmas, nose to right ankle
COCO_SIGMAS = (
    np.array(
        [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
         1.07, 1.07, .87, .87, .89, .89]
    )
    / 10.0
)  # fmt: skip


def _oks(truth: Instance, predicted: Instance) -> float:
    """COCO's object keypoint similarity over the points both sides see; NaN
    when they share none. The truth box's area stands in for the segment
    area COCO uses."""
    shared = truth.visible & predicted.visible
    if not shared.any():
        return float("nan")
    count = len(truth.keypoints)
    sigmas = COCO_SIGMAS if count == 17 else np.full(count, COCO_SIGMAS.mean())
    x1, y1, x2, y2 = truth.bbox
    area = max(float((x2 - x1) * (y2 - y1)), np.finfo(float).eps)
    d2 = ((truth.keypoints[shared] - predicted.keypoints[shared]) ** 2).sum(axis=1)
    k2 = (2 * sigmas[shared]) ** 2
    return float(np.mean(np.exp(-d2 / (2 * area * k2))))


def _keypoint_similarity(truth: Instance, predicted: Instance) -> float:
    """1 when the shared visible points coincide, falling off with their mean
    distance relative to the truth's size. Shaped like OKS, but not OKS: one
    scale for every keypoint."""
    shared = truth.visible & predicted.visible
    if not shared.any():
        return 0.0
    distance = np.linalg.norm(
        truth.keypoints[shared] - predicted.keypoints[shared], axis=1
    )
    x1, y1, x2, y2 = truth.bbox
    size = float(np.hypot(x2 - x1, y2 - y1))
    if size == 0:
        return 1.0 if distance.max() == 0 else 0.0
    return float(np.exp(-((distance.mean() / (0.1 * size)) ** 2) / 2))
