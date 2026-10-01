"""Match each ground-truth instance to at most one prediction.

Pairs are built by image name and similarity, never by list position: two
lists built by different code paths can drift out of order and still have the
same length. Within an image the most similar pair is taken first, so the
result does not depend on the order instances were listed in. Instances of
different classes are never paired.
"""

from dataclasses import dataclass, field
from pathlib import PurePath

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


def pair(
    truth: Dataset,
    predicted: Dataset,
    min_iou: float = 0.3,
    min_keypoint_similarity: float = 0.5,
    min_shared_points: int = 2,
) -> Pairing:
    """Boxes are compared by IoU. When either side has no real box (a results
    file without `bbox`, or `sv.KeyPoints` without detections), the keypoints
    themselves are compared, since a box around two points on a line is empty.

    Image names are compared as given. When none match but they do once the
    file extension is dropped (`x.jpg` against `x`), extensions are dropped and
    a warning says so.
    """
    result = Pairing()
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
        candidates, near = [], []
        for i, t in enumerate(truths):
            for j, p in enumerate(preds):
                if use_classes and t.class_id != p.class_id:
                    continue
                if t.boxed and p.boxed:
                    score, floor = _iou(t.bbox, p.bbox), min_iou
                elif (t.visible & p.visible).sum() < min_shared_points:
                    continue
                else:
                    score, floor = _keypoint_similarity(t, p), min_keypoint_similarity
                if score >= floor:
                    candidates.append((score, i, j))
                elif score > 0:
                    near.append((i, j))
        used_t: set[int] = set()
        used_p: set[int] = set()
        for _score, i, j in sorted(candidates, key=lambda c: -c[0]):
            if i not in used_t and j not in used_p:
                used_t.add(i)
                used_p.add(j)
                result.pairs.append(Pair(image, truths[i], preds[j], i, j))
        result.missed.extend(
            (image, t) for i, t in enumerate(truths) if i not in used_t
        )
        result.extra.extend((image, p) for j, p in enumerate(preds) if j not in used_p)
        result.near_misses.extend(
            (image, truths[i])
            for i in sorted({i for i, j in near if i not in used_t and j not in used_p})
        )
    if truth and predicted and not set(truth) & set(predicted):
        result.warnings.append(
            "No image name appears in both truth and predictions: check that "
            "both sides name images the same way."
        )
    return result


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
        raise ValueError(
            f"skeletons differ: truth has {sorted(sizes['truth'])} keypoints, "
            f"predictions {sorted(sizes['predictions'])}"
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
        (n for n in predicted if n not in truth and PurePath(n).stem in stem_t), None
    ) or next(iter(predicted))
    warnings.append(
        f"Image names matched only without their extensions (e.g. {odd!r}); "
        "extensions dropped."
    )
    return stem_t, stem_p


def _by_stem(dataset: Dataset) -> Dataset | None:
    """None when dropping extensions would merge two images."""
    out: Dataset = {}
    for name, instances in dataset.items():
        stem = PurePath(name).stem if PurePath(name).suffix.lower() in _IMAGE else name
        if stem in out:
            return None
        out[stem] = instances
    return out


_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


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
