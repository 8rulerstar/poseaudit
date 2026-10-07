"""`sv.KeyPoints` from roboflow/supervision (0.30.6 or later), without importing it."""

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

from poseaudit.types import Dataset, Instance


def from_supervision(
    keypoints_by_image: Mapping[str, Any],
    detections_by_image: Mapping[str, Any] | None = None,
) -> Dataset:
    """Map image name -> `sv.KeyPoints`.

    Both arguments are mappings keyed by image name, even for one image: a
    bare `sv.KeyPoints` raises a TypeError, since its image could not be paired
    with the truth's by name.

    Visibility is the container's: its `visible` mask when set. Filter by
    confidence on the container first, e.g. `kp.visible = kp.keypoint_confidence
    > 0.5`. As in supervision, a point at (0, 0) or with a non-finite coordinate
    is missing.

    `sv.KeyPoints.from_ultralytics` drops the model's boxes. Pass the same
    model's `sv.Detections` (one row per keypoint row) to match by box IoU,
    as for COCO results; without them matching compares keypoints. Build them
    from the same result with `sv.Detections.from_ultralytics`, not with
    `kp.as_detections()`, which drops rows with no visible point. Boxes a
    model keeps in `kp.data["xyxy"]` (RF-DETR) are used when no detections
    are given. A model that fills `visible` itself (RF-DETR: confidence > 0)
    keeps nearly every point; set your own threshold on it.
    """
    _by_image(keypoints_by_image, "keypoints_by_image", "sv.KeyPoints")
    if detections_by_image is not None:
        _by_image(detections_by_image, "detections_by_image", "sv.Detections")
    dataset: Dataset = {}
    unfiltered, suspect = [], []
    if detections_by_image is not None:
        stray = sorted(set(detections_by_image) - set(keypoints_by_image))
        if stray:
            warnings.warn(
                f"detections for images with no keypoints: {stray[:3]}", stacklevel=2
            )
        boxless = sorted(
            name
            for name in set(keypoints_by_image) - set(detections_by_image)
            if "xyxy" not in (getattr(keypoints_by_image[name], "data", None) or {})
        )
        if boxless:
            warnings.warn(
                f"{len(boxless)} images have keypoints but no detections, e.g. "
                f"{boxless[:3]}; those are matched by keypoints instead of boxes",
                stacklevel=2,
            )
    for name, key_points in keypoints_by_image.items():
        if not hasattr(key_points, "xy"):
            raise TypeError(f"{name}: expected sv.KeyPoints, got {type(key_points)}")
        xy = np.asarray(key_points.xy, float)
        mask = getattr(key_points, "visible", None)
        if (
            mask is None
            and getattr(key_points, "keypoint_confidence", None) is not None
        ):
            unfiltered.append(name)
        classes = getattr(key_points, "class_id", None)
        boxes = _boxes(name, detections_by_image, len(xy))
        if boxes is None:
            boxes = _carried_boxes(name, key_points, len(xy))
        instances = []
        for i in range(len(xy)):
            present = np.isfinite(xy[i]).all(axis=1) & (xy[i] != 0).any(axis=1)
            if mask is not None:
                if np.asarray(mask).dtype != bool:
                    raise ValueError(
                        f"{name}: visible must be a boolean mask, got "
                        f"{np.asarray(mask).dtype}; e.g. kp.visible = "
                        "kp.keypoint_confidence > 0.5"
                    )
                row = np.asarray(mask[i], bool)
                if row.shape != present.shape:
                    raise ValueError(
                        f"{name}: visible mask has {row.shape[0]} points, "
                        f"xy has {present.shape[0]}"
                    )
                present &= row
            points = np.where(np.isfinite(xy[i]), xy[i], 0.0)
            class_id = None if classes is None else int(classes[i])
            if boxes is None:
                instances.append(Instance.from_keypoints(points, present, class_id))
            else:
                instances.append(Instance(boxes[i], points, present, True, class_id))
        if (
            boxes is not None
            and detections_by_image is not None
            and name in detections_by_image
            and not _same_rows(name, key_points, detections_by_image[name])
        ):
            suspect += _swapped_rows(name, instances)
        dataset[name] = instances
    if suspect:
        warnings.warn(
            f"{len(suspect)} images have rows whose keypoints fit each other's "
            f"boxes better than their own, e.g. {suspect[:3]}; if the detections "
            "were filtered or sorted apart from the keypoints, matching is wrong. "
            "Crowded or sparsely labelled rows can do this too; detections with "
            "the same scores as the keypoints are matched by score instead.",
            stacklevel=2,
        )
    if unfiltered:
        warnings.warn(
            f"{len(unfiltered)} images carry keypoint_confidence but no visible mask, "
            "so every point counts as seen; set kp.visible = "
            "kp.keypoint_confidence > 0.5 first",
            stacklevel=2,
        )
    return dataset


def _by_image(given, argument: str, kind: str) -> None:
    if not isinstance(given, Mapping):
        raise TypeError(
            f"{argument} must map image name -> {kind}, got "
            f"{type(given).__name__}; for one image pass {{'image.jpg': value}} "
            "with the name the truth uses"
        )


def _boxes(name: str, detections_by_image, rows: int) -> np.ndarray | None:
    if detections_by_image is None or name not in detections_by_image:
        return None
    boxes = np.asarray(detections_by_image[name].xyxy, float)
    if not np.isfinite(boxes).all():
        raise ValueError(f"{name}: a detection box holds a non-finite value")
    if len(boxes) != rows:
        raise ValueError(
            f"{name}: {len(boxes)} detections for {rows} keypoint rows; "
            "they must come from the same model call, row for row"
        )
    return boxes


def _carried_boxes(name: str, key_points, rows: int) -> np.ndarray | None:
    """Boxes some models (RF-DETR) carry in the container's `data["xyxy"]`."""
    data = getattr(key_points, "data", None) or {}
    if "xyxy" not in data:
        return None
    boxes = np.asarray(data["xyxy"], float)
    if rows == 0 and boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.shape != (rows, 4):
        raise ValueError(
            f"{name}: data['xyxy'] has shape {boxes.shape}, expected ({rows}, 4)"
        )
    if not np.isfinite(boxes).all():
        raise ValueError(f"{name}: data['xyxy'] holds a non-finite value")
    return boxes


def _same_rows(name: str, key_points, detections) -> bool:
    """True when both containers carry the same per-row scores, all distinct:
    `from_ultralytics` puts the same box confidence on both, so the rows are
    known to line up. Scores whose order differs, or class ids that differ,
    mean the rows were filtered or sorted apart. Scores that agree only
    roughly (rounded on one side) prove nothing either way."""
    a = getattr(key_points, "class_id", None)
    b = getattr(detections, "class_id", None)
    if a is not None and b is not None and not np.array_equal(a, b):
        raise ValueError(
            f"{name}: keypoints and detections differ in class ids row for row; "
            "were the detections filtered or sorted apart from them?"
        )
    a = getattr(key_points, "detection_confidence", None)
    b = getattr(detections, "confidence", None)
    if a is None or b is None:
        return False
    a, b = np.asarray(a, float), np.asarray(b, float)
    distinct = len(np.unique(a)) == len(a) and len(np.unique(b)) == len(b)
    if distinct and not np.array_equal(np.argsort(a), np.argsort(b)):
        raise ValueError(
            f"{name}: keypoints and detections rank their scores differently "
            "row for row; were the detections filtered or sorted apart from them?"
        )
    return distinct and np.array_equal(a, b)


def _swapped_rows(name: str, instances: list[Instance]) -> list[str]:
    """Rows i and j look swapped when exchanging their boxes improves the fit of
    both together by a clear margin (0.2 IoU; correctly ordered yolo11n rows on
    COCO reach 0.07). Geometry cannot be sure: crowded human labels with few
    visible points reach 0.2 too, so this only warns."""
    fits = _fits(instances)
    n = len(instances)
    for i in range(n):
        for j in range(i + 1, n):
            here = fits[i, i] + fits[j, j]
            swapped = fits[i, j] + fits[j, i]
            if np.isfinite(here + swapped):
                clear = swapped > here + _SWAP_MARGIN
            else:  # a row too sparse to fit: judge by the other alone
                k, other = (i, j) if np.isfinite(fits[i, i]) else (j, i)
                clear = np.isfinite(fits[k, k]) and (
                    fits[k, other] > fits[k, k] + _SWAP_MARGIN
                )
            if clear:
                return [f"{name} rows {i}, {j}"]
    return []


_SWAP_MARGIN = 0.2


def _fits(instances: list[Instance]) -> np.ndarray:
    """IoU of each row's key point spread (rows) with each row's box (columns);
    NaN for a row with fewer than two visible points."""
    boxes = [inst.bbox for inst in instances]
    fits = np.full((len(instances), len(instances)), np.nan)
    for i, inst in enumerate(instances):
        points = inst.keypoints[inst.visible]
        if len(points) < 2:
            continue
        low, high = points.min(axis=0), points.max(axis=0)
        pad = max(1.0, 0.1 * float((high - low).max()))  # a line of points has no area
        spread = np.r_[low - pad, high + pad]
        fits[i] = [_iou(spread, box) for box in boxes]
    return fits


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
