"""COCO keypoint annotations, and detection results in the COCO results format."""

import json
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from poseaudit.io.yolo import CONFIDENCE_WARNING, _fractional
from poseaudit.types import Dataset, Instance


def _read(path: str | Path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"{path}: the file is empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error})") from None


def _annotation_file(path: str | Path) -> dict:
    """The parsed annotation file, or a clear error when it is a results list:
    the two files swapped (`--gt` and `--pred`)."""
    data = _read(path)
    if isinstance(data, list):
        raise ValueError(
            f"{path}: this is a results file (a list of detections), not an "
            "annotation file (an object with images and annotations); were the "
            "truth and the predictions swapped (--gt and --pred)?"
        )
    if not isinstance(data, dict) or "images" not in data:
        raise ValueError(f"{path}: not a COCO annotation file (no 'images' list)")
    return data


def load_coco(annotations: str | Path, classes: Iterable[int] | None = None) -> Dataset:
    """Crowd regions and people with no labelled keypoint are left out.
    `classes` keeps only those category ids."""
    data = _annotation_file(annotations)
    wanted = _wanted(classes)
    seen: set = set()
    names = {image["id"]: image["file_name"] for image in data["images"]}
    dataset: Dataset = {name: [] for name in names.values()}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) or ann.get("num_keypoints", 1) == 0:
            continue
        seen.add(ann.get("category_id"))
        if wanted is not None and ann.get("category_id") not in wanted:
            continue
        try:
            keypoints, visible, _ = _points(ann["keypoints"], 0.0)
            bbox = _box(ann["bbox"])
        except ValueError as error:
            raise ValueError(
                f"{annotations}: annotation {ann.get('id')}: {error}"
            ) from None
        instance = Instance(bbox, keypoints, visible, class_id=ann.get("category_id"))
        dataset[_name(names, ann["image_id"], annotations)].append(instance)
    _nothing_kept(annotations, wanted, seen)
    return dataset


def load_coco_results(
    results: str | Path,
    annotations: str | Path,
    min_confidence: float = 0.0,
    classes: Iterable[int] | None = None,
    min_score: float = 0.0,
) -> Dataset:
    """Results name images by id only, so the annotation file maps ids to names.

    Matching ignores detection scores, so `min_score` drops detections scored
    below it first: a low-scoring stray box that overlaps a person better
    would otherwise take the pair.

    The third value of each point is a confidence in most results files, not a
    visibility flag: set `min_confidence` (for example 0.5), or every point a
    model returns counts as seen.
    """
    images = _annotation_file(annotations)["images"]
    names = {image["id"]: image["file_name"] for image in images}
    dataset: Dataset = {}
    looks_like_confidence = False
    wanted = _wanted(classes)
    seen: set = set()
    rows = _read(results)
    if isinstance(rows, dict):
        raise ValueError(
            f"{results}: this is an annotation file, not a results file (a list "
            "of detections); were the truth and the predictions swapped (--gt "
            "and --pred)? Otherwise pass it with --gt-format coco, or give the "
            "model's results here"
        )
    for result in rows:
        if not isinstance(result, dict):
            raise ValueError(
                f"{results}: each result must be an object, got {result!r}"
            )
        seen.add(result.get("category_id"))
        if wanted is not None and result.get("category_id") not in wanted:
            continue
        if not result.get("score", 1.0) >= min_score:  # a NaN score is dropped
            continue
        name = _name(names, result["image_id"], annotations)
        try:
            keypoints, visible, third = _points(result["keypoints"], min_confidence)
            bbox = _box(result["bbox"]) if "bbox" in result else None
        except ValueError as error:
            raise ValueError(
                f"{results}: result for image {result['image_id']}: {error}"
            ) from None
        looks_like_confidence |= _fractional(third)
        class_id = result.get("category_id")
        if bbox is not None:
            instance = Instance(bbox, keypoints, visible, class_id=class_id)
        else:
            instance = Instance.from_keypoints(keypoints, visible, class_id)
        dataset.setdefault(name, []).append(instance)
    if looks_like_confidence and min_confidence == 0:
        warnings.warn(CONFIDENCE_WARNING.format(where=results), stacklevel=2)
    _nothing_kept(results, wanted, seen)
    return dataset


def _name(names: dict, image_id, annotations) -> str:
    if image_id not in names:
        raise ValueError(f"image_id {image_id} is not listed in {annotations}")
    return names[image_id]


def _points(flat: list[float], min_confidence: float):
    values = np.asarray(flat, float)
    if values.ndim != 1 or len(values) % 3:
        raise ValueError(
            "keypoints must be a flat list of x, y, v triples; got "
            f"{values.size} values"
        )
    points = values.reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("keypoints hold a non-finite value (NaN or Infinity)")
    return points[:, :2], points[:, 2] > min_confidence, points[:, 2]


def _box(values) -> np.ndarray:
    if len(values) != 4:
        raise ValueError(f"bbox must be x, y, width, height; got {len(values)} values")
    x, y, w, h = (float(v) for v in values)
    box = np.array([x, y, x + w, y + h])
    if not np.isfinite(box).all():
        raise ValueError("bbox holds a non-finite value (NaN or Infinity)")
    return box


def _wanted(classes) -> set[int] | None:
    return None if classes is None else {int(c) for c in classes}


def _nothing_kept(where, wanted, seen) -> None:
    if wanted is not None and seen and not seen & wanted:
        present = sorted(x for x in seen if x is not None)
        warnings.warn(
            f"{where}: no entry has category {sorted(wanted)}; the categories "
            f"present are {present}",
            stacklevel=3,
        )
