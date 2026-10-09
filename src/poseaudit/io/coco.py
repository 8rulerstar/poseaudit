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


def _annotation_file(
    path: str | Path, keys: tuple[str, ...] = ("images", "annotations")
) -> dict:
    """The parsed annotation file, or a clear error when it is a results list:
    the two files swapped (`--gt` and `--pred`). Results need only its
    `images`: an image-info file without annotations names them as well."""
    data = _read(path)
    if isinstance(data, list):
        raise ValueError(
            f"{path}: this is a results file (a list of detections), not an "
            "annotation file (an object with images and annotations); were the "
            "truth and the predictions swapped (--gt and --pred)?"
        )
    for key in keys:
        if not isinstance(data, dict) or not isinstance(data.get(key), list):
            raise ValueError(f"{path}: not a COCO annotation file (no {key!r} list)")
    return data


def _field(record, key: str, where: str, hint: str = ""):
    """A required field, or an error naming the file, the entry and the field
    instead of a bare KeyError."""
    if not isinstance(record, dict):
        raise ValueError(f"{where}: each entry must be an object, got {record!r}")
    if key not in record:
        raise ValueError(f"{where} has no {key!r}{hint}")
    return record[key]


# a detector's results give boxes and no keypoints
NO_KEYPOINTS = (
    ": these look like a box detector's results; poseaudit needs a pose model's, "
    "with keypoints"
)


def load_coco(annotations: str | Path, classes: Iterable[int] | None = None) -> Dataset:
    """Crowd regions and people with no labelled keypoint are left out; a
    warning counts the crowd regions, since a prediction on one then counts
    as unmatched, or is paired with a labelled person its box overlaps.
    `classes` keeps only those category ids."""
    data = _annotation_file(annotations)
    wanted = _wanted(classes)
    seen: set = set()
    crowds = 0
    names = _names(data["images"], annotations)
    dataset: Dataset = {name: [] for name in names.values()}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            crowds += wanted is None or ann.get("category_id") in wanted
            continue
        if not _labelled(ann):
            continue
        seen.add(ann.get("category_id"))
        if wanted is not None and ann.get("category_id") not in wanted:
            continue
        where = f"{annotations}: annotation {ann.get('id')}"
        try:
            keypoints, visible, _ = _points(_field(ann, "keypoints", where), 0.0)
            bbox = _box(_field(ann, "bbox", where))
        except ValueError as error:
            raise ValueError(
                str(error) if str(error).startswith(where) else f"{where}: {error}"
            ) from None
        instance = Instance(bbox, keypoints, visible, class_id=ann.get("category_id"))
        image_id = _field(ann, "image_id", where)
        dataset[_name(names, image_id, annotations)].append(instance)
    if crowds:
        regions = (
            "1 crowd region (iscrowd 1) was left out and is"
            if crowds == 1
            else f"{crowds} crowd regions (iscrowd 1) were left out and are"
        )
        warnings.warn(
            f"{annotations}: {regions} in no count. A prediction on one counts as "
            "unmatched, unless its box overlaps a labelled person by --min-iou "
            "(min_iou): then it is paired with that person and read as their error.",
            stacklevel=2,
        )
    _nothing_kept(annotations, wanted, seen, ("labelled person", "labelled people"))
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
    names = _names(_annotation_file(annotations, ("images",))["images"], annotations)
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
    for index, result in enumerate(rows):
        if not isinstance(result, dict):
            raise ValueError(
                f"{results}: each result must be an object, got {result!r}"
            )
        where = f"{results}: result {index}"
        hint = NO_KEYPOINTS if "bbox" in result else ""
        _field(result, "keypoints", where, hint)
        _field(result, "image_id", where)
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
    _nothing_kept(results, wanted, seen, ("detection", "detections"))
    return dataset


def _names(images, annotations) -> dict:
    """Image id to file name. A repeated id or file name would silently put
    the people of two images into one, where they get paired across images."""
    names: dict = {}
    for index, image in enumerate(images):
        for key in ("id", "file_name"):
            _field(image, key, f"{annotations}: images[{index}]")
        if image["id"] in names:
            raise ValueError(f"{annotations}: image id {image['id']} is listed twice")
        names[image["id"]] = image["file_name"]
    seen: set = set()
    for name in names.values():
        if name in seen:
            raise ValueError(
                f"{annotations}: file name {name} is listed under two image ids"
            )
        seen.add(name)
    return names


def _name(names: dict, image_id, annotations) -> str:
    if image_id not in names:
        alike = [i for i in names if str(i) == str(image_id)]
        if alike:
            raise ValueError(
                f"image_id {image_id!r} is not listed in {annotations}, which "
                f"lists {alike[0]!r}: the two files write image ids as different "
                "types (a number and a string)"
            )
        raise ValueError(
            f"image_id {image_id!r} is not listed in {annotations}: were the "
            "predictions made on images of another annotation file or split? "
            "--gt must be the annotations of the images the model was run on"
        )
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


def _labelled(ann) -> bool:
    """num_keypoints is a count derived from the points; a stale 0 beside
    points flagged v > 0 must not drop a labelled person."""
    if ann.get("num_keypoints", 1) != 0:
        return True
    flags = (ann.get("keypoints") or [])[2::3]  # null for no points too
    return any(isinstance(v, (int, float)) and v > 0 for v in flags)


def _nothing_kept(where, wanted, seen, what: tuple[str, str]) -> None:
    """`what` names what was looked at, one and many: a crowd region of a
    category kept is counted in a warning of its own, which "no entry has
    that category" would contradict."""
    if wanted is not None and seen == {None}:
        warnings.warn(
            f"{where}: no entry has a category_id, so keeping categories "
            f"{sorted(wanted)} keeps nothing; leave the class filter out",
            stacklevel=3,
        )
    elif wanted is not None and seen and not seen & wanted:
        present = sorted(x for x in seen if x is not None)
        warnings.warn(
            f"{where}: no {what[0]} has category {sorted(wanted)}; the "
            f"{what[1]} have categories {present}",
            stacklevel=3,
        )
