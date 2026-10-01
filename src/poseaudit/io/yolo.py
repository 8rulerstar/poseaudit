"""Ultralytics YOLO pose label files: one `<image>.txt` per image."""

import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from poseaudit.types import Dataset, Instance

ImageSize = tuple[int, int] | Callable[[str], tuple[int, int]]


def load_yolo(
    folder: str | Path,
    image_size: ImageSize,
    num_keypoints: int,
    classes: Iterable[int] | None = None,
    min_confidence: float = 0.0,
    per_point: int | None = None,
    min_score: float = 0.0,
) -> Dataset:
    """Read `class cx cy w h (x y [v])*K [score]` rows into pixel coordinates.

    `image_size` is required: the labels are normalised by width and height
    separately, so an angle measured on them is distorted on a non-square image.
    Pass one (width, height) or a function from image name to it.

    A point is visible when its third value exceeds `min_confidence`: 0 suits
    labels (v = 0, 1, 2), a threshold such as 0.5 suits predictions, whose
    third value is a confidence. `per_point` (2 or 3) settles rows whose length
    fits both layouts, which only happens with one keypoint.
    """
    size_of = image_size if callable(image_size) else (lambda _name: image_size)
    wanted = None if classes is None else {int(c) for c in classes}
    dataset: Dataset = {}
    looks_like_confidence = False
    seen: set[int] = set()
    if not Path(folder).is_dir():
        raise FileNotFoundError(f"{folder}: no such folder")
    rows = flagged = unscored = 0
    for path in sorted(Path(folder).glob("*.txt")):
        if path.name == "classes.txt":
            continue
        name = path.name[: -len(".txt")]  # not splitext: names may hold dots
        scale = np.array(size_of(name), float)
        instances = []
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            fields = line.split()
            if not fields:
                continue
            try:
                value = float(fields[0])
                label = int(value)
            except (ValueError, OverflowError):
                label, value = -1, float("nan")
            if label != value:
                raise ValueError(
                    f"{path}:{number}: class {fields[0]!r} is not a whole number"
                )
            seen.add(label)
            if wanted is not None and label not in wanted:
                continue
            try:
                instance, third, score = _instance(
                    fields, num_keypoints, scale, min_confidence, per_point
                )
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from None
            if score is not None and not score >= min_score:  # NaN is dropped
                unscored += score != score
                continue
            instances.append(instance)
            looks_like_confidence |= _fractional(third)
            rows += 1
            flagged += third is None and _reads_as_flags(fields[5:])
        dataset[name] = instances
    if looks_like_confidence and min_confidence == 0:
        warnings.warn(CONFIDENCE_WARNING.format(where=folder), stacklevel=2)
    if unscored:
        warnings.warn(
            f"{folder}: {unscored} rows with a NaN score were dropped", stacklevel=2
        )
    if per_point is None and rows and flagged == rows:
        warnings.warn(
            f"{folder}: every row read as {num_keypoints} points of x y also reads "
            f"as {len(fields[5:]) // 3} points of x y v with flags 0, 1, 2; is "
            "num_keypoints (--keypoints) right?",
            stacklevel=2,
        )
    if wanted is not None and seen and not seen & wanted:
        warnings.warn(
            f"{folder}: no row has class {sorted(wanted)}; the classes present are "
            f"{sorted(seen)}",
            stacklevel=2,
        )
    return dataset


CONFIDENCE_WARNING = (
    "{where}: point values look like confidences, not visibility flags, and "
    "min_confidence is 0, so every point counts as seen"
)


def _reads_as_flags(fields: list[str]) -> bool:
    """Keypoint values that fit x y v triples with integer flags 0, 1, 2."""
    values = np.array(fields, float)
    return len(values) % 3 == 0 and bool(np.isin(values[2::3], (0, 1, 2)).all())


def _fractional(values: np.ndarray | None) -> bool:
    return values is not None and bool(
        np.any((values > 0) & (values != np.round(values)))
    )


def _layout(values: int, k: int, per_point: int | None) -> int:
    fits = {n for n in (2, 3) if values in (n * k, n * k + 1)}
    if per_point is not None:
        if per_point not in fits:
            raise ValueError(f"{values} keypoint values do not fit {k} x {per_point}")
        return per_point
    if not fits:
        raise ValueError(
            f"{values} keypoint values fit neither {k} x 2 nor {k} x 3 "
            "(plus an optional score); is num_keypoints (--keypoints) right?"
        )
    if len(fits) == 2:
        raise ValueError(f"{values} values fit both layouts; pass per_point=2 or 3")
    return fits.pop()


def _instance(
    fields: list[str],
    k: int,
    scale: np.ndarray,
    min_confidence: float,
    per_point: int | None,
) -> tuple[Instance, np.ndarray | None, float | None]:
    values = np.array(fields[1:], float)
    if len(values) < 4:
        raise ValueError("a row needs a class, a box and keypoints")
    cx, cy, w, h = values[:4] * np.tile(scale, 2)
    rest = values[4:]
    n = _layout(len(rest), k, per_point)
    if not np.isfinite(values[: 4 + n * k]).all():
        raise ValueError("a value is not finite (NaN or inf)")
    points = rest[: n * k].reshape(k, n)
    normalised = np.r_[values[:2], points[:, :2].ravel()]
    if np.any((normalised < -0.05) | (normalised > 1.05)) or np.any(values[2:4] > 1.05):
        raise ValueError(
            "coordinates outside the image: YOLO values are fractions of the image "
            f"size; is num_keypoints ({k}) right?"
        )
    keypoints = points[:, :2] * scale
    third = points[:, 2] if n == 3 else None
    if third is not None:
        visible = third > min_confidence
    else:  # Ultralytics writes a missing point as 0 0 in this layout
        visible = (points[:, :2] != 0).any(axis=1)
    bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    class_id = int(float(fields[0]))
    score = float(rest[n * k]) if len(rest) > n * k else None
    return Instance(bbox, keypoints, visible, class_id=class_id), third, score
