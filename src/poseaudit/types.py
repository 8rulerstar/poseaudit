from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    """One posed object in one image, in pixel coordinates.

    `boxed` is False when the source gave no box and `bbox` is only the extent
    of the visible points; matching then compares keypoints instead.
    `class_id` is None when the source does not say.
    """

    bbox: np.ndarray  # (4,) x1, y1, x2, y2
    keypoints: np.ndarray  # (K, 2)
    visible: np.ndarray  # (K,) bool
    boxed: bool = True
    class_id: int | None = None

    def __post_init__(self) -> None:
        visible = np.asarray(self.visible)
        if visible.dtype != bool:
            raise ValueError(
                f"visible must be boolean, got {visible.dtype}; for COCO-style "
                "flags pass flags > 0"
            )
        if visible.shape != (len(self.keypoints),):
            raise ValueError(
                f"visible has shape {visible.shape}, keypoints "
                f"{np.shape(self.keypoints)}"
            )
        _check_finite(np.asarray(self.keypoints, float), visible)
        box = np.asarray(self.bbox, float)
        if box.shape != (4,) or box[2] < box[0] or box[3] < box[1]:
            raise ValueError(
                f"bbox must be x1, y1, x2, y2 with x1 <= x2, y1 <= y2: {box}"
            )

    @classmethod
    def from_keypoints(
        cls, keypoints: np.ndarray, visible: np.ndarray, class_id: int | None = None
    ) -> "Instance":
        keypoints = np.asarray(keypoints, float)
        visible = np.asarray(visible)
        if visible.dtype != bool:
            raise ValueError(
                f"visible must be boolean, got {visible.dtype}; for COCO-style "
                "flags pass flags > 0, for a visibility score e.g. score > 0.5"
            )
        _check_finite(keypoints, visible)
        points = keypoints[visible] if visible.any() else keypoints
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        box = np.array([x1, y1, x2, y2], float)
        return cls(box, keypoints, visible, boxed=False, class_id=class_id)


def _check_finite(keypoints: np.ndarray, visible: np.ndarray) -> None:
    if visible.shape != (len(keypoints),):
        return  # the shape check reports it
    bad = np.flatnonzero(visible & ~np.isfinite(keypoints).all(axis=-1))
    if len(bad):
        raise ValueError(
            f"keypoints {bad.tolist()} are visible but not finite (NaN or inf): "
            "mark points that were not found as not visible"
        )


# image name -> instances in that image
Dataset = dict[str, list[Instance]]
