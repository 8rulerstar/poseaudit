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
        # float64 throughout: unsigned pixel coordinates would wrap on
        # subtraction, and float32 loses digits far from the origin
        keypoints = np.asarray(self.keypoints, float)
        _check_shapes(keypoints, visible)
        _check_finite(keypoints, visible)
        box = np.asarray(self.bbox, float)
        object.__setattr__(self, "keypoints", keypoints)
        object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "bbox", box)
        if box.shape == (4,) and not np.isfinite(box).all():
            # NaN fails every comparison below, and its IoU would quietly
            # leave the instance unmatched
            raise ValueError(f"bbox holds a non-finite value (NaN or inf): {box}")
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
        _check_shapes(keypoints, visible)
        _check_finite(keypoints, visible)
        points = keypoints[visible] if visible.any() else keypoints
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        box = np.array([x1, y1, x2, y2], float)
        return cls(box, keypoints, visible, boxed=False, class_id=class_id)


def _check_shapes(keypoints: np.ndarray, visible: np.ndarray) -> None:
    """Keypoints (K, 2) and one flag per keypoint. Other shapes would fail
    later with an unpacking error, or pass and read the wrong numbers."""
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        hint = (
            "; for x, y, v rows pass xy[:, :2]" if keypoints.shape[1:] == (3,) else ""
        )
        raise ValueError(
            "keypoints must have shape (K, 2), x and y of each point; got "
            f"{keypoints.shape}{hint}"
        )
    if visible.shape != (len(keypoints),):
        raise ValueError(
            f"visible must have shape (K,), one flag per keypoint: "
            f"({len(keypoints)},) for keypoints of shape {keypoints.shape}; got "
            f"{visible.shape}"
        )


def _check_finite(keypoints: np.ndarray, visible: np.ndarray) -> None:
    bad = np.flatnonzero(visible & ~np.isfinite(keypoints).all(axis=-1))
    if len(bad):
        raise ValueError(
            f"keypoints {bad.tolist()} are visible but not finite (NaN or inf): "
            "mark points that were not found as not visible"
        )


# image name -> instances in that image
Dataset = dict[str, list[Instance]]
