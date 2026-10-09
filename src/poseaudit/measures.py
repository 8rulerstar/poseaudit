"""What is read off the keypoints, and how two readings are compared.

Every `read` returns NaN for geometry it cannot measure (two points on one
pixel), so such an instance is counted as unreadable rather than scored.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Measure:
    name: str
    points: tuple[int, ...]
    unit: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(i, (int, np.integer)) and not isinstance(i, bool)
            for i in self.points
        ):
            raise TypeError(
                f"{self.name}: keypoints are given by index, counted from 0, not "
                f"by name: got {self.points!r}. With a list of names, pass "
                "names.index('left_elbow')."
            )
        if any(i < 0 for i in self.points):
            raise ValueError(f"{self.name}: keypoint indices must be 0 or more")
        if self._repeats():
            raise ValueError(
                f"{self.name} {self.points}: a keypoint is repeated, which leaves "
                "nothing to measure"
            )

    def _repeats(self) -> bool:
        return len(set(self.points)) < len(self.points)

    def read(self, keypoints: np.ndarray) -> float:
        raise NotImplementedError

    def read_many(self, keypoints: np.ndarray) -> np.ndarray:
        """`read` over (n, K, 2) keypoints at once."""
        return np.array([self.read(k) for k in keypoints])

    def shift(self, value: float, by: float) -> float:
        """`value` measured from `by` instead of from zero."""
        return value - by

    def difference(self, predicted: float, truth: float) -> float:
        return predicted - truth

    def middle(self, truth: float, predicted: float) -> float:
        """The value halfway between the two readings."""
        return truth + self.difference(predicted, truth) / 2


@dataclass(frozen=True)
class Tilt(Measure):
    """Degrees of the axis through a and b from vertical, in [-90, 90).

    Positive when the upper end lies to the right of the lower end, as seen
    in the image. The axis has no direction: listing a and b the other way
    round reads the same.
    """

    def read(self, keypoints: np.ndarray) -> float:
        a, b = (keypoints[i] for i in self.points)
        dx, dy = b - a
        if dx == 0 and dy == 0:
            return float("nan")
        # image y grows downward, so -dy is "up"
        return _fold(float(np.degrees(np.arctan2(dx, -dy))))

    def read_many(self, keypoints: np.ndarray) -> np.ndarray:
        a, b = (keypoints[:, i] for i in self.points)
        dx, dy = (b - a).T
        out = (np.degrees(np.arctan2(dx, -dy)) + 90.0) % 180.0 - 90.0
        return np.where((dx == 0) & (dy == 0), np.nan, out)

    def difference(self, predicted: float, truth: float) -> float:
        return _fold(predicted - truth)

    def middle(self, truth: float, predicted: float) -> float:
        return _fold(super().middle(truth, predicted))

    def shift(self, value: float, by: float) -> float:
        return _fold(value - by)


@dataclass(frozen=True)
class Angle(Measure):
    """Interior angle at b between b->a and b->c, 0 to 180 degrees."""

    def read(self, keypoints: np.ndarray) -> float:
        a, b, c = (keypoints[i] for i in self.points)
        u, v = a - b, c - b
        if not (u.any() and v.any()):
            return float("nan")
        # the angle does not depend on length: scaling keeps huge values finite
        u, v = u / np.abs(u).max(), v / np.abs(v).max()
        # atan2 stays exact near 0 and 180 degrees, where arccos loses digits
        cross = u[0] * v[1] - u[1] * v[0]
        return float(np.degrees(np.arctan2(abs(cross), np.dot(u, v))))

    def read_many(self, keypoints: np.ndarray) -> np.ndarray:
        a, b, c = (keypoints[:, i] for i in self.points)
        u, v = a - b, c - b
        empty = ~u.any(axis=1) | ~v.any(axis=1)
        # the angle does not depend on length: scaling keeps huge values finite
        with np.errstate(invalid="ignore", divide="ignore"):
            u = u / np.abs(u).max(axis=1, keepdims=True)
            v = v / np.abs(v).max(axis=1, keepdims=True)
        cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        out = np.degrees(np.arctan2(np.abs(cross), (u * v).sum(axis=1)))
        return np.where(empty, np.nan, out)


@dataclass(frozen=True)
class Length(Measure):
    """|a-b| in pixels; NaN when a and b coincide, as for the other measures:
    two labels on one pixel say the part was not resolved, not that it has
    no length."""

    def read(self, keypoints: np.ndarray) -> float:
        a, b = (keypoints[i] for i in self.points)
        value = float(np.linalg.norm(b - a))
        return value if value > 0 else float("nan")


@dataclass(frozen=True)
class Ratio(Measure):
    """|a-b| / |c-d|: scale-free, so a far and a near object compare."""

    def _repeats(self) -> bool:
        # two segments may share a point (upper arm over forearm), not be one
        a, b, c, d = self.points
        return a == b or c == d or {a, b} == {c, d}

    def read(self, keypoints: np.ndarray) -> float:
        a, b, c, d = (keypoints[i] for i in self.points)
        denominator = np.linalg.norm(d - c)
        if denominator == 0:
            return float("nan")
        return float(np.linalg.norm(b - a) / denominator)


def _fold(degrees: float) -> float:
    return (degrees + 90.0) % 180.0 - 90.0


def tilt(a: int, b: int, name: str = "tilt") -> Tilt:
    return Tilt(name, (a, b), "deg")


def angle(a: int, b: int, c: int, name: str = "angle") -> Angle:
    return Angle(name, (a, b, c), "deg")


def length(a: int, b: int, name: str = "length") -> Length:
    return Length(name, (a, b), "px")


def ratio(a: int, b: int, c: int, d: int, name: str = "ratio") -> Ratio:
    return Ratio(name, (a, b, c, d), "")
