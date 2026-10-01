"""Does the prediction put each instance on the same side of a decision
threshold as the truth? What a defect or pass/fail rule actually depends on.

A model that reads large values too small needs a lower threshold on its own
side; `predicted` sets it, and `matched_threshold` finds the one that flags as
many instances as the truth does.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from poseaudit.confidence import wilson

Side = Literal["above", "below", "outside"]


@dataclass(frozen=True)
class ThresholdAgreement:
    threshold: float  # applied to the truth
    predicted_threshold: float  # applied to the prediction
    side: Side
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    sensitivity: float  # flagged truths the prediction also flags
    sensitivity_ci: tuple[float, float]
    precision: float  # prediction flags that the truth also flags
    precision_ci: tuple[float, float]
    matched: bool = False  # predicted_threshold was chosen to match the count


def flagged(values: np.ndarray, threshold: float, side: Side) -> np.ndarray:
    """`outside` flags |value| >= threshold: a tilt either way."""
    if side == "above":
        return values >= threshold
    if side == "below":
        return values <= threshold
    return np.abs(values) >= threshold


def matched_threshold(
    truth: np.ndarray, predicted: np.ndarray, threshold: float, side: Side
) -> float:
    """The predicted threshold that flags as many instances as the truth does."""
    count = int(flagged(truth, threshold, side).sum())
    if count == 0:
        return float("nan")
    if side == "below":
        return float(np.sort(predicted)[count - 1])
    scores = np.abs(predicted) if side == "outside" else predicted
    return float(np.sort(scores)[::-1][count - 1])


def agreement(
    truth: np.ndarray,
    predicted: np.ndarray,
    threshold: float,
    side: Side,
    predicted_threshold: float | None = None,
    matched: bool = False,
) -> ThresholdAgreement:
    cut = threshold if predicted_threshold is None else predicted_threshold
    t, p = flagged(truth, threshold, side), flagged(predicted, cut, side)
    tp, fn = int((t & p).sum()), int((t & ~p).sum())
    fp, tn = int((~t & p).sum()), int((~t & ~p).sum())
    nan = float("nan")
    return ThresholdAgreement(
        threshold,
        cut,
        side,
        tp,
        fn,
        fp,
        tn,
        tp / (tp + fn) if tp + fn else nan,
        wilson(tp, tp + fn),
        tp / (tp + fp) if tp + fp else nan,
        wilson(tp, tp + fp),
        matched,
    )
