"""matching="optimal": the assignment with the highest summed rank."""

import itertools
import json

import numpy as np
import pytest

from poseaudit import pair
from poseaudit.cli import main
from poseaudit.pairing import _iou, linear_sum_assignment
from poseaudit.types import Instance


def boxed(*box: float) -> Instance:
    """A box with no visible point, so its rank is its IoU alone."""
    return Instance(np.array(box, float), np.zeros((2, 2)), np.zeros(2, bool))


def crowd():
    """T0 overlaps P0 by 0.9 and P1 by 0.8, T1 overlaps P0 by 0.8 and P1 by
    0.7: greedy gives P0 to T0 and, at min_iou 0.75, leaves T1 alone."""
    truth = {"a": [boxed(0, 0, 10, 10), boxed(0, 1, 10, 10)]}
    predicted = {"a": [boxed(0, 0, 10, 9), boxed(0, 0, 10, 8)]}
    return truth, predicted


def total(pairing) -> float:
    """The summed IoU, which is the rank here."""
    return sum(_iou(p.truth.bbox, p.predicted.bbox) for p in pairing.pairs)


def test_optimal_beats_greedy_where_greedy_is_not_optimal() -> None:
    truth, predicted = crowd()
    greedy = pair(truth, predicted, min_iou=0.75)
    best = pair(truth, predicted, min_iou=0.75, matching="optimal")
    assert [(p.truth_index, p.predicted_index) for p in greedy.pairs] == [(0, 0)]
    assert sorted((p.truth_index, p.predicted_index) for p in best.pairs) == [
        (0, 1),
        (1, 0),
    ]
    assert total(best) == pytest.approx(1.6) and total(greedy) == pytest.approx(0.9)
    assert best.missed == [] and len(greedy.missed) == 1
    assert best.matching == "optimal" and best.changed_from_greedy == 2
    assert greedy.matching == "greedy" and greedy.changed_from_greedy == 0


def test_optimal_never_pairs_below_the_threshold() -> None:
    truth, predicted = crowd()
    none = pair(truth, predicted, min_iou=0.95, matching="optimal")
    assert none.pairs == [] and len(none.missed) == 2 and len(none.extra) == 2
    assert len(none.near_misses) == 2
    one = pair(truth, predicted, min_iou=0.85, matching="optimal")
    assert [(p.truth_index, p.predicted_index) for p in one.pairs] == [(0, 0)]
    assert one.changed_from_greedy == 0


def test_optimal_agrees_with_greedy_when_greedy_is_optimal() -> None:
    truth, predicted = crowd()  # at the default 0.3, greedy pairs both
    best = pair(truth, predicted, matching="optimal")
    assert total(pair(truth, predicted)) == pytest.approx(total(best))


def test_an_unknown_matching_is_refused() -> None:
    truth, predicted = crowd()
    with pytest.raises(ValueError, match="matching must be"):
        pair(truth, predicted, matching="hungarian")


def _brute(cost: np.ndarray) -> float:
    n, m = cost.shape
    if n > m:
        return _brute(cost.T)
    return min(
        sum(cost[i, c] for i, c in enumerate(cols))
        for cols in itertools.permutations(range(m), n)
    )


def test_the_assignment_is_least_cost_on_random_matrices() -> None:
    rng = np.random.default_rng(0)
    for k in range(400):
        n, m = rng.integers(1, 6, 2)
        # ties on half of them: small integers repeat
        cost = rng.integers(0, 4, (n, m)) if k % 2 else rng.random((n, m))
        cost = cost.astype(float)
        rows, cols = linear_sum_assignment(cost)
        assert len(rows) == min(n, m) == len(set(rows)) == len(set(cols))
        assert list(rows) == sorted(rows)
        assert cost[rows, cols].sum() == pytest.approx(_brute(cost))
        again = linear_sum_assignment(cost)
        assert np.array_equal(again[0], rows) and np.array_equal(again[1], cols)
        high = linear_sum_assignment(cost, maximize=True)
        assert cost[high].sum() == pytest.approx(-_brute(-cost))


def test_the_assignment_matches_scipy() -> None:
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(1)
    for k in range(300):
        n, m = rng.integers(1, 12, 2)
        cost = rng.integers(0, 5, (n, m)) if k % 3 == 0 else rng.normal(size=(n, m))
        for maximize in (False, True):
            ours = linear_sum_assignment(cost, maximize=maximize)
            theirs = scipy.linear_sum_assignment(cost, maximize=maximize)
            assert cost[ours].sum() == pytest.approx(cost[theirs].sum())


def test_cli_matching_flag(tmp_path, capsys) -> None:
    gt = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": i + 1, "image_id": 1, "category_id": 1, "bbox": box,
             "keypoints": [box[0] + 1, box[1] + 1, 2, box[0] + 5, box[1] + 1, 2]}
            for i, box in enumerate(([0, 0, 10, 10], [0, 1, 10, 9]))
        ],
    }  # fmt: skip
    pred = [
        {"image_id": 1, "category_id": 1, "bbox": box, "score": 0.9,
         "keypoints": [box[0] + 1, box[1] + 1, 2, box[0] + 5, box[1] + 1, 2]}
        for box in ([0, 0, 10, 9], [0, 0, 10, 8])
    ]  # fmt: skip
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "pred.json").write_text(json.dumps(pred))
    base = ["audit", "--format", "coco",
            "--gt", str(tmp_path / "gt.json"), "--pred", str(tmp_path / "pred.json"),
            "--tilt", "0,1", "--big-error", "5", "--min-iou", "0.75",
            "--resamples", "50", "--jitter-repeats", "0"]  # fmt: skip
    main(base + ["--json", str(tmp_path / "g.json")])
    greedy = capsys.readouterr().out
    assert "read 1 of 2" in greedy and "matching     " not in greedy
    assert "matching" not in json.loads((tmp_path / "g.json").read_text())["settings"]
    main(base + ["--matching", "optimal", "--json", str(tmp_path / "o.json")])
    out = capsys.readouterr().out
    assert "read 2 of 2" in out
    assert "matching     optimal, 2 truths paired differently than greedy" in out
    settings = json.loads((tmp_path / "o.json").read_text())["settings"]
    assert settings["matching"] == "optimal" and settings["matching_changed"] == 2
    with pytest.raises(SystemExit):
        main(base + ["--matching", "best"])


def test_optimal_maximises_the_sum_not_the_number_of_pairs():
    """One strong pair outweighs two weak ones, leaving a truth unpaired."""
    from poseaudit.pairing import _optimal

    candidates = [(0.9, 0.9, 0.9, 0, 0), (0.1, 0.1, 0.1, 0, 1), (0.3, 0.3, 0.3, 1, 0)]
    assert _optimal(candidates, 2, 2) == [(0, 0)]
