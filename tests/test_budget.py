import json
from pathlib import Path

import numpy as np
import pytest

from poseaudit import angle, length
from poseaudit.budget import budget, fit_spread
from poseaudit.cli import main
from poseaudit.pairing import Pair, Pairing
from poseaudit.types import Instance

DEMO = Path(__file__).parents[1] / "examples" / "coco_elbow"


def limbs(n, rng, straight=False):
    """(n, 3, 2) arms: segments 10 to 200 px, bent 60 to 140 degrees or
    straight 160 to 180."""
    length_ = rng.uniform(10, 200, n)
    turn = np.radians(rng.uniform(160, 180, n) if straight else rng.uniform(60, 140, n))
    a = np.stack([length_, np.zeros(n)], axis=1)
    c = np.stack([length_ * np.cos(turn), length_ * np.sin(turn)], axis=1)
    return np.stack([a, np.zeros((n, 2)), c], axis=1) + rng.uniform(0, 500, (n, 1, 2))


def noisy(truth, fixed, share, rng, shared=0.0):
    first = np.hypot(*(truth[:, 0] - truth[:, 1]).T)
    second = np.hypot(*(truth[:, 2] - truth[:, 1]).T)
    size = np.stack([first, (first + second) / 2, second], axis=1)
    spread = np.hypot(fixed, share * size)[..., None]
    moved = rng.normal(size=(len(truth), 1, 2)) * shared  # the same for all three
    return truth + rng.normal(size=truth.shape) * spread + moved


def pairing(truth, predicted):
    box = np.array([-1e4, -1e4, 1e4, 1e4])
    seen = np.ones(3, bool)
    return Pairing(
        pairs=[
            Pair(f"{i}.jpg", Instance(box, t, seen), Instance(box, p, seen))
            for i, (t, p) in enumerate(zip(truth, predicted, strict=True))
        ]
    )


def test_fit_recovers_the_spread_it_was_made_with():
    rng = np.random.default_rng(1)
    truth = limbs(20000, rng)
    fixed, share = fit_spread(truth, noisy(truth, 2.0, 0.05, rng))
    assert fixed == pytest.approx(2.0, abs=0.25)
    assert share == pytest.approx(0.05, abs=0.004)


def test_a_shift_all_three_points_share_is_left_out():
    """Moving the whole limb does not change its angle, so it must not
    count as error."""
    rng = np.random.default_rng(2)
    truth = limbs(3000, rng)
    plain = fit_spread(truth, noisy(truth, 2.0, 0.05, np.random.default_rng(3)))
    shifted = fit_spread(
        truth, noisy(truth, 2.0, 0.05, np.random.default_rng(3), shared=30.0)
    )
    assert shifted == pytest.approx(plain, abs=0.02)


def test_an_honest_fit_passes_its_check_and_gives_a_size():
    rng = np.random.default_rng(4)
    truth = np.concatenate([limbs(400, rng), limbs(400, rng, straight=True)])
    result = budget(
        pairing(truth, noisy(truth, 1.0, 0.01, rng)), angle(0, 1, 2), 10, 0.2
    )
    assert [g.name for g in result.groups] == ["straight", "bent"]
    for group in result.groups:
        assert group.passed
        assert group.floor_rate < 0.2
        assert 5 <= group.smallest_limb_px <= 200


def test_no_size_when_even_a_huge_limb_misses_the_target():
    rng = np.random.default_rng(5)
    truth = limbs(400, rng)
    result = budget(pairing(truth, noisy(truth, 1.0, 0.2, rng)), angle(0, 1, 2), 10)
    bent = result.groups[1]
    assert bent.floor_rate > 0.2
    assert bent.smallest_limb_px is None
    assert "very large limb" in bent.note


def test_no_size_when_the_fit_fails_its_check():
    """Errors the model of the fit cannot describe: every other prediction
    has its wrist moved far off."""
    rng = np.random.default_rng(6)
    truth = limbs(400, rng)
    predicted = noisy(truth, 1.0, 0.01, rng)
    predicted[::2, 2] += 60
    bent = budget(pairing(truth, predicted), angle(0, 1, 2), 10).groups[1]
    assert not bent.passed
    assert bent.smallest_limb_px is None
    assert "failed" in bent.note


def test_a_small_group_is_not_fitted():
    rng = np.random.default_rng(7)
    truth = limbs(10, rng, straight=True)
    straight = budget(
        pairing(truth, noisy(truth, 1.0, 0.01, rng)), angle(0, 1, 2), 10
    ).groups[0]
    assert straight.n == 10
    assert np.isnan(straight.fixed_px)
    assert "too few" in straight.note


def test_only_an_angle_is_budgeted():
    with pytest.raises(ValueError, match="angle"):
        budget(Pairing(), length(0, 1), 10)


@pytest.mark.parametrize("target", [0, 1, 1.5])
def test_target_must_be_a_share(target):
    with pytest.raises(ValueError, match="target"):
        budget(Pairing(), angle(0, 1, 2), 10, target)


def test_cli_on_the_demo(tmp_path, capsys):
    out = tmp_path / "budget.json"
    main([
        "budget", "--format", "coco",
        "--gt", str(DEMO / "gt_200.json"),
        "--pred", str(DEMO / "pred_yolo11n.json"),
        "--angle", "5,7,9", "--big-error", "15", "--json", str(out),
    ])  # fmt: skip
    printed = capsys.readouterr().out
    assert "straight (from 150" in printed and "bent (below 150" in printed
    figures = json.loads(out.read_text(encoding="utf-8"))
    assert [g["name"] for g in figures["groups"]] == ["straight", "bent"]
    assert sum(g["n"] for g in figures["groups"]) == 322


def test_cli_takes_one_model(capsys):
    with pytest.raises(SystemExit, match="one model"):
        main([
            "budget", "--format", "coco",
            "--gt", str(DEMO / "gt_200.json"),
            "--pred", f"a={DEMO / 'pred_yolo11n.json'}",
            "--pred", f"b={DEMO / 'pred_yolo11n.json'}",
            "--angle", "5,7,9", "--big-error", "15",
        ])  # fmt: skip


def test_a_group_under_a_thousand_readings_is_flagged():
    rng = np.random.default_rng(8)
    truth = limbs(200, rng)
    result = budget(pairing(truth, noisy(truth, 1.0, 0.01, rng)), angle(0, 1, 2), 10)
    bent = result.groups[1]
    assert bent.few
    assert "few readings" in result.summary()


def test_limbs_all_one_length_skip_the_check_by_size():
    """With every limb one length the shorter half is empty: it used to put
    NaN in the checks and in the JSON, and fail the group for it."""
    rng = np.random.default_rng(9)
    # exact 50 px segments, bent between 37 and 143 degrees: 3-4-5 triangles
    ends = [(0, 50), (30, 40), (40, 30), (-30, 40), (-40, 30)]
    truth = np.array(
        [[(50, 0), (0, 0), ends[i % 5]] for i in range(200)], float
    ) + rng.integers(0, 500, (200, 1, 2))
    bent = budget(pairing(truth, noisy(truth, 1.0, 0.01, rng)), angle(0, 1, 2), 10)
    bent = bent.groups[1]
    assert [part for part, *_ in bent.checks] == ["all"]
    assert "by size" in bent.note
    result = budget(pairing(truth, noisy(truth, 1.0, 0.01, rng)), angle(0, 1, 2), 10)
    json.dumps(result.to_dict(), allow_nan=False)
    assert "nan" not in result.summary()


def test_a_keypoint_past_the_data_is_an_error():
    rng = np.random.default_rng(10)
    truth = limbs(50, rng)
    with pytest.raises(ValueError, match="keypoint 99"):
        budget(pairing(truth, truth), angle(0, 1, 99), 10)


def test_nothing_paired_is_an_error():
    with pytest.raises(ValueError, match="no prediction was paired"):
        budget(Pairing(), angle(0, 1, 2), 10)


@pytest.mark.parametrize("straight_from", [0, 180, -5])
def test_straight_from_must_split_the_angles(straight_from):
    rng = np.random.default_rng(11)
    truth = limbs(50, rng)
    with pytest.raises(ValueError, match="straight_from"):
        budget(pairing(truth, truth), angle(0, 1, 2), 10, straight_from=straight_from)


def test_a_size_at_the_bottom_of_the_search_reads_or_less():
    rng = np.random.default_rng(12)
    truth = limbs(400, rng)
    result = budget(pairing(truth, noisy(truth, 0.2, 0.0, rng)), angle(0, 1, 2), 30)
    bent = result.groups[1]
    assert bent.below_search
    assert "5 px or less" in result.summary()


def test_the_verdict_does_not_turn_on_the_seed():
    """One random half used to decide some groups by luck."""
    rng = np.random.default_rng(13)
    truth = limbs(300, rng)
    predicted = noisy(truth, 2.0, 0.09, rng)
    verdicts = {
        budget(pairing(truth, predicted), angle(0, 1, 2), 15, seed=s).groups[1].passed
        for s in range(8)
    }
    assert len(verdicts) == 1


def test_the_size_is_the_first_that_meets_the_target():
    from poseaudit.budget import _LIMBS, _limb, _rate

    rng = np.random.default_rng(14)
    truth = limbs(1000, rng)
    bent = budget(
        pairing(truth, noisy(truth, 1.5, 0.03, rng)), angle(0, 1, 2), 15, 0.2
    ).groups[1]
    size = bent.smallest_limb_px
    assert size is not None and size > _LIMBS[0]
    before = _LIMBS[np.searchsorted(_LIMBS, size) - 1]
    draws = np.random.default_rng(0).normal(size=(200_000, 3, 2))
    rate = lambda length: _rate(  # noqa: E731
        _limb(length, bent.typical_angle), bent.fixed_px, bent.share, 15, draws,
        angle(0, 1, 2),
    )  # fmt: skip
    assert rate(size) <= 0.2 + 0.005
    assert rate(before) > 0.2 - 0.005


def test_budget_help_does_not_offer_comparing_models(capsys):
    with pytest.raises(SystemExit):
        main(["budget", "--help"])
    assert "compare models" not in capsys.readouterr().out
