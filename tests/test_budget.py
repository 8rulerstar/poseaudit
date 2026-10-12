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


def group_rate(truth, length, fixed, share, big_error, seed=0, draws=4000):
    """Written apart from the module: the large-error rate of these limbs,
    each scaled to a mean segment of `length` and read `draws` times."""
    rng = np.random.default_rng(seed)
    rates = []
    for limb in truth - truth[:, 1:2]:
        first, second = (np.hypot(*(limb[i] - limb[1])) for i in (0, 2))
        limb = limb * length / ((first + second) / 2)
        sizes = (
            np.array([first, (first + second) / 2, second])
            * length
            / ((first + second) / 2)
        )
        spread = np.sqrt(fixed**2 + (share * sizes) ** 2)
        drawn = limb + rng.normal(size=(draws, 3, 2)) * spread[None, :, None]
        u, v = drawn[:, 0] - drawn[:, 1], drawn[:, 2] - drawn[:, 1]
        read = np.degrees(
            np.arccos(np.clip((u * v).sum(1) / np.hypot(*u.T) / np.hypot(*v.T), -1, 1))
        )
        a, c = limb[0] - limb[1], limb[2] - limb[1]
        true = np.degrees(np.arccos(a @ c / np.hypot(*a) / np.hypot(*c)))
        rates.append(np.mean(np.abs(read - true) >= big_error))
    return float(np.mean(rates))


@pytest.mark.parametrize("straight", [False, True])
def test_the_size_meets_the_target_on_the_groups_own_limbs(straight):
    """The size is judged on limbs bent like the group's: one median angle
    gave straight groups sizes that missed the target (28% for 20%)."""
    from poseaudit.budget import _LIMBS

    rng = np.random.default_rng(14)
    truth = limbs(600, rng, straight=straight)
    result = budget(pairing(truth, noisy(truth, 1.5, 0.03, rng)), angle(0, 1, 2), 15)
    group = result.groups[0 if straight else 1]
    size = group.smallest_limb_px
    assert size is not None and size > _LIMBS[0]
    before = _LIMBS[np.searchsorted(_LIMBS, size) - 1]
    shapes = truth[rng.choice(len(truth), 200, replace=False)]
    args = (group.fixed_px, group.share, 15)
    assert group_rate(shapes, size, *args) <= 0.2 + 0.015
    assert group_rate(shapes, before, *args) > 0.2 - 0.015


def test_flags_say_when_a_size_is_outside_the_data_or_near_the_floor():
    rng = np.random.default_rng(15)
    truth = limbs(600, rng)
    near = budget(pairing(truth, noisy(truth, 0.5, 0.09, rng)), angle(0, 1, 2), 15)
    bent = near.groups[1]
    assert bent.smallest_limb_px is not None
    assert bent.near_floor == (bent.floor_rate > 0.15)
    assert bent.near_floor
    short = truth * 0.5  # segments of 5 to 100 px, a size of about 200 needed
    far = budget(pairing(short, noisy(short, 18.0, 0.0, rng)), angle(0, 1, 2), 15)
    bent = far.groups[1]
    assert bent.smallest_limb_px > np.hypot(*(short[:, 0] - short[:, 1]).T).max()
    assert bent.outside_data
    assert "extrapolation" in far.summary()


def test_no_readable_pair_is_an_error():
    rng = np.random.default_rng(16)
    truth = limbs(50, rng)
    seen = np.array([True, False, True])
    box = np.array([-1e4, -1e4, 1e4, 1e4])
    hidden = Pairing(
        pairs=[Pair("a.jpg", Instance(box, t, seen), Instance(box, t, seen))
               for t in truth]
    )  # fmt: skip
    with pytest.raises(ValueError, match="no pair shows all three"):
        budget(hidden, angle(0, 1, 2), 10)


def test_budget_help_does_not_offer_comparing_models(capsys):
    with pytest.raises(SystemExit):
        main(["budget", "--help"])
    assert "compare models" not in capsys.readouterr().out


def test_frames_of_one_subject_stay_together_and_are_counted_once(monkeypatch):
    """A clip copied ten times used to lose the few-readings warning and
    narrow the check as if it were ten times the people."""
    import poseaudit.budget as module

    rng = np.random.default_rng(17)
    truth = limbs(200, rng)
    predicted = noisy(truth, 3.0, 0.12, rng)
    box, seen = np.array([-1e4, -1e4, 1e4, 1e4]), np.ones(3, bool)
    frames = Pairing(
        pairs=[
            Pair(f"s{i}_{copy}.jpg", Instance(box, t, seen), Instance(box, p, seen))
            for copy in range(10)
            for i, (t, p) in enumerate(zip(truth, predicted, strict=True))
        ]
    )
    fitted = []
    real_fit = module.fit_spread

    def recording_fit(t, p):
        fitted.append({tuple(np.round(x.ravel(), 6)) for x in t})
        return real_fit(t, p)

    real_rate = module.large_rate

    def checking_rate(t, *args, **kwargs):
        # every limb predicted is one the latest fit did not see
        assert not fitted[-1] & {tuple(np.round(x.ravel(), 6)) for x in t}
        return real_rate(t, *args, **kwargs)

    monkeypatch.setattr(module, "fit_spread", recording_fit)
    monkeypatch.setattr(module, "large_rate", checking_rate)
    result = budget(frames, angle(0, 1, 2), 15, cluster=lambda name: name.split("_")[0])
    bent = result.groups[1]
    assert bent.n == 2000 and bent.clusters == 200
    assert bent.few
    rate = bent.read_rate  # the check's width is that of 200 subjects, not 2,000
    assert bent.tolerance == pytest.approx(
        max(0.06, 2 * np.sqrt(rate * (1 - rate) / 200))
    )
    assert bent.tolerance > 0.06
    assert "in 200 clusters" in result.summary()


def test_a_fit_never_predicts_what_it_was_fitted_on(monkeypatch):
    import poseaudit.budget as module

    rng = np.random.default_rng(18)
    truth = limbs(300, rng)
    fitted = []
    real_fit = module.fit_spread
    real_rate = module.large_rate

    def recording_fit(t, p):
        fitted.append({tuple(x.ravel()) for x in t})
        return real_fit(t, p)

    def checking_rate(t, *args, **kwargs):
        assert not fitted[-1] & {tuple(x.ravel()) for x in t}
        return real_rate(t, *args, **kwargs)

    monkeypatch.setattr(module, "fit_spread", recording_fit)
    monkeypatch.setattr(module, "large_rate", checking_rate)
    budget(pairing(truth, noisy(truth, 1.0, 0.05, rng)), angle(0, 1, 2), 15)
    assert len(fitted) >= 2 * 5  # two halves, five splits, in each group


def test_the_tolerance_is_two_standard_errors_and_at_least_six_points():
    from poseaudit.budget import _tolerance

    large = np.array([True] * 30 + [False] * 70)
    assert _tolerance(large) == pytest.approx(2 * np.sqrt(0.3 * 0.7 / 100))
    assert _tolerance(large, 25) == pytest.approx(2 * np.sqrt(0.3 * 0.7 / 25))
    assert _tolerance(np.array([True] * 30 + [False] * 970)) == 0.06


def test_a_target_out_of_reach_gives_no_size(monkeypatch):
    """The search stops at its largest size and says so, rather than give it."""
    import poseaudit.budget as module

    monkeypatch.setattr(module, "_LIMBS", np.arange(5.0, 11.0))
    rng = np.random.default_rng(19)
    truth = limbs(400, rng)
    bent = budget(
        pairing(truth, noisy(truth, 3.0, 0.0, rng)), angle(0, 1, 2), 15
    ).groups[1]
    assert bent.passed and bent.floor_rate < 0.2
    assert bent.smallest_limb_px is None
    assert "not reached by 10 px" in bent.note


def test_redraw_reports_how_the_answer_moves_with_the_data():
    rng = np.random.default_rng(20)
    truth = limbs(400, rng)
    data = pairing(truth, noisy(truth, 1.5, 0.03, rng))
    result = budget(data, angle(0, 1, 2), 15, redraw=6)
    bent = result.groups[1]
    assert result.redraws == 6
    assert 0 <= bent.redraw_passed <= 6
    assert len(bent.redraw_sizes) <= bent.redraw_passed
    assert "data drawn again 6 times" in result.summary()
    json.dumps(result.to_dict(), allow_nan=False)
    # the first answer is the one without redrawing
    plain = budget(data, angle(0, 1, 2), 15)
    assert bent.smallest_limb_px == plain.groups[1].smallest_limb_px
    assert plain.groups[1].redraw_passed is None


def test_redraw_draws_whole_clusters():
    """A subject drawn again brings all its frames, and a subject drawn twice
    counts as two."""
    rng = np.random.default_rng(21)
    truth = limbs(100, rng)
    predicted = noisy(truth, 1.5, 0.05, rng)
    box, seen = np.array([-1e4, -1e4, 1e4, 1e4]), np.ones(3, bool)
    frames = Pairing(
        pairs=[
            Pair(f"s{i}_{copy}.jpg", Instance(box, t, seen), Instance(box, p, seen))
            for copy in range(3)
            for i, (t, p) in enumerate(zip(truth, predicted, strict=True))
        ]
    )
    result = budget(
        frames, angle(0, 1, 2), 15, cluster=lambda n: n.split("_")[0], redraw=3
    )
    assert result.groups[1].redraw_passed is not None
    assert result.groups[1].clusters == result.groups[1].n // 3


def test_redraw_must_not_be_negative():
    rng = np.random.default_rng(22)
    truth = limbs(50, rng)
    with pytest.raises(ValueError, match="redraw"):
        budget(pairing(truth, truth), angle(0, 1, 2), 10, redraw=-1)


def test_the_classes_are_importable_and_the_module_stays_a_module():
    """0.4.0 code that imports poseaudit.budget and reads its names works."""
    import types

    import poseaudit
    import poseaudit.budget

    assert isinstance(poseaudit.budget, types.ModuleType)
    assert poseaudit.budget.budget is budget
    assert poseaudit.BudgetResult is poseaudit.budget.BudgetResult
    assert poseaudit.budget.Group is poseaudit.BudgetGroup


def test_redraw_never_fits_on_a_copy_of_what_it_predicts(monkeypatch):
    """An image drawn twice: its copies are split as one."""
    import poseaudit.budget as module

    rng = np.random.default_rng(23)
    truth = limbs(200, rng)
    fitted, real_fit, real_rate = [], module.fit_spread, module.large_rate

    def recording_fit(t, p):
        fitted.append({tuple(x.ravel()) for x in t})
        return real_fit(t, p)

    def checking_rate(t, *args, **kwargs):
        assert not fitted[-1] & {tuple(x.ravel()) for x in t}
        return real_rate(t, *args, **kwargs)

    monkeypatch.setattr(module, "fit_spread", recording_fit)
    monkeypatch.setattr(module, "large_rate", checking_rate)
    budget(pairing(truth, noisy(truth, 1.0, 0.05, rng)), angle(0, 1, 2), 15, redraw=3)


def test_one_cluster_has_no_check_and_no_size():
    rng = np.random.default_rng(24)
    truth = limbs(200, rng)
    result = budget(
        pairing(truth, noisy(truth, 1.0, 0.02, rng)),
        angle(0, 1, 2),
        15,
        cluster=lambda name: "everyone",
        redraw=2,
    )
    bent = result.groups[1]
    assert not bent.passed and bent.smallest_limb_px is None
    assert "from one" in bent.note
    assert bent.redraw_passed == 0


def test_redraw_counts_draws_that_pass_fail_or_cannot_fit():
    rng = np.random.default_rng(25)
    truth = limbs(400, rng)
    good = budget(
        pairing(truth, noisy(truth, 1.5, 0.03, rng)), angle(0, 1, 2), 15, redraw=5
    ).groups[1]
    assert good.passed and good.redraw_passed >= 3  # a right fit mostly passes
    assert len(good.redraw_sizes) <= good.redraw_passed
    assert len(set(good.redraw_sizes)) > 1  # the draws differ
    predicted = noisy(truth, 1.0, 0.01, rng)
    predicted[::2, 2] += 60  # errors the fit cannot describe
    bad = budget(pairing(truth, predicted), angle(0, 1, 2), 10, redraw=4).groups[1]
    assert bad.redraw_passed == 0 and bad.redraw_sizes == []
    few = budget(
        pairing(truth[:35], noisy(truth[:35], 1.0, 0.02, rng)),
        angle(0, 1, 2),
        15,
        redraw=6,
    )
    assert few.groups[1].redraw_unfitted is not None
    text = few.summary()
    assert "data drawn again 6 times" in text
    keys = few.to_dict()
    assert keys["redraws"] == 6 and "clustered" in keys
    assert {"redraw_passed", "redraw_unfitted", "redraw_sizes"} <= set(
        keys["groups"][1]
    )


def test_cli_passes_redraw(tmp_path):
    out = tmp_path / "budget.json"
    main([
        "budget", "--format", "coco",
        "--gt", str(DEMO / "gt_200.json"),
        "--pred", str(DEMO / "pred_yolo11n.json"),
        "--angle", "5,7,9", "--big-error", "15", "--redraw", "2",
        "--json", str(out),
    ])  # fmt: skip
    figures = json.loads(out.read_text(encoding="utf-8"))
    assert figures["redraws"] == 2
    assert all(g["redraw_passed"] is not None for g in figures["groups"])


def test_a_handful_of_clusters_is_too_wide_to_check():
    """Five clusters made the check pass a fit within about 37 points."""
    rng = np.random.default_rng(26)
    truth = limbs(300, rng)
    predicted = noisy(truth, 3.0, 0.12, rng)
    result = budget(
        pairing(truth, predicted),
        angle(0, 1, 2),
        15,
        cluster=lambda n: str(int(n[:-4]) % 5),
    )
    bent = result.groups[1]
    assert bent.clusters == 5
    assert not bent.passed and bent.smallest_limb_px is None
    assert "too few clusters to check" in bent.note
    text = result.summary()
    assert "no check, so no size" in text
    assert "predicted vs read large:  -" not in text and "nan" not in text


def test_redraw_counts_draws_too_small_to_fit():
    """Thirty bent arms and one straight one: draws that leave the straight
    group under thirty readings are counted apart from failed checks."""
    rng = np.random.default_rng(27)
    truth = np.concatenate([limbs(31, rng, straight=True), limbs(400, rng)])
    result = budget(
        pairing(truth, noisy(truth, 1.0, 0.02, rng)), angle(0, 1, 2), 15, redraw=8
    )
    straight = result.groups[0]
    assert straight.redraw_unfitted >= 1
    assert straight.redraw_unfitted + straight.redraw_passed <= 8
    assert "had too few readings to fit" in result.summary()


@pytest.mark.parametrize("command", ["audit", "budget"])
def test_a_cluster_group_that_matches_nothing_is_named(command, capsys):
    argv = [
        command, "--format", "coco",
        "--gt", str(DEMO / "gt_200.json"),
        "--pred", str(DEMO / "pred_yolo11n.json"),
        "--angle", "5,7,9", "--big-error", "15",
        "--cluster", "^0*([0-4])?",
    ]  # fmt: skip
    if command == "audit":
        argv += ["--jitter-repeats", "0", "--resamples", "50"]
    with pytest.raises(SystemExit, match="first group matched nothing"):
        main(argv)


def test_the_count_not_the_rate_decides_whether_a_group_is_checked():
    """Sixty readings with no large error read used to get a narrow check
    by luck, and a size; a hundred readings are too few either way."""
    rng = np.random.default_rng(28)
    for n, noise in [(60, 0.0), (100, 0.05), (130, 0.05)]:
        truth = limbs(n, rng)
        result = budget(
            pairing(truth, noisy(truth, noise, noise, rng)), angle(0, 1, 2), 15
        )
        bent = result.groups[1]
        assert not bent.checks and bent.smallest_limb_px is None, n
        assert "70 are needed" in bent.note
        assert "next: more readings" in result.summary()
    truth = limbs(150, rng)
    bent = budget(
        pairing(truth, noisy(truth, 1.0, 0.03, rng)), angle(0, 1, 2), 15
    ).groups[1]
    assert bent.checks  # 75 a half: checked


def test_the_summary_says_what_to_try_when_there_is_no_size():
    rng = np.random.default_rng(29)
    truth = limbs(400, rng)
    floor = budget(pairing(truth, noisy(truth, 1.0, 0.2, rng)), angle(0, 1, 2), 10)
    assert "next: a looser --target or --big-error" in floor.summary()
    predicted = noisy(truth, 1.0, 0.01, rng)
    predicted[::2, 2] += 60
    failed = budget(pairing(truth, predicted), angle(0, 1, 2), 10)
    assert "audit --size-bands" in failed.summary()


def test_one_image_with_many_readings_has_no_check():
    """Without --cluster, 400 people in one image are 400 readings, enough by
    count, but no fit can be made without that image."""
    rng = np.random.default_rng(30)
    truth = limbs(400, rng)
    predicted = noisy(truth, 1.0, 0.02, rng)
    box, seen = np.array([-1e4, -1e4, 1e4, 1e4]), np.ones(3, bool)
    one = Pairing(
        pairs=[
            Pair("crowd.jpg", Instance(box, t, seen), Instance(box, p, seen))
            for t, p in zip(truth, predicted, strict=True)
        ]
    )
    bent = budget(one, angle(0, 1, 2), 15).groups[1]
    assert not bent.checks and bent.smallest_limb_px is None
    assert "these come from one" in bent.note
