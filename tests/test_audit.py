import json
import os
import re

import numpy as np
import pytest

from poseaudit import agreement as ag
from poseaudit import angle, audit, tilt
from poseaudit.cli import main
from poseaudit.confidence import wilson
from poseaudit.pairing import Pair, Pairing, pair
from poseaudit.types import Instance


def leaning(degrees: float, visible: bool = True, class_id=None) -> Instance:
    """An axis whose upper end leans `degrees` to the right."""
    rad = np.radians(degrees)
    lower = np.array([0.0, 100.0])
    upper = lower + 100 * np.array([np.sin(rad), -np.cos(rad)])
    kp = np.array([lower, upper])
    box = np.array([0, 0, 100, 100], float)
    return Instance(box, kp, np.array([True, visible]), class_id=class_id)


def pairing(truths, preds, images=None) -> Pairing:
    names = images or [f"img{i}" for i in range(len(truths))]
    rows = zip(names, truths, preds, strict=True)
    return Pairing(pairs=[Pair(n, leaning(t), leaning(p)) for n, t, p in rows])


def noisy(seed, n, real_spread, label_sd, model_sd, gain=1.0):
    rng = np.random.default_rng(seed)
    real = rng.uniform(-real_spread, real_spread, n)
    return real + rng.normal(0, label_sd, n), gain * real + rng.normal(0, model_sd, n)


def test_errors_rates_and_limits() -> None:
    truths = [0, 0, 0, 0, 10, 10, 10, 10]
    preds = [1, -1, 1, -1, 4, 4, 4, 4]
    r = audit(pairing(truths, preds), tilt(0, 1), big_error=5)

    assert r.n == 8
    assert r.mean_abs_error == pytest.approx(3.5)
    assert r.big_error_rate == pytest.approx(0.5)
    assert r.bias == pytest.approx(-3.0)
    assert r.rmse == pytest.approx(np.sqrt((4 * 1 + 4 * 36) / 8))
    low, high = r.limits
    assert low < r.bias < high
    assert r.empirical_limits == pytest.approx(tuple(np.percentile(
        np.array(preds) - np.array(truths), [2.5, 97.5])))  # fmt: skip


def test_gain_reads_a_model_that_compresses_both_ways() -> None:
    """+15 read as +9 and -15 as -9: a gain of 0.6, not a bias of zero."""
    truths = np.random.default_rng(0).uniform(-20, 20, 400)
    r = audit(pairing(truths, 0.6 * truths), tilt(0, 1), big_error=5)

    assert r.gain == pytest.approx(0.6, abs=0.01)
    assert r.gain_ci[1] < 1
    assert r.bands[0].bias > 2 and r.bands[-1].bias < -2


def test_gain_stays_honest_when_the_model_is_much_noisier_than_the_labels() -> None:
    """The case this tool is for. The Bland-Altman slope, which compares the
    two spreads, calls an unbiased model expanding and a compressing one
    unbiased; the gain gets both right."""
    t, p = noisy(1, 1500, 25, label_sd=1, model_sd=10)
    fair = audit(pairing(t, p), tilt(0, 1), big_error=5, resamples=300)
    lo, hi = fair.gain_ci
    assert lo < 1 < hi
    assert fair.ba_slope_ci[0] > 0  # the known failure, kept visible

    t, p = noisy(2, 1500, 25, label_sd=1, model_sd=8, gain=0.8)
    squashed = audit(pairing(t, p), tilt(0, 1), big_error=5, resamples=300)
    assert squashed.gain_ci[1] < 0.9


def test_ba_slope_is_fair_when_both_readings_are_equally_noisy() -> None:
    t, p = noisy(3, 2000, 30, label_sd=6, model_sd=6)
    r = audit(pairing(t, p), tilt(0, 1), big_error=5, band_by="mean", resamples=300)
    lo, hi = r.ba_slope_ci
    assert lo < 0 < hi
    assert all(abs(b.bias) < 0.6 for b in r.bands)


def test_deming_recovers_the_gain_when_the_noise_ratio_is_known() -> None:
    t, p = noisy(4, 3000, 30, label_sd=5, model_sd=5, gain=0.8)
    r = audit(pairing(t, p), tilt(0, 1), big_error=5, noise_ratio=1.0, resamples=200)
    assert r.gain < 0.78  # attenuated by the label noise
    assert r.deming == pytest.approx(0.8, abs=0.03)


def test_icc_matches_a_reference_implementation() -> None:
    """pingouin 0.7.0, intraclass_corr, row ICC(A,1): 0.959377."""
    t = np.array([46.4, 26.3, 51.5, 41.8, 5.7, 58.5, 45.7, 47.2, 7.7, 27.0, 22.2,
                  55.6, 38.6, 49.4, 26.6, 13.6, 33.3, 3.8, 49.7, 37.9, 45.5, 21.3,
                  58.2, 53.6, 46.7])  # fmt: skip
    p = np.array([40.7, 28.2, 47.7, 40.1, 11.3, 60.4, 39.9, 40.7, 7.9, 29.1, 27.3,
                  49.0, 32.5, 41.2, 28.9, 18.9, 33.8, 5.4, 45.7, 35.8, 42.3, 25.5,
                  52.5, 50.6, 42.6])  # fmt: skip
    assert ag.icc_a1(t, p) == pytest.approx(0.959377, abs=1e-6)
    assert ag.icc_a1(t, t) == pytest.approx(1.0)
    assert ag.ccc(t, t) == pytest.approx(1.0)
    assert ag.ccc(t, t + 10) < ag.ccc(t, t + 1)


def test_repeated_limits_widen_when_a_subject_is_consistently_off() -> None:
    rng = np.random.default_rng(5)
    offsets = rng.normal(0, 4, 20)
    truths = np.zeros(200)
    preds = np.repeat(offsets, 10) + rng.normal(0, 0.5, 200)
    subjects = [f"s{i // 10}_frame{i % 10}" for i in range(200)]
    r = audit(
        pairing(truths, preds, subjects),
        tilt(0, 1),
        big_error=5,
        cluster=lambda image: image.split("_")[0],
    )
    naive, repeated = r.limits, r.repeated_limits
    assert repeated is not None
    assert repeated[1] - repeated[0] >= 0.99 * (naive[1] - naive[0])


def test_intervals_resample_whole_clusters() -> None:
    """Ten readings copied from one image are one piece of evidence, not ten."""
    base = np.random.default_rng(2).normal(0, 3, 20)
    truths, preds = np.zeros(200), np.repeat(base, 10)
    by_reading = audit(pairing(truths, preds), tilt(0, 1), big_error=5)
    images = [f"img{i // 10}" for i in range(200)]
    by_image = audit(pairing(truths, preds, images), tilt(0, 1), big_error=5)
    width = lambda r: r.bias_ci[1] - r.bias_ci[0]  # noqa: E731
    assert width(by_image) > 2 * width(by_reading)


def test_rate_intervals_resample_whole_images_by_default() -> None:
    """Ten copies of each image's error: the rate's interval must widen like
    the bias's does, without naming clusters."""
    rng = np.random.default_rng(2)
    truth = np.repeat(rng.normal(0, 6, 20), 10)
    preds = truth + np.repeat(rng.normal(0, 6, 20), 10)
    images = [f"img{i // 10}" for i in range(200)]
    r = audit(pairing(truth, preds, images), tilt(0, 1), big_error=5, thresholds=[3])
    big = round(r.big_error_rate * r.n)
    width = lambda ci: ci[1] - ci[0]  # noqa: E731
    assert width(r.big_error_rate_ci) > 2 * width(wilson(big, r.n))
    row = r.thresholds[0]
    caught, positives = row.true_positive, row.true_positive + row.false_negative
    assert width(row.sensitivity_ci) > 1.5 * width(wilson(caught, positives))


def test_the_rate_interval_never_collapses_to_nothing() -> None:
    r = audit(pairing([0] * 40, [1] * 40), tilt(0, 1), big_error=5)
    assert r.big_error_rate == 0.0
    assert r.big_error_rate_ci[1] > 0.05


def test_wilson_covers_close_to_95_percent_for_rare_events() -> None:
    rng = np.random.default_rng(3)
    p, n, hits = 0.05, 30, 0
    for _ in range(2000):
        low, high = wilson(int(rng.binomial(n, p)), n)
        hits += low <= p <= high
    assert hits / 2000 > 0.93


def test_not_read_says_whose_fault_it_was() -> None:
    collapsed = Instance(
        np.zeros(4), np.array([[5.0, 5.0], [5.0, 5.0]]), np.ones(2, bool)
    )
    unlabelled = leaning(0, visible=False)
    p = Pairing(
        pairs=[
            Pair("a", unlabelled, leaning(0)),
            Pair("b", leaning(0), leaning(3, visible=False)),
            Pair("c", leaning(0), collapsed),
            Pair("d", leaning(0), leaning(1)),
        ],
        missed=[("e", leaning(0)), ("f", leaning(0, visible=False))],
        extra=[("g", leaning(0))],
    )
    r = audit(p, tilt(0, 1), big_error=5)
    nr = r.not_read
    assert (nr.unlabelled, nr.no_predicted_point, nr.unmeasurable, nr.missed) == (
        2,
        1,
        1,
        1,
    )
    assert nr.unmatched_predictions == 1
    assert r.n == 1 and r.measurable == 4


def test_a_keypoint_beyond_either_skeleton_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="keypoint 9"):
        audit(pairing([0], [1]), angle(0, 1, 9), big_error=5)
    short = Instance(np.zeros(4), np.zeros((1, 2)), np.ones(1, bool))
    with pytest.raises(ValueError, match="prediction"):
        audit(Pairing(pairs=[Pair("a", leaning(0), short)]), tilt(0, 1), big_error=5)


def test_negative_keypoint_indices_are_refused() -> None:
    with pytest.raises(ValueError, match="0 or more"):
        tilt(-1, 0)


def test_thresholds_count_decisions_on_each_side() -> None:
    truths = [2, 8, 9, -10, 1, 3]
    preds = [2, 5, 9, -10, 8, 3]  # misses the 8, false alarm on the 1
    r = audit(pairing(truths, preds), tilt(0, 1), big_error=5, thresholds=[7])
    th = r.thresholds[0]
    assert th.side == "outside"
    assert (
        th.true_positive,
        th.false_negative,
        th.false_positive,
        th.true_negative,
    ) == (
        2,
        1,
        1,
        2,
    )
    assert th.sensitivity == pytest.approx(2 / 3)


def test_relative_to_the_frame_removes_a_camera_roll() -> None:
    """Each photo is rolled by its own angle; the model reads the roll too.
    Relative to each frame's median, only the parts' own tilts remain."""
    rng = np.random.default_rng(6)
    own = rng.normal(0, 2, (30, 5))
    roll = rng.normal(0, 8, (30, 1))
    truths = (own + roll).ravel()
    preds = (own + roll + 3).ravel()  # a constant offset, gone once relative
    images = [f"img{i // 5}" for i in range(150)]
    absolute = audit(pairing(truths, preds, images), tilt(0, 1), big_error=2)
    relative = audit(
        pairing(truths, preds, images), tilt(0, 1), big_error=2, relative_to="median"
    )
    assert absolute.bias == pytest.approx(3.0)
    assert abs(relative.bias) < 1e-9


def test_relative_mode_leaves_out_frames_too_small_for_a_baseline() -> None:
    images = ["a", "a", "b", "b", "b"]
    r = audit(
        pairing([0] * 5, [1] * 5, images), tilt(0, 1), big_error=5, relative_to="median"
    )
    assert r.not_read.too_few_in_frame == 2 and r.n == 3


def test_mixed_classes_and_other_traps_are_named() -> None:
    p = Pairing(
        pairs=[Pair(f"i{k}", leaning(0, class_id=k % 2), leaning(1)) for k in range(40)]
    )
    with pytest.raises(ValueError, match="mix classes"):
        audit(p, tilt(0, 1), big_error=5)
    notes = " ".join(audit(p, tilt(0, 1), big_error=5, mixed_classes=True).warnings)
    assert "mix classes" in notes

    flat = audit(pairing([85] * 40, [84] * 40), tilt(0, 1), big_error=5)
    assert any("horizontal" in w for w in flat.warnings)

    cut = audit(pairing([-5, 5] * 20, [-5, 5] * 20), tilt(0, 1), 5, bands=[0, 3, 90])
    assert any("outside the band edges" in w for w in cut.warnings)


def test_heavy_tails_point_to_the_empirical_limits() -> None:
    errors = np.r_[np.zeros(90), np.linspace(40, 80, 10)]
    r = audit(pairing(np.zeros(100), errors), tilt(0, 1), big_error=5)
    assert r.tail_shares[1] > 0.04
    assert any("percentile limits" in w for w in r.warnings)


def test_reports_are_reproducible_and_complete(tmp_path) -> None:
    one = audit(pairing([0, 0, 0], [1, 8, 2]), tilt(0, 1), big_error=5)
    two = audit(pairing([0, 0, 0], [1, 8, 2]), tilt(0, 1), big_error=5)
    dump = lambda r: json.dumps(r.to_dict(), default=str)  # noqa: E731
    assert dump(one) == dump(two)

    text = one.to_markdown(str(tmp_path / "r.md"))
    assert "## Limits of agreement" in text and "## Error by size" in text
    rows = [line for line in text.splitlines() if line.startswith("| img")]
    assert rows[0].startswith("| img1 ")
    lines = one.to_csv(str(tmp_path / "r.csv")).splitlines()
    assert lines[0].startswith("image,truth_index,predicted_index,class_id,cluster")
    assert len(lines) == 4

    one.plot(str(tmp_path / "r.png"))
    assert (tmp_path / "r.png").stat().st_size > 1000


def _yolo_pair(tmp_path, dx: float = 0.1, images: int = 1):
    for folder, shift in (("gt", 0.0), ("pred", dx)):
        (tmp_path / folder).mkdir()
        for i in range(images):
            (tmp_path / folder / f"a{i}.txt").write_text(
                f"0 0.5 0.5 1 1 0.5 0.0 2 {0.5 + shift} 1.0 2\n"
            )


def _cli(tmp_path, *extra):
    base = ["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
            "--pred", str(tmp_path / "pred"), "--image-size", "100", "100",
            "--keypoints", "2", "--big-error", "5"]  # fmt: skip
    main(base + list(extra))


def test_cli_end_to_end(tmp_path, capsys) -> None:
    _yolo_pair(tmp_path, images=4)
    _cli(tmp_path, "--tilt", "0,1", "--bands", "0,10,90", "--threshold", "3",
         "--cluster", r"^(a)\d", "--report", str(tmp_path / "r.md"),
         "--csv", str(tmp_path / "r.csv"), "--json", str(tmp_path / "r.json"),
         "--plot", str(tmp_path / "r.png"))  # fmt: skip
    out = capsys.readouterr().out
    assert "read 4 of 4" in out and "caught" in out
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["n"] == 4 and data["settings"]["cluster"] == "custom"
    for name in ("r.md", "r.csv", "r.png"):
        assert (tmp_path / name).exists()


def test_cli_reads_sizes_from_an_image_folder(tmp_path, capsys) -> None:
    from PIL import Image

    _yolo_pair(tmp_path, images=1)
    (tmp_path / "img").mkdir()
    Image.new("RGB", (200, 100)).save(tmp_path / "img" / "a0.jpg")
    main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
          "--pred", str(tmp_path / "pred"), "--images", str(tmp_path / "img"),
          "--keypoints", "2", "--tilt", "0,1", "--big-error", "5"])  # fmt: skip
    assert "read 1 of 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--tilt", "0,9"], "keypoint 9"),
        (["--tilt=-1,0"], "0 or more"),
        (["--tilt", "0,1", "--report", "/no/such/dir/r.md"], "does not exist"),
        (["--tilt", "a,b"], "indices"),
        (["--tilt", "0,1", "--bands", "abc"], "--bands"),
        (["--tilt", "0,1", "--bands", "0"], "--bands"),
        (["--tilt", "0,1", "--cluster", "("], "--cluster"),
        (["--tilt", "1,1"], "repeated"),
    ],
)
def test_cli_errors_are_messages_not_tracebacks(tmp_path, extra, message) -> None:
    _yolo_pair(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, *extra)
    assert message in str(stop.value)


def arms(truth_deg, predicted_deg, size=100.0, jitter=0.0, seed=0):
    """Elbows: shoulder, elbow, wrist, with the given interior angles."""
    rng = np.random.default_rng(seed)
    pairs = []
    for i, (t, p) in enumerate(zip(truth_deg, predicted_deg, strict=True)):

        def arm(angle_deg):
            elbow = np.array([200.0, 200.0])
            shoulder = elbow + [0.0, -size]
            a = np.radians(angle_deg)
            wrist = elbow + size * np.array([np.sin(a), -np.cos(a)])
            return np.array([shoulder, elbow, wrist])

        truth_kp = arm(t)
        predicted_kp = arm(p) + rng.normal(0, jitter, (3, 2))
        box = np.array([0, 0, 400, 400], float)
        pairs.append(Pair(f"i{i}", Instance(box, truth_kp, np.ones(3, bool)),
                          Instance(box, predicted_kp, np.ones(3, bool))))  # fmt: skip
    return Pairing(pairs=pairs)


def test_jitter_alone_sits_inside_its_own_reference() -> None:
    """A model with no tendency, only scattered points, whose angles near 180
    can only be read smaller: its gain is below 1 and the reference says why."""
    truths = np.random.default_rng(7).uniform(90, 179, 300)
    r = audit(arms(truths, truths, jitter=20, seed=8), angle(0, 1, 2), big_error=15)
    assert r.gain < 0.95
    lo, hi = r.jitter_gain_range
    assert lo - 0.05 <= r.gain <= hi + 0.05


def test_a_real_squash_falls_below_the_jitter_reference() -> None:
    truths = np.random.default_rng(9).uniform(20, 178, 300)
    squashed = 100 + 0.5 * (truths - 100)
    r = audit(arms(truths, squashed, jitter=3, seed=10), angle(0, 1, 2), big_error=15)
    assert r.gain_ci[1] < r.jitter_gain_range[0]


def test_errors_are_split_by_the_size_of_the_part() -> None:
    small = arms([90] * 30, [90] * 30, size=20, jitter=4, seed=1)
    large = arms([90] * 30, [90] * 30, size=200, jitter=4, seed=2)
    both = Pairing(
        pairs=small.pairs
        + [Pair("L" + p.image, p.truth, p.predicted) for p in large.pairs]
    )
    r = audit(both, angle(0, 1, 2), big_error=10, size_bands=[50])
    first, second = r.size_bands
    assert first.high == 50 and first.n == 30 and second.n == 30
    assert first.mean_abs_error > 3 * second.mean_abs_error


def test_relative_readings_give_no_reading_a_free_zero() -> None:
    """With the median of an odd count, the median object used to be its own
    baseline and scored exactly 0 on both sides."""
    rng = np.random.default_rng(11)
    truths = rng.normal(0, 3, 150)
    preds = truths + rng.normal(0, 1, 150)
    images = [f"img{i // 5}" for i in range(150)]
    r = audit(pairing(truths, preds, images), tilt(0, 1), 5, relative_to="median")
    assert sum(1 for x in r.readings if x.error == 0) == 0


def test_a_percentile_baseline_of_sizes_encodes_a_straightest_part_rule() -> None:
    """Excess over the straightest parts of the same photo: |tilt| minus the
    15th percentile of the others' |tilt|."""
    truths = [1, -2, 1.5, 12, -1]
    r = audit(
        pairing(truths, truths, ["a"] * 5), tilt(0, 1), 5,
        relative_to="p15", relative_abs=True, thresholds=[6],
    )  # fmt: skip
    excess = {round(x.truth, 6) for x in r.readings}
    assert round(12 - np.percentile([1, 2, 1.5, 1], 15), 6) in excess
    assert r.thresholds[0].side == "above"


def test_a_lower_predicted_threshold_can_be_set_or_matched() -> None:
    truths = np.array([2, 8, 9, 10, 12, 1, 3, 4])
    preds = 0.6 * truths  # everything reads 40% small
    r = audit(pairing(truths, preds), tilt(0, 1), big_error=5,
              thresholds=[7], predicted_thresholds=[4])  # fmt: skip
    same, lower, matched = r.thresholds
    assert same.true_positive == 1 and lower.true_positive == 4
    assert (
        matched.matched and matched.true_positive == 4 and matched.false_positive == 0
    )


def test_big_error_must_be_positive() -> None:
    with pytest.raises(ValueError, match="above 0"):
        audit(pairing([0], [1]), tilt(0, 1), big_error=0)


def test_too_few_named_clusters_are_flagged() -> None:
    r = audit(pairing([0] * 40, [1] * 40, [f"s{i % 4}_f{i}" for i in range(40)]),
              tilt(0, 1), 5, cluster=lambda image: image.split("_")[0])  # fmt: skip
    assert any("clusters" in w for w in r.warnings)


def test_json_output_is_strict(tmp_path) -> None:
    _yolo_pair(tmp_path)
    _cli(tmp_path, "--tilt", "0,1", "--json", str(tmp_path / "r.json"))
    text = (tmp_path / "r.json").read_text()
    assert "NaN" not in text
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard {c}"))


def correlated_arms(truth_deg, factor, seed, n_common=10.0, n_own=4.0, size=60.0):
    """Model errors as they come: the whole limb shifts together (which barely
    moves its angle) plus a little per-point scatter."""
    rng = np.random.default_rng(seed)
    truths = np.asarray(truth_deg)
    base = arms(truths, 100 + factor * (truths - 100), size=size).pairs
    pairs = []
    for p in base:
        moved = (
            p.predicted.keypoints
            + rng.normal(0, n_common, 2)
            + rng.normal(0, n_own, (3, 2))
        )
        pairs.append(
            Pair(p.image, p.truth, Instance(p.predicted.bbox, moved, np.ones(3, bool)))
        )
    return Pairing(pairs=pairs)


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_the_jitter_reference_stays_calibrated_when_errors_move_together(seed) -> None:
    """Breaking the link between an object's points would inflate its angle
    noise and call a model with no tendency expanding (or hide a squash)."""
    truths = np.random.default_rng(seed).uniform(40, 178, 300)
    r = audit(
        correlated_arms(truths, 1.0, seed + 100), angle(0, 1, 2), 15, resamples=400
    )
    lo, hi = r.gain_gap_ci
    assert lo < 0 < hi
    assert 0.025 < r.jitter_p < 0.975


def test_the_jitter_reference_catches_a_modest_squash() -> None:
    truths = np.random.default_rng(31).uniform(40, 178, 300)
    r = audit(correlated_arms(truths, 0.85, 131), angle(0, 1, 2), 15, resamples=400)
    assert r.gain_gap_ci[1] < 0 and r.jitter_p < 0.05


def test_band_labels_use_the_edges_given() -> None:
    r = audit(pairing([1, 2, 5, 6, 20, 30], [1, 2, 5, 6, 20, 30]), tilt(0, 1), 5,
              bands=[0, 3, 7, 90])  # fmt: skip
    assert [(b.low, b.high) for b in r.bands] == [(0, 3), (3, 7), (7, 90)]


def test_relative_sizes_need_a_baseline() -> None:
    with pytest.raises(ValueError, match="relative_to"):
        audit(pairing([0], [1]), tilt(0, 1), 5, relative_abs=True)


def test_yolo_coordinates_outside_the_image_point_to_a_wrong_skeleton(tmp_path) -> None:
    from poseaudit import load_yolo

    (tmp_path / "x.txt").write_text("0 .5 .5 .2 .2 .1 .1 2 .2 .2 2 0.9\n")
    # read as 3 points of x y (the third "point" is 2, 0.2): off the image
    with pytest.raises(ValueError, match="outside the image"):
        load_yolo(tmp_path, (100, 100), 3, per_point=2)


def test_a_class_filter_that_matches_nothing_names_the_classes_present(
    tmp_path,
) -> None:
    from poseaudit import load_yolo

    (tmp_path / "x.txt").write_text("5 .5 .5 .2 .2 .1 .1 2 .2 .2 2\n")
    with pytest.warns(UserWarning, match=r"classes present are \[5\]"):
        load_yolo(tmp_path, (100, 100), 2, classes=[0])


def test_the_robust_gain_shrugs_off_a_few_gross_failures() -> None:
    truths = np.random.default_rng(41).uniform(0, 40, 300)
    preds = truths.copy()
    preds[:15] = np.random.default_rng(42).uniform(-40, 0, 15)  # 5% read wildly off
    r = audit(pairing(truths, preds), tilt(0, 1), big_error=5)
    assert r.gain < 0.95
    assert r.robust_gain == pytest.approx(1.0, abs=0.02)


def test_plotting_leaves_the_matplotlib_backend_alone(tmp_path) -> None:
    import matplotlib

    before = matplotlib.get_backend()
    audit(pairing([0, 5, 9], [1, 4, 8]), tilt(0, 1), 5).plot(str(tmp_path / "p.png"))
    assert matplotlib.get_backend() == before


def test_a_ratio_is_sized_by_its_two_segments_not_the_gap() -> None:
    from poseaudit import ratio

    kp = np.array([[0.0, 0], [10, 0], [100, 0], [110, 0]])
    inst = Instance(np.array([0, 0, 110, 10], float), kp, np.ones(4, bool))
    r = audit(Pairing(pairs=[Pair("a", inst, inst)]), ratio(0, 1, 2, 3), 0.1)
    assert r.readings[0].size == pytest.approx(10.0)


def test_a_class_filter_that_reads_nothing_fails(tmp_path, capsys) -> None:
    _yolo_pair(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, "--tilt", "0,1", "--classes", "5")
    assert "nothing was read" in str(stop.value)


def test_a_thin_part_matched_too_strictly_is_flagged() -> None:
    """A long thin box read a few degrees off barely overlaps its truth: the
    worst predictions drop out as missed instead of counting as errors."""

    def bar(degrees):
        rad = np.radians(degrees)
        ends = np.array([[0.0, 0.0], [200 * np.cos(rad), 200 * np.sin(rad)]])
        ends += [300, 300]
        return Instance.from_keypoints(ends, np.array([True, True]))

    truth = {f"i{k}": [bar(0)] for k in range(40)}
    predicted = {f"i{k}": [bar(0.5 if k < 30 else 20)] for k in range(40)}
    r = audit(pair(truth, predicted), angle_free_tilt(), big_error=5)
    assert r.not_read.missed == 10
    assert any("overlap an unmatched prediction" in w for w in r.warnings)


def angle_free_tilt():
    return tilt(0, 1)


def test_a_same_count_threshold_against_the_fit_is_flagged() -> None:
    """Squashed readings plus many wild ones just above the threshold: counting
    from the top lands on the wild ones and raises the threshold the fit says
    to lower."""
    t = np.linspace(-20, 20, 200)
    p = 0.6 * t
    p[80:120] = 11.0  # gross failures among flat truths
    r = audit(
        pairing(t, p), tilt(0, 1), big_error=5, thresholds=[10], threshold_side="above"
    )
    same = [x for x in r.thresholds if x.matched][0]
    assert same.predicted_threshold > 10
    assert any("opposite way from the fitted line" in w for w in r.warnings)


def test_relative_readings_say_why_there_is_no_jitter_reference() -> None:
    t = np.tile([0.0, 1.0, 2.0, 12.0], 10)
    r = audit(
        pairing(t, t + 1, images=[f"img{i // 4}" for i in range(40)]),
        tilt(0, 1),
        big_error=5,
        relative_to="median",
    )
    assert "not computed" in r.summary()


def straight_noisier_arms(seed: int, n: int = 300) -> Pairing:
    """No tendency; every point scatters 8% of the arm, three times that on
    arms straighter than 150 degrees; arms of many sizes and directions."""
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n):
        t, size = rng.uniform(40, 178), rng.uniform(40, 160)
        truth = arms([t], [t], size=size).pairs[0].truth.keypoints
        noise = rng.normal(0, 0.08 * size, (3, 2)) * (3 if t > 150 else 1)
        turn = rng.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        t_kp, p_kp = truth @ rot.T, (truth + noise) @ rot.T
        box, seen = np.array([-400, -400, 800, 800], float), np.ones(3, bool)
        pairs.append(
            Pair(f"i{i}", Instance(box, t_kp, seen), Instance(box, p_kp, seen))
        )
    return Pairing(pairs=pairs)


@pytest.mark.parametrize("seed", [51, 52, 53])
def test_noise_that_grows_on_straight_arms_is_not_called_squashing(seed) -> None:
    """Donors drawn from any angle carry straight arms' noise onto folded ones
    and call an honest model squashed; drawn from similar angles they do not."""
    r = audit(straight_noisier_arms(seed), angle(0, 1, 2), 15, resamples=300)
    assert r.gain_gap_ci[1] > 0 and r.jitter_p > 0.05


def test_tilt_thresholds_apply_to_readings_as_read() -> None:
    """A tilt of 89 read as -89 is 2 degrees off, but the decision sees -89,
    and no predicted threshold may leave [-90, 90)."""
    truths = [89.0] * 5 + [0.0] * 20
    preds = [-89.0] * 5 + [0.5] * 20
    r = audit(pairing(truths, preds), tilt(0, 1), 5, thresholds=[80])
    same = [x for x in r.thresholds if x.matched][0]
    assert -90 <= same.predicted_threshold < 90
    assert r.thresholds[0].true_positive == 5


def test_relative_tilts_across_the_seam_are_not_torn_apart() -> None:
    """Near-horizontal parts at +89 and -89 are 2 degrees apart; a plain
    median of them is vertical and would call a near-perfect model 90 off."""
    rng = np.random.default_rng(3)
    truths, preds, images = [], [], []
    for k in range(40):
        frame = rng.uniform(-3, 3)
        for _ in range(5):
            t = ((90 + frame + rng.normal(0, 1.5)) + 90) % 180 - 90
            truths.append(t)
            preds.append(((t + 0.3) + 90) % 180 - 90)
            images.append(f"img{k}")
    r = audit(pairing(truths, preds, images), tilt(0, 1), 5, relative_to="median")
    assert r.mean_abs_error < 0.5


def test_parts_of_one_size_make_one_size_band() -> None:
    r = audit(pairing([0, 1, 2, 3, 20, 30], [0, 1, 2, 3, 22, 31]), tilt(0, 1), 5)
    assert len(r.size_bands) == 1 and r.size_bands[0].n == 6


def bisector_noise_arms(seed: int, n: int = 300) -> Pairing:
    """No tendency; the elbow scatters along the bisector of its angle, and
    half the arms bend the other way (mirror images)."""
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n):
        t, size = rng.uniform(40, 178), rng.uniform(60, 140)
        kp = arms([t], [t], size=size).pairs[0].truth.keypoints.copy()
        if rng.random() < 0.5:
            kp[:, 0] = 400 - kp[:, 0]  # mirror image
        u = (kp[0] - kp[1]) / np.linalg.norm(kp[0] - kp[1])
        v = (kp[2] - kp[1]) / np.linalg.norm(kp[2] - kp[1])
        bisector = (u + v) / np.linalg.norm(u + v)
        pred = kp + rng.normal(0, 0.03 * size, (3, 2))
        pred[1] += bisector * rng.normal(0, 0.25 * size)
        box, seen = np.array([0, 0, 400, 400], float), np.ones(3, bool)
        pairs.append(Pair(f"i{i}", Instance(box, kp, seen), Instance(box, pred, seen)))
    return Pairing(pairs=pairs)


@pytest.mark.parametrize("seed", [61, 62, 63])
def test_noise_toward_the_inside_of_the_bend_is_not_called_squashing(seed) -> None:
    r = audit(bisector_noise_arms(seed), angle(0, 1, 2), 15, resamples=300)
    assert r.gain_gap_ci[1] > 0 and r.jitter_p > 0.05


def test_named_clusters_give_no_jitter_p() -> None:
    truths = np.random.default_rng(5).uniform(40, 170, 60)
    r = audit(
        arms(truths, truths, jitter=5, seed=6),
        angle(0, 1, 2),
        15,
        cluster=lambda image: int(image[1:]) // 6,
        resamples=200,
    )
    assert r.jitter_p is None and r.gain_gap_ci is not None
    assert "no p with named clusters" in r.summary()


def test_a_misspelt_side_is_refused() -> None:
    with pytest.raises(ValueError, match="threshold_side"):
        audit(pairing([0, 1, 2], [0, 1, 2]), tilt(0, 1), 5, threshold_side="bellow")


def test_integer_visibility_flags_are_refused() -> None:
    with pytest.raises(ValueError, match="boolean"):
        Instance(np.zeros(4), np.zeros((2, 2)), np.array([2, 0]))


def test_side_without_a_threshold_is_refused(tmp_path) -> None:
    _yolo_pair(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, "--tilt", "0,1", "--side", "above")
    assert "--threshold" in str(stop.value)


def test_keypoint_matching_misses_point_to_the_similarity_knob() -> None:
    """Without boxes, pairs form by keypoint closeness: the advice must name
    that threshold, not the box one."""

    def arm(wrist_dx):
        kp = np.array([[200.0, 100.0], [200.0, 200.0], [300.0 + wrist_dx, 200.0]])
        return Instance.from_keypoints(kp, np.ones(3, bool))

    truth = {f"i{k}": [arm(0)] for k in range(40)}
    predicted = {f"i{k}": [arm(2 if k < 30 else 150)] for k in range(40)}
    r = audit(pair(truth, predicted), angle(0, 1, 2), big_error=5)
    assert r.not_read.missed == 10
    note = [w for w in r.warnings if "overlap an unmatched" in w]
    assert note and "--min-similarity" in note[0]
    loose = audit(
        pair(truth, predicted, min_keypoint_similarity=0.001), angle(0, 1, 2), 5
    )
    assert loose.n == 40


def test_a_range_that_is_only_rounding_noise_has_no_slope() -> None:
    t = np.full(10, 120.0) + np.arange(10) * 1e-13
    assert np.isnan(ag.gain(t, t + np.arange(10))[0])


def test_decision_intervals_resample_named_clusters() -> None:
    """Six subjects, each read many times the same way: the catch rate is
    known from six subjects, not from 120 readings."""
    t = np.tile(np.r_[np.full(10, 20.0), np.full(10, 0.0)], 6)
    p = t.copy()
    p[:20] = 0.0  # one subject's flags are all missed
    images = [f"s{k // 20}_f{k}" for k in range(120)]
    common = dict(thresholds=[10], threshold_side="above", resamples=400)
    loose = audit(pairing(t, p, images), tilt(0, 1), 5, **common)
    named = audit(
        pairing(t, p, images), tilt(0, 1), 5, cluster=lambda n: n.split("_")[0],
        **common,
    )  # fmt: skip
    width = lambda r: np.diff(r.thresholds[0].sensitivity_ci)[0]  # noqa: E731
    assert width(named) > 1.5 * width(loose)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--bands", "5,-5,0"], "increasing"),
        (["--bands", "nan,1"], "increasing"),
        (["--noise-ratio", "-1"], "above 0"),
        (["--min-in-frame", "0"], "1 or more"),
        (["--threshold", "nan"], "finite"),
        (["--resamples", "0"], "--resamples"),
        (["--jitter-repeats", "-1"], "--jitter-repeats"),
        (["--size-bands", "0,30"], "--size-bands"),
        (["--threshold", "1", "--pred-threshold", "nan"], "finite"),
        (["--min-iou", "2"], "at most 1"),
        (["--min-iou", "0"], "above 0"),
        (["--min-similarity", "1.5"], "at most 1"),
        (["--min-score", "nan"], "finite"),
        (["--resamples", "10"], "50 or more"),
        (["--bands", "0,5,5"], "increasing"),
        (["--min-conf", "nan"], "finite"),
        (["--keypoints", "0"], "1 or more"),
        (["--image-size", "0", "0"], "above 0"),
    ],
)
def test_values_the_audit_cannot_use_are_refused(tmp_path, extra, message) -> None:
    _yolo_pair(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, "--tilt", "0,1", *extra)
    assert message in str(stop.value)


def test_an_output_may_not_overwrite_an_input(tmp_path) -> None:
    _yolo_pair(tmp_path)
    (tmp_path / "gt" / "a0.txt").rename(tmp_path / "gt" / "keep.txt")
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, "--tilt", "0,1", "--json", str(tmp_path / "gt"))
    assert "folder" in str(stop.value) or "input" in str(stop.value)


def test_nothing_read_still_writes_the_json(tmp_path, capsys) -> None:
    _yolo_pair(tmp_path)
    with pytest.raises(SystemExit):
        _cli(
            tmp_path,
            "--tilt",
            "0,1",
            "--classes",
            "5",
            "--json",
            str(tmp_path / "r.json"),
        )
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["n"] == 0 and data["poseaudit"] and data["schema"] == 1
    assert any("class" in w for w in data["warnings"])


def test_small_ratio_errors_do_not_print_as_zero() -> None:
    from poseaudit import ratio

    def rod(scale):
        kp = np.array([[0.0, 0.0], [100.0 * scale, 0.0], [0.0, 50.0], [0.0, 150.0]])
        return Instance.from_keypoints(kp, np.ones(4, bool))

    truth = {f"i{k}": [rod(1.0)] for k in range(40)}
    pred = {f"i{k}": [rod(1.0 + 0.002 * (k % 3 - 1))] for k in range(40)}
    text = audit(pair(truth, pred), ratio(0, 1, 2, 3), big_error=0.01).summary()
    assert re.search(r"mean 0\.\d{4}\b", text), text  # four decimals, not two


def test_settings_record_every_choice_that_changes_the_figures() -> None:
    r = audit(
        pairing([0, 5, 10], [0, 5, 10]),
        tilt(0, 1),
        5,
        thresholds=[3],
        threshold_side="above",
        predicted_thresholds=[2],
        min_in_frame=2,
    )
    for key, value in {
        "thresholds": [3],
        "threshold_side": "above",
        "predicted_thresholds": [2],
        "min_in_frame": 2,
        "mixed_classes": False,
    }.items():
        assert r.settings[key] == value


def test_named_clusters_never_shrink_a_rate_interval_to_a_point() -> None:
    """With no large error, every resample gives a rate of 0: the interval
    must keep Wilson's width, not collapse to [0, 0]."""
    t = np.linspace(-10, 10, 120)
    images = [f"s{k // 10}_f{k}" for k in range(120)]
    r = audit(
        pairing(t, t + 0.5, images),
        tilt(0, 1),
        5,
        cluster=lambda n: n.split("_")[0],
        thresholds=[5],
        threshold_side="above",
        resamples=200,
    )
    assert r.big_error_rate == 0 and r.big_error_rate_ci[1] > 0.02
    assert r.thresholds[0].sensitivity == 1 and r.thresholds[0].sensitivity_ci[0] < 1


def test_outputs_may_not_land_in_an_input_folder_or_on_each_other(tmp_path) -> None:
    _yolo_pair(tmp_path)
    label = tmp_path / "pred" / "a0.txt"
    before = label.read_text()
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, "--tilt", "0,1", "--csv", str(label))
    assert "input" in str(stop.value) and label.read_text() == before
    out = str(tmp_path / "out")
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, "--tilt", "0,1", "--csv", out, "--json", out)
    assert "same file" in str(stop.value)


def test_cli_resampling_settings_reach_the_audit(tmp_path) -> None:
    _yolo_pair(tmp_path, images=4)
    _cli(tmp_path, "--tilt", "0,1", "--resamples", "60", "--jitter-repeats", "0",
         "--json", str(tmp_path / "r.json"))  # fmt: skip
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["settings"]["resamples"] == 60
    assert data["settings"]["jitter_repeats"] == 0 and data["jitter_gain"] is None


def test_the_summary_qualifies_p_squash() -> None:
    truths = np.random.default_rng(3).uniform(40, 178, 120)
    r = audit(arms(truths, 0.8 * truths + 25, jitter=2, seed=4), angle(0, 1, 2), 15,
              resamples=100)  # fmt: skip
    text = r.summary()
    assert "p(squash)" in text and "labels make p small" in text


def test_ratio_band_labels_keep_their_decimals(tmp_path) -> None:
    from poseaudit import ratio

    def rod(scale):
        kp = np.array([[0.0, 0.0], [100.0 * scale, 0.0], [0.0, 50.0], [0.0, 150.0]])
        return Instance.from_keypoints(kp, np.ones(4, bool))

    truth = {f"i{k}": [rod(1.0 + 0.01 * (k % 5))] for k in range(40)}
    pred = {f"i{k}": [rod(1.0 + 0.01 * (k % 5) + 0.002)] for k in range(40)}
    r = audit(pair(truth, pred), ratio(0, 1, 2, 3), big_error=0.01)
    r.to_markdown(str(tmp_path / "r.md"))
    section = (tmp_path / "r.md").read_text().split("## Error by truth value")[1]
    labels = [line.split("|")[1] for line in section.splitlines() if " to " in line]
    assert len(set(labels)) == len(labels) > 1


def test_an_image_folder_may_hold_labels_and_any_image_format(tmp_path, capsys) -> None:
    from PIL import Image

    _yolo_pair(tmp_path, images=1)
    (tmp_path / "img").mkdir()
    Image.new("RGB", (200, 100)).save(tmp_path / "img" / "a0.gif")
    (tmp_path / "img" / "a0.txt").write_text("not an image")
    main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
          "--pred", str(tmp_path / "pred"), "--images", str(tmp_path / "img"),
          "--keypoints", "2", "--tilt", "0,1", "--big-error", "5"])  # fmt: skip
    assert "read 1 of 1" in capsys.readouterr().out


def test_outputs_spelled_differently_still_may_not_overwrite_inputs(tmp_path) -> None:
    """A hard link, or another case on a case-insensitive file system, is the
    same file."""
    _yolo_pair(tmp_path)
    label = tmp_path / "pred" / "a0.txt"
    before = label.read_text()
    link = tmp_path / "link.txt"
    os.link(label, link)
    (tmp_path / "pred2").mkdir()
    with pytest.raises(SystemExit):
        _cli(tmp_path, "--tilt", "0,1", "--csv", str(link))
    shouted = tmp_path / "PRED" / "a0.txt"
    if shouted.exists():  # only where the file system ignores case
        with pytest.raises(SystemExit):
            _cli(tmp_path, "--tilt", "0,1", "--csv", str(shouted))
    assert label.read_text() == before
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, "--tilt", "0,1", "--csv", str(tmp_path / "out.x"),
             "--json", str(tmp_path / "OUT.x"))  # fmt: skip
    assert "same file" in str(stop.value)


def test_an_output_that_is_a_folder_is_refused(tmp_path) -> None:
    _yolo_pair(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, "--tilt", "0,1", "--report", str(tmp_path / "out"))
    assert "folder" in str(stop.value)


def test_images_that_share_a_name_are_refused(tmp_path) -> None:
    from PIL import Image

    _yolo_pair(tmp_path)
    (tmp_path / "img").mkdir()
    Image.new("RGB", (200, 100)).save(tmp_path / "img" / "a0.jpg")
    Image.new("RGB", (100, 200)).save(tmp_path / "img" / "a0.png")
    with pytest.raises(SystemExit) as stop:
        main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
              "--pred", str(tmp_path / "pred"), "--images", str(tmp_path / "img"),
              "--keypoints", "2", "--tilt", "0,1", "--big-error", "5"])  # fmt: skip
    assert "share a name" in str(stop.value)


def test_a_label_name_with_a_dot_does_not_borrow_another_image(tmp_path) -> None:
    from PIL import Image

    for folder in ("gt", "pred"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "155010.457_x.txt").write_text(
            "0 .5 .5 1 1 .5 0 2 .5 1 2\n"
        )
    (tmp_path / "img").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "img" / "155010.jpg")
    with pytest.raises(SystemExit) as stop:
        main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
              "--pred", str(tmp_path / "pred"), "--images", str(tmp_path / "img"),
              "--keypoints", "2", "--tilt", "0,1", "--big-error", "5"])  # fmt: skip
    assert "no image" in str(stop.value)


def test_nothing_read_writes_no_report(tmp_path) -> None:
    _yolo_pair(tmp_path)
    with pytest.raises(SystemExit):
        _cli(tmp_path, "--tilt", "0,1", "--classes", "5",
             "--report", str(tmp_path / "r.md"))  # fmt: skip
    assert not (tmp_path / "r.md").exists()


def test_loader_warnings_print_once(tmp_path, capsys) -> None:
    _yolo_pair(tmp_path)
    with pytest.raises(SystemExit):
        _cli(tmp_path, "--tilt", "0,1", "--classes", "5")
    captured = capsys.readouterr()
    assert "no row has class" in captured.err
    assert "no row has class" not in captured.out


def test_precision_intervals_also_keep_wilsons_width() -> None:
    t = np.linspace(-10, 10, 120)
    images = [f"s{k // 10}_f{k}" for k in range(120)]
    r = audit(pairing(t, t, images), tilt(0, 1), 5,
              cluster=lambda n: n.split("_")[0], thresholds=[5],
              threshold_side="above", resamples=200)  # fmt: skip
    assert r.thresholds[0].precision == 1 and r.thresholds[0].precision_ci[0] < 1


def test_caller_settings_cannot_override_the_audits_own() -> None:
    r = audit(pairing([0, 5, 10], [0, 5, 10]), tilt(0, 1), 5, seed=3,
              settings={"seed": 99, "note": "kept"})  # fmt: skip
    assert r.settings["seed"] == 3 and r.settings["note"] == "kept"


def test_a_new_file_inside_an_input_folder_is_refused(tmp_path) -> None:
    """Not an overwrite, but the next run would read it as a label."""
    _yolo_pair(tmp_path)
    with pytest.raises(SystemExit) as stop:
        _cli(tmp_path, "--tilt", "0,1", "--csv", str(tmp_path / "pred" / "new.txt"))
    assert "input folder" in str(stop.value)
    assert not (tmp_path / "pred" / "new.txt").exists()


def test_an_input_folder_spelled_in_nfd_is_the_same_folder(tmp_path) -> None:
    import unicodedata

    nfc, nfd = (unicodedata.normalize(f, "예측") for f in ("NFC", "NFD"))
    _yolo_pair(tmp_path)
    (tmp_path / "pred").rename(tmp_path / nfc)
    base = ["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
            "--pred", str(tmp_path / nfc), "--image-size", "100", "100",
            "--keypoints", "2", "--big-error", "5", "--tilt", "0,1"]  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        main(base + ["--csv", str(tmp_path / nfd / "new.csv")])
    assert "input" in str(stop.value)


def test_a_name_clash_among_images_no_label_needs_is_ignored(tmp_path, capsys) -> None:
    from PIL import Image

    _yolo_pair(tmp_path, images=1)
    (tmp_path / "img").mkdir()
    Image.new("RGB", (200, 100)).save(tmp_path / "img" / "a0.jpg")
    Image.new("RGB", (10, 10)).save(tmp_path / "img" / "zz.jpg")
    Image.new("RGB", (10, 10)).save(tmp_path / "img" / "zz.png")
    main(["audit", "--format", "yolo", "--gt", str(tmp_path / "gt"),
          "--pred", str(tmp_path / "pred"), "--images", str(tmp_path / "img"),
          "--keypoints", "2", "--tilt", "0,1", "--big-error", "5"])  # fmt: skip
    assert "read 1 of 1" in capsys.readouterr().out


def test_visible_points_must_be_finite() -> None:
    kp = np.array([[0.0, 0.0], [np.nan, 5.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match=r"keypoints \[1\] are visible but not finite"):
        Instance.from_keypoints(kp, np.array([True, True, True]))
    with pytest.raises(ValueError, match="not finite"):
        Instance(np.zeros(4), kp, np.array([True, True, True]))
    hidden = Instance.from_keypoints(kp, np.array([True, False, True]))
    assert np.allclose(hidden.bbox, [0, 0, 3, 4])
    listed = Instance.from_keypoints([[0, 0], [3, 4]], np.array([True, True]))
    assert listed.keypoints.dtype == float


def test_normalised_coordinates_are_flagged() -> None:
    rng = np.random.default_rng(0)

    def arm(a, scale):
        r = np.radians(a)
        kp = np.array(
            [[0.5, 0.2], [0.5, 0.5], [0.5 + 0.3 * np.sin(r), 0.5 + 0.3 * np.cos(r)]]
        )
        return Instance.from_keypoints(kp * scale, np.ones(3, bool))

    def run(scale):
        a = rng.uniform(30, 150, 40)
        pairs = [
            Pair(f"i{k}", arm(t, scale), arm(t + rng.normal(0, 3), scale))
            for k, t in enumerate(a)
        ]
        r = audit(
            Pairing(pairs=pairs), angle(0, 1, 2), 10, jitter_repeats=0, resamples=50
        )
        return any("within 0 to 1" in w for w in r.warnings)

    assert run(1.0) and not run(640.0)


def test_an_index_out_of_range_says_how_indices_count() -> None:
    """MediaPipe's 33 points are 0 to 32: counted from 1, the last is 33."""
    with pytest.raises(
        ValueError, match=r"has 2 keypoints \(indices 0-1\); indices count from 0"
    ):
        audit(pairing([0], [1]), tilt(0, 2), big_error=5)
    with pytest.raises(ValueError, match="has 2 keypoints"):
        audit(Pairing(missed=[("a", leaning(0))]), tilt(0, 2), big_error=5)


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan")])
def test_a_noise_ratio_must_be_above_0(ratio) -> None:
    """The CLI refuses these already; the function gave a NaN or negative slope."""
    with pytest.raises(ValueError, match="noise_ratio must be above 0"):
        audit(pairing([0, 5, 10], [1, 4, 9]), tilt(0, 1), 5, noise_ratio=ratio)


def landmarks(truth_scale, predicted_scale, past_frame=False):
    """Forty frames of 33 MediaPipe-style landmarks, fractions of the frame,
    each side multiplied by its own scale."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.2, 0.8, (40, 33, 2))
    moved = xy + rng.normal(0, 0.01, xy.shape)
    if past_frame:  # MediaPipe lets points run a little past the edges
        xy[0, 25], moved[1, 27] = (1.01, 0.5), (0.4, -0.002)
    seen = np.ones(33, bool)
    truth = {f"f{k}": [Instance.from_keypoints(xy[k] * truth_scale, seen)]
             for k in range(40)}  # fmt: skip
    predicted = {f"f{k}": [Instance.from_keypoints(moved[k] * predicted_scale, seen)]
                 for k in range(40)}  # fmt: skip
    return audit(pair(truth, predicted), angle(23, 25, 27), big_error=10,
                 resamples=100, jitter_repeats=0)  # fmt: skip


FRAME = np.array([1280.0, 720.0])


def test_points_just_past_the_frame_still_look_normalised() -> None:
    r = landmarks(np.ones(2), np.ones(2), past_frame=True)
    assert any("within 0 to 1" in w for w in r.warnings)
    assert not any("normalised" in w for w in landmarks(FRAME, FRAME).warnings)


@pytest.mark.parametrize(
    ("truth_scale", "predicted_scale", "odd", "other"),
    [
        (FRAME, np.ones(2), "predicted", "ground-truth"),
        (np.ones(2), FRAME, "ground-truth", "predicted"),
    ],
)
def test_one_side_normalised_explains_why_nothing_pairs(
    truth_scale, predicted_scale, odd, other
) -> None:
    """Pixel labels against MediaPipe or Ultralytics xyn predictions: nothing
    pairs, and the warning says why."""
    r = landmarks(truth_scale, predicted_scale)
    assert r.n == 0
    assert any(
        w.startswith(f"The {odd} coordinates look normalised")
        and f"the {other} ones look like pixels" in w
        and "cannot pair" in w
        for w in r.warnings
    )


def test_a_flat_truth_has_no_ba_slope_icc_or_ccc() -> None:
    """With no spread in the truth the BA slope is exactly 2 and ICC and CCC
    are 0 whatever the model does: figures about nothing."""
    t = np.full(20, 50.0)
    p = 50 + np.random.default_rng(0).normal(0, 5, 20)
    assert np.isnan([ag.ba_slope(t, p), ag.icc_a1(t, p), ag.ccc(t, p)]).all()
    assert np.isnan(ag.deming(t, p, 1.0))


def test_large_values_with_a_real_spread_are_not_flat() -> None:
    t = 1e13 + np.linspace(0, 1000, 10)
    assert ag.gain(t, t)[0] == pytest.approx(1.0)
    assert ag.ccc(t, t) == pytest.approx(1.0)


def test_a_cluster_drawn_twice_counts_as_two_in_repeated_limits() -> None:
    from poseaudit.confidence import clustered_bootstrap

    groups = np.array(["a", "a", "b", "b", "c", "c"])
    draws = []

    def record(idx, draw):
        draws.append((groups[idx], draw))
        return np.zeros(1)

    clustered_bootstrap(groups, record, resamples=200, seed=0, copies=True)
    twice = next(d for g, d in draws if (g == "a").sum() == 4)
    assert len(np.unique(twice)) == 3  # three draws, one of them "a" again
    assert all(len(d) == len(g) for g, d in draws)


def test_repeated_limits_interval_covers_the_estimate() -> None:
    """Forty subjects read twice, 5 above and 5 below their mean. Merging a
    subject's copies in a resample would understate the spread within subjects
    and put the whole interval below the estimate."""
    names = [f"s{k // 2}_f{k}" for k in range(80)]
    r = audit(pairing(np.zeros(80), np.tile([5.0, -5.0], 40), names), tilt(0, 1),
              10, cluster=lambda n: n.split("_")[0], resamples=200,
              jitter_repeats=0)  # fmt: skip
    assert r.repeated_limits is not None and r.repeated_upper_ci is not None
    low, high = r.repeated_upper_ci
    assert low - 1e-6 <= r.repeated_limits[1] <= high + 1e-6


def test_one_image_gives_no_interval_rather_than_a_point() -> None:
    """Every resample of a single image holds the same readings: an interval
    from them is a point, which reads as certainty."""
    t = np.linspace(-10, 10, 30)
    r = audit(pairing(t, t + np.linspace(-2, 3, 30), ["one"] * 30), tilt(0, 1), 5,
              resamples=100, jitter_repeats=0)  # fmt: skip
    assert np.isnan(r.bias_ci).all() and np.isnan(r.gain_ci).all()
    assert "[no interval]" in r.summary()
    assert any("Only one image" in w for w in r.warnings)


def test_one_cluster_gives_no_rate_interval_either() -> None:
    """Wilson's interval would treat the readings of one image as independent:
    with one cluster the rates get no interval, like every other figure."""
    t = np.linspace(-10, 10, 30)
    r = audit(pairing(t, t + 1, ["one"] * 30), tilt(0, 1), 5, thresholds=[5],
              threshold_side="above", resamples=100, jitter_repeats=0)  # fmt: skip
    assert np.isnan(r.big_error_rate_ci).all()
    assert all(np.isnan(b.big_error_rate_ci).all() for b in r.size_bands)
    th = r.thresholds[0]
    assert np.isnan(th.sensitivity_ci).all() and np.isnan(th.precision_ci).all()
    assert "[no interval]" in r.to_markdown()
    from poseaudit.cli import _finite

    assert _finite(r.to_dict())["big_error_rate_ci"] == [None, None]


def test_too_few_images_are_flagged_like_too_few_clusters() -> None:
    r = audit(pairing([0] * 40, [1] * 40, [f"img{i % 5}" for i in range(40)]),
              tilt(0, 1), 5, resamples=100, jitter_repeats=0)  # fmt: skip
    assert any("Only 5 images" in w for w in r.warnings)
