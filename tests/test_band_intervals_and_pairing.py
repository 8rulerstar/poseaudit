"""Size bands resample whole clusters, as the overall figures do; the summary
gives the percentile limits' intervals; pairing says when keypoints, not box
overlap alone, chose among several predictions for one label."""

import numpy as np

import poseaudit as pa
from poseaudit import angle, audit, pair
from poseaudit.confidence import wilson

BOX = np.array([0.0, 0.0, 1000.0, 1000.0])
SEEN = np.ones(3, bool)


def _points(degrees: float, length: float) -> np.ndarray:
    """Shoulder, elbow, wrist: an arm bent to `degrees`, both segments
    `length` px long."""
    elbow = np.array([500.0, 500.0])
    turn = np.radians(degrees)
    wrist = elbow + length * np.array([np.cos(turn), np.sin(turn)])
    return np.array([elbow + (length, 0.0), elbow, wrist])


def _clips(clips: int = 20, frames: int = 10, seed: int = 0):
    """Clips whose every frame is read alike: off by 30 degrees in the even
    clips and by about 1 in the odd ones. Half of each clip's arms are 40 px,
    half 100 px."""
    rng = np.random.default_rng(seed)
    truth, predicted = {}, {}
    for c in range(clips):
        for k in range(frames):
            name = f"clip{c:02d}_{k:03d}"
            degrees = rng.uniform(40, 150)
            length = 40.0 if k < frames // 2 else 100.0
            off = 30.0 if c % 2 == 0 else rng.normal(0, 1)
            truth[name] = [pa.Instance(BOX, _points(degrees, length), SEEN)]
            predicted[name] = [pa.Instance(BOX, _points(degrees + off, length), SEEN)]
    return pair(truth, predicted)


def _from_clips(per_clip, frames: int = 10, seed: int = 0):
    """One clip of `frames` arms for each (length, off, low, high): segments
    `length` px long, bent between `low` and `high` degrees and read `off`
    degrees off."""
    rng = np.random.default_rng(seed)
    truth, predicted = {}, {}
    for c, (length, off, low, high) in enumerate(per_clip):
        for k in range(frames):
            name = f"clip{c:02d}_{k:03d}"
            degrees = rng.uniform(low, high)
            truth[name] = [pa.Instance(BOX, _points(degrees, length), SEEN)]
            predicted[name] = [pa.Instance(BOX, _points(degrees + off, length), SEEN)]
    return pair(truth, predicted)


def _clip(name: str) -> str:
    return name.split("_")[0]


FAST = {"resamples": 500, "jitter_repeats": 0, "size_bands": [60]}


def test_a_size_band_rate_resamples_whole_clusters() -> None:
    """The bands' rates were Wilson intervals, which take every reading as
    independent: with errors shared within a clip they covered the true rate
    about half the time. They now come from the cluster bootstrap, never
    narrower than Wilson's."""
    r = audit(_clips(), angle(0, 1, 2), 15, cluster=_clip, **FAST)
    assert [b.n for b in r.size_bands] == [100, 100]
    for band in r.size_bands:
        big = round(band.big_error_rate * band.n)
        low, high = wilson(big, band.n)
        band_low, band_high = band.big_error_rate_ci
        assert band_low <= low and band_high >= high
        assert band_high - band_low > 1.5 * (high - low)


def test_a_size_band_rate_stays_wilson_with_one_reading_to_a_cluster() -> None:
    r = audit(_clips(), angle(0, 1, 2), 15, **FAST)
    for band in r.size_bands:
        big = round(band.big_error_rate * band.n)
        assert band.big_error_rate_ci == wilson(big, band.n)


def test_a_size_band_in_a_few_clusters_has_no_interval() -> None:
    """The band under 60 px is two clips, one read all wrong and one all
    right: it is missing from about 13% of the resamples. Its rate used to
    fall back to Wilson's interval for 20 independent readings (30% to 70%);
    with its readings in two clusters there is none."""
    small = [(40.0, 30.0, 40, 140), (40.0, 0.5, 40, 140)]
    large = [(100.0, 30.0 if c % 2 == 0 else 0.5, 40, 140) for c in range(38)]
    r = audit(_from_clips(small + large), angle(0, 1, 2), 15, cluster=_clip, **FAST)
    few, many = r.size_bands
    assert (few.n, few.big_error_rate) == (20, 0.5)
    assert np.isnan(few.big_error_rate_ci).all() and np.isnan(few.gain_ci).all()
    assert np.isfinite(many.big_error_rate_ci).all()
    section = r.to_markdown().split("## Error by size")[1].split("\n## ")[0]
    (row,) = [x for x in section.splitlines() if x.startswith("| 0 to 60 px |")]
    assert "| 50.0% [no interval] |" in row


def test_a_decision_rate_in_a_few_clusters_has_no_interval() -> None:
    """Only two of 40 clips hold arms bent below 60 degrees, so the catch
    rate is undefined in about 13% of the resamples: no interval, as for
    the size bands, rather than Wilson's for 20 independent readings."""
    bent = [(100.0, 0.5, 30, 50)] * 2
    open_ = [(100.0, 0.5, 80, 140)] * 38
    r = audit(
        _from_clips(bent + open_), angle(0, 1, 2), 15, cluster=_clip,
        thresholds=[60], threshold_side="below", **FAST,
    )  # fmt: skip
    assert r.thresholds
    for row in r.thresholds:
        assert row.true_positive == 20
        assert np.isnan(row.sensitivity_ci).all()
        assert np.isnan(row.precision_ci).all()


def test_a_size_band_slope_has_its_interval() -> None:
    r = audit(_clips(), angle(0, 1, 2), 15, cluster=_clip, **FAST)
    report = r.to_markdown()
    section = report.split("## Error by size")[1].split("\n## ")[0]
    for band in r.size_bands:
        low, high = band.gain_ci
        assert low < band.gain < high
        assert f"{band.gain:.3f} [{low:.3f} to {high:.3f}] |" in section
    assert "stay Wilson" not in report
    assert r.to_dict()["size_bands"][0]["gain_ci"] == r.size_bands[0].gain_ci


def test_the_report_says_how_size_bands_are_resampled() -> None:
    """One sentence names the band rates among the rates that resample whole
    clusters, rather than a sentence after Wilson's that read as if they
    used Wilson's; paired values have no size bands to name."""
    with_bands = (
        "The rates of large errors, overall and per size band, and the "
        "decision rates resample whole"
    )
    without = "The overall rate of large errors and the decision rates resample whole"
    keypoints = audit(_clips(), angle(0, 1, 2), 15, cluster=_clip, **FAST)
    text = " ".join(keypoints.to_markdown().split())
    assert with_bands in text and "rates per size band." not in text
    values = pa.audit_values(np.arange(30.0) + 1, np.arange(30.0), 5, resamples=50)
    assert not values.size_bands
    text = " ".join(values.to_markdown().split())
    assert without in text and "size band" not in text


def test_the_full_summary_gives_the_percentile_limits_intervals() -> None:
    r = audit(_clips(), angle(0, 1, 2), 15, cluster=_clip, **FAST)
    lines = r.summary(full=True).splitlines()
    (first,) = [x for x in lines if x.startswith("  limit CIs")]
    low, high = r.empirical_lower_ci, r.empirical_upper_ci
    assert first == (
        f"  limit CIs    lower [{low[0]:+.2f} to {low[1]:+.2f}], upper "
        f"[{high[0]:+.2f} to {high[1]:+.2f}] (percentile, cluster bootstrap)"
    )
    following = lines[lines.index(first) + 1]
    assert following.startswith("               lower [")
    assert following.endswith("(normal, cluster bootstrap)")


def test_few_readings_give_no_percentile_limit_interval() -> None:
    r = audit(_clips(clips=4, frames=2), angle(0, 1, 2), 15, **FAST)
    text = r.summary(full=True)
    assert "(percentile, cluster bootstrap)" not in text
    assert "(normal, cluster bootstrap)" in text


# --- pairing among several predictions -----------------------------------------


def _person(box, points) -> pa.Instance:
    return pa.Instance(np.array(box, float), np.array(points, float), SEEN)


LABEL = [[20.0, 40.0], [50.0, 100.0], [80.0, 160.0]]


def test_a_warning_counts_labels_whose_pair_the_keypoints_chose() -> None:
    """Among two predictions of one person, the one whose keypoints sit
    closest to the label wins although the other's box overlaps it more:
    the audited model is scored on its best try. The pairing stays as it was;
    a warning now says so."""
    truth = {"a.jpg": [_person([0, 0, 100, 200], LABEL)]}
    near_box = _person([0, 0, 100, 210], np.array(LABEL) + 40)  # IoU 0.95
    near_points = _person([0, 0, 100, 250], LABEL)  # IoU 0.80, the label's points
    pairing = pair(truth, {"a.jpg": [near_box, near_points]})
    (chosen,) = pairing.pairs
    assert chosen.predicted_index == 1
    (warning,) = pairing.warnings
    assert warning.startswith(
        "1 labelled instance had two or more predictions overlapping enough to "
        "pair, and keypoint similarity (OKS), not box overlap alone, chose "
        "which one: the prediction box overlap alone would have taken was left "
        "unpaired (e.g. 'a.jpg')."
    )
    assert "--min-score" in warning


def test_no_warning_when_the_keypoints_swap_two_people() -> None:
    """Two people whose boxes nearly coincide, each prediction's box on one
    of them and its points on the other. Box overlap alone would swap them;
    the keypoints pair them right, and no prediction is left over, so no
    try was dropped. Both used to be counted as best-of-several tries."""
    other = np.array(LABEL) + [60.0, 0.0]
    truth = {
        "a.jpg": [_person([0, 0, 100, 200], LABEL), _person([10, 0, 110, 200], other)]
    }
    predicted = {
        "a.jpg": [_person([0, 0, 100, 200], other), _person([10, 0, 110, 200], LABEL)]
    }
    pairing = pair(truth, predicted)
    pairs = sorted((p.truth_index, p.predicted_index) for p in pairing.pairs)
    assert pairs == [(0, 1), (1, 0)]
    assert pairing.warnings == []


def test_only_the_label_with_a_try_left_over_is_counted() -> None:
    """The same two people and a third prediction close to the first. The
    first person's pair by box overlap alone went to the second person, so
    it is not counted; the second's was left over, so it is."""
    other = np.array(LABEL) + [60.0, 0.0]
    truth = {
        "a.jpg": [_person([0, 0, 100, 200], LABEL), _person([10, 0, 110, 200], other)]
    }
    predicted = {
        "a.jpg": [
            _person([0, 0, 100, 200], other),
            _person([10, 0, 110, 200], LABEL),
            _person([0, 0, 100, 202], np.array(LABEL) + [5.0, 0.0]),
        ]
    }
    pairing = pair(truth, predicted)
    pairs = sorted((p.truth_index, p.predicted_index) for p in pairing.pairs)
    assert pairs == [(0, 2), (1, 0)]
    (warning,) = pairing.warnings
    assert warning.startswith("1 labelled instance had")


def test_no_warning_when_box_overlap_alone_pairs_alike() -> None:
    truth = {"a.jpg": [_person([0, 0, 100, 200], LABEL)]}
    better = _person([0, 0, 100, 210], LABEL)
    worse = _person([0, 0, 100, 250], np.array(LABEL) + 40)
    assert pair(truth, {"a.jpg": [better, worse]}).warnings == []
    assert pair(truth, {"a.jpg": [worse]}).warnings == []
