"""Round 7: non-finite boxes, EXIF-rotated images, fractional supervision
classes, cluster names, conflicting image sizes, and comparison invariants."""

import csv

import numpy as np
import pytest

from poseaudit import Instance, angle, audit, compare, pair, tilt
from poseaudit.cli import main
from poseaudit.io import from_supervision, load_yolo

ALL = np.ones(3, bool)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_box_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        Instance([0, 0, bad, 40], np.zeros((3, 2)), ALL)


def _data(n=30, seed=1):
    rng = np.random.default_rng(seed)
    truth, pred = {}, {}
    for i in range(n):
        name = f"clip{i % 6}_f{i:03d}"
        kp = rng.uniform(0, 200, (3, 2))
        truth[name] = [Instance([0, 0, 200, 200], kp, ALL)]
        noisy = kp + rng.normal(0, 3, (3, 2))
        pred[name] = [Instance([0, 0, 200, 200], noisy, ALL)]
    return truth, pred


def test_renaming_clusters_does_not_change_the_intervals() -> None:
    truth, pred = _data()
    names = {"clip0": "z", "clip1": "a", "clip2": "m", "clip3": "b", "clip4": "y"}
    names["clip5"] = "c"
    p = pair(truth, pred)
    one = audit(p, angle(0, 1, 2), 5, cluster=lambda n: n.split("_")[0],
                resamples=200, jitter_repeats=0)  # fmt: skip
    two = audit(p, angle(0, 1, 2), 5, cluster=lambda n: names[n.split("_")[0]],
                resamples=200, jitter_repeats=0)  # fmt: skip
    # the summary per cluster names the clusters; its figures are the same
    skip = {"settings": None, "by_subject": None}
    assert one.to_dict() | skip == two.to_dict() | skip
    figures = [sorted((g.n, g.bias) for g in r.by_subject) for r in (one, two)]
    assert figures[0] == figures[1]


def test_a_regex_string_as_cluster_is_explained() -> None:
    truth, pred = _data()
    with pytest.raises(TypeError, match="function or a mapping"):
        audit(pair(truth, pred), angle(0, 1, 2), 5, cluster=r"^(clip\d)_")


def test_a_model_compared_with_itself_differs_by_nothing() -> None:
    truth, pred = _data()
    c = compare({"x": pred, "y": pred}, truth, [angle(0, 1, 2), tilt(0, 1)], 5,
                resamples=100, jitter_repeats=0)  # fmt: skip
    for d in c.differences:
        assert d.n_shared == 30
        assert d.mean_abs_error_diff == 0 and d.big_error_rate_diff == 0
        assert d.mean_abs_error_diff_ci == (0, 0)


def test_an_instance_one_model_cannot_read_is_left_out_of_the_shared() -> None:
    truth, pred = _data()
    blind = dict(pred)
    first = sorted(blind)[0]
    kp = blind[first][0].keypoints
    blind[first] = [Instance([0, 0, 200, 200], kp, np.array([True, False, True]))]
    c = compare({"x": pred, "y": blind}, truth, angle(0, 1, 2), 5,
                resamples=100, jitter_repeats=0)  # fmt: skip
    (d,) = c.differences
    assert (d.n_shared, d.n_a, d.n_b) == (29, 30, 29)


class _KeyPoints:
    def __init__(self, xy, class_id):
        self.xy, self.class_id = xy, class_id
        self.visible = None
        self.keypoint_confidence = None


def test_a_fractional_supervision_class_is_refused() -> None:
    xy = np.array([[[10, 10], [20, 20], [30, 15]]], float)
    with pytest.raises(ValueError, match="whole number"):
        from_supervision({"a": _KeyPoints(xy, np.array([0.5]))})
    (inst,) = from_supervision({"a": _KeyPoints(xy, np.array([2.0]))})["a"]
    assert inst.class_id == 2


def _label(folder, name):
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.txt").write_text(
        "0 0.5 0.5 0.2 0.4 0.45 0.4 2 0.5 0.5 2 0.6 0.45 2\n", encoding="utf-8"
    )


def test_an_exif_rotated_image_gives_its_shown_size(tmp_path, capsys) -> None:
    pil = pytest.importorskip("PIL.Image")
    images = tmp_path / "img"
    images.mkdir()
    image = pil.new("RGB", (640, 480))
    exif = image.getexif()
    exif[0x0112] = 6  # stored on its side, shown 480 x 640
    image.save(images / "e.jpg", exif=exif)
    _label(tmp_path / "gt", "e")
    _label(tmp_path / "pr", "e")
    out = tmp_path / "s.csv"
    main(f"audit --format yolo --gt {tmp_path / 'gt'} --pred {tmp_path / 'pr'} "
         f"--keypoints 3 --images {images} --length 0,2 --big-error 5 "
         f"--resamples 50 --jitter-repeats 0 --csv {out}".split())  # fmt: skip
    (row,) = csv.DictReader(out.read_text("utf-8").splitlines())
    # 0.15 of 480 across and 0.05 of 640 down
    assert float(row["truth"]) == pytest.approx(np.hypot(0.15 * 480, 0.05 * 640))


def test_images_and_image_size_together_are_refused(tmp_path) -> None:
    _label(tmp_path / "gt", "a")
    (tmp_path / "img").mkdir()
    with pytest.raises(SystemExit, match="not both"):
        main(f"audit --format yolo --gt {tmp_path / 'gt'} --pred {tmp_path / 'gt'} "
             f"--keypoints 3 --images {tmp_path / 'img'} --image-size 640 480 "
             "--length 0,2 --big-error 5".split())  # fmt: skip


def test_an_empty_label_file_needs_no_image(tmp_path) -> None:
    _label(tmp_path, "a")
    (tmp_path / "stray.txt").write_text("\n", encoding="utf-8")

    def size_of(name):
        if name != "a":
            raise ValueError(f"no image for {name}")
        return (640, 480)

    data = load_yolo(tmp_path, size_of, 3)
    assert data["stray"] == [] and len(data["a"]) == 1
