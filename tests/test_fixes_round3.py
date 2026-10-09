import json
from pathlib import Path

import numpy as np
import pytest

import poseaudit as pa
from poseaudit.confidence import clustered_bootstrap
from poseaudit.measures import length, ratio
from poseaudit.types import Instance

DEMO = Path(__file__).parent.parent / "examples" / "coco_elbow"
# the demo data is in the repository, not in the sdist: tests that read it
# skip there rather than fail
needs_demo = pytest.mark.skipif(
    not (DEMO / "gt_200.json").exists(), reason="demo data is not in this checkout"
)


def test_length_and_ratio_of_huge_coordinates_stay_finite() -> None:
    k = np.array([[0.0, 0.0], [3e300, 4e300], [0.0, 0.0], [6e300, 8e300]])
    with np.errstate(all="raise"):
        assert length(0, 1).read(k) == pytest.approx(5e300)
        assert ratio(0, 1, 2, 3).read(k) == pytest.approx(0.5)


@needs_demo
@pytest.mark.parametrize("field", ["id", "file_name"])
def test_a_repeated_image_is_refused(tmp_path, field) -> None:
    gt = json.loads((DEMO / "gt_200.json").read_text(encoding="utf-8"))
    gt["images"][1][field] = gt["images"][0][field]
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt), encoding="utf-8")
    with pytest.raises(ValueError, match="gt.json.*twice|gt.json.*two image ids"):
        pa.load_coco(path)


def test_a_mostly_undefined_statistic_has_no_interval() -> None:
    def statistic(idx):
        # defined only when both groups are drawn, as an ICC of two images
        return np.array([1.0 if len(set(idx)) == 2 else np.nan, float(len(idx))])

    low, high = clustered_bootstrap(["a", "b"], statistic, resamples=200)
    assert np.isnan(low[0]) and np.isnan(high[0])
    assert low[1] == high[1] == 2.0


def test_two_images_give_no_icc_interval() -> None:
    def person(angle_deg, x):
        a = np.radians(angle_deg)
        k = np.array([[100.0, 0], [0, 0], [100 * np.cos(a), 100 * np.sin(a)]])
        return Instance.from_keypoints(k + [x, 0], np.ones(3, bool))

    gt = {"i0": [person(40, 0)], "i1": [person(90, 0)]}
    pred = {"i0": [person(43, 0)], "i1": [person(95, 0)]}
    r = pa.audit(pa.pair(gt, pred), pa.angle(0, 1, 2), big_error=5)
    assert np.isfinite(r.icc) and np.isnan(r.icc_ci).all()
    assert np.isnan(r.ccc_ci).all()
