import numpy as np

import poseaudit as pa


def _one_reading() -> pa.AuditResult:
    rng = np.random.default_rng(0)
    xy = rng.uniform(100, 500, (17, 2))
    seen = np.ones(17, bool)
    box = np.array([0.0, 0.0, 600.0, 600.0])
    truth = {"a": [pa.Instance(bbox=box, keypoints=xy, visible=seen)]}
    pred = {"a": [pa.Instance(bbox=box, keypoints=xy + 3.0, visible=seen)]}
    return pa.audit(pa.pair(truth, pred), pa.angle(5, 7, 9), big_error=15)


def test_no_limits_means_no_share_outside_them(tmp_path) -> None:
    result = _one_reading()
    assert np.isnan(result.limits[0])
    assert all(np.isnan(result.tail_shares))
    assert np.isnan(result.limits_coverage)
    report = tmp_path / "r.md"
    result.to_markdown(report)
    assert "fall below the normal limits" not in report.read_text(encoding="utf-8")
