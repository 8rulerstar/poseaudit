# Changelog

## 0.1.1

- Python 3.14 is tested in CI and listed as supported.
- PyPI page links to the source, the issue tracker and this changelog.
- The example and validation scripts read and write their JSON as UTF-8, so
  they behave the same on systems whose default encoding is not UTF-8.

## 0.1.0

First release.

- Measures: tilt, angle, length and ratio, optionally relative to the rest of
  each image (median or a percentile, of values or of their absolute values,
  leaving each reading out of its own baseline).
- Inputs: COCO annotations and results, Ultralytics YOLO pose, supervision
  `sv.KeyPoints` (with `sv.Detections` for boxes), or arrays; class filters
  and class-aware matching.
- Agreement: error, large-error rate, bias, percentile and normal limits of
  agreement, repeated-measures limits, ICC(A,1), CCC, Pearson r, error by size
  of the measured part.
- Gain and squashing: least-squares and robust (Theil-Sen) gain, Bland-Altman
  and Deming slopes, and the gain keypoint jitter alone gives, with the gap's
  interval and a one-sided p(squash).
- Decisions at thresholds, with a separate or count-matched predicted
  threshold.
- Intervals resample whole images or named clusters.
- Output: summary, markdown report, CSV, JSON and a two-panel plot.
