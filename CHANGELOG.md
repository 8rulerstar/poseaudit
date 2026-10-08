# Changelog

## 0.1.3

- Pairing: dropping an image extension keeps the folder, so `cam1/0001.jpg`
  and `cam2/0001.jpg` are no longer paired as one image.
- README: the demo figures are described as disagreement with COCO's labels,
  not as error; a box example for `Instance`, a note that the demo data is not
  in the package, and references to markerless validation studies and Ronchi
  and Perona (2017).

## 0.1.2

- Pairing: boxes that qualify by IoU are ranked by the mean of IoU and OKS,
  so two people with nearly the same box are paired by their keypoints, not
  swapped. Ties go to OKS, then IoU, so the order of the lists no longer
  matters; an exact tie between different predictions is warned about. On
  the COCO elbow demo 5 of 490 pairs change: mean error 19.55 to 19.48°,
  large errors 43.5 to 43.2%; the README is updated.
- A warning when images overlap but no truth, or only a few images, pair.
- The rate of large errors and the decision rates resample whole images (or
  named clusters) whenever some hold several readings, as the other
  intervals do.
- `Instance` and `Instance.from_keypoints` refuse a visible point with a NaN
  or infinite coordinate; `from_keypoints` accepts lists.
- `audit` warns when every coordinate read lies within 0 to 1.
- `length` reads NaN for two points on one pixel, like the other measures.
- Keypoints given by name raise a clear error: measures take indices.
- Swapped `--gt` and `--pred` (or loader arguments) are named as such.
- `from_supervision` asks for an `{image name: sv.KeyPoints}` mapping when
  given a bare container.

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
