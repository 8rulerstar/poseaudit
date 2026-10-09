# Changelog

## Unreleased

- Several measures in one run: `--angle`, `--tilt`, `--length` and `--ratio`
  may be repeated. The output is one table with a row per measure (n, bias,
  percentile limits, mean |error|, RMSE, the large-error rate, slope, ICC);
  `--full` prints each measure in full. The JSON then holds the shared
  settings and a `measures` list of single-measure entries; with one measure
  it is unchanged.
- Model comparison: `pa.compare({name: predicted}, truth, measures,
  big_error)` and repeated `--pred NAME=PATH`. For each measure and pair of
  models, the differences in mean |error| and in the large-error rate on the
  readings both made, with paired bootstrap intervals over images (or named
  clusters) and the shared and per-model counts. `Comparison.table()` gives
  flat rows and `to_csv()` writes them. A pair whose shared readings come
  from fewer than 20 images or clusters, or from none, is warned about
  under the differences and in the JSON's `warnings`.
- `AuditResult.to_rows()`: the headline figures as a flat row.
- `Instance` and `Instance.from_keypoints` refuse keypoints not of shape
  (K, 2), such as x, y, v rows, and a `visible` not of shape (K,), naming the
  shape expected.
- A measure with a repeated keypoint, such as `angle(25, 25, 27)`, is refused
  when made; the two segments of a ratio may still share a point.
- A keypoint index out of range says how many keypoints there are and that
  indices count from 0; unpaired instances are checked too.
- `audit(noise_ratio=...)` refuses a ratio of 0 or less, as the CLI does.
- Normalised coordinates are judged per side on every visible point, with
  0.05 of slack for points just past the frame. When only one side looks
  normalised, the warning says the two sides are in different units, which
  is why nothing pairs.
- The BA slope, ICC(A,1) and CCC are NaN when the truth does not vary,
  instead of 2, 0 and 0 whatever the model does. A truth counts as flat only
  when its range is below 1e-12 of its size, so large values with a real
  spread keep their slopes.
- In resamples, repeated-readings limits count a cluster drawn twice as two
  clusters, not one twice the size.
- With a single image or cluster every interval is NaN (`null` in the JSON,
  `[no interval]` in the summary) instead of a single point, the Wilson
  intervals of rates included. Fewer than 20 images get the same warning as
  fewer than 20 named clusters.
- A warning when the percentile limits rest on 40 readings or fewer.
- With no large error, the warning says how high the rate could still be.
- The default summary leads with the figures of a method comparison: bias,
  the percentile limits and (when the tails are close to normal) the normal
  limits, mean |error| and RMSE, the large-error rate, the slope and
  ICC(A,1). The jitter reference and its p, the BA slope, Deming, CCC and r
  are under `--full`. Each limits line names its JSON key.
- In text the gain is called the slope (pred on truth, the proportional
  bias) and the robust gain the Theil-Sen slope; `p(squash)` is
  `p(slope <= jitter)`, which compares the slope with the jitter reference
  and does not by itself measure squashing. JSON keys and attribute names
  (`gain`, `robust_gain`, `jitter_p`) are unchanged.
- The JSON also gives the percentile limits as `percentile_limits` (the same
  as `empirical_limits`); `limits` stay the normal ones.
- With fewer than 10 readings the percentile limits print as n/a, and are
  `null` in the JSON (`percentile_limits` and `empirical_limits` both).
- When the truth hardly varies, a warning gives its range and says the slope,
  ICC and CCC are not defined; any figure that is not defined prints as n/a
  in the summary and the report, never as nan.
- With named clusters the repeated-readings limits are in the default
  summary, not only under `--full`.
- `AuditResult.percentile_limits`, `.slope` and `.theil_sen` are aliases of
  `empirical_limits`, `gain` and `robust_gain`, and `AuditResult` has a
  docstring listing its main fields.
- The warning that coordinates look normalised comes before any figure, in
  the summary and the report, and sizes are then not labelled in px.
- Intervals and limits never print as -0.000.
- CLI: `--plot` without matplotlib fails before the audit runs. On a console
  that is not UTF-8 (Git Bash on Korean Windows, for one) the degree sign
  prints as " deg", with the columns kept aligned. The plot hint reads
  `pip install "poseaudit[plot]"`.
- `report.md` opens with the headline in words, naming a relative reading as
  such and never rounding a rare large error to 0%, and says which way an
  error points. The settings are a table at the end instead of a Python dict.
  Size band labels share one number of decimals.
- CLI: `--image-size` also takes `WxH`, and an error names a malformed size
  such as `1280x`; `--band-by` has a help line; `poseaudit audit --help` ends
  with examples.
- README: the Quick start shows the exact default output, and the block
  below it the real `--full` output (both checked by a test); a Python API
  section; a runnable arrays snippet; recipes for a validation paper and a CI
  gate under the intro; the
  JSON gates fail on a `null` interval (in jq `null <= 3` is
  true), with a Python alternative to `jq`, and a note on line continuations
  in PowerShell.

## 0.1.4

- Pairing: a warning when unmatched truth images share their last name (after
  the last `/` or `\`, without an image extension) with unmatched predicted
  images, as `imgs/a.jpg` and `a` do: folders differ, so they are not paired.
  The warning for names that share nothing gives the same hint.
- Pairing: the example in the warning about dropped extensions no longer
  depends on the order images were listed in.
- When nothing pairs because one side has no visible point (all labels
  unlabelled, or every predicted point under the confidence cut), the warning
  says so instead of pointing at image names.
- Summary: "Only 1 reading", "No large errors" and "1 cluster" in the singular,
  and limits print as n/a when too few readings give none.
- README: the share of arms that COCO's label spread alone puts off by 15° or
  more (25.9 to 39.9%) beside the headline; `from_keypoints` described as
  giving no box; the OKS used for pairing differs from COCO's own.

## 0.1.3

- Pairing: dropping an image extension keeps the folder, so `cam1/0001.jpg`
  and `cam2/0001.jpg` are no longer paired as one image.
- On Windows, image names are kept as given when the extension is dropped.
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
