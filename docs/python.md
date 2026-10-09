# Python

```python
import poseaudit as pa

truth = pa.load_coco("person_keypoints_val2017.json")
predicted = pa.load_coco_results(
    "results.json", "person_keypoints_val2017.json", min_confidence=0.5
)

result = pa.audit(pa.pair(truth, predicted), pa.angle(5, 7, 9), big_error=15)
print(result.summary())  # summary(full=True) adds every agreement statistic
result.to_markdown("report.md")  # sizes, bands, decisions, largest errors
result.to_csv("readings.csv")  # one row per reading, to recompute anything
result.plot("panels.png")  # predicted against true, and a Bland-Altman plot
```

In a notebook, `result.plot()` with no file shows the figure inline. It
returns a Matplotlib figure, with or without a file, to change or save
yourself (`fig.savefig("panels.pdf")`); pyplot is not needed.

## Your own arrays

From your own arrays: `xy` of shape (K, 2) in pixel coordinates (not 0 to 1)
and a boolean `visible` of shape (K,) per object, here a 33-point skeleton:

```python
import numpy as np
import poseaudit as pa

rng = np.random.default_rng(0)
xy_true = rng.uniform(100, 500, (33, 2))  # your labelled points, in pixels
xy_pred = xy_true + rng.normal(0, 3, (33, 2))  # your model's points
visible_pred = np.ones(33, bool)

truth = {"frame_001": [pa.Instance.from_keypoints(xy_true, np.ones(33, bool))]}
predicted = {"frame_001": [pa.Instance.from_keypoints(xy_pred, visible_pred)]}
result = pa.audit(pa.pair(truth, predicted), pa.angle(23, 25, 27), big_error=10)
```

`from_keypoints` gives no box (`boxed=False`): such instances are paired by
keypoint closeness, not by box IoU, and the extent of the visible points serves
only as the truth's scale. If you have the detector's box, pass it, so a person
whose left and right points were swapped still pairs by box and shows up as a
large error rather than as unmatched:

```python
box = np.array([90.0, 80.0, 520.0, 530.0])  # the detector's x1, y1, x2, y2 in pixels
pa.Instance(bbox=box, keypoints=xy_pred, visible=visible_pred)
```

## Compare models

`compare` takes any number of prediction sets and compares every pair on the
readings both made of the same labelled instance: the difference in mean
|error| and in the large-error rate (first minus second, so below 0 the first
is closer to the truth), with paired intervals that resample whole images.

```python
c = pa.compare(
    {"yolo11n": predicted_n, "yolo11s": predicted_s},
    truth,
    measures=[pa.angle(5, 7, 9), pa.angle(6, 8, 10)],
    big_error=15,
)
print(c.summary())
rows = c.table()  # one dict per measure and pair of models
c.to_csv("differences.csv")
```

The same model under two detection-score filters shows why comparing on shared
readings matters. Run from `examples/coco_elbow`:

```python
import poseaudit as pa

truth = pa.load_coco("gt_200.json")
models = {
    f"score>={s}": pa.load_coco_results(
        "pred_yolo11n.json", "gt_200.json", min_confidence=0.5, min_score=s
    )
    for s in (0.0, 0.7)
}
c = pa.compare(models, truth, measures=[pa.angle(5, 7, 9)], big_error=15)
print(c.summary())
```

```text
score>=0.0:
measure        n  bias    limits              mean |error|             RMSE    >= 15°                  slope  ICC(A,1)
angle 5,7,9  322  +2.27°  -50.90° to +79.99°  19.48° [17.05 to 22.13]  29.57°  43.2% [37.9% to 48.6%]  0.731  0.762
  ! angle 5,7,9: 2.5% of errors fall below the normal limits and 4.3% above them, against 2.5% each for a normal error: use the percentile limits.

score>=0.7:
measure        n  bias    limits              mean |error|             RMSE    >= 15°                  slope  ICC(A,1)
angle 5,7,9  254  +2.86°  -41.80° to +65.51°  17.32° [14.97 to 19.66]  26.00°  39.8% [33.9% to 45.9%]  0.787  0.816
  ! angle 5,7,9: 126 of 380 labelled instances were not read: the figures describe the ones that were, which are usually easier.
  ! angle 5,7,9: 2.4% of errors fall below the normal limits and 4.3% above them, against 2.5% each for a normal error: use the percentile limits.

measure      a - b                    shared  n a  n b  mean |error| a - b       large-error rate a - b
angle 5,7,9  score>=0.0 - score>=0.7     254  322  254  -0.06° [-0.20 to +0.00]  +0.0 pt [+0.0 to +0.0]
  ! angle 5,7,9, score>=0.0 - score>=0.7: the models read only 1 of the 254 shared readings differently; the paired intervals rest on that one and can be far too narrow.
```

Dropping detections scored under 0.7 lowers the mean |error| from 19.48° to
17.32°. The 254 readings both share are the same but for one, and the two
differ by -0.06°. The filter did not make the elbows more accurate. It removed 68 people that were read
worse. A plain mean over each set would call that a 2° improvement.

## Python API

- Loaders give `{image name: [Instance, ...]}`: `load_coco(annotations)`,
  `load_coco_results(results, annotations, min_confidence=0.5)`,
  `load_yolo(folder, image_size, num_keypoints)`, `from_supervision(...)`, or
  `Instance.from_keypoints(xy, visible)` for your own arrays.
- `pair(truth, predicted)` matches people image by image (box IoU, or keypoint
  similarity without boxes) and returns a `Pairing`.
- Measures: `angle(a, b, c)`, `tilt(a, b)`, `length(a, b)`,
  `ratio(a, b, c, d)`, with keypoint indices counted from 0.
- `audit(pairing, measure, big_error=...)` returns an `AuditResult`. Its main
  fields: `n`; `bias` and `median_error`; `percentile_limits` and the normal
  `limits`; `mean_abs_error`, `rmse`; `big_error_rate`; `slope` and
  `theil_sen` (`gain` and `robust_gain` in the JSON); `icc`, `ccc`;
  `size_bands`; `readings` and `worst()`; `warnings`. Most figures have a
  `_ci` interval. `summary()`, `to_markdown()`, `to_csv()`,
  `to_dict()` and `plot()` write it out (`plot()` also returns the
  Matplotlib figure, shown inline in a notebook), and `to_rows()` gives the
  headline figures as a flat row to stack into a table. In Jupyter or
  IPython a result on its own shows its summary; `repr()` is one line.
- `compare({name: predicted}, truth, measures, big_error)` returns a
  `Comparison`. With measures in different units `big_error` is a mapping
  by kind (`angle`, `tilt`, `length`, `ratio`) or unit (`deg`, `px`), such
  as `{"angle": 15, "length": 10, "ratio": 0.1}`; one number for degrees and
  pixels alike is refused. The `Comparison` holds `results` (each model's
  `AuditResult`s, one per measure), `differences`, `table()` (flat rows) and
  `to_csv()`. It too shows its summary in Jupyter.
- Types: a `Dataset` is `{image name: [Instance, ...]}`. `Pairing` holds the
  `Pair`s made by `pair()`. A `Measure` is what `angle()` and the others
  return. An `AuditResult` holds its `readings` (each a `Reading`), why
  instances were not read (`not_read`, a `NotRead` of counts by reason), its
  `bands` by level of the measure (each a `Band`) and its `thresholds` (one
  `ThresholdAgreement` per threshold given). A `Comparison` holds one `Difference` per measure and pair of
  models.
- `pa.__version__` is the installed version; the JSON and the report record
  it, with every setting and the `seed`, for a methods section.
- What the outputs reveal: the report and the JSON record `--gt`, `--pred`
  and `--images` as typed, so an absolute path shows your user name; pass
  relative paths before sharing them. The CSV, the JSON and the report's
  largest-error table name images by file name. The PNG holds only the plot
  and Matplotlib's version.
- The same inputs and `seed` give the same CSV and the same figures on any
  Python version. A different NumPy can change the last digit or two of a
  float in the JSON (seen: the Bland-Altman slope, around 1e-17), never a
  reported figure.
