# Command line

All examples run from [`examples/coco_elbow`](../examples/coco_elbow), on the
demo data. The commands continue lines with bash's `\`; in PowerShell put the
command on one line, or end each line with a backtick instead.

## On your own files

```bash
poseaudit audit --format coco --gt annotations.json --pred results.json \
    --min-conf 0.5 --angle 5,7,9 --big-error 15 --size-bands 30,60 \
    --report report.md --plot panels.png --csv readings.csv
```

`--big-error` is required: the error, in the measure's unit, that counts as
large. `report.md` explains every line of the summary. `poseaudit audit --help`
lists every option with examples.

## Every statistic: --full

`--full` adds the jitter reference and every agreement statistic. They are
explained in [statistics.md](statistics.md).

```text
$ poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json \
    --angle 5,7,9 --big-error 15 --size-bands 30,60 --full
angle (5, 7, 9): read 322 of 380 labelled instances
  not read     no matching prediction 35, prediction lacked a point 23, unmeasurable 0
  not counted  206 with a point unlabelled in the truth, 79 unmatched predictions
  bias         +2.27° [-1.21 to +6.15], median +0.52°
  limits       -50.90° to +79.99° (percentile, 2.5th to 97.5th; JSON percentile_limits)
  normal       -55.60° to +60.14° (bias +/- 1.96 SD; JSON limits)
  |error|      mean 19.48° [17.05 to 22.13], median 12.59°, 95th pct 68.99°
  RMSE         29.57° [25.30 to 33.94]
  >= 15°       43.2% [37.9% to 48.6%]
  by size      0-30 px 58% (n 134), 30-60 px 37% (n 112), 60+ px 26% (n 76)
  slope        0.731 [0.639 to 0.814] (pred on truth, 1 is ideal); Theil-Sen 0.786
  ICC(A,1)     0.762 [0.677 to 0.827]
  vs jitter    0.871 from keypoint jitter alone; gap -0.140 [-0.199 to -0.076], p(slope <= jitter) <= 0.002*
               * swaps and gross failures lower the slope too, and noisy
                 labels make p small for an honest model: read the gap
  BA slope     -0.049 [-0.119 to +0.016]
  agreement    CCC 0.761 [0.677 to 0.826], r 0.763
  ! 2.5% of errors fall below the normal limits and 4.3% above them, against 2.5% each for a normal error: use the percentile limits.
```

## Several joints

Repeat `--angle`, `--tilt`, `--length` or `--ratio` to read several measures
off the same pairing. The output is one row per measure; `--full` prints each
measure's full block instead. Both elbows and both knees on the demo data:

```text
$ poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json \
    --angle 5,7,9 --angle 6,8,10 --angle 11,13,15 --angle 12,14,16 --big-error 15
measure         n    bias    limits              mean |error|             RMSE    >= 15°                  slope  ICC(A,1)
angle 5,7,9     322  +2.27°  -50.90° to +79.99°  19.48° [17.05 to 22.13]  29.57°  43.2% [37.9% to 48.6%]  0.731  0.762
angle 6,8,10    308  +3.74°  -54.10° to +75.45°  18.07° [15.54 to 20.67]  30.11°  34.7% [29.5% to 40.2%]  0.751  0.803
angle 11,13,15  243  +5.19°  -38.92° to +80.06°  16.40° [12.93 to 20.26]  29.80°  30.5% [24.4% to 37.2%]  0.715  0.794
angle 12,14,16  242  +4.63°  -37.07° to +69.46°  14.62° [11.87 to 17.66]  24.38°  31.8% [25.0% to 39.0%]  0.796  0.871
  ! angle 5,7,9: 2.5% of errors fall below the normal limits and 4.3% above them, against 2.5% each for a normal error: use the percentile limits.
  ! angle 6,8,10: 2.3% of errors fall below the normal limits and 4.2% above them, against 2.5% each for a normal error: use the percentile limits.
  ! angle 11,13,15: 1.6% of errors fall below the normal limits and 4.1% above them, against 2.5% each for a normal error: use the percentile limits.
  ! angle 12,14,16: 1.7% of errors fall below the normal limits and 5.0% above them, against 2.5% each for a normal error: use the percentile limits.
```

## Compare models

Give each model as `--pred NAME=PATH`. Each model gets its table, and every
pair of models is compared on the readings both made of the same labelled
instance: the difference in mean |error| and in the large-error rate (first
model minus second, so below 0 the first is closer to the truth), with paired
intervals that resample whole images, the number of shared readings and each
model's own count. When the shared readings come from fewer than 20 images
(or clusters) a warning under the table says the paired intervals are too
narrow.

```bash
poseaudit audit --format coco --gt gt.json --angle 5,7,9 --angle 6,8,10 \
    --pred yolo11n=res_n.json --pred yolo11s=res_s.json --big-error 15 \
    --csv differences.csv
```

`--report` and `--plot` describe one model, so with several `--pred` they are
refused; `--csv` and `--json` hold the comparison.

The `--pred NAME=PATH` form compares prediction files. To compare one file
under two settings (a score filter, a confidence threshold), use Python:
[python.md](python.md#compare-models).

## Options

### Decisions at a threshold

When a measurement feeds a rule (flag a tilt of 10° or more, an elbow bent past
90°), what matters is whether the prediction lands on the same side of it. The
"truth" side is the same rule applied to the truth keypoints, not a defect
label: an inspector's pass or fail verdict can differ from any angle rule. A
model that reads angles differently may need its own threshold: give one with
`--pred-threshold`, and `poseaudit` also finds the one that flags as many
instances as the truth does. That one is fitted on the data it is scored on,
so its figures are optimistic until checked on other data. Only readings
count: a labelled instance that was not read is neither caught nor missed.

```text
$ poseaudit audit ... --threshold 90 --side below --pred-threshold 100
  below 90°: caught 53 of 84 (63% [52% to 74%]), false alarms 11 of 238
  below 90°, predicted 100°: caught 63 of 84 (75% [65% to 85%]), false alarms 23 of 238
  below 90°, predicted 98.40° (same count, fitted here): caught 61 of 84 (73% [62% to 83%]), false alarms 23 of 238
```

### Tilts relative to the rest of the photo

If a rule compares each object with the others in its photo, a tilted camera
moves every tilt together. `--relative-to median` reads each tilt against the
median tilt of the other objects in the image, on the truth and the prediction
separately, so a camera roll drops out. `--relative-to p20 --relative-abs`
reads how much more each object leans than the 20th percentile of the others'
absolute tilts. Each object's baseline leaves the object itself out, so none
is scored against itself. With `--relative-abs` a value below 0 means
straighter than the baseline, so band edges should start below 0. This is for
tilts; an angle inside a limb does not change when the camera rolls, and the
report says so.

### Several readings of one subject

Frames of one video, or photos of one subject, are not independent. Name the
group with `--cluster REGEX` (its first capture group in the image name) or
`cluster=` in Python, for example `cluster=lambda name: name.split("_")[0]`
for names such as `athlete3_trial2_f0041`. Intervals then resample whole
groups, and limits of agreement for repeated readings (Bland and Altman 2007)
are added. Leave the groups out and every frame counts as independent: the
intervals come out far too narrow, and nothing warns about it.

By default the group is the image. With fewer than 20 groups the intervals are
too narrow, and the report says so; nested groups (subject, trial, frame) are
not supported. With named groups the jitter reference gives no p: it treats
readings as independent, and when a subject carries the same error from
reading to reading it would flag honest models as squashed far more often than
it says. Read the gap's interval, which resamples whole groups.

## Measures

| measure | points | reads |
|---|---|---|
| `tilt(a, b)` | 2 | degrees of the axis through a and b from vertical, -90 to 90, positive when the upper end leans right; the axis has no direction, so errors wrap (89° vs -89° is 2° off) and a part read upside down is not an error |
| `angle(a, b, c)` | 3 | interior angle at `b`, 0 to 180° (180 is straight; a flexion angle is 180 minus this, which flips the sign of the bias); unsigned, so an angle read bending the other way is not an error |
| `length(a, b)` | 2 | pixels; two points on one pixel are unmeasurable, not 0 |
| `ratio(a, b, c, d)` | 4 | \|a-b\| / \|c-d\|, independent of scale |

The size of a reading (the "arm segment" and "by size" figures above) is the
mean length, in the truth, of the segments its points span: a-b and b-c for
an angle, a-b for a tilt or length, a-b and c-d for a ratio. There
is no directed angle (0 to 360) and no fixed axis other than vertical yet;
`--relative-to` reads tilts against the rest of the image instead.

## Inputs

| source | loader |
|---|---|
| COCO keypoint annotations | `load_coco(path, classes)` (crowd regions skipped) |
| COCO results format | `load_coco_results(path, annotations_path, min_confidence, classes)` |
| Ultralytics YOLO pose labels or predictions | `load_yolo(folder, image_size, num_keypoints, classes, min_confidence)` |
| `sv.KeyPoints` (supervision 0.30.6+) | `from_supervision(keypoints_by_image, detections_by_image=None)` |
| anything else | `Instance.from_keypoints(xy, visible)` per object, in a dict of image name to list |

COCO annotations with no labelled keypoint (`num_keypoints` 0) and crowd
regions are skipped as they load, so they appear in no count. A warning
gives the number of crowd regions left out: a prediction on one has no
truth to pair with and counts among the unmatched predictions.

- **Bad values.** NaN or infinite coordinates, scores or boxes stop the load
  with the file and the row.
- **Confidence.** In a results file or a YOLO prediction the third value of a
  point is a confidence. Without a threshold every returned point counts as
  seen, and a warning says so.
- **Missing points.** A YOLO point written as `0 0` is missing, as in
  supervision.
- **Pixels.** Angles are measured on the coordinates given. YOLO coordinates
  are fractions of width and height, so an angle measured on them is
  distorted on any non-square image. Give one size, a function of the image
  name, or `--images DIR` to read each image's size. A size given in the
  wrong order (height, width) passes every check and quietly distorts the
  figures (on upright parts it made them look better); `--images` avoids that.
  Arrays passed to `Instance.from_keypoints` must be in pixels too.
- **Classes.** Instances of different classes are not paired, and an audit
  whose readings mix classes stops with an error: parts of different kinds can
  hide each other's errors. Filter with `--classes` (any format), or allow it
  with `--mixed-classes`. When the two sides number classes differently (COCO
  calls a person 1, Ultralytics 0) they are matched ignoring classes, with a
  warning. A class filter that keeps nothing names the classes that are
  there.
- **Wrong keypoint counts.** A YOLO load stops on coordinates outside the
  image, which usually means `--keypoints` is wrong, and on a missing folder.
  If every row could also be read as fewer points in x y v form with flags 0,
  1 and 2, a warning asks you to check `--keypoints`.
- **Skeletons** of different sizes on the two sides are refused. Keypoints are
  named by index, counted from 0, in Python and on the command line; with a
  list of names, pass `names.index("left_elbow")`.
- **Swapped files.** A results list given as the truth, or an annotation file
  given as the predictions, stops the load with a message saying so.
- **supervision.**
  - The container's `visible` mask decides visibility: filter with
    `kp.visible = kp.keypoint_confidence > 0.5`, not with a 2D mask such as
    `kp[mask]`, which drops columns and so changes what each keypoint index
    means.
  - `from_ultralytics` drops the model's boxes; to match by box, pass
    `sv.Detections.from_ultralytics` of the same result, row for row (not
    `kp.as_detections()`, which drops rows with no visible point). Boxes a
    model keeps in `kp.data["xyxy"]` (RF-DETR) are used when no detections
    are given; RF-DETR also fills `visible` with confidence > 0, so set your
    own threshold on it.
  - When both carry the model's scores, rows whose scores rank differently
    stop the load. Without scores only geometry is left: two rows whose
    keypoints clearly fit each other's boxes better than their own give a
    warning, which crowded human labels can also trip.

## How pairs, counts and intervals are made

- Predictions are matched to the truth within each image, by image name and
  never by list position (a file extension is ignored when only that differs,
  with a warning; folders are not, so `cam1/0001.jpg` and `cam2/0001.jpg` stay
  apart, and a warning says when unmatched images share their last name with
  the other side's, as `imgs/a.jpg` and `a` do). A pair qualifies by box IoU (`--min-iou`, default 0.3), or
  by keypoint closeness when a side has no box (`--min-similarity`, default
  0.5). Within an image the best qualifying pairs are taken first, ranked by
  the mean of IoU and OKS (COCO's sigmas for 17 points, their mean for other
  skeletons; the truth box's area as scale, where COCO uses the segment area,
  so the values differ from COCO's own OKS), so two people with nearly the
  same box are told apart by their keypoints. Remaining ties go to keypoint
  similarity, then IoU, so the order instances are listed in does not matter;
  an exact tie between different predictions is warned about. A warning also
  says when images overlap but nothing, or only a few images' worth, pairs.
  Detection scores are ignored
  in matching: drop low-scoring detections first (`--min-score`, or
  `min_score=` in the loaders), or a stray box that overlaps better can take
  the pair.
- A prediction too far off to reach the matching threshold counts as missed,
  not as an error. When many such misses overlap a prediction that was left
  over, a warning names the threshold to lower.
- A reading needs every point of the measure in both. The report separates what
  was not read: points **unlabelled in the truth** (not the model's doing, and
  out of the denominator), no matching prediction, a predicted point missing,
  or geometry that cannot be measured.
- Intervals are percentile bootstraps over whole clusters (by default images),
  2,000 resamples with a fixed seed. The overall rate of large errors and the
  decision rates resample whole clusters too when some hold several readings
  (never narrower than Wilson's); with one reading to each they use Wilson
  intervals, which treat readings as independent. Rates per size band stay
  Wilson. With a single cluster there is nothing to resample, and every
  interval, Wilson's included, is left out (`null` in the JSON) rather than
  shown as a point or as if the readings were independent.


Exit codes and gating on the output are in [json.md](json.md).
