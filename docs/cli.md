# Command line

[README](../README.md) · [index](README.md) · **command line** · [Python](python.md) · [statistics](statistics.md) · [JSON, CSV and exit codes](json.md)

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
lists every option with examples, and so does [Every option](#every-option)
below.

`--plot` draws predicted against true values and a Bland-Altman plot, 7 inches
wide with 12 point text at 300 dpi: the full width of a two-column paper, or
one 3.5 inch column with its text at 6 points. The format follows the
extension (`.png`, `.svg`, `.pdf`). The colours stay apart for colour-blind
readers, and every line differs in dash pattern too, so a grayscale print
still matches each line to the legend.

## Every statistic: --full

`--full` adds every agreement statistic, the bootstrap intervals of the
normal limits (and exact ones when every reading is its own cluster), and,
set apart under `experimental`, the jitter reference. They are explained in
[statistics.md](statistics.md).

```text
$ poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json \
    --angle 5,7,9 --big-error 15 --size-bands 30,60 --full
angle (5, 7, 9): read 322 of 380 labelled instances
  not read     no matching prediction 35, prediction lacked a point 23, unmeasurable 0
  not counted  206 with a point unlabelled in the truth, 79 unmatched predictions
  bias         +2.27° [-1.21 to +6.15], median +0.52°
  limits       -50.90° to +79.99° (percentile, 2.5th to 97.5th; JSON percentile_limits)
  normal       -55.60° to +60.14° (bias +/- 1.96 SD; JSON limits)
  limit CIs    lower [-63.68 to -47.32], upper [+50.60 to +70.54] (cluster bootstrap)
  |error|      mean 19.48° [17.05 to 22.13], median 12.59°, 95th pct 68.99°
  RMSE         29.57° [25.30 to 33.94]
  >= 15°       43.2% [37.9% to 48.6%]
  by size      0-30 px 58% (n 134), 30-60 px 37% (n 112), 60+ px 26% (n 76)
  slope        0.731 [0.639 to 0.814] (pred on truth, 1 is ideal); Theil-Sen 0.786
  ICC(A,1)     0.762 [0.677 to 0.827]
  BA slope     -0.049 [-0.119 to +0.016]
  agreement    CCC 0.761 [0.677 to 0.826], r 0.763
  experimental, still being validated:
  vs jitter    0.871 from keypoint jitter alone; gap -0.140 [-0.199 to -0.076], p(slope <= jitter) <= 0.002*
               * swaps and gross failures lower the slope too, and noisy
                 labels make p small for an honest model: read the gap
  ! 2.5% of errors fall below the normal limits and 4.3% above them, against 2.5% each for a normal error: use the percentile limits.
```

## Several joints

Repeat `--angle`, `--tilt`, `--length` or `--ratio` to read several measures
off the same pairing. The output is one row per measure; `--full` prints each
measure's full block instead. On a terminal narrower than the table the
columns come in blocks that fit, each led by the measure, and long lines break
between words; piped or redirected output keeps one line per row, as below.
Both elbows and both knees on the demo data:

```text
$ poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json \
    --angle 5,7,9 --angle 6,8,10 --angle 11,13,15 --angle 12,14,16 --big-error 15
measure           n  bias    limits              mean |error|             RMSE    >= 15°                  slope  ICC(A,1)
angle 5,7,9     322  +2.27°  -50.90° to +79.99°  19.48° [17.05 to 22.13]  29.57°  43.2% [37.9% to 48.6%]  0.731  0.762
angle 6,8,10    308  +3.74°  -54.10° to +75.45°  18.07° [15.54 to 20.67]  30.11°  34.7% [29.5% to 40.2%]  0.751  0.803
angle 11,13,15  243  +5.19°  -38.92° to +80.06°  16.40° [12.93 to 20.26]  29.80°  30.5% [24.4% to 37.2%]  0.715  0.794
angle 12,14,16  242  +4.63°  -37.07° to +69.46°  14.62° [11.87 to 17.66]  24.38°  31.8% [25.0% to 39.0%]  0.796  0.871
  ! angle 5,7,9: 2.5% of errors fall below the normal limits and 4.3% above them, against 2.5% each for a normal error: use the percentile limits.
  ! angle 6,8,10: 2.3% of errors fall below the normal limits and 4.2% above them, against 2.5% each for a normal error: use the percentile limits.
  ! angle 11,13,15: 1.6% of errors fall below the normal limits and 4.1% above them, against 2.5% each for a normal error: use the percentile limits.
  ! angle 12,14,16: 1.7% of errors fall below the normal limits and 5.0% above them, against 2.5% each for a normal error: use the percentile limits.
```

Measures in different units each take their own large error: by kind
(`angle`, `tilt`, `length`, `ratio`) or by unit (`deg` for angles and tilts,
`px` for lengths), comma-separated or with `--big-error` repeated. A kind
wins over its unit, and a bare number covers what the others do not name, as
long as those measures share one unit: a single 15 would count 15° on an
angle and 15 on a ratio as large alike, where no ratio ever is, so it is
refused. With more than one threshold the table gives each row its own
before the rate:

```bash
poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json \
    --angle 5,7,9 --tilt 5,11 --length 5,7 --ratio 5,7,7,9 \
    --big-error angle:15,tilt:5,length:10,ratio:0.5
```

`--threshold`, `--pred-threshold` and the edges given to `--bands` are in the
measure's unit as well and apply to every measure, so with measures in
different units they are refused; audit each unit in a run of its own. A
warning that several measures raise alike comes once under the table,
naming them, and so does the advice that ends a warning several measures
raise with counts of their own.

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

`--plot` draws one model, so with several `--pred` it is refused. `--report`
writes the comparison as Markdown, ready for a paper or a blog: one table of
every model's figures on every measure, measure by measure in the order
given, then the differences, the warnings and the settings. Each model's
figures cover the readings it made; rank models on the differences, which
use only the readings both made. `--csv` writes the differences, with the
large-error threshold of each row and each interval beside its estimate, and
`--json` everything ([json.md](json.md#comparing-models)); in Python,
`Comparison.to_rows()` gives each model's figures as rows for a CSV.

The `--pred NAME=PATH` form compares prediction files. To compare one file
under two settings (a score filter, a confidence threshold), use Python:
[python.md](python.md#compare-models).

## Paired values

Joint angles that a pipeline such as Pose2Sim, Sports2D or OpenSim's inverse
kinematics already computed, against motion capture or a goniometer, need no
keypoints and no matching. `poseaudit paired` takes them as they are and
gives the same summary, tables, JSON, CSV, report and plot. Run from
[`examples/paired`](../examples/paired), whose data is synthetic (made by its
`make_data.py`):

```text
$ poseaudit paired --table angles.csv --big-error 5
measure          n  bias    limits            mean |error|          RMSE   >= 5°                   slope  ICC(A,1)
knee_angle_r   720  +0.55°  -7.57° to +9.15°  3.51° [3.19 to 3.90]  4.35°  24.7% [19.6% to 30.4%]  0.892  0.978
hip_flexion_r  720  -0.05°  -5.15° to +5.34°  2.15° [1.96 to 2.33]  2.67°  6.8% [4.6% to 9.3%]     1.002  0.988
  ! every measure: Only 6 subjects: bootstrap intervals are too narrow with this few.
```

**A long-format table** (`--table`, a CSV; in Python also a DataFrame or a
dict of columns) has one row per value:

| column | |
|---|---|
| `pred`, `ref` | required: the predicted and the reference value |
| `measure` | which measure the row belongs to; without it every row is one measure, `value` |
| `subject` | the readings of one subject are resampled together and get repeated-measures limits; without it every row counts as independent, and a warning says so |
| `trial` | the trial within its subject, for the summary per trial |
| `frame` or `time` | kept in the CSV and in the report's largest errors |
| `unit` | the measure's unit; `deg` (or `degrees`) and `rad` make differences wrap, so 179° against -179° is 2° off |

**OpenSim .mot files** (`--pred-mot P.mot --ref-mot R.mot`, with
`--subject` and `--trial` to name them, or `--mot-pairs pairs.csv` listing
`subject, trial, pred, ref` for several) are lined up by column name and by
time: each prediction frame inside the reference's span is compared with the
reference interpolated linearly there (angles the short way round, so 170°
to -170° passes through 180°). A reference value next to a missing one is
missing. `--time-offset S` moves the prediction onto the reference's clock.
Columns ending `_tx`, `_ty` or `_tz` are translations in metres; the rest are
angles, in degrees unless the header says `inDegrees=no`.

```bash
poseaudit paired --pred-mot S01_walk1_pose.mot --ref-mot S01_walk1_mocap.mot \
    --subject S01 --trial walk1 --measure knee_angle_r --big-error 5 --full
```

With subjects, the summary adds the limits for repeated readings (Bland and
Altman 2007) and the spread of the bias across subjects; the report has a
table per subject and per subject and trial. Every interval resamples whole
subjects, so with fewer than 20 they are too narrow, as the warning says.

| option | what it does |
|---|---|
| `--table CSV` | the long-format table |
| `--pred-mot MOT`, `--ref-mot MOT` | one predicted and one reference .mot file |
| `--mot-pairs CSV` | several .mot pairs: columns `subject`, `trial` (optional), `pred`, `ref`, paths relative to the CSV |
| `--subject NAME`, `--trial NAME` | with `--pred-mot`: whose files they are |
| `--time-offset S` | seconds added to the prediction's times (default 0) |
| `--measure NAME` | audit this measure only; repeat it for several |
| `--unit NAME=UNIT` | a measure's unit, over the table's `unit` column |
| `--no-wrap` | angle differences taken as they are, not the short way round |
| `--big-error E` or `KEY:E` | required: one number, or by measure name or unit, as in `knee_angle_r:5,deg:10,m:0.02` |

`--bands`, `--band-by`, `--threshold`, `--pred-threshold`, `--side`,
`--noise-ratio`, `--seed`, `--resamples` and the outputs (`--report`,
`--csv`, `--json`, `--plot`, `--full`) work as for `audit`. The CSV's columns
are `subject, trial, frame, ref, pred, error, mean`.

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

## Every option

`poseaudit audit --help` prints them too. Values in the measure's unit are
degrees for an angle or a tilt, pixels for a length, and plain numbers for a
ratio.

**Data**

| option | what it does |
|---|---|
| `--gt PATH` | required: the truth, a YOLO label folder or COCO keypoint annotations |
| `--pred PATH` | required: the predictions, a YOLO folder or a COCO results file; repeated as `--pred NAME=PATH`, it [compares models](#compare-models) |
| `--format yolo\|coco` | the format of both |
| `--gt-format yolo\|coco` | the format of `--gt`, over `--format` |
| `--pred-format yolo\|coco` | the format of `--pred`, over `--format` |
| `--image-size W H` | YOLO: the image width then height in pixels, or `WxH` ([why](#inputs)) |
| `--images DIR` | YOLO: each image's size read from its file (needs `pip install "poseaudit[images]"`) |
| `--keypoints K` | YOLO: keypoints per instance |
| `--classes ID ...` | keep only these class ids, on both sides |
| `--min-conf C` | a predicted point counts as seen above this confidence (default 0) |
| `--min-score S` | drop predicted detections scored below this (default 0); matching ignores scores |
| `--min-iou IOU` | the smallest box IoU that pairs a prediction with a truth (default 0.3) |
| `--min-similarity OKS` | the smallest keypoint similarity that pairs them when a side has no box (default 0.5) |

**Measure** (repeat these for [several](#several-joints); keypoints by index, counted from 0)

| option | what it does |
|---|---|
| `--angle A,B,C` | the angle at B, 0 to 180° ([measures](#measures)) |
| `--tilt A,B` | the tilt of the axis A-B from vertical, -90 to 90° |
| `--length A,B` | the distance from A to B, in pixels |
| `--ratio A,B,C,D` | length A-B over length C-D |
| `--relative-to median\|pNN` | each value read against the rest of its image ([why](#tilts-relative-to-the-rest-of-the-photo)) |
| `--relative-abs` | with `--relative-to`: \|value\| minus the baseline of the others' \|values\| |
| `--min-in-frame N` | with `--relative-to`: the fewest readings an image needs (default 3) |

**Analysis**

| option | what it does |
|---|---|
| `--big-error E` or `KIND:E` | required: the error, in the measure's unit, that counts as large; by kind or unit when the measures' units differ ([how](#several-joints)) |
| `--bands N\|EDGES` | bands of the measured value for the bias by band: a count (default 4), or edges in the measure's unit |
| `--band-by truth\|mean\|predicted` | what those bands sort readings by (default the truth; [why it matters](statistics.md#which-way-you-sort-decides-the-story)) |
| `--size-bands N\|EDGES` | bands by the size of the measured part: a count (default 3), or pixel edges such as `30,60` |
| `--cluster REGEX` | readings grouped by the regex's first capture group in the image name ([why](#several-readings-of-one-subject)) |
| `--threshold T ...` | [decision thresholds](#decisions-at-a-threshold) applied to the truth, in the measure's unit |
| `--pred-threshold T ...` | the prediction held to these thresholds as well |
| `--side above\|below\|outside` | which side of a threshold is flagged (default outside for a tilt, else above) |
| `--noise-ratio R` | the variance of the prediction's noise over that of the labels', when known: adds a Deming slope |
| `--mixed-classes` | readings of several classes allowed in one audit |
| `--seed N` | the seed for resampling and the jitter rebuilds (default 0) |
| `--resamples N` | bootstrap resamples for every interval (default 2000, at least 50) |
| `--jitter-repeats N` | rebuilds for the jitter reference (default 500; 0 turns it off) |

**Output**

| option | what it does |
|---|---|
| `--report FILE` | a Markdown report that explains every line; comparing models, their figures and differences as Markdown tables |
| `--csv FILE` | one row per reading; comparing models, one per difference ([columns](json.md)) |
| `--json FILE` | every figure ([keys](json.md#names)) |
| `--plot FILE` | the two panels, PNG, SVG or PDF by the extension (needs `pip install "poseaudit[plot]"`) |
| `--full` | every statistic in the summary |

`--resamples` and `--jitter-repeats` trade precision for speed: 50,000
readings took about four and a half minutes for each measure at the
defaults on one desktop, and just under a minute with `--jitter-repeats 0`.
Before a run that long, poseaudit says so on stderr.

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
truth of its own and counts among the unmatched predictions, unless its box
overlaps a labelled person by `--min-iou`, when it is paired with that person
and read as their error. On crowd-heavy data, read the warning about
unmatched truths beside unmatched predictions with this in mind: a lower
`--min-iou` would pair crowd members with the wrong person.

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

## When it stops

Errors print on stderr after `poseaudit:`, and the run exits with 1 (2, with
the usage, when the options themselves cannot be read). A message about one
file may start with its name (and a YOLO label's line number); after that,
each starts as below, `...` standing for a name or a number:

| message | what to do |
|---|---|
| `--big-error ... would count an error of ... as large alike` | Measures in different units: give each kind its own value, as in `--big-error angle:15,length:10,ratio:0.1` ([Several joints](#several-joints)). |
| `--big-error has no value for ...` | Add `KIND:VALUE` for that measure, or a bare number for every measure the others do not name. |
| `... in the measure's unit and would apply to every measure` | `--threshold`, `--pred-threshold` and `--bands` edges hold one unit: audit the angles and the lengths in runs of their own. |
| `YOLO labels need --image-size W H or --images DIR` | YOLO coordinates are fractions of the image: give its size, width first, or the image folder ([Inputs](#inputs)). |
| `give --format, or both --gt-format and --pred-format` | `--format coco` or `--format yolo` for both files; `--gt-format` and `--pred-format` when they differ. |
| `COCO results name images by id: --gt must be COCO annotations` | A COCO results file needs the COCO annotations its image ids come from; YOLO labels cannot be paired with it. Convert one side, or give YOLO predictions. |
| `... uses keypoint ..., but a ... instance in ... has ... keypoints` | Keypoints count from 0: COCO's 17 run 0 to 16. Check the indices and the skeleton. |
| `... takes comma-separated keypoint indices counted from 0, not names` | Give numbers: the COCO left elbow is `--angle 5,7,9` (shoulder, elbow, wrist). |
| `... keypoint values fit neither ... x 2 nor ... x 3` | `--keypoints` does not match the label files: count the values after the box and divide by 3 (or 2). |
| `coordinates outside the image: YOLO values are fractions of the image size` | Usually a wrong `--keypoints`, or labels in pixels rather than fractions. |
| `skeletons differ: truth has ... keypoints, predictions ...` | The two sides use different skeletons (17 COCO points against 133 whole-body, say): use predictions of the labels' skeleton. |
| `this is a results file (a list of detections), not an annotation file` | `--gt` and `--pred` are swapped. |
| `this is an annotation file, not a results file` | `--gt` and `--pred` are swapped, or `--pred` should be the model's results. |
| `... has no ...: these look like a box detector's results` | The results have boxes and no keypoints: run a pose model, not a detector. |
| `image_id ... is not listed in ...` | The results were made on images of another annotation file or split; give the annotations the model was run on. |
| `readings mix classes ...: audit one class at a time` | Keep one class with `--classes`, or allow the mix with `--mixed-classes`. |
| `--cluster ... matches nothing in image name ...` | Every image name must match: `--cluster "^(clip\d+)_"` reads the cluster `clip3` from `clip3_000123.jpg` ([why](#several-readings-of-one-subject)). |
| `--plot draws one measure` | Leave out `--plot`, or give one measure. |
| `give one of --table, --pred-mot with --ref-mot, or --mot-pairs` | `poseaudit paired` reads one source at a time ([Paired values](#paired-values)). |
| `the files do not overlap in time` | The two .mot files cover different times: `--time-offset` moves the prediction onto the reference's clock. |
| `the two files share no column name` | The .mot columns must have the same names in both files (`knee_angle_r` in each); rename one side's columns. |
| `the table needs columns pred and ref` | Name the columns `pred` and `ref` (any case); `measure`, `subject`, `trial`, `frame` and `unit` are optional. |
| `nothing was read` | The summary above it says why: no pairs, points not labelled or not predicted. The JSON, CSV and report are still written. |

Exit codes and gating on the output are in [json.md](json.md).
