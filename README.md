# poseaudit

[![PyPI](https://img.shields.io/pypi/v/poseaudit)](https://pypi.org/project/poseaudit/) [![Python](https://img.shields.io/pypi/pyversions/poseaudit)](https://pypi.org/project/poseaudit/) [![CI](https://github.com/8rulerstar/poseaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/8rulerstar/poseaudit/actions/workflows/ci.yml) [![License](https://img.shields.io/pypi/l/poseaudit)](https://github.com/8rulerstar/poseaudit/blob/main/LICENSE) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/8rulerstar/poseaudit/blob/main/examples/poseaudit_demo.ipynb)

**How far off are the joint angles, tilts and lengths you read from a pose model?**
OKS and mAP say whether the points land near the right place; poseaudit says whether the
angle computed from them is good enough to use, with the statistics used to validate a
measuring device: bias, limits of agreement, ICC and the rate of large errors. Angles
already computed (Pose2Sim, OpenSim, your own code) go in as a CSV or `.mot` files with
`poseaudit paired` ([paired values](https://github.com/8rulerstar/poseaudit/blob/main/docs/cli.md#paired-values)).

![poseaudit run on the demo data in a terminal, and the summary it prints](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/demo.gif)

## What plain numpy misses

The mean |error| is one line of numpy. The work is around it (brackets are 95% intervals):

- **Who was skipped.** Keep only the detections `yolo11n-pose` scored 0.7 or more and its
  mean elbow error drops from 19.48° to 17.32°: the filter dropped 68 people it read worse.
  The 254 people both settings read were read alike but for one, and on them the
  difference, every detection minus filtered, is -0.06° [-0.20 to +0.00]. Models given as
  `--pred a=a.json --pred b=b.json` are compared the same way, on what both read.
- **Pairing.** People are matched to labels image by image by box overlap and keypoint
  similarity (OKS); points left unlabelled in the truth stay out of the count instead of
  counting as errors.
- **Honest intervals.** Readings from one image, or from one subject with `--cluster`,
  are resampled together.
- **Where it goes wrong.** 58% of elbows disagree with COCO's labels by 15° or more when
  the arm segments average under 30 px, against 26% above 60 px. How much of that is the
  labels' is not known: COCO's sigmas give the size of their spread, not its shape. As
  independent, isotropic noise on each point, that spread alone would put 30 to 44% of the
  small arms there, 17 to 31% of the large ones and 25.9 to 39.9% of all arms (43%
  observed). Yet on val2017 `yolo11x-pose` disagrees on 9 to 10% of straight left elbows
  with segments of 35 px or more, where that noise alone gives 22 to 25% on the demo's
  straight arms of the same lengths: most likely the labels' noise is shaped otherwise, or
  models learn it ([statistics](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#reading-the-demo)).

![The score filter person by person: the mean drops only because the worst-read people were dropped](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/score-filter.gif)

## Install

```bash
python -m pip install poseaudit
```

That needs only NumPy and runs every audit. `python -m pip install "poseaudit[plot]"` adds
`--plot` (Matplotlib); the extra `[images]` adds `--images`, which reads YOLO image sizes
from the images (Pillow), and `[supervision]` adds `pa.from_supervision`.

## Quick start

```bash
git clone https://github.com/8rulerstar/poseaudit   # the demo data (or GitHub's Code > Download ZIP)
cd poseaudit/examples/coco_elbow
poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json --angle 5,7,9 --big-error 15 --size-bands 30,60
```

```text
angle (5, 7, 9): read 322 of 380 labelled instances
  not read     no matching prediction 35, prediction lacked a point 23, unmeasurable 0
  not counted  206 with a point unlabelled in the truth, 79 unmatched predictions
  bias         +2.27° [-1.21 to +6.15], median +0.52°
  limits       -50.90° to +79.99° (percentile, 2.5th to 97.5th; JSON percentile_limits)
  |error|      mean 19.48° [17.05 to 22.13], median 12.59°, 95th pct 68.99°
  RMSE         29.57° [25.30 to 33.94]
  >= 15°       43.2% [37.9% to 48.6%]
  by size      0-30 px 58% (n 134), 30-60 px 37% (n 112), 60+ px 26% (n 76)
  slope        0.731 [0.639 to 0.814] (pred on truth, 1 is ideal); Theil-Sen 0.786
  ICC(A,1)     0.762 [0.677 to 0.827]
```

Two warnings follow it, as in the GIF above. That is the left elbow angle of `yolo11n-pose` against COCO's labels on 200 val2017 images.
Keypoints count from 0 in COCO's order (5, 7, 9: left shoulder, elbow, wrist; each right-side
point is one higher). `--big-error` is required: the error that counts as large, in the
measure's unit. Add `--report report.md` for a report that explains every line, `--full` for
every statistic and `--plot panels.png` for [this figure](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/panels.png) (with `[plot]`). If `poseaudit` is not found, use `python -m poseaudit`.

<details>
<summary><b>The same audit from Python</b></summary>

```python
import poseaudit as pa

truth = pa.load_coco("gt_200.json")
predicted = pa.load_coco_results("pred_yolo11n.json", "gt_200.json")
result = pa.audit(
    pa.pair(truth, predicted), pa.angle(5, 7, 9), big_error=15, size_bands=[30, 60]
)
print(result.summary())
```

Your own arrays, YOLO labels and `result.to_markdown()` / `to_csv()` / `plot()` are in
[Python](https://github.com/8rulerstar/poseaudit/blob/main/docs/python.md).

</details>

## Per cycle

Gait and rehabilitation studies report figures per stride or repetition, not per trial.
`poseaudit paired --table angles.csv --big-error 5 --cycles events.csv` cuts each trial at
the start times in your CSV of events (or `--cycles auto` finds them where the reference repeats),
stretches each cycle to 0 to 100% and gives the peak, trough and range of every cycle, the
RMSE over the cycle and the mean error with its SD at each point, where an error that comes
at one phase shows. The GIF draws the knee strides of the synthetic data in
`examples/paired` (`cycles_gif.py`) ([cycles](https://github.com/8rulerstar/poseaudit/blob/main/docs/cli.md#paired-values)).

![Knee cycles of a prediction and motion capture stretched to 0 to 100%, then the mean error and its SD over the cycle](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/cycles.gif)

<details>
<summary><b>For a validation paper</b></summary>

```bash
poseaudit audit --format coco --gt labels.json --pred results.json --angle 5,7,9 --big-error 15 --full --json figures.json --report report.md --plot panels.png
```

`--gt` takes COCO keypoint annotations and `--pred` a COCO results file (or, with
`--format yolo`, YOLO label folders). Report how many were read, the bias with its limits of
agreement, the mean absolute error, ICC and the large-error rate, each with its interval;
[docs/statistics.md](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#writing-it-up)
words this for the demo. Frames of one video are not independent: if image names carry the
clip or participant, as `clip3_000123.jpg` does, add `--cluster "^(clip\d+)_"` (its first
group names the cluster), or the intervals come out far too narrow.

</details>

## As a CI gate

```bash
poseaudit audit --format coco --gt labels.json --pred results.json --angle 5,7,9 --big-error 15 --json figures.json
jq -e '(.mean_abs_error_ci[1] // 1e9) <= 25 and (.big_error_rate_ci[1] // 1) <= 0.5' figures.json
```

It fails unless the 95% intervals end at or below 25° of mean |error| and 50% of large errors.
Resampling is seeded, so reruns agree; one image or `--cluster` group gives `null`, which fails.

## How large must a limb be

`poseaudit budget --format coco --gt labels.json --pred results.json --angle 5,7,9 --big-error 15`
gives the smallest limb, in pixels, whose angle is off by 15° or more in at most 20% of
readings, for that model on images like those, after checking its fit on readings it did
not see (to within at least 6 percentage points, so a size can miss the target by that
much). It needs about 140 readings of the joint, or 140 subjects or clips with `--cluster`,
in each of its two groups (straight and bent limbs): on the 200-image demo it gives no size
and says why. On COCO val2017, over four YOLO11 models at these settings, the check passed
all 16 elbow groups, 11 of which gave a size (the others never get below 20%), and 3 of 16
knee groups; one elbow group passed in only 10 of 20 redraws of the data (`--redraw`)
([budget](https://github.com/8rulerstar/poseaudit/blob/main/docs/cli.md#how-large-must-a-limb-be)).

<details>
<summary><b>Caveats</b></summary>

- The figures are agreement with one human label, not error against the world ([demo](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#reading-the-demo)).
- 2D angles read off an image are not 3D joint angles.
- Name the subject (`subject` column, `--cluster`), or every frame counts as independent.
- The jitter check in `--full` is experimental and assumes labels much cleaner than the model.

</details>

<details>
<summary><b>What is new versus OKS, PCK and pycocotools</b></summary>

They score where the points land, yet points scattered by the label spread OKS is built on,
as independent isotropic noise, turn these elbow angles by 11 to 16° on average. poseaudit
scores the angle itself, with the statistics clinicians use to compare two measuring
devices, from COCO, YOLO or supervision.

</details>

## More

[docs/README.md](https://github.com/8rulerstar/poseaudit/blob/main/docs/README.md) indexes the docs: every option, comparing models, the
Python API, the statistics, the JSON output, the [limitations](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#limitations) and the [CHANGELOG](https://github.com/8rulerstar/poseaudit/blob/main/CHANGELOG.md).

Code: MIT. Demo data: a subset of [COCO](https://cocodataset.org)'s 2017 keypoint labels
(© COCO Consortium, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) and predictions
from Ultralytics `yolo11n-pose` (AGPL-3.0), which poseaudit does not depend on or include.
