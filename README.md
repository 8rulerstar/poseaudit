# poseaudit

[![PyPI](https://img.shields.io/pypi/v/poseaudit)](https://pypi.org/project/poseaudit/) [![Python](https://img.shields.io/pypi/pyversions/poseaudit)](https://pypi.org/project/poseaudit/) [![CI](https://github.com/8rulerstar/poseaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/8rulerstar/poseaudit/actions/workflows/ci.yml) [![License](https://img.shields.io/pypi/l/poseaudit)](https://github.com/8rulerstar/poseaudit/blob/main/LICENSE) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/8rulerstar/poseaudit/blob/main/examples/poseaudit_demo.ipynb)

**How far off are the joint angles, tilts and lengths you read from a pose model?**
OKS and mAP say whether the points land near the right place; poseaudit says whether the
angle you compute from them is good enough to use. It judges your model against your labels
as a scale is validated: bias (the mean offset), limits of agreement (where 95% of the
differences fall), ICC (agreement, 1 at best) and how often it is off by too much.

![poseaudit run on the demo data in a terminal, and the summary it prints](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/demo.gif)

## What plain numpy misses

The mean |error| is one line of numpy. The work is around it (brackets are 95% intervals):

- **Who was skipped.** Keep only the detections `yolo11n-pose` scored 0.7 or more and its
  mean elbow error drops from 19.48° to 17.32°: the filter dropped 68 people it read worse.
  The 254 people both settings read were read alike but for one, and on them the
  difference, every detection minus filtered, is -0.06° [-0.20 to +0.00]. Models given as
  `--pred a=a.json --pred b=b.json` are compared the same way, on what both read.
- **Pairing.** People are matched to labels image by image by box overlap; points left
  unlabelled in the truth stay out of the count instead of counting as errors.
- **Honest intervals.** Readings from one image, or from one subject with `--cluster`,
  are resampled together.
- **Where it goes wrong.** 58% of elbows are off by 15° or more when the arm segments
  average under 30 px, against 26% above 60 px.

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
git clone https://github.com/8rulerstar/poseaudit   # for the demo data, which pip leaves out
cd poseaudit/examples/coco_elbow
poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json --angle 5,7,9 --big-error 15 --size-bands 30,60
```

This prints the summary in the GIF above: the left elbow angle of `yolo11n-pose` against
COCO's labels on 200 val2017 images. Keypoints count from 0 in COCO's order (5, 7, 9: left
shoulder, elbow, wrist; 11, 13, 15: left hip, knee, ankle; each right-side point is one
higher). `--big-error` is required: the error that counts as large, in the measure's unit.
Add `--report report.md` for a report that explains every line, `--full` for every statistic
and `--plot panels.png` for [this figure](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/panels.png) (with `[plot]`). If `poseaudit` is not found, use `python -m poseaudit`.

## For a validation paper

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

## As a CI gate

```bash
poseaudit audit --format coco --gt labels.json --pred results.json --angle 5,7,9 --big-error 15 --json figures.json
jq -e '(.mean_abs_error_ci[1] // 1e9) <= 25 and (.big_error_rate_ci[1] // 1) <= 0.5' figures.json
```

It fails unless the 95% intervals end at or below 25° of mean |error| and 50% of large errors.
Resampling is seeded, so reruns agree; one image or `--cluster` group gives `null`, which fails.

## Caveats

- The figures are agreement with one human label, not error against the world ([demo](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#reading-the-demo)).
- 2D angles read off an image are not 3D joint angles.
- Every reading counts once; there is no per-subject summary yet.
- The jitter check in `--full` assumes labels much cleaner than the model.

## What is new versus OKS, PCK and pycocotools

They score where the points land, yet points scattered by the label spread OKS is built on
turn these elbow angles by 11 to 16° on average. poseaudit scores the angle itself, with the
statistics clinicians use to compare two measuring devices, from COCO, YOLO or supervision.

## More

[docs/README.md](https://github.com/8rulerstar/poseaudit/blob/main/docs/README.md) indexes the docs: every option, comparing models, the
Python API, the statistics, the JSON output, the [limitations](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#limitations) and the [CHANGELOG](https://github.com/8rulerstar/poseaudit/blob/main/CHANGELOG.md).

Code: MIT. Demo data: a subset of [COCO](https://cocodataset.org)'s 2017 keypoint labels
(© COCO Consortium, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) and predictions
from Ultralytics `yolo11n-pose` (AGPL-3.0), which poseaudit does not depend on or include.
