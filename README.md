# poseaudit

[![PyPI](https://img.shields.io/pypi/v/poseaudit)](https://pypi.org/project/poseaudit/)
[![Python](https://img.shields.io/pypi/pyversions/poseaudit)](https://pypi.org/project/poseaudit/)
[![CI](https://github.com/8rulerstar/poseaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/8rulerstar/poseaudit/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/poseaudit)](https://github.com/8rulerstar/poseaudit/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/8rulerstar/poseaudit/blob/main/examples/poseaudit_demo.ipynb)

**poseaudit tells you how far off the joint angles, tilts and lengths you read
from a pose model are, judged the way a measuring instrument is judged.**

![Left: predicted against true angle. Right: Bland-Altman plot](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/panels.png)

mAP says the points land near the right place. It does not say how wrong the
elbow angle you compute from them is. poseaudit reads the same angle from your
labels and from your model and reports the difference.

## What a few lines of numpy would miss

The mean |error| itself is one line of numpy. The work is around it:

- **Who was skipped.** Keep only `yolo11n-pose` detections scored 0.7 or more
  and its mean elbow error drops from 19.48° to 17.32°. On the 254 people both
  settings read, the difference is -0.06° [-0.20 to +0.00]. The model did not
  get better. It dropped 68 people it read worse. poseaudit counts what was not read
  and compares models only on what both read.
- **Where it goes wrong.** On the demo, 58% of elbows are off by 15° or more
  when the arm segments average under 30 px, against 26% above 60 px. The
  overall bias is +2.3°.
- **Pairing.** People are matched to labels by box or keypoints, image by
  image, with unlabelled points kept out of the denominator.
- **Honest intervals.** Readings from one image, or one subject with
  `--cluster`, are resampled together.
- **Agreement statistics.** Bland-Altman limits, ICC, slopes, and a check of
  whether a slope below 1 is more than keypoint jitter.

## Install and run

```bash
pip install poseaudit
git clone https://github.com/8rulerstar/poseaudit && cd poseaudit/examples/coco_elbow
poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json --angle 5,7,9 --big-error 15 --size-bands 30,60
```

That is the left elbow (COCO keypoints 5, 7, 9) of `yolo11n-pose` against
COCO's labels on 200 val2017 images:

```text
$ poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json \
    --angle 5,7,9 --big-error 15 --size-bands 30,60
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
  ! 2.5% of errors fall below the normal limits and 4.3% above them, against 2.5% each for a normal error: use the percentile limits.
```

`--big-error` is required. It is the error, in the measure's unit, that counts
as large. Add `--report report.md` for a report that explains every line,
`--plot panels.png` for the figure above, and `--full` for every statistic.
`pip install` does not ship the demo data, hence the clone.

Do not read 19.48° as the model's error. This is the weakest YOLO pose model,
and COCO's own label spread alone puts 26 to 40% of these arms off by 15° or
more ([details](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#reading-the-demo)).

## For a validation paper

```bash
poseaudit audit --format coco --gt gt.json --pred res.json --angle 5,7,9 \
    --big-error 15 --full --json figures.json --report report.md
```

Report the bias with its limits of agreement, the mean absolute error, ICC and
the large-error rate, each with its interval. For the demo: "Against COCO's
labels, the elbow angle had a bias of +2.3° (95% CI -1.2 to +6.2) with limits of agreement
of -50.9° to +80.0° (2.5th to 97.5th percentiles), a mean absolute error of
19.5°, ICC(A,1) 0.76, and 43% of readings off by 15° or more."

Frames of one video are not independent. Name the subject or clip with
`--cluster`, or the intervals come out far too narrow
([why](https://github.com/8rulerstar/poseaudit/blob/main/docs/cli.md#several-readings-of-one-subject)).

## As a CI gate

```bash
poseaudit audit --format coco --gt gt.json --pred res.json --angle 5,7,9 \
    --big-error 15 --json figures.json
jq -e '(.mean_abs_error_ci[1] // 1e9) <= 25 and (.big_error_rate_ci[1] // 1) <= 0.5' figures.json
```

A single cluster gives `null` intervals. The `//` makes the gate fail on them.

## Caveats

- The figures are agreement with one human label, not error against the world.
- 2D angles read off an image are not 3D joint angles.
- Every reading counts once. There is no per-subject summary yet.
- The jitter check is new and checked by simulation only.

More in [Limitations](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#limitations).

## What is new versus OKS, PCK and pycocotools

OKS, PCK and pycocotools score where the points land. They answer "is this a
good detector?". poseaudit answers "can I use the angle this detector gives
me?". It is for sports and rehabilitation motion analysis, ergonomics and
industrial measurement. It uses the statistics clinicians already use to compare two measuring
devices. What it adds is the path from computer vision formats (COCO, YOLO,
supervision) to those statistics: pairing people, counting who was not read,
intervals that respect images or subjects, error by size, decisions at a
threshold, and comparing models on the readings they share.

## More

- [docs/cli.md](https://github.com/8rulerstar/poseaudit/blob/main/docs/cli.md): every option, several joints, comparing
  models, decisions at a threshold, inputs, pairing, exit codes
- [docs/python.md](https://github.com/8rulerstar/poseaudit/blob/main/docs/python.md): the Python API, your own arrays, the
  comparison above in code
- [docs/statistics.md](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md): reading the demo, label noise,
  the sorting trap, the slope and the jitter reference, definitions,
  references, limitations
- [docs/json.md](https://github.com/8rulerstar/poseaudit/blob/main/docs/json.md): the JSON and CSV output
- [CHANGELOG.md](https://github.com/8rulerstar/poseaudit/blob/main/CHANGELOG.md)

## Feedback

If you measured something with poseaudit, a line in
[Discussions](https://github.com/8rulerstar/poseaudit/discussions/categories/show-and-tell)
saying what (a joint, a tilt, a length; which model) helps decide what to
build next. Bugs go to [issues](https://github.com/8rulerstar/poseaudit/issues).

## License and data

Code: MIT. The demo's annotations are a subset of the
[COCO](https://cocodataset.org) 2017 keypoint annotations (keypoint fields
only), © COCO Consortium, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the images are not
included. The demo's predictions were made with Ultralytics `yolo11n-pose`
(AGPL-3.0), which this package does not depend on or include.
