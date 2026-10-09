# poseaudit

[![PyPI](https://img.shields.io/pypi/v/poseaudit)](https://pypi.org/project/poseaudit/) [![Python](https://img.shields.io/pypi/pyversions/poseaudit)](https://pypi.org/project/poseaudit/) [![CI](https://github.com/8rulerstar/poseaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/8rulerstar/poseaudit/actions/workflows/ci.yml) [![License](https://img.shields.io/pypi/l/poseaudit)](https://github.com/8rulerstar/poseaudit/blob/main/LICENSE) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/8rulerstar/poseaudit/blob/main/examples/poseaudit_demo.ipynb)

**How far off are the joint angles, tilts and lengths you read from a pose model?**
poseaudit reads the same measure from your labels and from your model, pairs them
person by person, and judges the difference the way a measuring instrument is judged:
bias, limits of agreement, ICC and how often it is off by more than you can accept.

![poseaudit run on the demo data in a terminal, and the summary it prints](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/demo.gif)

## What plain numpy misses

The mean |error| is one line of numpy. The work is around it:

- **Who was skipped.** Keep only the detections `yolo11n-pose` scored 0.7 or more and
  its mean elbow error drops from 19.48° to 17.32°. The model did not get better: the
  filter dropped 68 people it read worse, and on the 254 people both settings read the
  difference is -0.06° [-0.20 to +0.00]. poseaudit compares only what both read.
- **Pairing.** People are matched to labels image by image; points left unlabelled in
  the truth stay out of the count instead of counting as errors.
- **Honest intervals.** Readings from one image, or from one subject with `--cluster`,
  are resampled together.
- **Where it goes wrong.** 58% of elbows are off by 15° or more when the arm segments
  average under 30 px, against 26% above 60 px.

![The score filter person by person: the mean drops only because the worst-read people were dropped](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/score-filter.gif)

## Install

```bash
pip install poseaudit
```

That needs only NumPy and runs every audit. Extras add `--plot`
(`pip install "poseaudit[plot]"`, Matplotlib), `--images` to read YOLO image sizes
from the images (`[images]`, Pillow) and `pa.from_supervision` (`[supervision]`).

## Quick start

```bash
git clone https://github.com/8rulerstar/poseaudit   # for the demo data, which pip leaves out
cd poseaudit/examples/coco_elbow
poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json --angle 5,7,9 --big-error 15 --size-bands 30,60
```

This audits the left elbow angle (COCO keypoints 5, 7, 9: shoulder, elbow, wrist) of
`yolo11n-pose` against COCO's labels on 200 val2017 images and prints the summary in
the GIF above. `--big-error` is required: the error that counts as large, in the
measure's unit. Add `--report report.md` for a report that explains every line,
`--plot panels.png` for [this figure](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/panels.png)
and `--full` for every statistic. If `poseaudit` is not found, use `python -m poseaudit`.

## For a validation paper

```bash
poseaudit audit --format coco --gt gt.json --pred res.json --angle 5,7,9 --big-error 15 --full --json figures.json --report report.md --plot panels.png
```

Report how many were read, the bias with its limits of agreement, the mean absolute
error, ICC and the large-error rate, each with its interval. [docs/statistics.md](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#writing-it-up)
words this for the demo, with references. Frames of one video are not independent: name
the clip with `--cluster "^(clip\d+)_"` (a regex on the image name), or the intervals
come out far too narrow.

## As a CI gate

```bash
poseaudit audit --format coco --gt gt.json --pred res.json --angle 5,7,9 --big-error 15 --json figures.json
jq -e '(.mean_abs_error_ci[1] // 1e9) <= 25 and (.big_error_rate_ci[1] // 1) <= 0.5' figures.json
```

This fails when the mean |error| could be above 25° or the large-error rate above 50%.
A single cluster gives `null` intervals, and the `//` fails on them too.

## Caveats

- The figures are agreement with one human label, not error against the world.
- On the demo, COCO's own label spread [explains much of the 19.48°](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#reading-the-demo).
- 2D angles read off an image are not 3D joint angles.
- Every reading counts once; there is no per-subject summary yet.
- The jitter check in `--full` assumes labels much cleaner than the model.

## What is new versus OKS, PCK and pycocotools

They score where the points land: is this a good detector? poseaudit asks whether you
can use the angle it gives you, for sports, rehabilitation or ergonomics. Its statistics
are the ones clinicians use to compare two measuring devices; what is new is the path
to them from COCO, YOLO and supervision formats.

## More

[docs/README.md](https://github.com/8rulerstar/poseaudit/blob/main/docs/README.md) indexes
the docs: every option, comparing models, the Python API, the statistics, the JSON
output and the [limitations](https://github.com/8rulerstar/poseaudit/blob/main/docs/statistics.md#limitations).
See also the [CHANGELOG](https://github.com/8rulerstar/poseaudit/blob/main/CHANGELOG.md) and [issues](https://github.com/8rulerstar/poseaudit/issues).

Code: MIT. Demo data: a subset of [COCO](https://cocodataset.org)'s 2017 keypoint labels
(© COCO Consortium, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) and predictions
from Ultralytics `yolo11n-pose` (AGPL-3.0), which poseaudit does not depend on or include.
