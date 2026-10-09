# Demo: elbow angle on COCO

Left elbow angle (keypoints 5, 7, 9) of `yolo11n-pose` on the first 200
val2017 images that have a person with a labelled left arm.

Both inputs are in this folder, so the audit runs straight away:

```bash
poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json \
  --angle 5,7,9 --big-error 15 --report report.md --plot panels.png
```

- `gt_200.json`: the COCO annotations of those images, keypoint fields only.
  © COCO Consortium, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- `pred_yolo11n.json`: the model's results, made with Ultralytics
  `yolo11n-pose`. `predict.py` imports Ultralytics, which is AGPL-3.0; the
  weights are downloaded, not included. The file was rebuilt byte for byte
  with ultralytics 8.4.150 and torch 2.14.0 on CPU, Ultralytics' default
  settings (image size 640, box confidence 0.25), and weights with sha256
  `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`. Other
  versions or a GPU may change it slightly. Points are stored as seen (2) or
  not (0), so `--min-conf` cannot be varied afterwards; edit
  `MIN_POINT_CONFIDENCE` and rebuild for that. Load only weights you trust:
  they are pickled. A point counts as seen when its
  confidence was at least 0.5 (`MIN_POINT_CONFIDENCE` in `predict.py`).

To rebuild them:

```bash
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -j annotations_trainval2017.zip annotations/person_keypoints_val2017.json -d data
python select_images.py data/person_keypoints_val2017.json data   # writes gt_200.json
(cd data/images && xargs -n 1 -P 8 curl -sO < ../urls.txt)         # about 32 MB
pip install ultralytics==8.4.150
python predict.py gt_200.json data/images pred_yolo11n.json
python figures.py ../../docs                                        # README figures
python gifs.py ../../docs                                           # README GIFs
```
