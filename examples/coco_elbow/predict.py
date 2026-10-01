"""Run a public YOLO pose model on the demo images and save COCO-format results.

pip install ultralytics
python predict.py gt_200.json data/images pred_yolo11n.json
"""

import json
import sys
from pathlib import Path

from ultralytics import YOLO

# a point the model is unsure of is reported as unseen rather than guessed
MIN_POINT_CONFIDENCE = 0.5


def main(annotations: str, images: str, out: str, weights: str = "yolo11n-pose.pt"):
    ids = {
        i["file_name"]: i["id"]
        for i in json.loads(Path(annotations).read_text())["images"]
    }
    model = YOLO(weights)
    results = []
    for name, image_id in ids.items():
        pred = model(str(Path(images) / name), verbose=False)[0]
        if pred.keypoints is None:
            continue
        xy = pred.keypoints.xy.tolist()
        conf = pred.keypoints.conf.tolist() if pred.keypoints.conf is not None else None
        for i, box in enumerate(pred.boxes.xyxy.tolist()):
            flat = []
            for j, (x, y) in enumerate(xy[i]):
                seen = conf is None or conf[i][j] >= MIN_POINT_CONFIDENCE
                flat += [round(x, 2), round(y, 2), 2 if seen else 0]
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [round(v, 2) for v in (x1, y1, x2 - x1, y2 - y1)],
                    "keypoints": flat,
                    "score": round(float(pred.boxes.conf[i]), 4),
                }
            )
    Path(out).write_text(json.dumps(results))
    print(f"{len(results)} people in {len(ids)} images -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
