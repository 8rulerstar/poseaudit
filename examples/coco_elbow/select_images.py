"""Pick the demo images: the first 200 val2017 images, by id, with a person whose
left shoulder, elbow and wrist are all labelled. Steps are in README.md.
"""

import json
import sys
from pathlib import Path

LEFT_ARM = (5, 7, 9)
COUNT = 200


def main(annotations: str, out: str) -> None:
    data = json.loads(Path(annotations).read_text(encoding="utf-8"))
    usable = {
        a["image_id"]
        for a in data["annotations"]
        if not a["iscrowd"] and all(a["keypoints"][3 * i + 2] > 0 for i in LEFT_ARM)
    }
    images = sorted(
        (i for i in data["images"] if i["id"] in usable), key=lambda i: i["id"]
    )
    images = images[:COUNT]
    ids = {i["id"] for i in images}
    keep = ("id", "image_id", "category_id", "bbox", "keypoints", "num_keypoints",
            "iscrowd", "area")  # fmt: skip
    subset = {
        "info": {"description": "200 COCO val2017 images, keypoints only"},
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": [
            {k: a[k] for k in keep if k in a}
            for a in data["annotations"]
            if a["image_id"] in ids and not a["iscrowd"]
        ],
        "categories": data["categories"],
    }
    folder = Path(out)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "urls.txt").write_text(
        "\n".join(i["coco_url"] for i in images), encoding="utf-8"
    )
    Path(__file__).with_name("gt_200.json").write_text(
        json.dumps(subset), encoding="utf-8"
    )
    print(f"{len(images)} images, {len(subset['annotations'])} people")


if __name__ == "__main__":
    main(*sys.argv[1:])
