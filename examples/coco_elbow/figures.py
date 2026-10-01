"""The two README figures, from the committed demo data.

python figures.py ../../docs
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import poseaudit as pa  # noqa: E402

HERE = Path(__file__).parent


def main(out: str) -> None:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    truth = pa.load_coco(HERE / "gt_200.json")
    predicted = pa.load_coco_results(HERE / "pred_yolo11n.json", HERE / "gt_200.json")
    pairing = pa.pair(truth, predicted)
    elbow = pa.angle(5, 7, 9, name="elbow angle")

    results = {
        by: pa.audit(pairing, elbow, big_error=15, band_by=by)
        for by in ("truth", "predicted", "mean")
    }
    results["truth"].plot(str(folder / "panels.png"))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = {
        "truth": "sorted by the true angle",
        "predicted": "sorted by the predicted angle",
        "mean": "sorted by the mean of both",
    }
    for by, result in results.items():
        middles = [(b.low + b.high) / 2 for b in result.bands]
        ax.plot(middles, [b.bias for b in result.bands], marker="o", label=labels[by])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="elbow angle of the band (°)", ylabel="mean error in the band (°)")
    ax.set_title("The same 322 readings, sorted three ways", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "trap.png", dpi=150)
    for by, result in results.items():
        print(by, [f"{b.bias:+.1f}" for b in result.bands])


if __name__ == "__main__":
    main(*sys.argv[1:])
