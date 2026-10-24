"""Set the authors' published RMSE beside poseaudit's bias and RMSE.

python compare.py data out

`data` holds the unpacked `results/` (the authors' summary tables), `out`
the four `<model>/report.json` written by `poseaudit paired ... --json`.
The authors remove each signal's own mean before their RMSE, so it leaves
out any constant gap between the two sides; poseaudit's bias is that gap
and its RMSE keeps it."""

import csv
import json
import sys
from pathlib import Path

MODELS = {
    "bodytrack": "BodyTrack",
    "mmpose": "MMPose",
    "motionagformer": "MotionAGFormer",
    "motionbert": "MotionBERT",
}


def main(data: str, out: str) -> None:
    published: dict[tuple[str, str], float] = {}
    for table in sorted(
        (Path(data) / "results" / "evaluation").glob("A??_SummaryTable.csv")
    ):
        with open(table, newline="") as f:
            for row in csv.DictReader(f):
                published[table.name[:3], row["Model"]] = float(
                    row["RMSE"].split("±")[0]
                )
    rows = []
    for folder, model in MODELS.items():
        report = json.loads((Path(out) / folder / "report.json").read_text())
        for m in report["measures"]:
            activity = m["measure"]["name"].split()[0]
            rows.append(
                (
                    activity,
                    m["measure"]["name"].split()[1],
                    model,
                    published[activity, model],
                    m["bias"],
                    m["rmse"],
                )
            )
    rows.sort()
    head = f"{'activity':8} {'joint':13} {'model':15}"
    print(f"{head} {'published RMSE':>15} {'bias':>8} {'RMSE':>7}")
    for activity, joint, model, pub, bias, rmse in rows:
        row = f"{activity:8} {joint:13} {model:15}"
        print(f"{row} {pub:>14.2f}° {bias:>+7.1f}° {rmse:>6.1f}°")
    print("\nMean over the 13 activities, models in the authors' order")
    print("(lowest published RMSE first):")
    means = {}
    for model in MODELS.values():
        mine = [r for r in rows if r[2] == model]
        means[model] = [sum(r[i] for r in mine) / len(mine) for i in (3, 4, 5)]
        means[model][1] = sum(abs(r[4]) for r in mine) / len(mine)
    print(f"{'model':15} {'published RMSE':>15} {'mean |bias|':>12} {'RMSE':>7}")
    for model, (pub, bias, rmse) in sorted(means.items(), key=lambda kv: kv[1][0]):
        print(f"{model:15} {pub:>14.1f}° {bias:>11.1f}° {rmse:>6.1f}°")


if __name__ == "__main__":
    main(*sys.argv[1:])
