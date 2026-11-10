"""Three checks on the knee comparison, after build.py: the peak gap by
speed, with both sides low-pass filtered, and against the marker angle
projected onto the pelvis's sagittal plane.

python checks.py data out checks    # writes checks/, prints checks.txt

- speed: the pairs of each of Walk_2km, Walk_5km and Walk_8km audited apart;
- filter: both sides through a 6 Hz zero-lag Butterworth (2nd order run
  forward and back), each unbroken run of samples on its own;
- sagittal: hip joint centre, knee and ankle markers projected onto the
  plane normal to the pelvis's left-right axis, and the signed angle there,
  its sign set so that the larger excursion is flexion.
Needs pandas and scipy, like build.py."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402

CUTOFF = 6.0


def read_mot(p: Path):
    lines = p.read_text().splitlines()
    i = lines.index("endheader")
    a = np.array([[float(x) for x in row.split("\t")] for row in lines[i + 2 :]])
    return a[:, 0], a[:, 1]


def lowpass(t, y):
    rate = 1 / np.median(np.diff(t))
    b, a = butter(2, CUTOFF / (rate / 2))  # filtfilt doubles to 4th order
    # gaps: split into runs of consecutive samples, filter each long run
    out = np.full_like(y, np.nan)
    step = np.median(np.diff(t))
    breaks = np.where(np.diff(t) > 1.5 * step)[0] + 1
    for seg in np.split(np.arange(len(t)), breaks):
        if len(seg) > 3 * 3 * max(len(a), len(b)):
            out[seg] = filtfilt(b, a, y[seg])
    return out


def run(pairs: Path, events: Path, js: Path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "poseaudit",
            "paired",
            "--mot-pairs",
            str(pairs),
            "--cycles",
            str(events),
            "--big-error",
            "5",
            "--no-wrap",
            "--json",
            str(js),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with open(js) as f:
        r = json.load(f)
    peak = next(c for c in r["cycle"] if c["feature"] == "peak")
    trough = next(c for c in r["cycle"] if c["feature"] == "trough")
    return peak["bias"], peak["bias_ci"], trough["bias"], peak["n"]


def write_pairs(rows, path: Path, base: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["subject", "trial", "pred", "ref"])
        for s, tr, p, r in rows:
            w.writerow([s, tr, str(base / p), str(base / r)])


def main(data: str, out: str, work: str) -> None:
    data, out, work = Path(data).resolve(), Path(out).resolve(), Path(work).resolve()
    work.mkdir(exist_ok=True)
    with open(out / "pairs.csv") as f:
        rows = list(csv.reader(f))[1:]
    events = out / "events.csv"
    res = {}

    write_pairs(rows, work / "all.csv", out)
    res["all"] = run(work / "all.csv", events, work / "all.json")
    for sp in ("Walk_2km", "Walk_5km", "Walk_8km"):
        sub = [r for r in rows if r[1] == sp]
        write_pairs(sub, work / f"{sp}.csv", out)
        res[sp] = run(work / f"{sp}.csv", events, work / f"{sp}.json")

    # filtered
    (work / "mot_f").mkdir(exist_ok=True)
    frows = []
    for s, tr, p, r in rows:
        for f in (p, r):
            t, y = read_mot(out / f)
            build.write_mot(work / "mot_f" / Path(f).name, t, lowpass(t, y))
        frows.append([s, tr, "mot_f/" + Path(p).name, "mot_f/" + Path(r).name])
    write_pairs(frows, work / "filt.csv", work)
    res["filtered"] = run(work / "filt.csv", events, work / "filt.json")

    # sagittal-projected marker angle
    (work / "mot_s").mkdir(exist_ok=True)
    srows = []
    with open(out / "trials.csv") as f:
        legs = {(r[0], r[1]): r[2] for r in list(csv.reader(f))[1:]}
    for s, tr, p, _r in rows:
        subj = s[1:]
        time, m = build.read_trc(data / "02_markerbased_data" / subj / f"{tr}.trc")
        hips = build.hip_centres(m)
        right = build.unit(m["RASI"] - m["LASI"])

        def proj(v, right=right):
            return v - (v * right).sum(-1, keepdims=True) * right

        leg = legs[(s, tr)]
        thigh = proj(m[f"{leg}KNE"] - hips[leg])
        shank = proj(m[f"{leg}ANK"] - m[f"{leg}KNE"])
        signed = np.degrees(
            np.arctan2(
                (np.cross(thigh, shank) * right).sum(-1), (thigh * shank).sum(-1)
            )
        )
        # sign: flexion is the large excursion, so its extreme sets the sign
        hi, lo = np.nanpercentile(signed, [99, 1])
        ref = signed if hi > -lo else -signed
        name = f"{subj}_{tr}_marker_sag.mot"
        build.write_mot(work / "mot_s" / name, time, ref)
        srows.append([s, tr, p, "mot_s/" + name])
    with open(work / "sag.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["subject", "trial", "pred", "ref"])
        for s, tr, p, r in srows:
            w.writerow([s, tr, str(out / p), str(work / r)])
    res["sagittal"] = run(work / "sag.csv", events, work / "sag.json")

    # sagittal + filtered
    (work / "mot_sf").mkdir(exist_ok=True)
    sfrows = []
    for s, tr, p, r in srows:
        t, y = read_mot(work / r)
        build.write_mot(work / "mot_sf" / Path(r).name, t, lowpass(t, y))
        sfrows.append([s, tr, "mot_f/" + Path(p).name, "mot_sf/" + Path(r).name])
    write_pairs(sfrows, work / "sagf.csv", work)
    res["sagittal+filtered"] = run(work / "sagf.csv", events, work / "sagf.json")

    print("peak and trough bias, each subject's mean counting once; n is strides")
    for k, (pb, ci, tb, n) in res.items():
        print(
            f"{k:20s} peak {pb:+6.2f} [{ci[0]:+6.2f} {ci[1]:+6.2f}]"
            f"  trough {tb:+6.2f}  n={n}"
        )
    print("peak bias per subject by speed, S02 to S10 | S11 on")
    for sp in ("Walk_2km", "Walk_5km", "Walk_8km"):
        with open(work / f"{sp}.json") as f:
            cycles = json.load(f)["cycles"]
        per: dict[str, list[float]] = {}
        for c in cycles:
            per.setdefault(c["subject"], []).append(c["pred_peak"] - c["ref_peak"])
        means = {s: float(np.mean(v)) for s, v in per.items()}
        early = sorted(round(v, 1) for s, v in means.items() if int(s[1:]) <= 10)
        late = sorted(round(v, 1) for s, v in means.items() if int(s[1:]) > 10)
        print(f"{sp:20s} {early} | {late}")


if __name__ == "__main__":
    main(*sys.argv[1:])
