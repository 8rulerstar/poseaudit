# Real data: knee flexion per stride against marker motion capture

Knee flexion of a side-view YOLOv8, read in 2D from the image, against the
3D angle from marker motion capture, over 31 walking and jogging trials of
17 subjects on a treadmill. The speeds, 2, 5 and 8 km/h, are the file
names'; the pelvis stays within 0.35 m over the two minutes of a trial,
where 8 km/h would cover 270 m. At
8 km/h some subjects ran. `summary.txt` is what the audit prints.

```bash
python download.py data             # 1.1 GB from Zenodo, MD5 checked, 3 GB unpacked
pip install pandas scipy "poseaudit[plot,images]"
python build.py data out            # .mot pairs, stride starts, one row per trial
poseaudit paired --mot-pairs out/pairs.csv --cycles out/events.csv \
  --big-error 5 --offset-shifts 0.1 --json out/report.json
python cycles_gif.py out ../../docs # the README's cycles.gif
```

`cycles_gif.py` borrows the GIF helpers of `../coco_elbow`, so run it from a
clone of the repository. The dataset has 21 subjects; 4 have no walking
trial in the marker files, and some trials have only one side.

## What it shows

- Per stride, the peak flexion is read 17.3° lower [12.8 to 22.2] on
  average, and lower in all 17 subjects: by 0.5° in S15, by 40° in S06.
  With 17 subjects that interval is likely too narrow (the audit says so).
  The trough is read 2.6° lower, so the range of motion shrinks by 14.7°.
  Over every frame the bias is -6.2°.
- The subjects differ more than any figure across them shows: S02 to S10
  are read 13 to 40° low at the peak, S11 to S21 0.5 to 21°, mostly under
  15°. Why is not known. The thigh spans 2 to 2.6 times as many pixels in
  S02 and S03 as in the others (three of their trials are also the only
  ones at 60 frames a second), so the camera's distance or setup changed
  at least once; the dataset
  does not describe it.
- `--offset-shifts 0.1`: moving the time offset 0.1 s either way moves the
  RMSE by 7.2° and the peak bias by 0.05°. The peak and range per stride
  and the bias barely move with how the two recordings were lined up, for
  shifts this small; the RMSE (13.1°) and the mean |error| do not.

What the gap is not: a figure for YOLOv8 alone. It mixes the model, a 2D
angle against a 3D one, a camera whose angle to the walking direction is
not known, a hip keypoint against a hip joint centre, knee and ankle
markers on the outside of the joint against keypoints, and a time offset
estimated here. The audit says how large the gap is and where in the
stride it comes; it cannot say which of these makes it.

## How the two sides are made comparable

All of it is fixed in `build.py`, and none of it uses the knee angles:

- Marker knee: the angle at the lateral knee marker between the hip joint
  centre (Harrington et al. 2007, from the pelvis markers) and the lateral
  ankle marker. This is not the Plug-in Gait or OpenSim knee angle.
- YOLO knee: the angle at the knee keypoint between the hip and ankle
  keypoints, points under 0.3 confidence left out. The leg is the one the
  camera sees more confidently, as YOLO names it (the right in every
  trial).
- Time offset: the dataset is described as synchronised, but the two
  recordings of a trial are up to about a second apart. The offset is the
  lag where the fore-aft gap between the ankles correlates best (0.95 to
  0.999). Two ways were tried first and dropped: lining up by the knee
  angles, which is circular, and by foot strikes, where a swap of legs and
  half a stride look alike.
- Strides start where the heel marker of that leg is lowest. That is not
  initial contact: in walking it comes a few percent of the stride after
  it, in some running strides at mid-stance. 0% in the figures is this
  point.

## Data

Derived from "A Synchronized Dual-System Marker-based and Markerless Motion
Capture Dataset for Gait and Fundamental Fitness Activities", Capital
University of Physical Education and Sports (2026),
https://doi.org/10.5281/zenodo.19720803,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The keypoints are
the dataset authors' YOLOv8 predictions; the variant is not stated. Changes:
knee flexion computed from the marker trajectories and from the keypoints,
lined up in time and cut into strides as above. Nothing of the dataset is
stored here: `download.py` fetches it. The description mentions synchronised
video, which the files on Zenodo do not include.
