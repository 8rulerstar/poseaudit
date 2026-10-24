# Public benchmark: four video pose models against IMU joint angles

The VIDIMU benchmark ("Paving the Way Towards Kinematic Assessment Using
Monocular Video", [10.5281/zenodo.15088423](https://doi.org/10.5281/zenodo.15088423),
CC BY 4.0) compares the joint angles of four monocular 3D pose models
(BodyTrack, MMPose, MotionAGFormer, MotionBERT) with IMU inverse kinematics
in OpenSim, over 16 subjects and 13 daily activities. Its authors published
an RMSE per activity and model. This example runs the same angles through
`poseaudit paired`. `summary.txt` is what it prints.

```bash
python download.py data            # 465 MB from Zenodo, MD5 checked
pip install pandas poseaudit
python build.py data out           # one .mot pair per subject and activity, per model
for m in bodytrack mmpose motionagformer motionbert; do
  (cd out/$m && poseaudit paired --mot-pairs pairs.csv --time-offset auto --big-error 10 --json report.json)
done
python compare.py data out         # the published RMSE beside poseaudit's bias and RMSE
```

`build.py` follows the authors' choices: one joint per activity, the first
trial every model and the IMU hold, and the video angle as the 3D angle
between two keypoint vectors (the definitions are in its docstring).

## What it shows

- The published RMSE leaves out the constant gap between the two sides:
  the authors remove each signal's own mean before computing it. With the
  same preprocessing, that RMSE reproduces the published figures within
  0.2° on average over the 52 activity and model cells (checked outside
  this example, with the authors' code).
- Kept in, that gap reorders the models. MotionAGFormer has the lowest
  published RMSE (9.2° on average over the activities) but the largest
  bias (24.8° on average, in absolute value) and the largest RMSE (30.8°);
  BodyTrack, 11.0° published, has the smallest (12.9° and 20.7°).
- An example: shoulder flexion in A13, published RMSE 4.99° for
  MotionAGFormer; poseaudit reads it 48.6° above the IMU on average
  [45.9 to 51.2], with an RMSE of 49.5°.
- Lining the clocks up (`--time-offset auto`) lowers the RMSE by 1.0° in
  the median cell (0.2 to 4.6°) and moves the bias by 0.2° or less in 48 of
  the 52 cells (at most 1.7°, MotionBERT in A06): the gap is not a timing
  artifact (compared outside this example, against no offset).

## What it does not show

- The video angle (between keypoint vectors) and the IMU coordinate
  (OpenSim's `arm_flex`, `elbow_flex`, `knee_angle`) are defined differently,
  so a constant gap between them is expected and removing it can be a fair
  choice. That part is the same for all four models; the differences
  between the models' biases on the same joint (in A13, -10.6° for BodyTrack
  against +48.6° for MotionAGFormer) are not.
- The reference is IMU inverse kinematics, not marker motion capture.
- With 16 subjects the intervals are likely too narrow (the audit says so).
  The offset search flags 57 of 208 MotionAGFormer pairs whose offset is
  more than two frames from the median, 3 with a rival lag and 5 on the
  edge of the search.
