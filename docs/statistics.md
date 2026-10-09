# Statistics

What the figures mean, what the demo shows, and where they stop being valid.

## Reading the demo

The README's output is the left elbow angle (COCO keypoints 5, 7, 9: shoulder, elbow, wrist;
indices count from 0) of `yolo11n-pose` against COCO's human labels on 200
val2017 images. The data is in
[`examples/coco_elbow`](https://github.com/8rulerstar/poseaudit/tree/main/examples/coco_elbow);
run the command from that folder.

These figures are disagreement between the model and **one human label**, not
the model's error against the world. COCO's own annotators disagree: placing
the three points with the spread COCO publishes for repeated labels (its OKS
sigmas) moves the elbow angle by 11 to 16° on average on these same arms,
depending on how those sigmas are read (a rough estimate that treats each
label's points as independent; [`validation/label_noise.py`](https://github.com/8rulerstar/poseaudit/blob/main/validation/label_noise.py)).
That spread alone puts 25.9 to 39.9% of the arms off by 15° or more, against
the 43.2% observed ([`validation/label_noise.out.md`](https://github.com/8rulerstar/poseaudit/blob/main/validation/label_noise.out.md)).
A good part of the 19.48° may be the labels'. With a reference better than
the model (motion capture, careful relabelling), the same figures describe
the model.

### What a few lines of NumPy would miss

- **What was not read, and why.** Unlabelled points, unmatched people and
  missing predicted points are counted apart. The error figures cover what
  was read, so a model that skips its hard cases does look better on them:
  the count on the first line, and a warning past 30% unread, show it.
- **Intervals that respect the data.** Readings from one image are resampled
  together, and with `--cluster` so are the frames of one video or the photos
  of one subject (left out, frames count as independent).
- **The sorting trap.** When the labels are noisy, bias in bands sorted by the
  truth leans toward squashing even for an unbiased model, the more so the
  noisier they are ([below](#which-way-you-sort-decides-the-story)).
- **Decisions.** Whether the prediction lands on the same side of a threshold
  as the truth, which is what a pass or fail rule depends on.

### The statistics

When keypoints are used to *measure* something (a joint angle, a tilt, a length),
position metrics such as OKS, PCK or pixel error do not say how far off the
measurement is. `poseaudit` reads the measurement off the ground truth and off
the prediction and reports the difference the way a measuring instrument is
judged: how large, how often large, where, biased which way, and how sure you
can be of each figure. The statistics are the familiar agreement ones (Bland-Altman
limits, ICC, CCC, Deming and Theil-Sen slopes); what `poseaudit` adds is the
path to them from the formats computer vision evaluates in (COCO, YOLO,
supervision): matching, counting what could not be read and why, intervals
that respect images or subjects, and a check of whether a slope below 1 is
more than keypoint jitter.

## What the demo shows

**Disagreement grows as the arms get smaller.** When the arm segments average
under 30 px, 58% of elbows disagree with the label by 15° or more; above 60 px, 26%. The
signed bias of +2° hides all of this, and so does a single mAP. The link with
size is an association; size is not the whole cause. Some of it is geometry:
a pixel of error turns a short segment further than a long one, for the label
as much as for the model, and an arm pointing toward the camera looks short
and is hard to read even on a large person. Small people in COCO are also
more often occluded, blurred and loosely labelled.

| arm segment | n | mean abs error | 15° or more apart | slope (no jitter reference per band) |
|---|---|---|---|---|
| 0 to 30 px | 134 | 25.55° | 58.2% [49.7% to 66.2%] | 0.526 |
| 30 to 60 px | 112 | 16.57° | 36.6% [28.3% to 45.8%] | 0.791 |
| 60+ px | 76 | 13.06° | 26.3% [17.7% to 37.2%] | 0.909 |

## Which way you sort decides the story

The natural next check is the bias in bands of angle. Sort the same readings
three ways and you get three stories:

![Mean error per band when sorting by the truth, the prediction, or their mean](https://raw.githubusercontent.com/8rulerstar/poseaudit/main/docs/trap.png)

Whichever reading you sort by, its own noise puts its extreme values in the
extreme bands, where they read as bias (regression to the mean). Trust the
sort by the truth when the truth is much less noisy than the model (the
default, `--band-by truth`); sort by the mean of both (`--band-by mean`) and
read the Bland-Altman slope when the two are about equally noisy. Under the
wrong assumption, either view misleads: with a model much noisier than its
labels, the Bland-Altman slope calls an unbiased model expanding.

## Does the model squash large angles?

The **slope** (pred on truth, the proportional bias; earlier versions called
it the gain, still its name in the JSON) is the least-squares slope of the
predicted angle on the true one: 0.73 means that, on average over these readings, a 10° difference comes out as
about 7°. Keypoint jitter alone pulls a slope below 1 too: a straight arm can
only be read as more bent, a folded one only as more open. The **jitter
reference** (`vs jitter`, shown with `--full`) is the slope of predictions rebuilt from the truth
plus this model's own point displacements ([how](#the-jitter-reference)):
0.87 here. The model's slope is 0.140 lower (interval 0.076 to 0.199, paired
within resamples). With the default seed none of the 500 rebuilds comes out
as low as the model's (`p(slope <= jitter) <= 0.002`); with some other seeds one does
(0.004).
The model reads angle differences as smaller than its own scatter explains.
What that is, the check cannot say:

- **Gross failures and swaps.** Gross failures tied to the true angle (a
  straight arm read as folded) and left and right swapped on one side lower
  the slope just as squashing does; failures in random directions are part of
  the rebuilds and do not widen the gap. Leaving out the 31 readings off by
  45° or more leaves a gap of -0.068 [-0.109 to -0.028] (about half the gap),
  and leaving out the 61 off by 30° or more, -0.038 [-0.073 to -0.003]. Those
  subsets are chosen by the outcome, so they describe where the gap comes from
  rather than test it, and trimming by the outcome changes a gap by itself, so
  what is left is not a clean measure of squashing either. Some of the largest errors may be
  the labels' (a point on the wrong limb), and the 58 instances not read may
  not be a random part of the rest.
- **Noise in the truth.** The rebuild takes its displacements against the
  labels, so label noise is already in its scatter. On these same arms, honest
  models with labels as noisy as COCO's annotators fell 0.02 to 0.04 below the
  reference, far less than 0.140
  ([`validation/label_noise.py`](https://github.com/8rulerstar/poseaudit/blob/main/validation/label_noise.py), with independent
  normal noise on each point). Under that model label noise does not explain
  this gap, but it does trip the p: those honest models got
  `p(slope <= jitter) <= 0.05` in 7 to 23% of 30 runs each. With noisy labels, read
  the size of the gap, not the p.
- **Squashing** is what remains, and it is not separated from the two above.
  The slopes that suit equal noise, the Bland-Altman slope (-0.049 [-0.119 to
  +0.016]) and Deming at a noise ratio of 1 (0.945 [0.870 to 1.019]), include
  no squashing, but do not rule out a small one. Which to trust
  depends on the noise ratio, which this data does not pin down. Deming for
  other ratios (`--noise-ratio`, the variance of prediction noise over that
  of label noise):

  | noise ratio | 1 | 4 | 10 |
  |---|---|---|---|
  | Deming slope | 0.945 [0.870 to 1.019] | 0.797 [0.707 to 0.877] | 0.758 [0.667 to 0.842] |

  On these arms honest models give a Deming slope at a ratio of 1 near 1
  whether the labels are clean or as noisy as the model, with independent
  normal noise and no gross failures
  ([`validation/label_noise.py`](https://github.com/8rulerstar/poseaudit/blob/main/validation/label_noise.py)): there, jitter
  at the ends of the range does not drag it down the way it drags the least-squares slope.

For this demo, then: a real gap, about half of it in the largest errors.
Whether the rest is squashing of a few percent or nothing, this data cannot
tell.

A few gross failures move a least-squares slope a lot. The **Theil-Sen
slope**, which they barely move, is 0.79 here against the least-squares
0.73. Noise that grows with the angle, or on smaller arms, also separates the
two, so their difference alone does not say how much comes from gross
failures, and a gross failure can also be a label on the wrong limb.

## The jitter reference

Each rebuilt arm takes all its point shifts from one arm with a similar true
angle (one of up to six equal-count groups; possibly itself), scaled to its
own size. Shifts are expressed along and across each point's own segment, and
mirrored for arms that bend the other way, so an error along the limb stays
along it and one toward the inside of the bend stays there. The shift a whole
group shares is removed first, because that shared shift is the tendency
being tested. Drawing each point from a different arm instead would break the
correlation within an arm (a whole limb shifting at once barely changes its
angle) and set the reference too low.

## Statistics used

| figure | definition |
|---|---|
| slope | least-squares slope of predicted on truth (`gain` in the JSON) |
| Theil-Sen | Theil-Sen slope (`robust_gain` in the JSON): median of the slopes between pairs of readings |
| vs jitter | median slope over 500 rebuilds (see [The jitter reference](#the-jitter-reference)); the gap's interval averages 10 rebuilds per resample; p(slope <= jitter) = (1 + rebuilds with a slope at or below the model's) / 501 |
| BA slope | slope of the error on the mean of both readings (Bland and Altman 1999) |
| Deming | slope of predicted on truth with a known ratio of noise variances (Deming 1943; Linnet 1993); the ratio is prediction over label |
| limits | 2.5th and 97.5th percentiles of the error; normal limits are bias ± 1.96 SD (Bland and Altman 1986) |
| repeated limits | bias ± 1.96 √(between + within variance) (Bland and Altman 2007) |
| ICC(A,1) | two-way, absolute agreement, single rating (McGraw and Wong 1996); matches pingouin; its interval is the bootstrap, not the F interval |
| CCC | Lin's concordance correlation (Lin 1989) |
| intervals | percentile bootstrap over clusters (Davison and Hinkley 1997); Wilson score intervals for rates without named clusters (Wilson 1927) |

The Theil-Sen slope uses every pair of readings up to about 1,000 readings and
500,000 random pairs above that.

### References

- Bland JM, Altman DG (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet* 327(8476):307-310.
- Bland JM, Altman DG (1999). Measuring agreement in method comparison studies. *Statistical Methods in Medical Research* 8(2):135-160.
- Bland JM, Altman DG (2007). Agreement between methods of measurement with multiple observations per individual. *Journal of Biopharmaceutical Statistics* 17(4):571-582.
- Davison AC, Hinkley DV (1997). *Bootstrap Methods and their Application*. Cambridge University Press.
- Deming WE (1943). *Statistical Adjustment of Data*. Wiley.
- Kanko RM, Laende EK, Davis EM, Selbie WS, Deluzio KJ (2021). Concurrent assessment of gait kinematics using marker-based and markerless motion capture. *Journal of Biomechanics* 127:110665.
- Lin LI (1989). A concordance correlation coefficient to evaluate reproducibility. *Biometrics* 45(1):255-268.
- Linnet K (1993). Evaluation of regression procedures for methods comparison studies. *Clinical Chemistry* 39(3):424-432.
- McGraw KO, Wong SP (1996). Forming inferences about some intraclass correlation coefficients. *Psychological Methods* 1(1):30-46.
- Nakano N, Sakura T, Ueda K, et al. (2020). Evaluation of 3D markerless motion capture accuracy using OpenPose with multiple video cameras. *Frontiers in Sports and Active Living* 2:50.
- Ronchi MR, Perona P (2017). Benchmarking and error diagnosis in multi-instance pose estimation. *ICCV*, 369-378.
- Sen PK (1968). Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association* 63(324):1379-1389.
- Theil H (1950). A rank-invariant method of linear and polynomial regression analysis. *Indagationes Mathematicae* 12:85-91.
- Wilson EB (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association* 22(158):209-212.

## Limitations

- The figures describe agreement with the ground truth, not with the world:
  the truth's own error is inside them, and 2D angles are not 3D joint angles.
  Studies that validate markerless angles against motion capture (Nakano et
  al. 2020, Kanko et al. 2021) answer the question against a better
  reference; Ronchi and Perona (2017) break keypoint errors into jitter,
  inversion, swap and miss, which `poseaudit` does not.
- Every reading counts once: frames pooled from a few trials inflate ICC and
  CCC, and there is no per-trial or per-subject summary (peak angle, range of
  motion) yet.
- Tilts near horizontal wrap from +90 to -90; the report warns when many are
  close.
- The jitter reference is new, not a published method, and was checked by
  simulation only ([`validation/jitter_reference.py`](https://github.com/8rulerstar/poseaudit/blob/main/validation/jitter_reference.py),
  seeded). For models with no tendency and clean labels, with 300 arms and
  600 runs per case, the one-sided p was 0.05 or less in 2.3 to 5.7% of runs,
  and the gap's interval lay wholly below 0 in 2.2 to 4.3% with the 200
  resamples that table uses, and in 1.3 and 2.7% for the two cases rerun at
  the tool's defaults (150 runs each; 2.5% nominal). With 40 arms and noise
  that grows on straight arms the p ran to 8%, hence the warning under 60
  readings (where exactly to draw that line was not tested). **With labels as
  noisy as the model, honest models were flagged in about 35% of runs** (37%
  at 200 resamples, 35% at the defaults): the check assumes labels much
  cleaner than the model. A 7% squash was caught in every run, with point
  noise of 8% of the arm; with noisier points it is caught less often (not
  measured here). It assumes independent objects whose errors, apart from a
  shift shared by similar angles, depend only on the true value and the
  object's own frames. The p and the interval can disagree near the edge.
- Matching is greedy and ignores scores (filter them first with `--min-score`);
  in dense crowds an optimal assignment may pair differently.
- One measure per run, one level of clusters, no time series.

## Roadmap

- a label-noise reference beside the jitter one (from annotator spread or
  repeated labels)
- left-right swap detection
- auditing angles given directly (filtered or from inverse kinematics)
- directed angles (0 to 360), signed joint angles, a chosen reference axis
- regression-based limits of agreement; per-subject summaries
