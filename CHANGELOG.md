# Changelog

## Unreleased

### Added

- Several measures in one run: `--angle`, `--tilt`, `--length` and `--ratio`
  may be repeated. The output is one table with a row per measure (n, bias,
  percentile limits, mean |error|, RMSE, the large-error rate, slope, ICC);
  `--full` prints each measure in full. The JSON then holds the shared
  settings and a `measures` list of single-measure entries; with one measure
  it is unchanged. In the table the figures line up on the decimal point.
  When the measures share one large-error threshold the rate's header names
  it (`>= 15°`); when they differ a `large if` column gives each row its own.
- `--big-error` for each kind of measure,
  `angle:15,tilt:5,length:10,ratio:0.1`, or by unit, `deg:15,px:10`,
  comma-separated or repeated; a kind wins over its unit, and a bare number covers the measures the others do
  not name. A bare number over measures in different units is refused, since
  15 counts as large on an angle and never on a ratio: the table showed 0.0%
  for every ratio. In Python, `compare(..., big_error=...)` takes the same
  as a mapping, `{"angle": 15, "length": 10, "ratio": 0.1}`, and refuses one
  number for measures in different units.
- `AuditResult.plot()` without a file returns the figure, which a notebook
  shows inline, sharp on a high-density screen, with or without pyplot
  imported; with a file it saves it and returns it too. The demo notebook
  shows it this way.
- docs/cli.md lists the messages a run stops with and what to do about
  each; a test checks that the code still says them.
- With several measures, `--csv` leads each row with a `measure` column,
  and the run fails only when no measure read anything. Giving no measure
  now says "give at least one of".
- Model comparison: `pa.compare({name: predicted}, truth, measures,
  big_error)` and repeated `--pred NAME=PATH`. For each measure and pair of
  models, the differences in mean |error| and in the large-error rate on the
  readings both made, with paired bootstrap intervals over images (or named
  clusters) and the shared and per-model counts. `Comparison.table()` gives
  flat rows and `to_csv()` writes them. A pair whose shared readings come
  from fewer than 20 images or clusters, or from none, is warned about
  under the differences and in the JSON's `warnings`. Comparing models on
  the command line, `--csv` writes the differences, one row per measure and
  pair, not the readings; `--json` writes `models`, `differences` and
  `warnings` (the loader's warnings included, each printed once);
  `--report` writes the comparison as Markdown and `--plot` is refused.
- Comparing models, each difference counts the shared readings the two models
  read differently (`n_differing`, a new column in the comparison CSV and in
  the JSON rows), and a warning says when fewer than 20 differ: the paired
  intervals rest on those alone. In the demo's score-filter comparison one of
  254 does; docs/python.md now says so.
- In Jupyter or IPython a result (`AuditResult`) or a `Comparison` on its own
  shows its summary, and `repr()` of either, or of a `Pairing`, is one line.
  The dataclass repr listed every reading and keypoint array: 80,000
  characters for the demo's 322 readings, 780,000 for its pairing. The demo
  notebook shows a result this way.
- On a terminal, output fits the window: a table of several measures or of
  model differences wider than the terminal is cut into blocks of columns,
  each led by the measure, and long summary lines and warnings break between
  words, indented under what they continue. At 80 columns the table of five
  measures was 128 wide and every row wrapped. Piped or redirected output is
  unchanged.
- CLI: `--image-size` also takes `WxH`, and an error names a malformed size
  such as `1280x`; `--band-by` has a help line; `poseaudit audit --help` ends
  with examples.
- `python -m poseaudit` runs the command, as `poseaudit` does.
- Before a long run the CLI says on stderr that it can take a minute or more
  (or several minutes) and names the flags that make it faster; until now it
  printed nothing until done. It weighs the pairs by the resamples, the
  jitter rebuilds (each costs about as much as 20 resamples) and the number
  of measures.
- Loading COCO annotations warns how many crowd regions (`iscrowd` 1) were
  left out: they are in no count, and a prediction on one counts as
  unmatched, unless its box overlaps a labelled person by `--min-iou`.
- With no large error, the warning says how high the rate could still be.
- A warning when the percentile limits rest on 40 readings or fewer.
- The JSON also gives the percentile limits as `percentile_limits` (the same
  as `empirical_limits`); `limits` stay the normal ones.
- `AuditResult.percentile_limits`, `.slope` and `.theil_sen` are aliases of
  `empirical_limits`, `gain` and `robust_gain`, and `AuditResult` has a
  docstring listing its main fields.
- `AuditResult.to_rows()`: the headline figures as a flat row.
- A comparison as tables for a paper, a blog or a leaderboard:
  `--report` with several `--pred` (until now refused) and
  `Comparison.to_markdown()` write one Markdown table of every model's
  figures on every measure, one of the differences on shared readings, the
  warnings and the settings; `Comparison.to_rows()` gives the first table as
  flat rows led by `model`, for a CSV. With 3 models and 4 joints the
  per-model figures were only in three space-aligned text tables or nested
  in the JSON.
- docs/README.md, an index of which page holds what, and how to install the
  extras (`poseaudit[plot]`, `[images]`, `[supervision]`), which no page
  named: the README's own `--plot` step failed after its
  `pip install poseaudit`. Every docs page starts with links to the others;
  json.md, python.md and statistics.md linked to none.
- docs/cli.md lists every option of `poseaudit audit` in tables by group, as
  the README said it did; `--band-by`, `--jitter-repeats`, `--min-in-frame`,
  `--noise-ratio`, `--resamples` and `--seed` were in no page. A test keeps
  the tables in step with the parser.
- docs/json.md gives each figure's name in the summary, the JSON, Python and
  `to_rows()` side by side (the summary's `limits` are the JSON's
  `percentile_limits`, its `normal` the JSON's `limits`, its `slope` the
  JSON's `gain`), and a test checks the keys exist.
- The README figures are drawn by the new plot code, and
  `docs/social-preview.png` (1280 x 640, with an SVG) is a plain card for the
  repository's social preview. `examples/coco_elbow/figures.py` makes all
  three from the demo data.

### Changed

- The default summary leads with the figures of a method comparison: bias,
  the percentile limits and (when the tails are close to normal) the normal
  limits, mean |error| and RMSE, the large-error rate, the slope and
  ICC(A,1). The jitter reference and its p, the BA slope, Deming, CCC and r
  are under `--full`. Each limits line names its JSON key.
- In text the gain is called the slope (pred on truth, the proportional
  bias) and the robust gain the Theil-Sen slope; `p(squash)` is
  `p(slope <= jitter)`, which compares the slope with the jitter reference
  and does not by itself measure squashing. JSON keys and attribute names
  (`gain`, `robust_gain`, `jitter_p`) are unchanged.
- The `--plot` figure is drawn for print. It is 7 inches wide with 12 point
  text at 300 dpi, so shrunk to one 3.5 inch column of a two-column paper its
  text is still 6 points; until now it was 10 inches wide with 8 point
  legends, under 3 points at that width. Its colours are Okabe and Ito's, and
  every line also differs in dash pattern: the line of equality and the
  fitted line were both solid and nearly the same gray when printed in black
  and white. The bias and the limits carry their values and unit at the right
  of the Bland-Altman panel, the fitted slope its interval, one legend sits
  under both panels, and the axes read "predicted minus truth" and share
  their range. Many readings are drawn fainter, and as an image inside an SVG
  or PDF above 5,000. PDFs keep their text as TrueType. The style applies to
  the figure only: your matplotlib settings are left alone. Tick labels carry
  their whole value, with no offset or power of ten in the corner (100,000
  and more take an SI prefix such as 1.6M); degree axes spanning under 90
  degrees tick in whole steps; lines that print the same value share one
  label; and under 10 readings the points are drawn over the lines, so a lone
  reading is not hidden under the bias line.
- Under 10 readings the plot no longer draws percentile limits; the summary
  and the JSON give none there, since they are only the smallest and largest
  errors.
- `poseaudit --help` says what the tool does. `poseaudit audit --help` uses
  short placeholders (`--image-size W H`, not `W [H ...]`), says which
  options are required, gives every default, an example comparing two models,
  and that `--plot` writes PNG, SVG or PDF by the file's extension. Each
  measure's option names its points and what it reads (`--angle A,B,C` is the
  angle at B between B-A and B-C; `--tilt A,B`, `--length A,B`,
  `--ratio A,B,C,D`), with COCO's shoulder, elbow, wrist and hip indices,
  where all four read `I,J,...` before. `poseaudit` with no arguments prints
  the help instead of an error about a missing COMMAND.
- `report.md` opens with the headline in words: how far off the labels, on
  how many of the labelled instances, that the figures are agreement with the
  labels rather than error against the true value, and which way an error
  points. It names a relative reading as such and never rounds a rare large
  error to 0%. The settings are a table at the end instead of a Python dict.
  Size band labels share one number of decimals.
- The report has a Summary and a How to read this heading, and the numbers in
  its tables are right-aligned. Its pointer to "Which way you sort decides the
  story" named a README section that has moved; it now links to
  docs/statistics.md.
- The warning that coordinates look normalised comes before any figure, in
  the summary and the report, and sizes are then not labelled in px.
- With fewer than 10 readings the percentile limits print as n/a, and are
  `null` in the JSON (`percentile_limits` and `empirical_limits` both).
- When the truth hardly varies, a warning gives its range and says the slope,
  ICC and CCC are not defined; any figure that is not defined prints as n/a
  in the summary and the report, never as nan.
- With named clusters the repeated-readings limits are in the default
  summary, not only under `--full`.
- The warning about tilts near horizontal names the ICC, CCC and r too:
  readings either side of the wrap count as far apart, which makes them look
  far too good. The errors are not affected.
- Normalised coordinates are judged per side on every visible point, with
  0.05 of slack for points just past the frame. When only one side looks
  normalised, the warning says the two sides are in different units, which
  is why nothing pairs.
- A keypoint index out of range says how many keypoints there are and that
  indices count from 0; unpaired instances are checked too.
- Loader warnings print on stderr as they are raised, before a long run
  rather than after it, and are no longer lost when the run then fails.
- Every output (`--json`, `--csv`, `--report`, `--plot` and the `to_csv`,
  `to_markdown` and `plot` methods) is written to a temporary file beside it
  and then moved into place, so a run killed part way leaves the old file or
  none, never half of one, and parallel runs writing the same path each leave
  a whole file (the last to finish wins).
- The jitter reference rebuilds faster with the same numbers: each object's
  frames are computed once and taken by row in every resample, and the
  rebuilt points are read without being laid out first. Lengths and ratios
  are read all at once rather than one object at a time, and angles without
  reductions along an axis of two. On 50,000 readings at the defaults one
  measure took 4.5 minutes instead of 7 on the machine measured; every
  figure is the same to the last digit.
- Upgrading from 0.1.4, measures are computed in a slightly different order of
  floating-point steps, so the readings in the CSV and a few dozen figures in
  the JSON can differ in their last two or three digits, up to about 1e-13
  relative (the demo's median error moves from 0.5241505123347991 to
  0.5241505123348205). No figure changes at the precision printed.
- README: 100 lines. It opens on a GIF of the demo command typed in a
  terminal and the summary it prints, and a second GIF draws the
  score-filter comparison person by person; `examples/coco_elbow/gifs.py`
  makes both from the demo data, and a test checks the first shows the
  Quick start's command and output. Commands are on one line, so they paste
  into PowerShell as they are. Short recipes for a validation paper and a
  CI gate; the worked paper paragraph is in docs/statistics.md (Writing it
  up) and the real `--full` output in docs/cli.md, both checked by a test.
  The options, the Python API (with a runnable arrays snippet), the
  statistics and the JSON moved to docs/; the JSON gates fail on a `null`
  interval (in jq `null <= 3` is true), with a Python alternative to `jq`.
- docs/python.md covers every public name, says what the outputs reveal
  (paths as typed, image names) and how far results agree across machines.
  Its box snippet runs.
- README: the jitter check's caveat says it assumes labels much cleaner than
  the model.
- The demo notebook prints the slope under that name, as the summary does
  (`gain` stays in the JSON).
- With several measures in different units, `--threshold`,
  `--pred-threshold` and the edges given to `--bands`, which are in the
  measure's unit and apply to every measure, are refused.
- Under a table of several measures, a warning they raise alike comes once,
  naming them ("every measure" for all), and the advice that ends the
  near-miss warning or the warning about few readings for the jitter
  reference comes once after each measure's counts. On the demo's tilt and
  length the near-miss advice was printed twice, about 370 characters each.
- Plot ticks step by 1, 2 or 5 times a power of ten (degrees across a joint's
  range by 45), as many as the axis has room for: matplotlib's own choice
  put the trap figure's at -8, 0, 8, 16. The predicted-against-truth panel
  has the same ticks on both axes, as does the Bland-Altman panel's x-axis;
  on lengths the y-axis had 0, 50, 100, 150, 200 beside an x-axis of 0, 100,
  200. `docs/trap.png` has the same angle axis as `docs/panels.png`.
- The comparison CSV and `Comparison.table()` put each interval beside its
  estimate (`mean_abs_error_diff`, then its `_ci_low` and `_ci_high`) and
  give each row's large-error threshold, `big_error`, which a run with a
  threshold for each kind of measure left out. `Difference.big_error` holds
  it.
- `compare()` records each model's name in its results' settings, as the
  command line did, and the JSON's top-level `settings` are those every
  model shares: they named the first model and its file.
- `audit()` takes `big_error` as a mapping by kind or unit, as `compare()`
  does, and keeps it as a float, so the JSON reads `15.0` from Python as
  from the command line.
- The report says "An error of 15 degrees or more counts as large" where
  it said "big error >= 15°", the one place outside code that called a
  large error big.
- "What the outputs reveal" and the runtime at 50,000 readings moved to
  where a reader looks for them: docs/json.md and the option tables in
  docs/cli.md.
- `--images` without Pillow says `pip install "poseaudit[images]"`, as
  `--plot` names its extra.
- docs/statistics.md says how to recompute the figures from the CSV: z is
  1.959964, percentiles are NumPy's default (type 7), CCC uses population
  moments, Theil-Sen leaves out pairs with the same true value, and a tilt's
  slopes use truth plus error.

### Fixed

- A COCO results file without keypoints (a box detector's) stopped with
  `a file is not in the format given: KeyError('keypoints')`; it now names
  the file and the result and says a pose model's results are needed. A
  missing `bbox`, `image_id`, `file_name` or `annotations` list is named
  the same way, and an image id that is not in the annotations asks whether
  the predictions come from another annotation file or split.
- CLI: `--plot` without matplotlib fails before the audit runs. On a console
  that is not UTF-8 (Git Bash on Korean Windows, for one) the degree sign
  prints as " deg", with the columns kept aligned. The plot hint reads
  `pip install "poseaudit[plot]"`.
- The crowd-region warning said a prediction on a crowd region always counts
  as unmatched. It is paired with a labelled person when its box overlaps
  them by `--min-iou`, and the warning now says so. The warning about
  unmatched truths beside unmatched predictions, which suggests a lower
  `--min-iou`, adds that this pairs crowd members and other unlabelled people
  with the wrong person.
- With several measures and nothing read, `--report` wrote an empty file;
  with one it wrote none, leaving any report from an earlier run in place.
  The report is now written either way and says what was not read, as the
  JSON and CSV do. The closing message says when `--plot` drew nothing, and
  when the file there is from an earlier run.
- A class filter that kept a crowd region's class counted the region as left
  out while the next warning said no entry had that class; the warning now
  says no labelled person (or no detection) has it.
- A COCO person written with `"keypoints": null` and `num_keypoints` 0 is left
  out again, as in 0.1.4, instead of stopping the load.
- `--min-in-frame` (and `min_in_frame` with `relative_to`) must be 2 or more.
  At 1 an image's only reading had no others to be read against, and the run
  failed with "SVD did not converge".
- With a single image or cluster every interval is NaN (`null` in the JSON,
  `[no interval]` in the summary) instead of a single point, the Wilson
  intervals of rates included. Fewer than 20 images get the same warning as
  fewer than 20 named clusters.
- The BA slope, ICC(A,1) and CCC are NaN when the truth does not vary,
  instead of 2, 0 and 0 whatever the model does. A truth counts as flat only
  when its range is below 1e-12 of its size, so large values with a real
  spread keep their slopes.
- A figure undefined in more than 2.5% of bootstrap resamples, such as the
  ICC of two images when a draw repeats one, now has no interval instead of
  one drawn from the resamples where it happened to be defined.
- Renaming the clusters (`--cluster`, `cluster=`) no longer changes the
  bootstrap intervals: groups are drawn in order of first appearance. A
  regular expression string passed as `cluster` in Python gets a clear
  TypeError.
- In resamples, repeated-readings limits count a cluster drawn twice as two
  clusters, not one twice the size.
- Intervals and limits never print as -0.000.
- A COCO person whose `num_keypoints` says 0 but whose points are flagged
  as labelled is kept; only a person with no labelled point is left out.
- A class filter on a COCO results file with no `category_id` says so,
  instead of listing an empty set of categories present.
- `from_supervision` names a NaN or infinite `class_id` instead of failing
  with a bare conversion error.
- `Instance` refuses a box holding NaN or infinity; such a person could
  never be matched and was left out silently.
- `--images` reads the size of a photo stored on its side (EXIF orientation
  5 to 8) as it is shown, width and height swapped. Giving `--images` and
  `--image-size` together is refused instead of one being ignored. An empty
  YOLO label file needs no image size.
- `from_supervision` refuses a fractional `class_id` instead of truncating it.
- An empty COCO file, or one that is not valid JSON, is named in the error;
  one starting with a byte order mark is read. `angle`, `length` and `ratio`
  stay finite on huge coordinates.
- An image id written as a number in one COCO file and as a string in the
  other is named as such, instead of reported as missing.
- Keypoints given as unsigned integers (such as uint8) no longer wrap round
  when subtracted: an `Instance` and every measure read them as float64.
  float32 keypoints are widened the same way.
- A measure given twice with its keypoints listed the other way round
  (`--angle 5,7,9 --angle 9,7,5`) is refused like any other duplicate.
- A COCO annotation file that lists an image id twice, or one file name
  under two ids, is refused: the people of two images would be paired
  together.
- `Instance` and `Instance.from_keypoints` refuse keypoints not of shape
  (K, 2), such as x, y, v rows, and a `visible` not of shape (K,), naming the
  shape expected.
- A measure with a repeated keypoint, such as `angle(25, 25, 27)`, is refused
  when made; the two segments of a ratio may still share a point.
- `audit(noise_ratio=...)` refuses a ratio of 0 or less, as the CLI does.
- The CI gate in docs/json.md no longer divides by zero when nothing was
  measurable.
- Python: `big_error` in a wrong form stopped with an internal error. The
  command line's `"angle:15,length:10"` passed to `compare` gave
  `Unknown format code 'g' for object of type 'str'`, a mapping passed to
  `audit` gave `'>' not supported between instances of 'dict' and 'int'`,
  and a list of measures passed to `audit` an `AttributeError`. Each now
  says what to pass. A mapping key that is neither a kind nor a unit
  (`lenght`, `ANGLE`) is refused, as the command line refuses it; it was
  dropped silently when no measure needed it.
- `--plot` with two measures and no Matplotlib asked for Matplotlib, and
  once it was installed said `--plot draws one measure`; the second comes
  first now.
- `skeletons differ: truth has [17] keypoints, predictions [5]` reads
  `truth has 17 keypoints, predictions 5`.
- Nine tests failed without Pillow or Matplotlib instead of skipping, as
  the others do; they skip, or leave out only the step that needs the
  extra.
- docs/cli.md: the error table says that a message about a file may start
  with its name, and lists the refusal of `--threshold`, `--pred-threshold`
  and `--bands` edges across units. The demo notebook calls the instances
  labelled, as the summary does, and prints what was not read in words.
- README: the paper text says the interval of the share of large errors spans
  both the bootstrap interval and Wilson's score interval. The demo notebook
  links into docs/statistics.md instead of README sections that moved, and
  its saved outputs come from this version.

## 0.1.4

- Pairing: a warning when unmatched truth images share their last name (after
  the last `/` or `\`, without an image extension) with unmatched predicted
  images, as `imgs/a.jpg` and `a` do: folders differ, so they are not paired.
  The warning for names that share nothing gives the same hint.
- Pairing: the example in the warning about dropped extensions no longer
  depends on the order images were listed in.
- When nothing pairs because one side has no visible point (all labels
  unlabelled, or every predicted point under the confidence cut), the warning
  says so instead of pointing at image names.
- Summary: "Only 1 reading", "No large errors" and "1 cluster" in the singular,
  and limits print as n/a when too few readings give none.
- README: the share of arms that COCO's label spread alone puts off by 15° or
  more (25.9 to 39.9%) beside the headline; `from_keypoints` described as
  giving no box; the OKS used for pairing differs from COCO's own.

## 0.1.3

- Pairing: dropping an image extension keeps the folder, so `cam1/0001.jpg`
  and `cam2/0001.jpg` are no longer paired as one image.
- On Windows, image names are kept as given when the extension is dropped.
- README: the demo figures are described as disagreement with COCO's labels,
  not as error; a box example for `Instance`, a note that the demo data is not
  in the package, and references to markerless validation studies and Ronchi
  and Perona (2017).

## 0.1.2

- Pairing: boxes that qualify by IoU are ranked by the mean of IoU and OKS,
  so two people with nearly the same box are paired by their keypoints, not
  swapped. Ties go to OKS, then IoU, so the order of the lists no longer
  matters; an exact tie between different predictions is warned about. On
  the COCO elbow demo 5 of 490 pairs change: mean error 19.55 to 19.48°,
  large errors 43.5 to 43.2%; the README is updated.
- A warning when images overlap but no truth, or only a few images, pair.
- The rate of large errors and the decision rates resample whole images (or
  named clusters) whenever some hold several readings, as the other
  intervals do.
- `Instance` and `Instance.from_keypoints` refuse a visible point with a NaN
  or infinite coordinate; `from_keypoints` accepts lists.
- `audit` warns when every coordinate read lies within 0 to 1.
- `length` reads NaN for two points on one pixel, like the other measures.
- Keypoints given by name raise a clear error: measures take indices.
- Swapped `--gt` and `--pred` (or loader arguments) are named as such.
- `from_supervision` asks for an `{image name: sv.KeyPoints}` mapping when
  given a bare container.

## 0.1.1

- Python 3.14 is tested in CI and listed as supported.
- PyPI page links to the source, the issue tracker and this changelog.
- The example and validation scripts read and write their JSON as UTF-8, so
  they behave the same on systems whose default encoding is not UTF-8.

## 0.1.0

First release.

- Measures: tilt, angle, length and ratio, optionally relative to the rest of
  each image (median or a percentile, of values or of their absolute values,
  leaving each reading out of its own baseline).
- Inputs: COCO annotations and results, Ultralytics YOLO pose, supervision
  `sv.KeyPoints` (with `sv.Detections` for boxes), or arrays; class filters
  and class-aware matching.
- Agreement: error, large-error rate, bias, percentile and normal limits of
  agreement, repeated-measures limits, ICC(A,1), CCC, Pearson r, error by size
  of the measured part.
- Gain and squashing: least-squares and robust (Theil-Sen) gain, Bland-Altman
  and Deming slopes, and the gain keypoint jitter alone gives, with the gap's
  interval and a one-sided p(squash).
- Decisions at thresholds, with a separate or count-matched predicted
  threshold.
- Intervals resample whole images or named clusters.
- Output: summary, markdown report, CSV, JSON and a two-panel plot.
