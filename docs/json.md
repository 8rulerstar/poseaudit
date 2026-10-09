# JSON, CSV and exit codes

`--json` writes every figure, with `poseaudit` (the version) and `schema`
(1) at the top and the settings used. NaN and infinity become `null`. There
`limits` there are the normal limits; the percentile limits the summary
prints are `percentile_limits` (`empirical_limits` is the same pair under its
old name),
both `null` under 10 readings. Warnings, including those raised while
loading, are in `warnings`. `--csv` writes one row per reading:
`image, truth_index, predicted_index, class_id, cluster, size, truth,
predicted, error, mean`. Same inputs and seed, same bytes.

With several measures the JSON is `{"poseaudit", "schema", "settings",
"measures": [...]}`, each entry of `measures` what a single measure's JSON
would be, and the CSV leads each row with a `measure` column. With one measure
the JSON is as above, unchanged. Comparing models, it is `{"poseaudit",
"schema", "settings", "models": {name: [one entry per measure]},
"differences": [...], "warnings": [...]}`, the differences being the rows of
`Comparison.table()`, which `--csv` writes, and the warnings those about
paired intervals (too few shared images, or none).

The command exits 0 on success (warnings included); 1 on bad input or
settings, a failed write, or nothing read (the JSON and CSV are still
written); and 2 when the arguments cannot be parsed. There is no pass or fail
threshold built in; gate on the JSON. An interval that cannot be computed is
`null` (every interval is, with a single cluster), and in jq `null <= 3` is
true, so make a gate fail on `null`: for example
`jq -e '(.mean_abs_error_ci[1] // 1e9) <= 3 and .n / .measurable >= 0.9'`, or
without jq, in Python:

```python
import json

r = json.load(open("figures.json", encoding="utf-8"))  # written by --json
high = r["mean_abs_error_ci"][1]  # None when the interval is null
if not (high is not None and high <= 3 and r["n"] / r["measurable"] >= 0.9):
    raise SystemExit("poseaudit gate failed")
```

`--resamples` and `--jitter-repeats` trade precision for speed: 50,000
readings take about three minutes at the defaults.
