# JSON, CSV and exit codes

[README](../README.md) · [index](README.md) · [command line](cli.md) · [Python](python.md) · [statistics](statistics.md) · **JSON, CSV and exit codes**

`--json` writes every figure, with `poseaudit` (the version) and `schema`
(1) at the top and the settings used. NaN and infinity become `null`.
`limits` there are the normal limits; the percentile limits the summary
prints are `percentile_limits` (`empirical_limits` is the same pair under its
old name),
both `null` under 10 readings. Warnings, including those raised while
loading, are in `warnings`. `--csv` writes one row per reading:
`image, truth_index, predicted_index, class_id, cluster, size, truth,
predicted, error, mean`. Same inputs and seed, same bytes.

For paired values (`poseaudit paired`, `audit_paired`) the CSV's columns
are `subject, trial, frame, ref, pred, error, mean`, and `settings.input` is
`paired`. With subjects the JSON adds `by_subject` and `by_trial`, one entry
each of `subject, trial, n, bias, sd, mean_abs_error, rmse` (`trial` null in
`by_subject`); keypoint audits with `--cluster` fill `by_subject` per
cluster. `experimental` lists the keys whose method is still being validated
(the jitter reference): read them apart from the core figures.

With several measures the JSON is `{"poseaudit", "schema", "settings",
"measures": [...]}`, each entry of `measures` what a single measure's JSON
would be, and the CSV leads each row with a `measure` column. With one measure
the JSON is as above, unchanged.

## Comparing models

Comparing models, the JSON is `{"poseaudit", "schema", "settings", "models":
{name: [one entry per measure]}, "differences": [...], "warnings": [...]}`.
`settings` are those every model shares; each entry under `models` has its
own, with `model` (its name) and, from the command line, `pred` (its file).
The warnings are those about paired intervals (too few shared images, or
none, or too few shared readings read differently: `n_differing` counts
those) and the loaders' warnings.

`--csv` writes the differences, one row per measure and pair of models, as
`Comparison.table()` gives them: `measure, a, b, n_shared, n_a, n_b,
clusters, n_differing, big_error`, then `mean_abs_error_a`, `_b`, `_diff`
and its interval `mean_abs_error_diff_ci_low`, `_high`, then the same five
for `big_error_rate`. `--report` writes the comparison as Markdown, for a
paper or a blog: one table of every model's figures on every measure, one of
the differences, the warnings and the settings. In Python,
`Comparison.to_rows()` gives the first table as flat rows, led by `model`,
for a CSV of your own.

## Names

A figure is named for its reader: the summary and the report use words
("large errors"), the JSON, the CSV and Python use keys ("big_error_rate"),
after the flag `--big-error`. The same figure in each:

| summary line | JSON key | Python `AuditResult` | `to_rows()` column |
|---|---|---|---|
| read ... of ... labelled instances | `n`, `measurable` | `n`, `measurable` | `n` |
| not read, not counted | `not_read` | `not_read` | |
| bias, median | `bias`, `bias_ci`, `median_error` | the same | `bias`, `bias_ci_low`, `bias_ci_high` |
| limits (percentile) | `percentile_limits`, `empirical_lower_ci`, `empirical_upper_ci` | `percentile_limits` | `limits_low`, `limits_high` |
| normal (bias +/- 1.96 SD) | `limits`, `lower_limit_ci`, `upper_limit_ci` | `limits` | |
| limit CIs (`--full`): cluster bootstrap, then exact | `lower_limit_ci`, `upper_limit_ci`, `lower_limit_exact_ci`, `upper_limit_exact_ci` | the same | |
| by subject | `by_subject`, `by_trial` | the same | |
| \|error\| mean, median, 95th pct | `mean_abs_error`, `mean_abs_error_ci`, `median_abs_error`, `p95_abs_error` | the same | `mean_abs_error`, `mean_abs_error_ci_low`, `mean_abs_error_ci_high` |
| RMSE | `rmse`, `rmse_ci` | the same | `rmse` |
| `>= 15°` (in a table of measures that differ in it, `large if` and `large errors`) | `big_error_rate`, `big_error_rate_ci`; the threshold `big_error` | the same | `big_error`, `big_error_rate`, `big_error_rate_ci_low`, `big_error_rate_ci_high` |
| by size | `size_bands` | `size_bands` | |
| slope; Theil-Sen | `gain`, `gain_ci`; `robust_gain` | `slope` (also `gain`); `theil_sen` (also `robust_gain`) | `slope` |
| ICC(A,1) | `icc`, `icc_ci` | the same | `icc` |
| vs jitter (`--full`, experimental) | `jitter_gain`, `gain_gap`, `gain_gap_ci`, `jitter_p`; listed in `experimental` | the same | |
| BA slope (`--full`) | `ba_slope`, `ba_slope_ci` | the same | |
| Deming (with `--noise-ratio`) | `deming`, `deming_ci` | the same | |
| agreement: CCC, r (`--full`) | `ccc`, `ccc_ci`, `pearson` | the same | |

Comparing models, the summary's `mean |error| a - b` and `large-error rate
a - b` are `mean_abs_error_diff` and `big_error_rate_diff` in the JSON's
`differences`, in the CSV and on a `Difference`.

## Exit codes and gating

The command exits 0 on success (warnings included); 1 on bad input or
settings, a failed write, or nothing read (the JSON, CSV and report are
still written and say what was not read; no plot is drawn); and 2 when the
arguments cannot be parsed, or when `poseaudit` is run with none, which
prints the help. There is no pass or fail
threshold built in; gate on the JSON. An interval that cannot be computed is
`null` (every interval is, with a single cluster), and in jq `null <= 3` is
true, so make a gate fail on `null`: for example
`jq -e '(.mean_abs_error_ci[1] // 1e9) <= 3 and .measurable > 0 and .n / .measurable >= 0.9'`, or
without jq, in Python:

```python
import json

r = json.load(open("figures.json", encoding="utf-8"))  # written by --json
high = r["mean_abs_error_ci"][1]  # None when the interval is null
ok = high is not None and high <= 3
ok = ok and r["measurable"] > 0 and r["n"] / r["measurable"] >= 0.9
if not ok:
    raise SystemExit("poseaudit gate failed")
```

## What the outputs reveal

The report and the JSON record `--gt`, `--pred` and `--images` as typed, so
an absolute path shows your user name; pass relative paths before sharing
them. The CSV, the JSON and the report's largest-error table name images by
file name. The PNG holds only the plot and Matplotlib's version.

The same inputs and `seed` give the same CSV and the same figures on any
Python version. A different NumPy can change the last digit or two of a
float in the JSON (seen: the Bland-Altman slope, around 1e-17), never a
reported figure.
