# poseaudit docs

[README](../README.md) · **index** · [command line](cli.md) · [Python](python.md) · [statistics](statistics.md) · [JSON, CSV and exit codes](json.md)

Start with the [README](../README.md): what poseaudit is for, installing it,
and the demo. Then:

| page | read it for |
|---|---|
| [cli.md](cli.md) | running `poseaudit audit` on your files; [every option](cli.md#every-option); several joints; [comparing models](cli.md#compare-models); decisions at a threshold; [frames of one video](cli.md#several-readings-of-one-subject); the [measures](cli.md#measures); [input formats](cli.md#inputs); [how people are paired](cli.md#how-pairs-counts-and-intervals-are-made); [what an error message means](cli.md#when-it-stops) |
| [python.md](python.md) | the same from Python; [your own arrays](python.md#your-own-arrays); [comparing models in code](python.md#compare-models); [every public name](python.md#python-api) |
| [statistics.md](statistics.md) | [what the demo's figures say](statistics.md#reading-the-demo) and do not; label noise; [the sorting trap](statistics.md#which-way-you-sort-decides-the-story); [the slope and the jitter reference](statistics.md#does-the-model-squash-large-angles); [definitions](statistics.md#statistics-used); [references](statistics.md#references); [limitations](statistics.md#limitations) |
| [json.md](json.md) | the [JSON and CSV](json.md#json-csv-and-exit-codes); [which name a figure has where](json.md#names) (summary, JSON, Python, CSV); exit codes and gating CI on the JSON; [what the outputs reveal](json.md#what-the-outputs-reveal) |

## Installing the extras

`pip install poseaudit` needs NumPy alone and runs every audit. Three
features need one more package:

| for | install |
|---|---|
| `--plot` and `result.plot()` | `pip install "poseaudit[plot]"` (Matplotlib) |
| `--images DIR`, image sizes read from the images | `pip install "poseaudit[images]"` (Pillow) |
| `pa.from_supervision(...)` | `pip install "poseaudit[supervision]"` |

Without Matplotlib or Pillow, `--plot` and `--images` stop with the line to
run, before any slow work.

Elsewhere: [CHANGELOG.md](../CHANGELOG.md),
[CONTRIBUTING.md](../CONTRIBUTING.md), and the
[demo notebook](../examples/poseaudit_demo.ipynb).
