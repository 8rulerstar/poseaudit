# Contributing

Thank you for helping. poseaudit is small, so a short issue or pull request
is welcome.

## Ask a question or show what you measured

Use [Discussions](https://github.com/8rulerstar/poseaudit/discussions). A line
saying what you measured (a joint, a tilt, a length; which model) helps decide
what to build next.

## Report a bug

Open an [issue](https://github.com/8rulerstar/poseaudit/issues/new/choose)
with:

- `poseaudit --version`, your Python version and operating system;
- the command or Python lines you ran, and everything it printed;
- what you expected instead;
- if you can, a few lines of the input files that show the problem. Leave
  out anything private: the report and the JSON name image files and record
  paths as typed.

## Change the code

```bash
git clone https://github.com/8rulerstar/poseaudit
cd poseaudit
uv sync                      # Python 3.10 or later, with the dev tools
uv run pytest -q             # the tests
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy src              # types
```

Without uv, `pip install -e ".[plot,images,supervision]" pytest hypothesis
ruff mypy` in a virtual environment does the same.

- A change in behaviour comes with a test that fails without it.
- The README and the files in `docs/` quote the demo's real output, and
  `tests/test_readme.py` checks those blocks. If your change alters what the
  demo prints, run the command from `examples/coco_elbow` and paste the new
  output.
- Add a line to `CHANGELOG.md` under Unreleased, in Added, Changed or Fixed,
  saying what a user will notice.
- Messages and docs use plain words: say what happened and what to do, with
  the figure that shows it.

By contributing you agree that your work is released under the
[MIT License](LICENSE).
