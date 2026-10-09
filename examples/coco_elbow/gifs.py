"""The README's GIFs, from the committed demo data.

python gifs.py ../../docs

demo.gif types the README's demo command and prints, line by line, what
poseaudit prints for it on an 88-column terminal. score-filter.gif is the score
filter comparison in docs/python.md: the mean |error| over every detection,
over those scored 0.7 or more, and the difference on the people both read.
Every figure is computed here from the demo data. Needs Matplotlib and
Pillow: pip install "poseaudit[plot,images]".
"""

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

import poseaudit as pa  # noqa: E402
from poseaudit import plot  # noqa: E402
from poseaudit.cli import main as cli  # noqa: E402

HERE = Path(__file__).parent
COMMAND = [
    "poseaudit audit --format coco --gt gt_200.json --pred pred_yolo11n.json",
    "--angle 5,7,9 --big-error 15 --size-bands 30,60",
]
COLUMNS = 88
WIDTH = 800  # pixels: the GIFs are shown at their own size on GitHub

# a dark terminal
BACKGROUND = (13, 17, 23)
BAR = (30, 35, 43)
TITLE = (139, 148, 158)
PROMPT = (63, 185, 80)
TYPED = (240, 246, 252)
TEXT = (201, 209, 217)
LIGHTS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]


def main(out: str) -> None:
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    terminal(printed(COLUMNS), folder / "demo.gif")
    skipped(folder / "score-filter.gif")
    for name in ("demo.gif", "score-filter.gif"):
        ms = 0
        with Image.open(folder / name) as gif:
            for k in range(gif.n_frames):
                gif.seek(k)
                ms += gif.info["duration"]
        size = (folder / name).stat().st_size
        print(f"{name}: {size / 1024:.0f} KB, {ms / 1000:.1f} s")


class _Terminal(io.StringIO):
    """A UTF-8 terminal: poseaudit prints the degree sign and fits its lines
    to the width in COLUMNS, as on a real one."""

    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


def printed(columns: int) -> list[str]:
    """What `poseaudit audit` prints for the README's command, run here as
    the README runs it."""
    screen = _Terminal()
    before, where = os.environ.get("COLUMNS"), os.getcwd()
    os.environ["COLUMNS"] = str(columns)
    os.chdir(HERE)
    try:
        with redirect_stdout(screen):
            cli(" ".join(COMMAND).split()[1:])
    finally:
        os.chdir(where)
        if before is None:
            del os.environ["COLUMNS"]
        else:
            os.environ["COLUMNS"] = before
    return screen.getvalue().rstrip("\n").split("\n")


def _font(size: float) -> ImageFont.FreeTypeFont:
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    mpl = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"
    for path in (fonts / "CascadiaMono.ttf", fonts / "consola.ttf", mpl):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("no monospace font")


def terminal(output: list[str], path: Path) -> None:
    pad, bar = 16, 30
    command = [("$ ", COMMAND[0] + " \\"), ("", "    " + COMMAND[1])]
    rows = [*command, *(("", line) for line in output)]
    size = 20.0
    font = _font(size)
    while font.getlength("M" * (COLUMNS - 1)) > WIDTH - 2 * pad:
        size -= 0.25
        font = _font(size)
    step = round(size * 1.45)
    cell = font.getlength("M")
    height = bar + pad + (len(rows) + 1) * step + pad
    title = _font(13)

    def frame(lines: list[tuple[str, str]], cursor: bool) -> Image.Image:
        im = Image.new("RGB", (WIDTH, height), BACKGROUND)
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, WIDTH, bar), fill=BAR)
        for k, color in enumerate(LIGHTS):
            x = 18 + 20 * k
            draw.ellipse((x - 6, bar / 2 - 6, x + 6, bar / 2 + 6), fill=color)
        label = "examples/coco_elbow"
        draw.text(
            ((WIDTH - title.getlength(label)) / 2, bar / 2),
            label, font=title, fill=TITLE, anchor="lm",
        )  # fmt: skip
        for k, (prompt, text) in enumerate(lines):
            x, y = pad, bar + pad + k * step
            draw.text((x, y), prompt, font=font, fill=PROMPT)
            x += font.getlength(prompt)
            typed = k < len(command)  # the command is brighter than its output
            draw.text((x, y), text, font=font, fill=TYPED if typed else TEXT)
        if cursor:
            prompt, text = lines[-1]
            x = pad + font.getlength(prompt + text)
            y = bar + pad + (len(lines) - 1) * step
            draw.rectangle((x, y + 1, x + cell - 1, y + step - 4), fill=TEXT)
        return im

    blank = [("$ ", "")]
    frames = [(frame(blank, True), 500), (frame(blank, False), 300)]
    typed = list(blank)
    for k, (prompt, text) in enumerate(command):
        typed = typed[:k] + [(prompt, "")]
        for end in range(0, len(text), 4):
            typed[k] = (prompt, text[:end])
            frames.append((frame(typed, True), 30))
        typed[k] = (prompt, text)
        frames.append((frame(typed, True), 150))
    frames.append((frame(typed, False), 400))  # Enter: the audit runs
    for k in range(len(output)):
        frames.append((frame(rows[: len(command) + k + 1], False), 110))
    done = [*rows, ("$ ", "")]
    for _ in range(3):
        frames += [(frame(done, True), 550), (frame(done, False), 550)]
    _save(frames, path, [BACKGROUND, BAR, TITLE, PROMPT, TYPED, TEXT, *LIGHTS])


def _save(frames: list[tuple[Image.Image, int]], path: Path, exact: list) -> None:
    """One palette for every frame, so nothing flickers: the colours in
    `exact`, then those of the last two frames (which hold every colour)."""
    last, before = frames[-1][0], frames[-2][0]
    both = Image.new("RGB", (last.width, last.height * 2))
    both.paste(last, (0, 0))
    both.paste(before, (0, last.height))
    n = 128 - len(exact)
    found = both.quantize(colors=n, method=Image.Quantize.MEDIANCUT).getpalette()
    rgb = np.concatenate([np.array(exact), np.array(found[: 3 * n]).reshape(-1, 3)])
    images = []
    for im, _ in frames:  # each colour to its nearest in the palette, exactly
        pixels = np.asarray(im).reshape(-1, 3).astype(int)
        colors, where = np.unique(pixels, axis=0, return_inverse=True)
        far = ((colors[:, None, :] - rgb[None, :, :]) ** 2).sum(axis=2)
        index = far.argmin(axis=1)[where.ravel()].astype(np.uint8)
        image = Image.frombytes("P", im.size, index.tobytes())
        image.putpalette(rgb.flatten().tolist())
        images.append(image)
    images[0].save(
        path, save_all=True, append_images=images[1:], loop=0,
        duration=[ms for _, ms in frames], optimize=True,
    )  # fmt: skip


def _signed(x: float) -> str:
    return f"{round(x, 2) + 0.0:+.2f}"  # + 0.0: never "-0.00"


def _rgb(color: str) -> tuple[int, ...]:
    return tuple(round(255 * v) for v in matplotlib.colors.to_rgb(color))


def skipped(path: Path) -> None:
    """The comparison in docs/python.md, drawn person by person."""
    os.chdir(HERE)
    truth = pa.load_coco("gt_200.json")
    cut = 0.7
    models = {
        f"score>={s}": pa.load_coco_results(
            "pred_yolo11n.json", "gt_200.json", min_confidence=0.5, min_score=s
        )
        for s in (0.0, cut)
    }
    c = pa.compare(models, truth, measures=[pa.angle(5, 7, 9)], big_error=15)
    (d,) = c.differences
    every, scored = (c.results[name][0] for name in models)
    kept = {(r.image, r.truth_index) for r in scored.readings}
    error = np.array([abs(r.error) for r in every.readings])
    gone = np.array([(r.image, r.truth_index) not in kept for r in every.readings])
    # the same height for a person in both rows, so the shared readings line up
    height = np.random.default_rng(0).uniform(-0.28, 0.28, len(error))
    lo, hi = d.mean_abs_error_diff_ci
    stages = [
        (
            "Mean |error| of the elbow angle, every detection: "
            f"{every.mean_abs_error:.2f}°",
            "yolo11n-pose on the 200 demo images; one dot per person read",
        ),
        (
            f"Only detections scored {cut} or more: {scored.mean_abs_error:.2f}°",
            f"{every.mean_abs_error - scored.mean_abs_error:.2f}° lower. "
            "Did the model get better?",
        ),
        (
            f"No. The filter dropped {gone.sum()} people the model read worse",
            f"their mean |error| is {error[gone].mean():.2f}°",
        ),
        (
            f"On the {d.n_shared} people both read: {d.mean_abs_error_a:.2f}° "
            f"against {d.mean_abs_error_b:.2f}°",
            f"a difference of {_signed(d.mean_abs_error_diff)}° "
            f"[{_signed(lo)} to {_signed(hi)}]: the model did not get better",
        ),
    ]
    frames = []
    for stage, (headline, line) in enumerate(stages):
        fig = Figure(figsize=(WIDTH / 100, 4.2), dpi=100)
        with matplotlib.rc_context({**plot.STYLE, "font.size": 12}):
            fig.text(0.03, 0.93, headline, fontsize=14, fontweight="bold")
            fig.text(0.03, 0.855, line, fontsize=12, color=plot.MUTED)
            ax = fig.add_axes((0.25, 0.17, 0.72, 0.6))
            dot = {"s": 16, "color": plot.POINTS, "alpha": 0.55, "linewidths": 0}
            ax.scatter(error[~gone], 1 + height[~gone], **dot)
            if stage < 2:
                ax.scatter(error[gone], 1 + height[gone], **dot)
            else:
                ax.scatter(
                    error[gone], 1 + height[gone], s=26, facecolors="none",
                    edgecolors=plot.FIT, linewidths=1.3,
                    alpha=1 if stage == 2 else 0.3,
                    label=f"dropped by the filter ({gone.sum()})",
                )  # fmt: skip
                ax.legend(
                    loc="upper right", bbox_to_anchor=(1, 1.13), handletextpad=0.2
                )
            if stage >= 1:
                ax.scatter(error[~gone], height[~gone], **dot)
            means = [(1, every.mean_abs_error if stage < 3 else d.mean_abs_error_a)]
            if stage >= 1:
                means.append((0, scored.mean_abs_error))
            for row, value in means:
                ax.plot([value] * 2, [row - 0.36, row + 0.36], color=plot.INK, lw=2.5)
                ax.text(
                    value + 1.5, row + 0.3, f"mean {value:.2f}°", fontsize=12,
                    fontweight="bold", va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                )  # fmt: skip
            ax.set_ylim(-0.5, 1.5)
            shown = f"{d.n_shared} of {every.n}" if stage == 3 else every.n
            ax.set_yticks(
                [1, 0], [f"every detection\n{shown} people",
                         f"scored {cut} or more\n{scored.n} people"],
            )  # fmt: skip
            if stage < 1:
                ax.get_yticklabels()[1].set_alpha(0)
            right = 15 * np.ceil(error.max() / 15)
            ax.set_xlim(0, right)
            ax.set_xticks(np.arange(0, right + 1, 15))
            ax.set_xlabel("|error| of the elbow angle (°)")
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", color=plot.GRID, lw=0.8)
            ax.set_axisbelow(True)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        frames.append(Image.open(buffer).convert("RGB"))
    durations = [1500, 1800, 1800, 2600]
    colors = ["#ffffff", plot.INK, plot.MUTED, plot.POINTS, plot.FIT, plot.GRID]
    _save(list(zip(frames, durations, strict=True)), path, [_rgb(x) for x in colors])


if __name__ == "__main__":
    main(*sys.argv[1:])
