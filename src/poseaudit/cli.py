import argparse
import json
import os
import re
import shutil
import sys
import textwrap
import unicodedata
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import numpy as np

from poseaudit._files import write_text
from poseaudit._version import __version__
from poseaudit.audit import audit
from poseaudit.io import load_coco, load_coco_results, load_yolo
from poseaudit.measures import Measure, angle, length, ratio, tilt
from poseaudit.pairing import pair
from poseaudit.report import csv_rows_many, table
from poseaudit.types import Dataset

MEASURES: dict[str, tuple[Callable[..., Measure], int]] = {
    "tilt": (tilt, 2),
    "angle": (angle, 3),
    "length": (length, 2),
    "ratio": (ratio, 4),
}


# what each measure reads, for --help: the points it takes and where they go
MEANINGS = {
    "angle": ("A,B,C", "angle at B between B-A and B-C, 0 to 180 degrees"),
    "tilt": ("A,B", "tilt of the axis A-B from vertical, -90 to 90 degrees"),
    "length": ("A,B", "distance from A to B, in pixels"),
    "ratio": ("A,B,C,D", "length A-B over length C-D"),
}


class UsageError(Exception):
    pass


class _AddMeasure(argparse.Action):
    """--angle, --tilt, --length and --ratio may each be repeated; the
    measures keep the order they were given in."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        chosen = list(getattr(namespace, "measures", None) or [])
        chosen.append((self.dest, values))
        namespace.measures = chosen


def _measures(args: argparse.Namespace) -> list[Measure]:
    chosen = getattr(args, "measures", None) or []
    if not chosen:
        raise UsageError("give at least one of --tilt, --angle, --length, --ratio")
    made = [_measure(kind, spec) for kind, spec in chosen]
    keys = [m.key() for m in made]
    if len(set(keys)) < len(keys):
        raise UsageError(
            "a measure is given twice (listing its keypoints the other way "
            "round reads the same)"
        )
    return made


def _measure(kind: str, spec: str) -> Measure:
    make, count = MEASURES[kind]
    try:
        points = [int(p) for p in spec.split(",")]
    except ValueError:
        raise UsageError(
            f"--{kind} takes comma-separated keypoint indices counted from 0, "
            f"not names, got {spec!r}"
        ) from None
    if len(points) != count:
        raise UsageError(f"--{kind} takes {count} keypoint indices, got {len(points)}")
    return make(*points)


def _bands(spec: str, flag: str = "--bands", edges_only_positive: bool = False):
    try:
        if "," in spec:
            edges = [float(e) for e in spec.split(",")]
            if len(edges) < (1 if edges_only_positive else 2):
                raise ValueError
            if not all(np.isfinite(edges)) or edges != sorted(set(edges)):
                raise ValueError
            if edges_only_positive and edges[0] <= 0:
                raise ValueError
            return edges
        count = int(spec)
        if count < 1:
            raise ValueError
        return count
    except ValueError:
        raise UsageError(
            f"{flag} takes a count of 1 or more or increasing comma-separated edges, "
            f"not {spec!r}"
        ) from None


_IMAGES = {
    ".bmp", ".dng", ".gif", ".heic", ".jpeg", ".jpg", ".jp2", ".mpo", ".pfm",
    ".png", ".tif", ".tiff", ".webp",
}  # fmt: skip


_ORIENTATION = 0x0112


def _image_sizes(args):
    if args.images and args.image_size:
        raise UsageError("give --images or --image-size, not both")
    if args.images:
        folder = Path(args.images)
        if not folder.is_dir():
            raise UsageError(f"--images: {folder} is not a folder")
        try:
            from PIL import Image
        except ImportError:
            raise UsageError("--images needs Pillow: pip install pillow") from None
        files: dict[str, list[Path]] = {}
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGES:
                files.setdefault(p.stem, []).append(p)

        def size_of(name: str) -> tuple[int, int]:
            found = files.get(name)
            if found is None and Path(name).suffix.lower() in _IMAGES:
                found = files.get(Path(name).stem)  # a label named "x.jpg"
            if found is None:
                raise ValueError(f"no image for {name!r} in {folder}")
            if len(found) > 1:
                raise ValueError(
                    f"--images: {', '.join(p.name for p in found)} share a name; "
                    "the image size would be a guess"
                )
            path = found[0]
            with Image.open(path) as image:
                width, height = image.size
                # a phone photo stored on its side: YOLO labels are fractions of
                # the size as shown, after the EXIF rotation
                if image.getexif().get(_ORIENTATION) in (5, 6, 7, 8):
                    width, height = height, width
                return width, height

        return size_of
    if args.image_size:
        return tuple(args.image_size)
    raise UsageError("YOLO labels need --image-size W H or --images DIR")


def _load(args, kind: str, path: str, is_prediction: bool) -> Dataset:
    threshold = args.min_conf if is_prediction else 0.0
    if kind == "yolo":
        if not args.keypoints:
            raise UsageError("YOLO labels need --keypoints K")
        return load_yolo(
            path,
            _image_sizes(args),
            args.keypoints,
            args.classes,
            threshold,
            min_score=args.min_score if is_prediction else 0.0,
        )
    if not is_prediction:
        return load_coco(path, args.classes)
    if args.gt_format != "coco":
        raise UsageError(
            "COCO results name images by id: --gt must be COCO annotations"
        )
    return load_coco_results(path, args.gt, threshold, args.classes, args.min_score)


def _cluster(pattern: str | None):
    if pattern is None:
        return None
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        raise UsageError(f"--cluster: {error}") from None

    def cluster_of(image: str) -> str:
        found = compiled.search(image)
        if found is None:
            raise ValueError(f"--cluster {pattern!r} does not match image {image!r}")
        return found.group(1) if found.groups() else found.group(0)

    return cluster_of


def _finite(value):
    """NaN and infinity become null: strict JSON parsers refuse them."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _writable(path: str | None, flag: str, inputs=()) -> None:
    if not path:
        return
    target = Path(path).resolve()
    if not target.parent.is_dir():
        raise UsageError(f"{flag}: folder {Path(path).parent} does not exist")
    if target.is_dir():
        raise UsageError(f"{flag}: {path} is a folder, not a file")
    if any(_same(target, p) or _within(target, p) for p in inputs) or (
        target.is_file() and _inode(target) in _inodes(inputs)
    ):
        raise UsageError(
            f"{flag}: {path} is an input or inside an input folder; it would "
            "overwrite your data"
        )


def _key(path: Path) -> str:
    """A spelling of a path that ignores case and Unicode normalisation, as
    macOS and Windows file systems do."""
    return unicodedata.normalize("NFC", str(path)).casefold()


def _same(a: Path, b: Path) -> bool:
    """Same file: by inode when both exist (hard links, case, NFD), else by a
    spelling that ignores case and normalisation (erring toward refusing)."""
    if a.exists() and b.exists():
        return os.path.samefile(a, b)
    return _key(a) == _key(b)


def _inode(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def _inodes(inputs) -> set[tuple[int, int]]:
    """Every input file, and every file directly in an input folder: an output
    that is a hard link to one of them would overwrite it."""
    found = set()
    for p in inputs:
        for f in [p] if p.is_file() else (p.iterdir() if p.is_dir() else []):
            if f.is_file():
                found.add(_inode(f))
    return found


def _within(target: Path, folder: Path) -> bool:
    return folder.is_dir() and any(_same(parent, folder) for parent in target.parents)


def _check_values(args) -> None:
    """Values argparse lets through but the audit cannot use."""
    finite = all(np.isfinite(x) for x in (*args.threshold, *args.pred_threshold))
    if not finite:
        raise UsageError("--threshold and --pred-threshold must be finite numbers")
    if not np.isfinite(args.big_error):
        raise UsageError("--big-error must be a finite number")
    if args.noise_ratio is not None and not args.noise_ratio > 0:
        raise UsageError("--noise-ratio must be above 0")
    if args.image_size and min(args.image_size) <= 0:
        raise UsageError("--image-size takes a width and height above 0")
    if args.min_in_frame < 2:
        raise UsageError(
            "--min-in-frame must be 2 or more: each reading is read against the "
            "others in its image"
        )
    if args.resamples < 50 or args.jitter_repeats < 0:
        raise UsageError(
            "--resamples must be 50 or more (an interval from fewer is a guess), "
            "--jitter-repeats 0 or more"
        )
    for flag in ("min_iou", "min_similarity"):
        if not 0 < getattr(args, flag) <= 1:
            raise UsageError(
                f"--{flag.replace('_', '-')} must be above 0 and at most 1: at 0 "
                "objects that do not overlap at all would pair"
            )
    for flag in ("min_conf", "min_score"):
        if not np.isfinite(getattr(args, flag)):
            raise UsageError(f"--{flag.replace('_', '-')} must be a finite number")
    if args.keypoints is not None and args.keypoints < 1:
        raise UsageError("--keypoints must be 1 or more")


EXAMPLES = """\
examples:
  left elbow angle (COCO keypoints 5, 7, 9; indices count from 0), COCO
  annotations against COCO results, with a report:
    poseaudit audit --format coco --gt gt.json --pred res.json --angle 5,7,9 --big-error 15 --report report.md

  trunk tilt (shoulder 5 to hip 11) from YOLO label folders, each image's size
  read from the images:
    poseaudit audit --format yolo --gt labels --pred preds --keypoints 17 --images imgs --tilt 5,11 --big-error 5

  frames of one video resampled together (image names such as clip3_f0041):
    poseaudit audit ... --cluster "^(clip\\d+)_"

  two models compared on the readings both made, named for the output:
    poseaudit audit ... --pred small=res_s.json --pred large=res_l.json

more: https://github.com/8rulerstar/poseaudit (README and docs/cli.md)
"""  # noqa: E501


AUDIT = """\
Read measures off labelled and predicted keypoints, pair the people image by
image, and report how far apart the readings are. Required: --gt, --pred, a
format, at least one measure, and --big-error.
"""

SIZE_FORMS = "--image-size takes W H or WxH"


class _ImageSize(argparse.Action):
    """W H as two words, or WxH as one. The audit parser has no positional
    arguments, so the open count swallows nothing a correct line needs; a third
    word is named in the error."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        words = list(values) if len(values) > 1 else re.split(r"[xX]", values[0])
        if len(words) != 2 or not all(re.fullmatch(r"\d+", w) for w in words):
            parser.error(f"{SIZE_FORMS}, got {' '.join(values)!r}")
        setattr(namespace, self.dest, [int(w) for w in words])


class _Help(argparse.RawDescriptionHelpFormatter):
    """--image-size reads W H rather than argparse's W [H ...]."""

    def _format_args(self, action, default_metavar) -> str:
        if action.dest == "image_size":
            return "W H"
        return super()._format_args(action, default_metavar)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        # before Python 3.13 argparse reads a word such as -1x5 as an option, so
        # it never reaches _ImageSize and the size is reported as missing
        if message.startswith("argument --image-size") and "expected" in message:
            message += f"; {SIZE_FORMS}"
        super().error(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="poseaudit",
        description="How far off are the joint angles, tilts and lengths read "
        "from a pose model? Judged against labels, the way a measuring "
        "instrument is judged.",
        epilog="poseaudit audit --help lists every option, with examples.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(
        dest="command", required=True, title="commands", metavar="COMMAND"
    )
    run = sub.add_parser(
        "audit",
        help="how far measures read off predicted keypoints are from the truth",
        description=AUDIT,
        epilog=EXAMPLES,
        formatter_class=_Help,
    )
    data = run.add_argument_group("data")
    data.add_argument(
        "--gt", required=True, metavar="PATH", help="YOLO label folder or COCO json"
    )
    data.add_argument(
        "--pred",
        required=True,
        action="append",
        metavar="PATH",
        help="YOLO folder or COCO results json; to compare models, repeat it as "
        "--pred NAME=PATH",
    )
    data.add_argument(
        "--format", choices=["yolo", "coco"], help="format of both --gt and --pred"
    )
    data.add_argument(
        "--gt-format", choices=["yolo", "coco"], help="override --format for --gt"
    )
    data.add_argument(
        "--pred-format", choices=["yolo", "coco"], help="override --format for --pred"
    )
    data.add_argument(
        "--image-size",
        nargs="+",
        action=_ImageSize,
        metavar=("W", "H"),
        help="image width then height in pixels, as W H or WxH (YOLO)",
    )
    data.add_argument(
        "--images", metavar="DIR", help="image folder, to read each YOLO image's size"
    )
    data.add_argument(
        "--keypoints", type=int, metavar="K", help="keypoints per instance (YOLO)"
    )
    data.add_argument(
        "--classes",
        type=int,
        nargs="+",
        metavar="ID",
        help="class ids to keep, on both sides",
    )
    data.add_argument(
        "--min-conf",
        type=float,
        metavar="C",
        default=0.0,
        help="a predicted point counts as seen above this confidence (default 0)",
    )
    data.add_argument(
        "--min-score",
        type=float,
        metavar="S",
        default=0.0,
        help="drop predicted detections scored below this (default 0): matching "
        "ignores scores",
    )
    data.add_argument(
        "--min-iou",
        type=float,
        metavar="IOU",
        default=0.3,
        help="smallest box IoU that pairs a prediction with a truth (default 0.3)",
    )
    data.add_argument(
        "--min-similarity",
        type=float,
        metavar="OKS",
        default=0.5,
        help="smallest keypoint similarity that pairs them when a side has no box "
        "(default 0.5)",
    )
    what = run.add_argument_group(
        "measure",
        # the formatter keeps descriptions as written: lines under 80 columns
        "keypoints by index, counted from 0 (COCO: 5 left shoulder, 7 left\n"
        "elbow, 9 left wrist, 11 left hip); repeat these for several measures",
    )
    for kind, (metavar, meaning) in MEANINGS.items():
        what.add_argument(
            f"--{kind}", metavar=metavar, action=_AddMeasure, help=meaning
        )
    what.add_argument(
        "--relative-to",
        metavar="median|pNN",
        help="read each value against the rest of its image (median, or a "
        "percentile such as p15), e.g. to remove a camera roll from tilts",
    )
    what.add_argument(
        "--relative-abs",
        action="store_true",
        help="with --relative-to: |value| minus the baseline of the others' |values|",
    )
    what.add_argument(
        "--min-in-frame",
        type=int,
        metavar="N",
        default=3,
        help="with --relative-to: fewest readings an image needs (default 3)",
    )
    how = run.add_argument_group("analysis")
    how.add_argument(
        "--big-error",
        type=float,
        required=True,
        metavar="E",
        help="required: an error at least this large counts as large, in the "
        "measure's unit",
    )
    how.add_argument(
        "--bands",
        default="4",
        metavar="N|EDGES",
        help="a count, or comma-separated edges such as -90,-7,0,7,90",
    )
    how.add_argument(
        "--band-by",
        choices=["truth", "mean", "predicted"],
        default="truth",
        help="what to sort readings by for the bias bands: the truth (default) "
        "when the labels are much less noisy than the model, the mean of both when "
        "they are about equally noisy",
    )
    how.add_argument(
        "--cluster",
        metavar="REGEX",
        help="group readings by the first capture group of the image name",
    )
    how.add_argument(
        "--size-bands",
        default="3",
        metavar="N|EDGES",
        help="a count, or pixel edges such as 30,60",
    )
    how.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        metavar="T",
        default=[],
        help="decision thresholds applied to the truth",
    )
    how.add_argument(
        "--pred-threshold",
        type=float,
        nargs="+",
        metavar="T",
        default=[],
        help="also hold the prediction to these thresholds",
    )
    how.add_argument(
        "--side",
        choices=["above", "below", "outside"],
        help="which side of a threshold is flagged (default: outside for a tilt, "
        "else above)",
    )
    how.add_argument(
        "--noise-ratio",
        type=float,
        metavar="R",
        help="variance of prediction noise over variance of label noise; adds a "
        "Deming slope",
    )
    how.add_argument(
        "--mixed-classes",
        action="store_true",
        help="allow readings of several classes in one audit",
    )
    how.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="seed for resampling and rebuilds (default 0)",
    )
    how.add_argument(
        "--resamples",
        type=int,
        metavar="N",
        default=2000,
        help="bootstrap resamples for the intervals (default 2000; fewer is faster)",
    )
    how.add_argument(
        "--jitter-repeats",
        type=int,
        metavar="N",
        default=500,
        help="rebuilds for the jitter reference (default 500; 0 turns it off)",
    )
    out = run.add_argument_group("output")
    out.add_argument(
        "--report", metavar="FILE", help="markdown report that explains every line"
    )
    out.add_argument(
        "--csv",
        metavar="FILE",
        help="one row per reading; comparing models, one per difference",
    )
    out.add_argument("--json", metavar="FILE", help="every figure")
    out.add_argument(
        "--plot",
        metavar="FILE",
        help="the two panels, as PNG, SVG or PDF by the file's extension "
        "(needs matplotlib)",
    )
    out.add_argument(
        "--full", action="store_true", help="every statistic in the summary"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    if not (sys.argv[1:] if argv is None else argv):
        # a first try with no arguments: what the tool does and its command,
        # rather than an error about a missing COMMAND
        parser.print_help(sys.stderr)
        sys.exit(2)
    args = parser.parse_args(argv)
    loader_notes: list[str] = []

    def note(message, category, filename, lineno, file=None, line=None) -> None:
        """Each loader warning on stderr once, as it is raised: before a run
        that takes minutes, and even when the run then fails."""
        text = str(message).replace("min_confidence", "--min-conf")
        if text not in loader_notes:
            loader_notes.append(text)
            print(
                _console(f"poseaudit: warning: {text}", sys.stderr),
                file=sys.stderr,
                flush=True,
            )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = note
            result = _run(args)
    except (UsageError, ValueError, OSError, ImportError) as error:
        sys.exit(f"poseaudit: {error}")
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        sys.exit(f"poseaudit: a file is not in the format given: {error!r}")
    except RecursionError:
        sys.exit("poseaudit: a file is nested too deeply to read")
    except Exception as error:  # Pillow's DecompressionBombError and kin
        if type(error).__module__.startswith("PIL"):
            sys.exit(f"poseaudit: an image could not be read: {error}")
        raise
    from poseaudit.compare import Comparison

    if isinstance(result, Comparison):
        _finish_compare(result, args, loader_notes)
        return
    results = result
    if len(results) == 1:
        text = results[0].summary(full=args.full)
    elif args.full:
        text = "\n\n".join(r.summary(full=True) for r in results)
    else:
        text = table(results, _degree(sys.stdout), _columns(sys.stdout))
    print(_console(text, sys.stdout))
    # in the JSON too, but printed once: stderr above, not again in the summary
    for r in results:
        r.warnings = loader_notes + [w for w in r.warnings if w not in loader_notes]
    try:
        if len(results) == 1:
            _write(results[0], args)
        else:
            _write_many(results, args)
    except OSError as error:
        sys.exit(f"poseaudit: could not write: {error}")
    if all(r.n == 0 for r in results):
        # the JSON, CSV and report are written all the same (they say why, and
        # replace any from an earlier run); a plot has nothing to draw
        stale = args.plot and Path(args.plot).exists()
        sys.exit(
            "poseaudit: nothing was read"
            + ("; no plot was drawn" if args.plot else "")
            + (f", and {args.plot} is from an earlier run" if stale else "")
        )


def _finish_compare(comparison, args, loader_notes) -> None:
    """The loader's warnings are already on stderr; they go in the JSON too."""
    if args.full:
        blocks = [
            f"{name}:\n" + "\n\n".join(r.summary(full=True) for r in results)
            for name, results in comparison.results.items()
        ]
        from poseaudit.compare import differences_table

        text = (
            "\n\n".join(blocks)
            + "\n\n"
            + differences_table(
                comparison.differences,
                comparison.results,
                _degree(sys.stdout),
                _columns(sys.stdout),
            )
        )
    else:
        text = comparison.summary(_degree(sys.stdout), _columns(sys.stdout))
    print(_console(text, sys.stdout))
    try:
        if args.json:
            data = comparison.to_dict()
            data["warnings"] = loader_notes + data["warnings"]
            write_text(
                args.json,
                json.dumps(_finite(data), indent=2, default=str),
            )
        if args.csv:
            comparison.to_csv(args.csv)
    except OSError as error:
        sys.exit(f"poseaudit: could not write: {error}")
    if all(r.n == 0 for rs in comparison.results.values() for r in rs):
        sys.exit("poseaudit: nothing was read")


def _utf8(stream) -> bool:
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    return encoding in ("utf8", "utf8sig")


def _degree(stream) -> str:
    return "°" if _utf8(stream) else " deg"


def _columns(stream) -> int | None:
    """The terminal's width when `stream` is one (COLUMNS overrides it, as
    for --help); None for a pipe or a file, whose lines are left whole for
    whatever reads them."""
    try:
        if not stream.isatty():
            return None
    except (AttributeError, ValueError, OSError):
        return None
    return max(40, shutil.get_terminal_size().columns)


# a "  ! " warning, or a summary line's label column: where a wrapped line's
# continuation starts (two spaces deeper than the line for anything else)
_HANGING = re.compile(r" {2}! | {2}\S.*? {2,}(?=\S)")


def _fit(text: str, columns: int | None) -> str:
    """Lines wider than the terminal broken between words, each continuation
    indented under the start of the value it continues, rather than wrapped
    by the terminal mid-word at its first column. Table rows (no indent,
    cells two spaces apart) are left to `layout`, which cuts the table to
    fit."""
    if columns is None:
        return text
    width = columns - 1  # a line filling the last column wraps on Windows
    out = []
    for line in text.split("\n"):
        table_row = not line.startswith(" ") and "  " in line.strip()
        if len(line) <= width or table_row:
            out.append(line)
            continue
        found = _HANGING.match(line)
        if found and len(found.group(0)) <= 20:
            indent = len(found.group(0))
        else:
            indent = len(line) - len(line.lstrip(" ")) + 2
        out += textwrap.wrap(
            line,
            width,
            subsequent_indent=" " * indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
    return "\n".join(out)


def _console(text: str, stream) -> str:
    """ " deg" for the degree sign on a stream that is not UTF-8. On Korean
    Windows, Git Bash shows a pipe's bytes as UTF-8 while Python writes them in
    cp949, which can encode the sign, so it comes out garbled rather than
    failing. Reconfiguring the stream to UTF-8 would garble it instead for
    whatever reads the pipe as cp949; " deg" reads the same either way. Other
    text (image names) is left alone, and files are always written as UTF-8.
    On a terminal, long lines are then broken to its width (`_fit`)."""
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    if not _utf8(stream):
        # padding after a sign (the ">= 15°" label) gives up the three columns
        # " deg" adds, so the figures stay aligned
        text = re.sub(r"°( {4,})", lambda m: " deg" + m.group(1)[3:], text)
        text = text.replace("°", " deg")
        if encoding:  # never fail on a character the console cannot show
            text = text.encode(encoding, errors="replace").decode(encoding)
    return _fit(text, _columns(stream))


def _write(result, args) -> None:
    if args.json:
        write_text(
            args.json,
            json.dumps(_finite(result.to_dict()), indent=2, default=str),
        )
    if args.csv:
        result.to_csv(args.csv)
    if args.report:  # with nothing read too: it says what was not read and why
        result.to_markdown(args.report)
    if args.plot and result.n:
        try:
            result.plot(args.plot)
        except (ImportError, ValueError) as error:
            sys.exit(f"poseaudit: {error}")


def _run_compare(args, models):
    from poseaudit.compare import Comparison, _difference

    for flag in ("report", "plot"):
        if getattr(args, flag):
            raise UsageError(f"--{flag} takes one model; leave it out to compare")
    results = {}
    for name, path in models:
        args.pred = path
        results[name] = _run(args)
        for r in results[name]:
            r.settings["model"] = name
    names = list(results)
    differences = [
        _difference(results[a][k], results[b][k], a, b, args.resamples, args.seed)
        for k in range(len(results[names[0]]))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    return Comparison(results, differences)


def several(results) -> dict:
    """The JSON for several measures: the shared settings once, and each
    measure's figures as a single measure's JSON would give them."""
    from poseaudit._version import __version__

    return {
        "poseaudit": __version__,
        "schema": 1,
        "settings": results[0].settings,
        "measures": [r.to_dict() for r in results],
    }


def _write_many(results, args) -> None:
    if args.json:
        write_text(
            args.json,
            json.dumps(_finite(several(results)), indent=2, default=str),
        )
    if args.csv:
        write_text(args.csv, csv_rows_many(results))
    if args.report:  # every measure, those that read nothing included
        write_text(args.report, "\n".join(r.to_markdown() for r in results))


def _models(given: list[str]) -> list[tuple[str, str]]:
    """--pred PATH once, or --pred NAME=PATH for each model compared. A value
    that names an existing file or folder is a path even if it holds an =."""
    models = []
    for value in given:
        found = re.fullmatch(r"([^=/\\]+)=(.+)", value)
        if found and not Path(value).exists():
            models.append((found.group(1), found.group(2)))
        else:
            models.append(("", value))
    if len(models) > 1:
        if any(not name for name, _ in models):
            raise UsageError("to compare models give each as --pred NAME=PATH")
        names = [name for name, _ in models]
        if len(set(names)) < len(names):
            raise UsageError("two models given to --pred share a name")
    return models


def _run(args):
    if isinstance(args.pred, list):  # as parsed; a string once a model is chosen
        models = _models(args.pred)
        args.pred = models[0][1]
        if len(models) > 1:
            return _run_compare(args, models)
    args.gt_format = args.gt_format or args.format
    args.pred_format = args.pred_format or args.format
    if not (args.gt_format and args.pred_format):
        raise UsageError("give --format, or both --gt-format and --pred-format")
    inputs = {Path(p).resolve() for p in (args.gt, args.pred, args.images) if p}
    outputs = [getattr(args, f) for f in ("report", "csv", "json", "plot")]
    for flag in ("report", "csv", "json", "plot"):
        _writable(getattr(args, flag), f"--{flag}", inputs)
    given = [Path(o).resolve() for o in outputs if o]
    if any(_same(a, b) for i, a in enumerate(given) for b in given[i + 1 :]):
        raise UsageError(
            "two outputs name the same file; one would overwrite the other"
        )
    _check_values(args)
    if args.plot:  # before the audit, which can take minutes
        from poseaudit.plot import needs_matplotlib

        needs_matplotlib()
    if (args.side or args.pred_threshold) and not args.threshold:
        raise UsageError("--side and --pred-threshold need --threshold")
    measures = _measures(args)
    if args.plot and len(measures) > 1:
        raise UsageError("--plot draws one measure; give one, or leave out --plot")
    bands = _bands(args.bands)
    size_bands = _bands(args.size_bands, "--size-bands", edges_only_positive=True)
    truth = _load(args, args.gt_format, args.gt, is_prediction=False)
    predicted = _load(args, args.pred_format, args.pred, is_prediction=True)
    pairing = pair(
        truth,
        predicted,
        min_iou=args.min_iou,
        min_keypoint_similarity=args.min_similarity,
    )
    _say_if_slow(len(pairing.pairs), args.resamples, args.jitter_repeats, len(measures))
    return [_audit(args, pairing, m, bands, size_bands) for m in measures]


# A run's time grows with pairs x (resamples + _REBUILD x jitter rebuilds) for
# each measure: one rebuild of the jitter reference costs about as much as 20
# resamples (at 4,900 pairs, 3 seconds a measure without the rebuilds and 17
# with the default 500). Above _SLOW_WORK a run takes half a minute or more.
_REBUILD = 20
_SLOW_WORK = 100_000_000


def _say_if_slow(
    pairs: int, resamples: int, jitter_repeats: int, measures: int = 1
) -> None:
    """A note on stderr before a long run, which prints nothing until it is
    done, naming the flags that make it faster."""
    work = pairs * (resamples + _REBUILD * jitter_repeats) * measures
    if work < _SLOW_WORK:
        return
    how_long = "several minutes" if work >= 10 * _SLOW_WORK else "a minute or more"
    each = f"{measures} measures" if measures > 1 else "1 measure"
    if jitter_repeats:  # the larger cost, and its figures are under --full only
        faster = "--jitter-repeats 0 or a smaller --resamples is faster"
    else:
        faster = "A smaller --resamples is faster"
    print(
        f"poseaudit: {pairs:,} pairs, {each}, {resamples:,} resamples and "
        f"{jitter_repeats:,} jitter rebuilds; this can take {how_long}. {faster}.",
        file=sys.stderr,
        flush=True,
    )


def _audit(args, pairing, measure, bands, size_bands):
    return audit(
        pairing,
        measure,
        args.big_error,
        bands=bands,
        band_by=args.band_by,
        cluster=_cluster(args.cluster),
        size_bands=size_bands,
        thresholds=args.threshold,
        threshold_side=args.side,
        predicted_thresholds=args.pred_threshold,
        relative_to=args.relative_to,
        relative_abs=args.relative_abs,
        min_in_frame=args.min_in_frame,
        noise_ratio=args.noise_ratio,
        mixed_classes=args.mixed_classes,
        seed=args.seed,
        resamples=args.resamples,
        jitter_repeats=args.jitter_repeats,
        settings={
            "gt": args.gt,
            "pred": args.pred,
            "gt_format": args.gt_format,
            "pred_format": args.pred_format,
            "min_conf": args.min_conf,
            "min_score": args.min_score,
            "min_iou": args.min_iou,
            "min_similarity": args.min_similarity,
            "classes": args.classes,
            "image_size": args.image_size or args.images,
            "cluster_regex": args.cluster,
        },
    )


if __name__ == "__main__":
    main()
