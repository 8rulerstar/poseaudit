import argparse
import json
import os
import re
import sys
import unicodedata
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import numpy as np

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
    names = [(m.name, m.points) for m in made]
    if len(set(names)) < len(names):
        raise UsageError("a measure is given twice")
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


def _image_sizes(args):
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
                return image.size

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
    if args.min_in_frame < 1:
        raise UsageError("--min-in-frame must be 1 or more")
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
"""  # noqa: E501


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


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        # before Python 3.13 argparse reads a word such as -1x5 as an option, so
        # it never reaches _ImageSize and the size is reported as missing
        if message.startswith("argument --image-size") and "expected" in message:
            message += f"; {SIZE_FORMS}"
        super().error(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="poseaudit")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "audit",
        help="how far measures read off predicted keypoints are from the truth",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = run.add_argument_group("data")
    data.add_argument("--gt", required=True, help="YOLO label folder or COCO json")
    data.add_argument(
        "--pred",
        required=True,
        action="append",
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
    data.add_argument("--images", help="image folder, to read each YOLO image's size")
    data.add_argument("--keypoints", type=int, help="keypoints per instance (YOLO)")
    data.add_argument(
        "--classes", type=int, nargs="+", help="class ids to keep, on both sides"
    )
    data.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="a predicted point counts as seen above this confidence (default 0)",
    )
    data.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="drop predicted detections scored below this (default 0): matching "
        "ignores scores",
    )
    data.add_argument(
        "--min-iou",
        type=float,
        default=0.3,
        help="smallest box IoU that pairs a prediction with a truth (default 0.3)",
    )
    data.add_argument(
        "--min-similarity",
        type=float,
        default=0.5,
        help="smallest keypoint similarity that pairs them when a side has no box "
        "(default 0.5)",
    )
    what = run.add_argument_group("measure")
    for kind in MEASURES:
        what.add_argument(
            f"--{kind}",
            metavar="I,J,...",
            action=_AddMeasure,
            help="keypoint indices, counted from 0; repeat for several measures",
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
        default=3,
        help="with --relative-to: fewest readings an image needs (default 3)",
    )
    how = run.add_argument_group("analysis")
    how.add_argument(
        "--big-error",
        type=float,
        required=True,
        help="an error at least this large counts as large (unit of the measure)",
    )
    how.add_argument(
        "--bands",
        default="4",
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
        "--size-bands", default="3", help="a count, or pixel edges such as 30,60"
    )
    how.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[],
        help="decision thresholds applied to the truth",
    )
    how.add_argument(
        "--pred-threshold",
        type=float,
        nargs="+",
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
        help="variance of prediction noise over variance of label noise; adds a "
        "Deming slope",
    )
    how.add_argument(
        "--mixed-classes",
        action="store_true",
        help="allow readings of several classes in one audit",
    )
    how.add_argument(
        "--seed", type=int, default=0, help="seed for resampling and rebuilds"
    )
    how.add_argument(
        "--resamples",
        type=int,
        default=2000,
        help="bootstrap resamples for the intervals (default 2000; fewer is faster)",
    )
    how.add_argument(
        "--jitter-repeats",
        type=int,
        default=500,
        help="rebuilds for the jitter reference (default 500; 0 turns it off)",
    )
    out = run.add_argument_group("output")
    out.add_argument("--report", help="markdown report")
    out.add_argument(
        "--csv", help="one row per reading; comparing models, one per difference"
    )
    out.add_argument("--json", help="every figure")
    out.add_argument("--plot", help="PNG with two panels (needs matplotlib)")
    out.add_argument(
        "--full", action="store_true", help="every statistic in the summary"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
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
    loader_notes = []
    for w in caught:
        text = str(w.message).replace("min_confidence", "--min-conf")
        if text in loader_notes:
            continue
        loader_notes.append(text)
        print(_console(f"poseaudit: warning: {text}", sys.stderr), file=sys.stderr)
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
        text = table(results, "°" if _utf8(sys.stdout) else " deg")
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
        sys.exit("poseaudit: nothing was read")


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
            + differences_table(comparison.differences, comparison.results)
        )
    else:
        text = comparison.summary("°" if _utf8(sys.stdout) else " deg")
    print(_console(text, sys.stdout))
    try:
        if args.json:
            data = comparison.to_dict()
            data["warnings"] = loader_notes + data["warnings"]
            Path(args.json).write_text(
                json.dumps(_finite(data), indent=2, default=str),
                encoding="utf-8",
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


def _console(text: str, stream) -> str:
    """ " deg" for the degree sign on a stream that is not UTF-8. On Korean
    Windows, Git Bash shows a pipe's bytes as UTF-8 while Python writes them in
    cp949, which can encode the sign, so it comes out garbled rather than
    failing. Reconfiguring the stream to UTF-8 would garble it instead for
    whatever reads the pipe as cp949; " deg" reads the same either way. Other
    text (image names) is left alone, and files are always written as UTF-8."""
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    if _utf8(stream):
        return text
    # padding after a sign (the ">= 15°" label) gives up the three columns
    # " deg" adds, so the figures stay aligned
    text = re.sub(r"°( {4,})", lambda m: " deg" + m.group(1)[3:], text)
    text = text.replace("°", " deg")
    if encoding:  # never fail on a character the console cannot show
        text = text.encode(encoding, errors="replace").decode(encoding)
    return text


def _write(result, args) -> None:
    if args.json:
        Path(args.json).write_text(
            json.dumps(_finite(result.to_dict()), indent=2, default=str),
            encoding="utf-8",
        )
    if args.csv:
        result.to_csv(args.csv)
    if result.n == 0:
        return
    if args.report:
        result.to_markdown(args.report)
    if args.plot:
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
        Path(args.json).write_text(
            json.dumps(_finite(several(results)), indent=2, default=str),
            encoding="utf-8",
        )
    if args.csv:
        Path(args.csv).write_text(csv_rows_many(results), encoding="utf-8")
    if args.report:
        Path(args.report).write_text(
            "\n".join(r.to_markdown() for r in results if r.n), encoding="utf-8"
        )


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
    return [_audit(args, pairing, m, bands, size_bands) for m in measures]


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
