"""Round 10: outputs are written whole or not at all, and a long run says so
before it starts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from poseaudit import _files
from poseaudit.cli import _say_if_slow

DEMO = Path(__file__).parents[1] / "examples" / "coco_elbow"
# the demo data is in the repository, not in the sdist: tests that read it
# skip there rather than fail
needs_demo = pytest.mark.skipif(
    not (DEMO / "gt_200.json").exists(), reason="demo data is not in this checkout"
)


def test_an_interrupted_write_leaves_the_old_file_and_no_temporary(
    tmp_path,
) -> None:
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")

    def half(temporary: str) -> None:
        Path(temporary).write_text("hal", encoding="utf-8")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _files.write_with(target, half)
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_text_is_written_through_a_temporary_then_moved(tmp_path, monkeypatch) -> None:
    seen = []
    real = _files.os.replace

    def replace(source, destination):
        seen.append(Path(source).read_text(encoding="utf-8"))
        assert not Path(destination).exists()
        real(source, destination)

    monkeypatch.setattr(_files.os, "replace", replace)
    _files.write_text(tmp_path / "a.csv", "x,y\n", newline="")
    assert seen == ["x,y\n"]
    assert (tmp_path / "a.csv").read_bytes() == b"x,y\n"


@needs_demo
@pytest.mark.parametrize("call", ["markdown", "csv"])
def test_result_files_go_through_the_atomic_writer(tmp_path, monkeypatch, call) -> None:
    from poseaudit import audit, pair
    from poseaudit.io import load_coco, load_coco_results
    from poseaudit.measures import angle

    gt = DEMO / "gt_200.json"
    result = audit(
        pair(
            load_coco(gt),
            load_coco_results(DEMO / "pred_yolo11n.json", gt),
        ),
        angle(5, 7, 9),
        15,
        resamples=50,
        jitter_repeats=0,
    )
    written = []
    monkeypatch.setattr(
        _files, "write_text", lambda path, text, newline=None: written.append(path)
    )
    path = tmp_path / "out"
    getattr(result, f"to_{call}")(str(path))
    assert written == [str(path)]
    assert not path.exists()


@needs_demo
def test_parallel_runs_on_one_output_each_leave_a_whole_file(tmp_path) -> None:
    out = tmp_path / "same.json"
    command = [
        sys.executable,
        "-c",
        "import sys; from poseaudit.cli import main; main(sys.argv[1:])",
        "audit",
        "--format",
        "coco",
        "--gt",
        str(DEMO / "gt_200.json"),
        "--pred",
        str(DEMO / "pred_yolo11n.json"),
        "--angle",
        "5,7,9",
        "--big-error",
        "15",
        "--resamples",
        "50",
        "--jitter-repeats",
        "0",
        "--json",
        str(out),
    ]
    runs = [
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(6)
    ]
    assert [r.wait(timeout=300) for r in runs] == [0] * 6
    assert json.loads(out.read_text(encoding="utf-8"))["n"] > 0
    assert [p.name for p in tmp_path.iterdir()] == ["same.json"]


def test_a_long_run_is_announced(capsys) -> None:
    _say_if_slow(200_000, 2000, 500)
    err = capsys.readouterr().err
    assert "several minutes" in err
    assert "--jitter-repeats 0" in err
    _say_if_slow(200_000, 2000, 0)
    assert "--jitter-repeats" not in capsys.readouterr().err
    _say_if_slow(400, 2000, 500)
    assert capsys.readouterr().err == ""


@needs_demo
@pytest.mark.parametrize("name", ["plot.png", "plot.SVG", "plot"])
def test_a_plot_keeps_its_format_through_the_temporary(tmp_path, name) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from poseaudit import audit, pair
    from poseaudit.io import load_coco, load_coco_results
    from poseaudit.measures import angle

    gt = DEMO / "gt_200.json"
    result = audit(
        pair(load_coco(gt), load_coco_results(DEMO / "pred_yolo11n.json", gt)),
        angle(5, 7, 9),
        15,
        resamples=50,
        jitter_repeats=0,
    )
    result.plot(str(tmp_path / name))
    head = (tmp_path / name).read_bytes()[:64]
    assert (
        (b"<svg" in head or b"<?xml" in head)
        if name.endswith("SVG")
        else (head.startswith(b"\x89PNG"))
    )
    assert [p.name for p in tmp_path.iterdir()] == [name]
