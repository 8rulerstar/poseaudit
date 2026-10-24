"""The VIDIMU example's figures, quoted in its README and the main one, must
stay what its scripts printed. Its data is downloaded, not stored, so they
are checked against summary.txt, the output kept with the example."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
# examples/ is in the repository, not in the sdist: the check skips there
needs_example = pytest.mark.skipif(
    not (ROOT / "examples" / "vidimu" / "summary.txt").exists(),
    reason="examples are not in this checkout",
)


def doc(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


@needs_example
def test_the_vidimu_figures_are_what_its_scripts_printed() -> None:
    summary = doc("examples/vidimu/summary.txt")
    means = dict(
        (m, (float(p), float(b), float(r)))
        for m, p, b, r in re.findall(
            r"^(\w+) +([\d.]+)° +([\d.]+)° +([\d.]+)°$", summary, re.M
        )
    )
    # the CLI table gives the interval; the comparison table, the rounded bias
    a13 = re.search(
        r"^A13 arm_flex_r +\d+ +\+[\d.]+° \[\+([\d.]+) to \+([\d.]+)\]"
        r".* ([\d.]+)° +[\d.]+% \[",
        summary,
        re.M,
    )
    published = re.search(
        r"^A13 +arm_flex_r +MotionAGFormer +([\d.]+)° +\+([\d.]+)°", summary, re.M
    )
    body = re.search(
        r"^A13 +arm_flex_r +BodyTrack +[\d.]+° +([-+\d.]+)°", summary, re.M
    )
    assert len(means) == 4 and a13 and published and body
    example = doc("examples/vidimu/README.md")
    agf, bt = means["MotionAGFormer"], means["BodyTrack"]
    assert f"lowest\n  published RMSE ({agf[0]:.1f}°" in example
    assert f"largest\n  bias ({agf[1]:.1f}°" in example
    assert f"largest RMSE ({agf[2]:.1f}°)" in example
    assert (
        f"BodyTrack, {bt[0]:.1f}° published, has the smallest"
        f" ({bt[1]:.1f}° and {bt[2]:.1f}°)" in example
    )
    lo, hi, rmse = a13.groups()
    rmse_published, value = published.groups()
    assert f"published RMSE {rmse_published}° for" in example
    assert (
        f"{value}° above the IMU on average\n  [{float(lo):.1f} to {float(hi):.1f}]"
        in example
    )
    assert f"RMSE of {float(rmse):.1f}°" in example
    assert f"{float(body.group(1)):.1f}° for BodyTrack" in example
    main = " ".join(doc("README.md").split())
    assert f"lowest published RMSE ({agf[0]:.1f}° on average)" in main
    assert f"largest bias ({agf[1]:.1f}°)" in main
