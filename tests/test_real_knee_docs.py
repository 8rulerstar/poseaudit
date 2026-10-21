"""The real knee example's figures, quoted in the README and its own, must
stay what its audit printed. Its data is downloaded, not stored, so they are
checked against summary.txt, the audit's output kept with the example."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def doc(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_the_real_knee_figures_are_what_its_audit_printed() -> None:
    """The real knee data is downloaded, not stored: its README figures are
    checked against summary.txt, the audit's own output kept with it."""
    summary = doc("examples/real_knee/summary.txt")
    bias = re.search(r"  bias +([-+\d.]+)°", summary)
    peak = re.search(
        r"cycle peak +bias ([-\d.]+)° \[([-\d.]+) to ([-\d.]+)\].*\((\d+) cycles\)",
        summary,
    )
    trough = re.search(r"cycle trough bias ([-\d.]+)°", summary)
    span = re.search(r"cycle range +bias ([-\d.]+)°", summary)
    shift = re.search(
        r"offset \+/-0\.1 s moves .*?RMSE ([\d.]+)°, cycle peak ([\d.]+)°", summary
    )
    assert bias and peak and trough and span and shift
    value, lo, hi, cycles = peak.groups()
    peak_text = f"{-float(value):.1f}° below\n[{-float(hi):.1f} to {-float(lo):.1f}]"
    readme = doc("README.md")
    assert f"{int(cycles):,} strides" in readme
    assert f"{-float(bias.group(1)):.1f}° below the 3D" in readme
    assert peak_text in readme
    moves = f"RMSE by {float(shift.group(1)):.1f}° and the peak by {shift.group(2)}°"
    assert moves in readme
    example = doc("examples/real_knee/README.md")
    assert (
        f"read {-float(value):.1f}° lower [{-float(hi):.1f} to {-float(lo):.1f}]"
        in example
    )
    assert f"trough is read {-float(trough.group(1)):.1f}° lower" in example
    assert f"by {-float(span.group(1)):.1f}°" in example
    assert f"bias is {float(bias.group(1)):.1f}°" in example
    assert f"RMSE by {float(shift.group(1)):.1f}°" in example
