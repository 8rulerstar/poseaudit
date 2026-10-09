"""The figure `AuditResult.plot` returns. Imported once matplotlib is known to
be installed, so `poseaudit.plot` can say how to install it."""

import io
from typing import Any, cast

import matplotlib
from matplotlib.figure import Figure

# a notebook shows the figure at its size in inches times this; it is drawn at
# SHARP times that and declared at the smaller width, so it stays crisp on a
# high-density screen without taking twice the room
SCREEN_DPI = 100
SHARP = 2


class Panels(Figure):
    """A matplotlib Figure that a notebook shows inline whether or not pyplot
    was imported: without pyplot no inline backend has registered a way to
    show figures, and the notebook would print `<Figure size 700x420 ...>`."""

    def _repr_png_(self) -> tuple[bytes, dict]:
        from poseaudit.plot import STYLE

        buffer = io.BytesIO()
        with matplotlib.rc_context(cast(Any, STYLE)):
            self.savefig(buffer, format="png", dpi=SCREEN_DPI * SHARP)
        width, height = self.get_size_inches() * SCREEN_DPI
        return buffer.getvalue(), {"width": round(width), "height": round(height)}
