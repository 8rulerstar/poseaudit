"""Writing an output file whole or not at all."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path


@contextlib.contextmanager
def _temporary_beside(path: str | os.PathLike[str]) -> Iterator[str]:
    """A temporary file in the target's directory, moved over the target only
    once written in full. A run killed part way leaves the old file (or none),
    never half of one, and parallel runs writing the same path each leave a
    whole file: the last to finish wins."""
    target = Path(path)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent or "."
    )
    os.close(fd)
    try:
        yield temporary
        _replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _replace(source: str, target: Path) -> None:
    """``os.replace``, retried briefly on Windows, where moving over a file
    that another process is moving over at the same moment can be refused."""
    for attempt in range(50):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == 49:
                raise
            time.sleep(0.02 * (attempt + 1) ** 0.5)


def write_text(
    path: str | os.PathLike[str], text: str, newline: str | None = None
) -> None:
    """Write ``text`` as UTF-8, atomically."""
    with _temporary_beside(path) as temporary:
        with open(temporary, "w", encoding="utf-8", newline=newline) as file:
            file.write(text)


def write_with(path: str | os.PathLike[str], save: Callable[[str], None]) -> None:
    """Let ``save`` write a temporary file, then move it over ``path``."""
    with _temporary_beside(path) as temporary:
        save(temporary)
