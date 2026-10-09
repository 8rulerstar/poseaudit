"""Writing an output file whole or not at all."""

from __future__ import annotations

import contextlib
import os
import tempfile
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
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


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
