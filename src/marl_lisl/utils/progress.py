"""Small progress-bar wrapper with a safe no-tqdm fallback."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

try:  # pragma: no cover - fallback depends on optional runtime package
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def progress_iter(iterable: Iterable[T], **kwargs: object) -> Iterator[T]:
    """Return a tqdm-wrapped iterator when available, otherwise the original iterator."""
    if _tqdm is None:
        yield from iterable
    else:
        yield from _tqdm(iterable, **kwargs)
