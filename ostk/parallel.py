"""ostk.parallel — run a per-case measurement function across many cases.

`fn` must be a TOP-LEVEL function (picklable) that takes one item and returns a
result; ostk's primitives are pure/stateless, so per-case work parallelises
cleanly. Set workers=1 to run serially (debug / determinism check)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List


def map_cases(fn: Callable, items: Iterable, workers: int | None = None) -> List:
    items = list(items)
    if workers == 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
