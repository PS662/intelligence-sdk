"""Row filtering by id. Shared by every loader and by the runner's reference-level
filter so both apply identical semantics: include first, then exclude. Filters
subset the frame but never renumber it - the index (row ids) survives intact."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.recipe import FilterSpec


def _read_ids(spec: str | list[Any], path: str) -> list[Any]:
    if isinstance(spec, list):
        return spec
    file = Path(spec)
    if not file.exists():
        raise RecipeError(path, f"id file not found: {file}")
    frame = pd.read_csv(file)
    if frame.shape[1] != 1:
        raise RecipeError(path, f"id file {file} must have exactly one column, found {frame.shape[1]}")
    return frame.iloc[:, 0].tolist()


def apply_filter(frame: pd.DataFrame, spec: FilterSpec | None, path: str) -> pd.DataFrame:
    if spec is None or (spec.include is None and spec.exclude is None):
        return frame
    before = len(frame)
    out = frame
    if spec.include is not None:
        ids = _read_ids(spec.include, f"{path}.include")
        out = out[out.index.isin(ids)]
    if spec.exclude is not None:
        ids = _read_ids(spec.exclude, f"{path}.exclude")
        out = out[~out.index.isin(ids)]
    if len(out) == 0:
        raise RecipeError(
            path,
            f"no rows left after filtering (started with {before}); "
            f"include={spec.include!r} exclude={spec.exclude!r}",
        )
    return out


def filter_to_dict(spec: FilterSpec | None) -> dict[str, Any] | None:
    return None if spec is None else spec.model_dump(mode="json", exclude_none=True)
