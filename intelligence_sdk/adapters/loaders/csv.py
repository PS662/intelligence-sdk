"""``source: file`` loader. Version = MD5 of the raw bytes, recomputed on every load
and compared by the runner with what the model recorded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from intelligence_sdk.core.errors import RecipeError, did_you_mean
from intelligence_sdk.core.filters import apply_filter
from intelligence_sdk.core.recipe import FileDataset
from intelligence_sdk.core.types import DatasetVersion, LoadedDataset
from intelligence_sdk.interfaces import DataLoader

ROW_ID = "row_id"


class CsvLoader(DataLoader):
    def load(self, definition: FileDataset) -> LoadedDataset:
        path = Path(definition.path)
        if not path.exists():
            raise RecipeError("dataset.path", f"dataset file not found: {path}")
        raw = path.read_bytes()
        digest = hashlib.md5(raw).hexdigest()  # noqa: S324 - content identity, not security

        frame = pd.read_csv(path)
        if definition.id_column is not None:
            if definition.id_column not in frame.columns:
                raise RecipeError(
                    "dataset.id_column",
                    f"column '{definition.id_column}' not in {path}",
                    did_you_mean(definition.id_column, frame.columns),
                )
            frame = frame.set_index(definition.id_column, drop=True)
            if not frame.index.is_unique:
                raise RecipeError("dataset.id_column", f"column '{definition.id_column}' has duplicate ids")
        else:
            frame.index = pd.RangeIndex(len(frame), name=ROW_ID)

        version = DatasetVersion(
            source="file",
            locator=str(path),
            hash=digest,
            n_rows=len(frame),
            columns=[str(c) for c in frame.columns],
            provenance={"hash_algorithm": "md5", "bytes": len(raw)},
        )
        frame = apply_filter(frame, definition.filter, "dataset.filter")
        target = definition.target if definition.target in frame.columns else None
        return LoadedDataset(frame=frame, target=target, version=version)
