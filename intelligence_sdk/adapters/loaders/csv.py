"""``source: file`` loader: a CSV with one row per sample."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from intelligence_sdk.core.errors import RecipeError, did_you_mean
from intelligence_sdk.core.recipe import FileDataset
from intelligence_sdk.core.types import Dataset
from intelligence_sdk.interfaces import DataLoader

ROW_ID = "row_id"


class CsvLoader(DataLoader):
    def load(self, spec: FileDataset, name: str) -> Dataset:
        path = Path(spec.path)
        if not path.exists():
            raise RecipeError(f"{name}.path", f"dataset file not found: {path}")
        frame = pd.read_csv(path)
        if spec.id_column is not None:
            if spec.id_column not in frame.columns:
                raise RecipeError(
                    f"{name}.id_column",
                    f"column '{spec.id_column}' not in {path}",
                    did_you_mean(spec.id_column, frame.columns),
                )
            frame = frame.set_index(spec.id_column, drop=True)
            if not frame.index.is_unique:
                raise RecipeError(f"{name}.id_column", f"column '{spec.id_column}' has duplicate ids")
        else:
            frame.index = pd.RangeIndex(len(frame), name=ROW_ID)
        return Dataset(name=name, frame=frame, target=spec.target)
