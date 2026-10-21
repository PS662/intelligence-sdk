"""``source: dataset`` loader: a dataset this tool saved (``dataset.csv`` +
``metadata.json``), arriving with its recorded split and folds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.recipe import SavedDataset
from intelligence_sdk.core.types import Dataset
from intelligence_sdk.interfaces import DataLoader

DATASET_CSV = "dataset.csv"
METADATA_JSON = "metadata.json"


class SavedDatasetLoader(DataLoader):
    def load(self, spec: SavedDataset, name: str) -> Dataset:
        root = Path(spec.path)
        csv, meta_path = root / DATASET_CSV, root / METADATA_JSON
        if not csv.exists() or not meta_path.exists():
            raise RecipeError(
                f"{name}.path", f"no saved dataset at {root} (expected {DATASET_CSV} and {METADATA_JSON})"
            )
        meta = json.loads(meta_path.read_text())
        frame = pd.read_csv(csv, index_col=0)
        frame.index.name = meta.get("id_column") or frame.index.name or "row_id"
        dataset = Dataset.from_metadata(frame, meta)
        dataset.name = name
        dataset.lineage = [meta["name"]]
        return dataset
