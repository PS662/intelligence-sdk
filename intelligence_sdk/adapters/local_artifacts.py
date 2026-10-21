"""Local-directory implementation of the ``ArtifactStore`` port."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.types import Dataset
from intelligence_sdk.interfaces import ArtifactStore

DATASET_FILES = ("dataset.csv", "metadata.json", "train_ids.csv", "test_ids.csv")


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class LocalArtifactStore(ArtifactStore):
    def exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    def prepare(self, path: str | Path, overwrite: bool, what: str) -> Path:
        target = Path(path)
        if target.exists():
            if not overwrite:
                raise RecipeError(what, f"'{target}' already exists; set overwrite: true to replace it")
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, path: Path, name: str, payload: dict[str, Any]) -> Path:
        out = Path(path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=_json_default))
        return out

    def read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text())

    def write_yaml(self, path: Path, name: str, payload: dict[str, Any]) -> Path:
        out = Path(path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        clean = json.loads(json.dumps(payload, default=_json_default))
        out.write_text(yaml.safe_dump(clean, sort_keys=False, default_flow_style=False))
        return out

    def write_csv(self, path: Path, name: str, frame: pd.DataFrame) -> Path:
        out = Path(path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=frame.index.name is not None)
        return out

    def write_dataset(self, path: Path, dataset: Dataset) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for stale in DATASET_FILES:
            (root / stale).unlink(missing_ok=True)
        frame = dataset.frame.copy()
        frame.index.name = frame.index.name or "row_id"
        frame.to_csv(root / "dataset.csv", index=True)
        self.write_json(root, "metadata.json", dataset.to_metadata())
        if dataset.split is not None:
            pd.DataFrame({frame.index.name: dataset.split.train_ids}).to_csv(root / "train_ids.csv", index=False)
            pd.DataFrame({frame.index.name: dataset.split.test_ids}).to_csv(root / "test_ids.csv", index=False)
        return root
