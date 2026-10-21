from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from intelligence_sdk.core.types import Dataset


class ArtifactStore(ABC):
    """Where nodes write their outputs and datasets are saved. A local directory
    store in this package; an object store would be another subclass."""

    @abstractmethod
    def exists(self, path: str | Path) -> bool: ...

    @abstractmethod
    def prepare(self, path: str | Path, overwrite: bool, what: str) -> Path:
        """Return ``path`` ready for writing: created if absent, emptied if
        ``overwrite``; otherwise raise a ``RecipeError`` naming ``what``."""
        ...

    @abstractmethod
    def write_json(self, path: Path, name: str, payload: dict[str, Any]) -> Path: ...

    @abstractmethod
    def read_json(self, path: Path) -> dict[str, Any]: ...

    @abstractmethod
    def write_yaml(self, path: Path, name: str, payload: dict[str, Any]) -> Path: ...

    @abstractmethod
    def write_csv(self, path: Path, name: str, frame: pd.DataFrame) -> Path: ...

    @abstractmethod
    def write_dataset(self, path: Path, dataset: Dataset) -> Path:
        """Save ``dataset.csv``, ``metadata.json`` and, with a split, ``train_ids.csv``
        and ``test_ids.csv`` under ``path``."""
        ...
