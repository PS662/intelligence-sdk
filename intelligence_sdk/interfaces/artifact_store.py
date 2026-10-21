from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ArtifactStore(ABC):
    """Where runs leave their provenance bundle. Implemented by a local directory
    store; an object store would be another subclass."""

    @abstractmethod
    def new_run(self, section_name: str, output_dir: str) -> Path:
        """Create and return a fresh ``<output_dir>/<section_name>/<run_id>`` directory."""
        ...

    @abstractmethod
    def write_json(self, run_dir: Path, name: str, payload: dict[str, Any]) -> Path: ...

    @abstractmethod
    def read_json(self, path: Path) -> dict[str, Any]: ...

    @abstractmethod
    def write_yaml(self, run_dir: Path, name: str, payload: dict[str, Any]) -> Path: ...

    @abstractmethod
    def write_csv(self, run_dir: Path, name: str, frame: pd.DataFrame) -> Path: ...

    @abstractmethod
    def set_latest(self, section_name: str, output_dir: str, run_dir: Path) -> None:
        """Point ``<output_dir>/<section_name>/latest`` at ``run_dir``."""
        ...

    @abstractmethod
    def resolve(self, path: str | Path) -> Path:
        """Follow a ``latest`` pointer (symlink or json fallback) to a concrete run dir."""
        ...
