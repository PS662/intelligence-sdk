from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class ArtifactStore(Protocol):
    """Where runs leave their provenance bundle. Local directories in v1; an S3
    store is another implementation of this protocol."""

    def new_run(self, section_name: str, output_dir: str) -> Path:
        """Create and return a fresh ``<output_dir>/<section_name>/<run_id>`` directory."""
        ...

    def write_json(self, run_dir: Path, name: str, payload: dict[str, Any]) -> Path: ...

    def read_json(self, path: Path) -> dict[str, Any]: ...

    def write_yaml(self, run_dir: Path, name: str, payload: dict[str, Any]) -> Path: ...

    def write_csv(self, run_dir: Path, name: str, frame: pd.DataFrame) -> Path: ...

    def set_latest(self, section_name: str, output_dir: str, run_dir: Path) -> None:
        """Point ``<output_dir>/<section_name>/latest`` at ``run_dir``."""
        ...

    def resolve(self, path: str | Path) -> Path:
        """Follow a ``latest`` pointer (symlink or json fallback) to a concrete run dir."""
        ...
