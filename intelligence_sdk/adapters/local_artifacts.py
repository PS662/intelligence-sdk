"""Local-directory implementation of the ``ArtifactStore`` port.

Layout: ``<output_dir>/<section_name>/<run_id>/...`` with a ``latest`` pointer per
section (symlink, with a ``latest.json`` fallback on filesystems that refuse symlinks).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from intelligence_sdk.interfaces import ArtifactStore

LATEST = "latest"
LATEST_JSON = "latest.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class LocalArtifactStore(ArtifactStore):
    def new_run(self, section_name: str, output_dir: str) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(output_dir) / section_name / f"{stamp}_{uuid.uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def write_json(self, run_dir: Path, name: str, payload: dict[str, Any]) -> Path:
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=_json_default))
        return path

    def read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text())

    def write_yaml(self, run_dir: Path, name: str, payload: dict[str, Any]) -> Path:
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        clean = json.loads(json.dumps(payload, default=_json_default))
        path.write_text(yaml.safe_dump(clean, sort_keys=False, default_flow_style=False))
        return path

    def write_csv(self, run_dir: Path, name: str, frame: pd.DataFrame) -> Path:
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=frame.index.name is not None)
        return path

    def set_latest(self, section_name: str, output_dir: str, run_dir: Path) -> None:
        section_dir = Path(output_dir) / section_name
        link = section_dir / LATEST
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(run_dir.name, target_is_directory=True)
        except OSError:
            (section_dir / LATEST_JSON).write_text(json.dumps({"run_dir": str(run_dir)}))

    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_symlink() or p.is_dir():
            return p.resolve()
        if p.name == LATEST and (p.parent / LATEST_JSON).exists():
            return Path(json.loads((p.parent / LATEST_JSON).read_text())["run_dir"]).resolve()
        return p
