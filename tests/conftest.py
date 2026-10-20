from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
IRIS = REPO / "data" / "iris.csv"


@pytest.fixture
def iris_path() -> Path:
    return IRIS


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def file_dataset(path: Path | str = IRIS, **extra) -> dict:
    return {"source": "file", "path": str(path), "target": "species", **extra}


def train_block(dataset_name: str = "iris", model: str = "random_forest", params: dict | None = None, **extra) -> dict:
    block = {
        "type": "train",
        "datasets": [{"dataset_name": dataset_name}],
        "pipeline": [{"step": "standard_scaler"}],
        "model": {"name": model, "params": params or {}},
        "split": {"test_size": 0.2, "stratify": True},
        "seed": 7,
    }
    block.update(extra)
    return block
