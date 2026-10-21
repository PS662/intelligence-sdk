"""Shared value types. This module imports nothing from the rest of the SDK so that
``interfaces`` can depend on it without creating a cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ResultType(StrEnum):
    """What a backend's ``predict`` produced. The evaluator keys its metrics on this,
    so config never needs a task-type field: the model tells us what it emitted."""

    CLASS_LABEL = "class_label"
    CLASS_PROBABILITY = "class_probability"
    CONTINUOUS = "continuous"


class EstimatorKind(StrEnum):
    """Static nature of a registry model, known without touching data. Used at
    resolve time to reject metric/model mismatches before anything runs."""

    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


@dataclass(frozen=True)
class DatasetVersion:
    """Identity of the data a model was trained on: a content hash plus origin details.

    The loader computes it - file bytes for csv, the canonical frame for library
    datasets - and a loader backed by a dataset registry such as DVC supplies the hash
    it already stores. A test run against a different hash is reported as leakage.
    """

    source: str
    locator: str
    hash: str
    n_rows: int
    columns: list[str]
    origin: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "locator": self.locator,
            "hash": self.hash,
            "n_rows": self.n_rows,
            "columns": list(self.columns),
            "origin": dict(self.origin),
        }


@dataclass
class LoadedDataset:
    """A materialized dataset. ``frame.index`` holds the row identifiers (original CSV
    index or the configured id column). Filters subset the frame but never renumber
    it, so every recorded row id is stable across train and test."""

    frame: pd.DataFrame
    target: str | None
    version: DatasetVersion

    @property
    def ids(self) -> pd.Index:
        return self.frame.index

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c != self.target]

    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    def y(self) -> pd.Series | None:
        if self.target is None or self.target not in self.frame.columns:
            return None
        return self.frame[self.target]


@dataclass
class PredictionResult:
    """Backend output. ``frame`` is indexed by row id and always has a ``prediction``
    column; classifiers may add ``proba_<class>`` columns."""

    frame: pd.DataFrame
    result_type: ResultType

    @property
    def values(self) -> np.ndarray:
        return self.frame["prediction"].to_numpy()


@dataclass
class SplitRecord:
    """The train/holdout split as an artifact: computed once at train time and
    recorded by row id. Consumers select by these ids; they never re-split."""

    train_ids: list[Any]
    test_ids: list[Any]
    test_size: float
    stratify: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_ids": list(self.train_ids),
            "test_ids": list(self.test_ids),
            "test_size": self.test_size,
            "stratify": self.stratify,
            "seed": self.seed,
        }


@dataclass
class SectionOutput:
    """What one executed section leaves behind for its dependants."""

    name: str
    type: str
    run_dir: Path
    model_dir: Path | None = None
    dataset_json: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
