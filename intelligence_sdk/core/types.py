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


# ------------------------------------------------------------------------ dataset


@dataclass
class Split:
    """A train/test partition recorded by row id."""

    test_size: float
    stratify: bool
    seed: int
    train_ids: list[Any]
    test_ids: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_size": self.test_size,
            "stratify": self.stratify,
            "seed": self.seed,
            "train_ids": list(self.train_ids),
            "test_ids": list(self.test_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Split:
        return cls(
            test_size=float(data["test_size"]),
            stratify=bool(data["stratify"]),
            seed=int(data["seed"]),
            train_ids=list(data["train_ids"]),
            test_ids=list(data["test_ids"]),
        )

    def restricted(self, ids: pd.Index) -> Split:
        keep = set(ids.tolist())
        return Split(
            self.test_size,
            self.stratify,
            self.seed,
            [i for i in self.train_ids if i in keep],
            [i for i in self.test_ids if i in keep],
        )


@dataclass
class Folds:
    """A k-fold assignment recorded as the test ids of each fold."""

    k: int
    stratify: bool
    seed: int
    fold_test_ids: list[list[Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "stratify": self.stratify,
            "seed": self.seed,
            "fold_test_ids": [list(f) for f in self.fold_test_ids],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Folds:
        return cls(
            k=int(data["k"]),
            stratify=bool(data["stratify"]),
            seed=int(data["seed"]),
            fold_test_ids=[list(f) for f in data["fold_test_ids"]],
        )

    def pairs(self, ids: pd.Index) -> list[tuple[list[Any], list[Any]]]:
        """(train_ids, test_ids) per fold over the ids present in ``ids``."""
        present = ids.tolist()
        out = []
        for test in self.fold_test_ids:
            test_set = set(test)
            out.append(([i for i in present if i not in test_set], [i for i in present if i in test_set]))
        return out

    def restricted(self, ids: pd.Index) -> Folds:
        keep = set(ids.tolist())
        return Folds(self.k, self.stratify, self.seed, [[i for i in f if i in keep] for f in self.fold_test_ids])


@dataclass
class Dataset:
    """A materialized dataset. ``frame.index`` holds the row ids (positional
    ``row_id`` or the configured id column) and is unique. ``split`` and ``folds``
    are recorded by those ids, so a filter keeps them consistent by restriction."""

    name: str
    frame: pd.DataFrame
    target: str | None
    split: Split | None = None
    folds: Folds | None = None
    lineage: list[str] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ids(self) -> pd.Index:
        return self.frame.index

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c != self.target]

    def X(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        return (self.frame if frame is None else frame)[self.feature_columns]

    def y(self, frame: pd.DataFrame | None = None) -> pd.Series | None:
        source = self.frame if frame is None else frame
        if self.target is None or self.target not in source.columns:
            return None
        return source[self.target]

    def train_frame(self) -> pd.DataFrame:
        """Rows a model is fitted on: the train split, or everything without a split."""
        return self.frame.loc[self.split.train_ids] if self.split else self.frame

    def test_frame(self) -> pd.DataFrame | None:
        return self.frame.loc[self.split.test_ids] if self.split else None

    def restrict(self, ids: list[Any]) -> Dataset:
        """Keep only ``ids`` (in current order); split and folds follow."""
        keep = self.frame.index.isin(ids)
        frame = self.frame[keep]
        return Dataset(
            name=self.name,
            frame=frame,
            target=self.target,
            split=self.split.restricted(frame.index) if self.split else None,
            folds=self.folds.restricted(frame.index) if self.folds else None,
            lineage=list(self.lineage),
            filters=list(self.filters),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "id_column": self.frame.index.name,
            "columns": [str(c) for c in self.frame.columns],
            "n_rows": int(len(self.frame)),
            "lineage": list(self.lineage),
            "filters": list(self.filters),
            "split": self.split.to_dict() if self.split else None,
            "folds": self.folds.to_dict() if self.folds else None,
        }

    @classmethod
    def from_metadata(cls, frame: pd.DataFrame, meta: dict[str, Any]) -> Dataset:
        return cls(
            name=meta["name"],
            frame=frame,
            target=meta.get("target"),
            split=Split.from_dict(meta["split"]) if meta.get("split") else None,
            folds=Folds.from_dict(meta["folds"]) if meta.get("folds") else None,
            lineage=list(meta.get("lineage", [])),
            filters=list(meta.get("filters", [])),
        )


# --------------------------------------------------------------------- predictions


@dataclass
class PredictionResult:
    """Backend output. ``frame`` is indexed by row id and always has a ``prediction``
    column; classifiers may add ``proba_<class>`` columns."""

    frame: pd.DataFrame
    result_type: ResultType

    @property
    def values(self) -> np.ndarray:
        return self.frame["prediction"].to_numpy()


# ------------------------------------------------------------------- node results


@dataclass(kw_only=True)
class NodeResult:
    """What an executed node leaves behind for its dependants and the caller."""

    name: str
    type: str
    dataset: Dataset
    output_dir: Path | None = None


@dataclass(kw_only=True)
class DatasetResult(NodeResult):
    pass


@dataclass(kw_only=True)
class TrainResult(NodeResult):
    estimator: Any
    model_meta: dict[str, Any]
    search: dict[str, Any]


@dataclass(kw_only=True)
class TestResult(NodeResult):
    predictions: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(kw_only=True)
class EvalResult(NodeResult):
    report: pd.DataFrame
    summary: pd.DataFrame
