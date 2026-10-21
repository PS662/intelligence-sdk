"""Provider registry: the implementations a recipe's ``runtime`` block may name.

This is the only module that imports adapters. The tables are the allowlist;
``runtime.backend`` and ``runtime.artifact_store`` select from them by enum, never
by import path. Adding an implementation is one import and one table entry here.
"""

from __future__ import annotations

from intelligence_sdk.adapters.evaluator import DefaultEvaluator
from intelligence_sdk.adapters.loaders.csv import CsvLoader
from intelligence_sdk.adapters.loaders.library import LibraryLoader
from intelligence_sdk.adapters.loaders.saved import SavedDatasetLoader
from intelligence_sdk.adapters.local_artifacts import LocalArtifactStore
from intelligence_sdk.adapters.sklearn_backend import SklearnBackend
from intelligence_sdk.core.registry import ArtifactStoreKind, Backend
from intelligence_sdk.interfaces import ArtifactStore, DataLoader, Evaluator, ModelBackend

BACKENDS: dict[Backend, type[ModelBackend]] = {
    Backend.SKLEARN: SklearnBackend,
}

ARTIFACT_STORES: dict[ArtifactStoreKind, type[ArtifactStore]] = {
    ArtifactStoreKind.LOCAL: LocalArtifactStore,
}

# Keyed by the dataset definition's ``source`` discriminator.
LOADERS: dict[str, type[DataLoader]] = {
    "file": CsvLoader,
    "library": LibraryLoader,
    "dataset": SavedDatasetLoader,
}

EVALUATOR: type[Evaluator] = DefaultEvaluator


def available_providers() -> dict[str, list[str]]:
    """What a ``runtime`` block may name, for help output and docs."""
    return {
        "backend": [b.value for b in BACKENDS],
        "artifact_store": [s.value for s in ARTIFACT_STORES],
        "dataset_source": list(LOADERS),
    }
