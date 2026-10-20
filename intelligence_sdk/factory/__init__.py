"""Composition root. The only module that imports both the ports and the adapters.

Swapping a backend or a store is a one-line change here; nothing else in the SDK
names an adapter.
"""
from __future__ import annotations

from intelligence_sdk.adapters.evaluator import DefaultEvaluator
from intelligence_sdk.adapters.loaders.csv import CsvLoader
from intelligence_sdk.adapters.loaders.library import LibraryLoader
from intelligence_sdk.adapters.local_artifacts import LocalArtifactStore
from intelligence_sdk.adapters.sklearn_backend import SklearnBackend
from intelligence_sdk.core.runner import Runner


def default_runner() -> Runner:
    """Batteries-included assembly: CSV + library loaders, scikit-learn backend,
    local artifact directory, array-based evaluator."""
    return Runner(
        loaders={"file": CsvLoader(), "library": LibraryLoader()},
        backend=SklearnBackend(),
        store=LocalArtifactStore(),
        evaluator=DefaultEvaluator(),
    )
