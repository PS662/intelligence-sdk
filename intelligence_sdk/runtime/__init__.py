"""Runtime: turns a recipe's ``runtime`` block into a wired ``Runner``.

``core`` knows only the ports; this package is where provider names become
implementations (see ``providers.py`` for the registry).
"""

from __future__ import annotations

from intelligence_sdk.core.recipe import RuntimeSpec
from intelligence_sdk.core.runner import Runner
from intelligence_sdk.runtime import providers
from intelligence_sdk.runtime.providers import available_providers

__all__ = ["build_runner", "available_providers", "providers"]


def build_runner(runtime: RuntimeSpec | None = None) -> Runner:
    """Assemble a runner for ``runtime`` (defaults: sklearn backend, local artifacts)."""
    spec = runtime or RuntimeSpec()
    return Runner(
        loaders={source: cls() for source, cls in providers.LOADERS.items()},
        backend=providers.BACKENDS[spec.backend](),
        store=providers.ARTIFACT_STORES[spec.artifact_store](),
        evaluator=providers.EVALUATOR(),
    )
