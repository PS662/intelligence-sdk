"""intelligence-sdk: config-driven model building.

Public facade - three names are all a consumer needs::

    from intelligence_sdk import load_recipe, run_recipe, default_runner

``default_runner()`` is the batteries-included assembly; ``core.runner.Runner`` is
the override point for custom wiring (another backend, another artifact store).
"""
from __future__ import annotations

from pathlib import Path

from intelligence_sdk.core.recipe import Recipe, load_recipe
from intelligence_sdk.core.runner import Runner
from intelligence_sdk.core.types import SectionOutput
from intelligence_sdk.factory import default_runner

__all__ = ["load_recipe", "run_recipe", "default_runner", "Recipe", "Runner", "SectionOutput"]
__version__ = "0.1.0"


def run_recipe(
    recipe: str | Path | Recipe,
    *,
    allow_dataset_mismatch: bool = False,
    output_dir: str | None = None,
    runner: Runner | None = None,
) -> dict[str, SectionOutput]:
    """Validate and execute a recipe (path or already-loaded ``Recipe``)."""
    loaded = recipe if isinstance(recipe, Recipe) else load_recipe(recipe)
    return (runner or default_runner()).run(loaded, allow_dataset_mismatch=allow_dataset_mismatch, output_dir=output_dir)
