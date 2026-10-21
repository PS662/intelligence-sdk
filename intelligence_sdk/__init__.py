"""intelligence-sdk: config-driven model building.

Public facade - three names are all a consumer needs::

    from intelligence_sdk import load_recipe, run_recipe, build_runner

``build_runner`` assembles the implementations a recipe's ``runtime`` block names;
``core.runner.Runner`` is the override point for custom wiring.
"""

from __future__ import annotations

from pathlib import Path

from intelligence_sdk.core.recipe import Recipe, load_recipe
from intelligence_sdk.core.runner import Runner
from intelligence_sdk.core.types import NodeResult
from intelligence_sdk.runtime import build_runner

__all__ = ["load_recipe", "run_recipe", "build_runner", "Recipe", "Runner", "NodeResult"]
__version__ = "0.1.0"


def run_recipe(
    recipe: str | Path | Recipe,
    *,
    node: str | None = None,
    runner: Runner | None = None,
) -> dict[str, NodeResult]:
    """Validate and execute a recipe (path or loaded ``Recipe``). With ``node``, run
    only that node and the nodes it depends on."""
    loaded = recipe if isinstance(recipe, Recipe) else load_recipe(recipe)
    active = runner or build_runner(loaded.runtime)
    return active.run(loaded, node=node)
