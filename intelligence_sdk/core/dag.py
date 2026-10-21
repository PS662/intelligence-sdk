"""Recipe sections as a DAG.

Two kinds of edges:

* explicit - from ``depends_on``. A test section's explicit dependency names its
  model and, when it has no ``datasets`` block, its dataset.
* implicit - a section without ``depends_on`` (and not listed first) runs after the
  previously listed section. Implicit edges only order execution; the model-source
  rule and all inheritance consider explicit edges, so a standalone test with a
  ``model_path`` may be listed after a train section.

Listing order decides implicit chaining and breaks topological ties.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from intelligence_sdk.core.errors import RecipeError, RecipeValidationError, did_you_mean
from intelligence_sdk.core.recipe import Recipe, TestSection, TrainSection


@dataclass
class NodeInfo:
    name: str
    type: str
    position: int
    explicit_deps: list[str]
    implicit_dep: str | None


def build_graph(recipe: Recipe) -> nx.DiGraph:
    graph = nx.DiGraph()
    names = recipe.order
    for position, name in enumerate(names):
        graph.add_node(name, type=recipe.sections[name].type, position=position)
    for position, name in enumerate(names):
        section = recipe.sections[name]
        if section.depends_on:
            for dep in section.depends_on:
                graph.add_edge(dep, name, implicit=False)
        elif position > 0:
            graph.add_edge(names[position - 1], name, implicit=True)
    return graph


def validate_graph(recipe: Recipe, graph: nx.DiGraph) -> None:
    errors: list[RecipeError] = []
    names = recipe.order

    for name in names:
        section = recipe.sections[name]
        for i, dep in enumerate(section.depends_on):
            path = f"{name}.depends_on[{i}]"
            if dep == name:
                errors.append(RecipeError(path, "a section cannot depend on itself"))
            elif dep not in recipe.sections:
                errors.append(RecipeError(path, f"unknown section '{dep}'", did_you_mean(dep, names)))
    if errors:
        raise RecipeValidationError(errors)

    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        chain = " -> ".join(u for u, _ in cycle) + f" -> {cycle[0][0]}"
        raise RecipeError("", f"recipe is not a DAG; dependency cycle: {chain}")

    for name in names:
        section = recipe.sections[name]
        if isinstance(section, TestSection) and section.model_path is None:
            train_deps = [d for d in section.depends_on if isinstance(recipe.sections[d], TrainSection)]
            if len(section.depends_on) != 1 or len(train_deps) != 1:
                errors.append(
                    RecipeError(
                        f"{name}.depends_on",
                        "a test without 'model_path' must depend on exactly one train section "
                        f"(got: {', '.join(section.depends_on) or 'none'})",
                    )
                )
    if errors:
        raise RecipeValidationError(errors)


def execution_order(graph: nx.DiGraph) -> list[str]:
    """Topological order; ties broken by listing position so runs are reproducible."""
    return list(nx.lexicographical_topological_sort(graph, key=lambda n: graph.nodes[n]["position"]))


def explicit_train_dependency(recipe: Recipe, name: str) -> str | None:
    """The train section a test inherits its model from (explicit edges only)."""
    section = recipe.sections[name]
    for dep in section.depends_on:
        if isinstance(recipe.sections[dep], TrainSection):
            return dep
    return None


def describe(recipe: Recipe, graph: nx.DiGraph) -> list[NodeInfo]:
    """Rows for the ``inspect`` table, in execution order."""
    rows = []
    for name in execution_order(graph):
        section = recipe.sections[name]
        implicit = [u for u, _, d in graph.in_edges(name, data=True) if d.get("implicit")]
        rows.append(
            NodeInfo(
                name=name,
                type=section.type,
                position=graph.nodes[name]["position"],
                explicit_deps=list(section.depends_on),
                implicit_dep=implicit[0] if implicit else None,
            )
        )
    return rows


def validated_graph(recipe: Recipe) -> nx.DiGraph:
    graph = build_graph(recipe)
    validate_graph(recipe, graph)
    return graph
