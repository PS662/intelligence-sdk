"""Recipe nodes as graphs.

Two graphs, both validated before anything runs:

* the dependency graph over train/test/eval nodes, from explicit ``depends_on``.
  A test's dependency names its model source. Nodes with no edges are *isolated*:
  they run after the connected ones, in listing order.
* the reference graph over dataset nodes, from ``dataset_name`` entries. Dataset
  nodes are never scheduled on their own; they run when a reference reaches them
  (or when named as the ``--node`` target).

With a target node, execution is that node's transitive dependencies in
topological order followed by the node itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from intelligence_sdk.core.errors import RecipeError, RecipeValidationError, did_you_mean
from intelligence_sdk.core.recipe import ActionNodeTypes, DatasetNode, Recipe, TestNode, TrainNode


@dataclass
class ExecutionPlan:
    ordered: list[str]
    isolated: list[str]
    target: str | None = None

    @property
    def sequence(self) -> list[str]:
        return self.ordered + self.isolated


@dataclass
class NodeInfo:
    name: str
    type: str
    role: str  # dag | isolated | dataset
    position: int
    depends_on: list[str] = field(default_factory=list)
    referenced_by: list[str] = field(default_factory=list)


def _cycle_text(graph: nx.DiGraph) -> str:
    cycle = nx.find_cycle(graph)
    return " -> ".join(u for u, _ in cycle) + f" -> {cycle[0][0]}"


def dependency_graph(recipe: Recipe) -> nx.DiGraph:
    graph = nx.DiGraph()
    names = recipe.order
    for name, node in recipe.action_nodes.items():
        graph.add_node(name, type=node.type, position=names.index(name))
    errors: list[RecipeError] = []
    for name, node in recipe.action_nodes.items():
        for i, dep in enumerate(node.depends_on):
            path = f"{name}.depends_on[{i}]"
            if dep == name:
                errors.append(RecipeError(path, "a node cannot depend on itself"))
            elif dep not in recipe.nodes:
                errors.append(RecipeError(path, f"unknown node '{dep}'", did_you_mean(dep, list(recipe.action_nodes))))
            elif isinstance(recipe.nodes[dep], DatasetNode):
                errors.append(
                    RecipeError(path, f"'{dep}' is a dataset node; reference it from 'datasets', not 'depends_on'")
                )
            else:
                graph.add_edge(dep, name)
    if errors:
        raise RecipeValidationError(errors)
    if not nx.is_directed_acyclic_graph(graph):
        raise RecipeError("", f"recipe is not a DAG; dependency cycle: {_cycle_text(graph)}")

    for name, node in recipe.action_nodes.items():
        if isinstance(node, TestNode) and node.model_path is None:
            train_deps = [d for d in node.depends_on if isinstance(recipe.nodes[d], TrainNode)]
            if len(node.depends_on) != 1 or len(train_deps) != 1:
                errors.append(
                    RecipeError(
                        f"{name}.depends_on",
                        "a test without 'model_path' must depend on exactly one train node "
                        f"(got: {', '.join(node.depends_on) or 'none'})",
                    )
                )
    if errors:
        raise RecipeValidationError(errors)
    return graph


def reference_graph(recipe: Recipe) -> nx.DiGraph:
    """Dataset node -> dataset nodes it references. Registry entries are leaves."""
    graph = nx.DiGraph()
    dataset_nodes = recipe.dataset_nodes
    graph.add_nodes_from(dataset_nodes)
    for name, node in dataset_nodes.items():
        for ref in node.references():
            if ref in dataset_nodes:
                graph.add_edge(name, ref)
    if not nx.is_directed_acyclic_graph(graph):
        raise RecipeError("", f"dataset references form a cycle: {_cycle_text(graph)}")
    return graph


def validate(recipe: Recipe) -> tuple[nx.DiGraph, nx.DiGraph]:
    return dependency_graph(recipe), reference_graph(recipe)


def _topological(graph: nx.DiGraph, names: list[str]) -> list[str]:
    sub = graph.subgraph(names)
    return list(nx.lexicographical_topological_sort(sub, key=lambda n: graph.nodes[n]["position"]))


def plan(recipe: Recipe, graph: nx.DiGraph, node: str | None = None) -> ExecutionPlan:
    if node is not None:
        if node not in recipe.nodes:
            raise RecipeError("--node", f"unknown node '{node}'", did_you_mean(node, recipe.order))
        if isinstance(recipe.nodes[node], DatasetNode):
            return ExecutionPlan(ordered=[node], isolated=[], target=node)
        wanted = nx.ancestors(graph, node) | {node}
        return ExecutionPlan(ordered=_topological(graph, list(wanted)), isolated=[], target=node)

    connected = [n for n in recipe.action_nodes if graph.degree(n) > 0]
    isolated = [n for n in recipe.action_nodes if graph.degree(n) == 0]
    return ExecutionPlan(ordered=_topological(graph, connected), isolated=isolated)


def explicit_train_dependency(recipe: Recipe, name: str) -> str | None:
    node = recipe.nodes[name]
    for dep in getattr(node, "depends_on", []):
        if isinstance(recipe.nodes[dep], TrainNode):
            return dep
    return None


def referenced_by(recipe: Recipe) -> dict[str, list[str]]:
    """Dataset node -> nodes (of any type) that reference it directly."""
    out: dict[str, list[str]] = {name: [] for name in recipe.dataset_nodes}
    for name, node in recipe.nodes.items():
        for ref in node.references():
            if ref in out:
                out[ref].append(name)
    return out


def describe(recipe: Recipe, graph: nx.DiGraph, execution: ExecutionPlan) -> list[NodeInfo]:
    """Rows for ``inspect``: connected nodes in execution order, then isolated
    nodes, then dataset nodes with who references them."""
    names = recipe.order
    refs = referenced_by(recipe)
    rows = []
    for name in execution.ordered:
        node = recipe.nodes[name]
        role = "dataset" if isinstance(node, DatasetNode) else "dag"
        rows.append(NodeInfo(name, node.type, role, names.index(name), list(getattr(node, "depends_on", []))))
    for name in execution.isolated:
        rows.append(NodeInfo(name, recipe.nodes[name].type, "isolated", names.index(name)))
    listed = {r.name for r in rows}
    for name in recipe.dataset_nodes:
        if name not in listed:
            rows.append(NodeInfo(name, "dataset", "dataset", names.index(name), referenced_by=refs[name]))
    return rows


__all__ = [
    "ExecutionPlan",
    "NodeInfo",
    "ActionNodeTypes",
    "dependency_graph",
    "reference_graph",
    "validate",
    "plan",
    "explicit_train_dependency",
    "referenced_by",
    "describe",
]
