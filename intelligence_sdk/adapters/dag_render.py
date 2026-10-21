"""Optional PNG rendering of a recipe's graphs. matplotlib is an optional dependency
(``pip install intelligence-sdk[render]``); ``inspect`` degrades to its table
without it. Solid arrows are ``depends_on`` edges between train/test/eval nodes;
dotted arrows point from a node to the dataset nodes it references."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from intelligence_sdk.core.dag import referenced_by
from intelligence_sdk.core.recipe import Recipe

_COLORS = {"train": "#c6dbef", "test": "#c7e9c0", "eval": "#fdd0a2", "dataset": "#e5e5e5"}


def render_png(recipe: Recipe, graph: nx.DiGraph, out_path: Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = nx.DiGraph()
    for name, node in recipe.nodes.items():
        full.add_node(name, type=node.type)
    dep_edges = list(graph.edges)
    full.add_edges_from(dep_edges)
    ref_edges = [(user, ds) for ds, users in referenced_by(recipe).items() for user in users]
    full.add_edges_from(ref_edges)

    # Columns: dataset nodes on the left, then topological generations of the
    # dependency graph, isolated action nodes last.
    pos = {}
    datasets = [n for n in recipe.order if recipe.nodes[n].type == "dataset"]
    for i, name in enumerate(datasets):
        pos[name] = (0, -i)
    column = 1
    for layer in nx.topological_generations(graph.subgraph([n for n in graph if graph.degree(n) > 0])):
        for i, name in enumerate(sorted(layer, key=recipe.order.index)):
            pos[name] = (column, -i)
        column += 1
    for i, name in enumerate(n for n in graph if graph.degree(n) == 0):
        pos[name] = (column, -i)

    width = max(4.0, 2.8 * (max((x for x, _ in pos.values()), default=0) + 1))
    height = max(3.0, 1.5 * max(len(datasets), max((-y for _, y in pos.values()), default=0) + 1))
    fig, ax = plt.subplots(figsize=(width, height))
    colors = [_COLORS.get(full.nodes[n]["type"], "#eeeeee") for n in full.nodes]
    nx.draw_networkx_nodes(full, pos, node_color=colors, node_size=3400, edgecolors="#444", ax=ax)
    nx.draw_networkx_labels(
        full, pos, labels={n: f"{n}\n({full.nodes[n]['type']})" for n in full.nodes}, font_size=8, ax=ax
    )
    nx.draw_networkx_edges(full, pos, edgelist=dep_edges, arrows=True, arrowsize=18, node_size=3400, ax=ax)
    nx.draw_networkx_edges(
        full,
        pos,
        edgelist=ref_edges,
        arrows=True,
        arrowsize=14,
        style="dotted",
        edge_color="#888",
        node_size=3400,
        ax=ax,
    )
    ax.set_title(title or "recipe graphs  (solid = depends_on, dotted = dataset reference)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
