"""Optional PNG rendering of a recipe DAG. matplotlib is an optional dependency
(``pip install intelligence-sdk[render]``); ``inspect`` degrades to its table
without it. Explicit edges are solid, implicit (ordering-only) edges dashed."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

_COLORS = {"train": "#c6dbef", "test": "#c7e9c0", "eval": "#fdd0a2"}


def render_png(graph: nx.DiGraph, out_path: Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Layered layout: x = topological generation, y = position within the layer.
    pos = {}
    for depth, layer in enumerate(nx.topological_generations(graph)):
        layer = sorted(layer, key=lambda n: graph.nodes[n]["position"])
        for i, node in enumerate(layer):
            pos[node] = (depth, -i)

    fig, ax = plt.subplots(figsize=(max(4, 2.6 * (max(x for x, _ in pos.values()) + 1)), max(3, 1.4 * len(graph))))
    colors = [_COLORS.get(graph.nodes[n]["type"], "#eeeeee") for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=3400, edgecolors="#444", ax=ax)
    labels = {n: f"{n}\n({graph.nodes[n]['type']})" for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
    explicit = [(u, v) for u, v, d in graph.edges(data=True) if not d.get("implicit")]
    implicit = [(u, v) for u, v, d in graph.edges(data=True) if d.get("implicit")]
    nx.draw_networkx_edges(graph, pos, edgelist=explicit, arrows=True, arrowsize=18, node_size=3400, ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=implicit,
        arrows=True,
        arrowsize=18,
        style="dashed",
        edge_color="#888",
        node_size=3400,
        ax=ax,
    )
    ax.set_title(title or "recipe DAG  (solid = explicit depends_on, dashed = implicit listing order)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
