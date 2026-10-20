"""Implicit edges order, explicit edges mean."""
from __future__ import annotations

import pytest

from intelligence_sdk.core.dag import describe, execution_order, explicit_train_dependency, validated_graph
from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.recipe import parse_recipe
from tests.conftest import file_dataset, train_block

DS = {"datasets": {"iris": file_dataset()}}


def test_single_node_recipe():
    recipe = parse_recipe(DS | {"only": train_block()})
    assert execution_order(validated_graph(recipe)) == ["only"]


def test_implicit_chaining_follows_yaml_listing_order():
    # YAML mapping order is load-bearing: no depends_on anywhere, so b follows a and c follows b.
    recipe = parse_recipe(DS | {"a": train_block(), "b": train_block(), "c": train_block()})
    graph = validated_graph(recipe)
    assert execution_order(graph) == ["a", "b", "c"]
    assert graph.edges["a", "b"]["implicit"] is True
    # Reversing the listing reverses the chain.
    recipe2 = parse_recipe(DS | {"c": train_block(), "b": train_block(), "a": train_block()})
    assert execution_order(validated_graph(recipe2)) == ["c", "b", "a"]


def test_implicit_edge_does_not_break_test_xor_or_inheritance():
    # Standalone test listed after a train: ordered after it, but it does NOT inherit train_a's model.
    recipe = parse_recipe(DS | {"train_a": train_block(), "test_b": {"type": "test", "model_path": "artifacts/old/latest"}})
    graph = validated_graph(recipe)
    assert execution_order(graph) == ["train_a", "test_b"]
    assert graph.edges["train_a", "test_b"]["implicit"] is True
    assert explicit_train_dependency(recipe, "test_b") is None


def test_explicit_dependency_names_model_source():
    recipe = parse_recipe(DS | {"train_a": train_block(), "test_b": {"type": "test", "depends_on": ["train_a"]}})
    graph = validated_graph(recipe)
    assert graph.edges["train_a", "test_b"]["implicit"] is False
    assert explicit_train_dependency(recipe, "test_b") == "train_a"


def test_test_depending_on_non_train_is_rejected():
    recipe = parse_recipe(DS | {
        "e": {"type": "eval", "datasets": [{"dataset_name": "iris"}], "candidates": [{"name": "knn"}]},
        "t": {"type": "test", "depends_on": ["e"]},
    })
    with pytest.raises(RecipeError, match="exactly one train section"):
        validated_graph(recipe)


def test_unknown_dependency_suggests():
    recipe = parse_recipe(DS | {"train_a": train_block(), "t": {"type": "test", "depends_on": ["train_b"]}})
    with pytest.raises(RecipeError, match="t.depends_on\\[0\\].*did you mean 'train_a'"):
        validated_graph(recipe)


def test_cycle_is_rejected():
    recipe = parse_recipe(DS | {
        "a": train_block(depends_on=["b"]),
        "b": train_block(depends_on=["a"]),
    })
    with pytest.raises(RecipeError, match="cycle"):
        validated_graph(recipe)


def test_self_dependency_is_rejected():
    recipe = parse_recipe(DS | {"a": train_block(depends_on=["a"])})
    with pytest.raises(RecipeError, match="itself"):
        validated_graph(recipe)


def test_topological_order_with_fan_out_breaks_ties_by_listing():
    recipe = parse_recipe(DS | {
        "train_a": train_block(),
        "test_1": {"type": "test", "depends_on": ["train_a"]},
        "eval_x": {"type": "eval", "depends_on": ["train_a"], "datasets": [{"dataset_name": "iris"}], "candidates": [{"name": "knn"}]},
        "test_2": {"type": "test", "depends_on": ["train_a"]},
    })
    graph = validated_graph(recipe)
    assert execution_order(graph) == ["train_a", "test_1", "eval_x", "test_2"]
    rows = describe(recipe, graph)
    assert [r.name for r in rows] == ["train_a", "test_1", "eval_x", "test_2"]
    assert rows[1].explicit_deps == ["train_a"] and rows[1].implicit_dep is None
