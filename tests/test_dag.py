"""Explicit edges only; isolated nodes after the DAG; datasets resolved by reference."""

from __future__ import annotations

import pytest

from intelligence_sdk.core import dag
from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.recipe import parse_recipe
from tests.conftest import node_eval, node_test, node_train, registry


def _plan(data: dict, node: str | None = None) -> dag.ExecutionPlan:
    recipe = parse_recipe(data)
    graph, _ = dag.validate(recipe)
    return dag.plan(recipe, graph, node)


def test_single_node_recipe():
    assert _plan(registry() | {"only": node_train()}).sequence == ["only"]


def test_no_implicit_chaining_nodes_without_edges_are_isolated():
    plan = _plan(registry() | {"a": node_train(), "b": node_train(), "c": node_train()})
    assert plan.ordered == [] and plan.isolated == ["a", "b", "c"]


def test_connected_nodes_run_before_isolated_ones():
    plan = _plan(
        registry()
        | {
            "lonely": node_eval(),
            "train_a": node_train(),
            "test_a": node_test(depends_on=["train_a"]),
        }
    )
    assert plan.ordered == ["train_a", "test_a"] and plan.isolated == ["lonely"]
    assert plan.sequence == ["train_a", "test_a", "lonely"]


def test_node_mode_runs_ancestors_then_target():
    data = registry() | {
        "a": node_train(),
        "b": node_train(depends_on=["a"]),
        "c": node_test(depends_on=["b"]),
        "other": node_train(),
    }
    assert _plan(data, node="c").ordered == ["a", "b", "c"]
    assert _plan(data, node="b").ordered == ["a", "b"]
    assert _plan(data, node="other").ordered == ["other"]
    plan = _plan(data, node="other")
    assert plan.isolated == [] and plan.target == "other"


def test_node_mode_on_dataset_node_runs_only_that_node():
    data = registry() | {"prep": {"type": "dataset", "datasets": [{"dataset_name": "iris"}]}, "t": node_train("prep")}
    assert _plan(data, node="prep").ordered == ["prep"]
    assert _plan(data).sequence == ["t"]  # dataset nodes are never scheduled on their own


def test_unknown_target_node_suggests():
    with pytest.raises(RecipeError, match="unknown node 'trian'.*did you mean 'train'"):
        _plan(registry() | {"train": node_train()}, node="trian")


def test_dependency_cycle_is_rejected():
    with pytest.raises(RecipeError, match="dependency cycle: a -> b -> a"):
        _plan(registry() | {"a": node_train(depends_on=["b"]), "b": node_train(depends_on=["a"])})


def test_reference_cycle_between_dataset_nodes_is_rejected():
    data = registry() | {
        "a": {"type": "dataset", "datasets": [{"dataset_name": "b"}]},
        "b": {"type": "dataset", "datasets": [{"dataset_name": "a"}]},
        "t": node_train("a"),
    }
    with pytest.raises(RecipeError, match="dataset references form a cycle: a -> b -> a"):
        _plan(data)


def test_depends_on_cannot_name_a_dataset_node():
    data = registry() | {
        "prep": {"type": "dataset", "datasets": [{"dataset_name": "iris"}]},
        "t": node_train(depends_on=["prep"]),
    }
    with pytest.raises(RecipeError, match="t.depends_on\\[0\\].*dataset node"):
        _plan(data)


def test_unknown_dependency_suggests_and_self_dependency_rejected():
    with pytest.raises(RecipeError, match="t.depends_on\\[0\\].*did you mean 'train_a'"):
        _plan(registry() | {"train_a": node_train(), "t": node_test(depends_on=["train_b"])})
    with pytest.raises(RecipeError, match="itself"):
        _plan(registry() | {"a": node_train(depends_on=["a"])})


def test_test_must_depend_on_exactly_one_train():
    with pytest.raises(RecipeError, match="exactly one train node"):
        _plan(registry() | {"e": node_eval(), "t": node_test(depends_on=["e"])})


def test_describe_roles_and_references():
    recipe = parse_recipe(
        registry()
        | {
            "prep": {"type": "dataset", "datasets": [{"dataset_name": "iris"}]},
            "train_a": node_train("prep"),
            "test_a": node_test(depends_on=["train_a"]),
            "compare": node_eval(datasets=[{"dataset_name": "prep"}]),
        }
    )
    graph, _ = dag.validate(recipe)
    rows = dag.describe(recipe, graph, dag.plan(recipe, graph))
    assert [(r.name, r.role) for r in rows] == [
        ("train_a", "dag"),
        ("test_a", "dag"),
        ("compare", "isolated"),
        ("prep", "dataset"),
    ]
    assert rows[-1].referenced_by == ["train_a", "compare"]
