"""Every bad recipe must fail at load time with a config path and, where possible, a suggestion."""

from __future__ import annotations

import pytest

from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.recipe import InlineEntry, RefEntry, load_recipe, parse_recipe
from tests.conftest import file_dataset, node_eval, node_test, node_train, registry


def _err(data: dict) -> str:
    with pytest.raises(RecipeError) as info:
        parse_recipe(data)
    return str(info.value)


# ------------------------------------------------------------------ shapes


def test_reference_and_inline_entries_are_distinguished():
    recipe = parse_recipe(
        registry()
        | {
            "t": node_train()
            | {
                "datasets": [
                    {"dataset_name": "iris"},
                    {"name": "extra", "source": "library", "dataset": "iris", "target": "species"},
                ]
            }
        }
    )
    entries = recipe.nodes["t"].datasets
    assert isinstance(entries[0], RefEntry) and entries[0].dataset_name == "iris"
    assert isinstance(entries[1], InlineEntry) and entries[1].name == "extra" and entries[1].spec.source == "library"


def test_dataset_options_live_on_specs_and_node_roots():
    recipe = parse_recipe(registry() | {"t": node_train(split={"test_size": 0.3}, folds={"k": 4}, seed=1)})
    assert recipe.datasets["iris"].split.test_size == 0.2
    node = recipe.nodes["t"]
    assert node.split.test_size == 0.3 and node.folds.k == 4 and node.seed == 1


def test_runtime_block_defaults_and_validation():
    recipe = parse_recipe(registry() | {"t": node_train()})
    assert recipe.runtime.backend.value == "sklearn" and recipe.runtime.artifact_store.value == "local"
    msg = _err({"runtime": {"backend": "sklearnn"}} | registry() | {"t": node_train()})
    assert "runtime.backend" in msg and "did you mean 'sklearn'" in msg


def test_outputs_are_typed_per_node():
    recipe = parse_recipe(
        registry()
        | {
            "d": {
                "type": "dataset",
                "datasets": [{"dataset_name": "iris"}],
                "recipe_output": {"save_path": "x", "reuse": True},
            },
            "t": node_train(recipe_output={"save_path": "y", "overwrite": True}),
        }
    )
    assert recipe.nodes["d"].recipe_output.reuse is True
    assert recipe.nodes["t"].recipe_output.overwrite is True
    msg = _err(registry() | {"t": node_train(recipe_output={"save_path": "y", "reuse": True})})
    assert "t.recipe_output.reuse" in msg and "unknown field" in msg


def test_inline_entry_needs_a_name_and_suggests_reference_typos():
    assert "needs a 'name'" in _err({"t": node_train() | {"datasets": [file_dataset()]}})
    msg = _err(registry() | {"t": node_train() | {"datasets": [{"dataset_nam": "iris"}]}})
    assert "did you mean 'dataset_name'" in msg


def test_block_style_yaml_round_trip(tmp_path):
    text = """
datasets:
  iris:
    source: file
    path: data/iris.csv
    target: species
    split:
      test_size: 0.2

train_rf:
  type: train
  datasets:
    - dataset_name: iris
  model:
    name: random_forest
    params:
      n_estimators: [20, 50]
  hyperparam_search:
    cv: 3
    scoring: accuracy
"""
    p = tmp_path / "r.yaml"
    p.write_text(text)
    recipe = load_recipe(p)
    assert recipe.nodes["train_rf"].model.grid_params() == {"n_estimators": [20, 50]}
    assert recipe.nodes["train_rf"].hyperparam_search.cv == 3


# ------------------------------------------------------------------ errors


def test_unknown_field_names_path_and_suggests():
    msg = _err(registry() | {"train_rf": node_train() | {"model": {"name": "knn", "parms": {}}}})
    assert "train_rf.model.parms" in msg and "did you mean 'params'" in msg


def test_unknown_field_inside_inline_entry_and_registry():
    msg = _err({"t": node_train() | {"datasets": [{"name": "d", "source": "file", "pth": "x", "target": "y"}]}})
    assert "t.datasets[0].pth" in msg and "did you mean 'path'" in msg
    msg = _err({"datasets": {"iris": file_dataset(splt={})}, "t": node_train()})
    assert "datasets.iris.splt" in msg and "did you mean 'split'" in msg


def test_unknown_model_and_source_and_library_suggest():
    assert "did you mean 'random_forest'" in _err(registry() | {"t": node_train(model="randon_forest")})
    assert "t.datasets[0].source" in _err(
        {"t": node_train() | {"datasets": [{"name": "d", "source": "files", "path": "x", "target": "y"}]}}
    )
    msg = _err(
        {"t": node_train() | {"datasets": [{"name": "d", "source": "library", "dataset": "irs", "target": "y"}]}}
    )
    assert "t.datasets[0].dataset" in msg and "did you mean 'iris'" in msg


def test_unknown_node_type_suggests():
    msg = _err({"t": {"type": "trian"}})
    assert "t.type" in msg and "did you mean 'train'" in msg


def test_names_must_be_identifiers_and_reserved_keys_are_not_nodes():
    for bad in ("a/b", "../x", "has space", "1abc"):
        assert "identifiers" in _err({bad: node_train()})
    recipe = parse_recipe(registry() | {"t": node_train()})
    assert list(recipe.nodes) == ["t"] and "iris" in recipe.datasets


def test_namespace_is_shared():
    msg = _err(registry() | {"iris": {"type": "dataset", "datasets": [{"dataset_name": "iris"}]}})
    assert "iris: name already used by datasets.iris" in msg
    msg = _err(
        registry()
        | {"t": node_train() | {"datasets": [{"name": "iris", "source": "library", "dataset": "iris", "target": "y"}]}}
    )
    assert "t.datasets[0].name" in msg and "already used" in msg


def test_unknown_reference_suggests_registry_and_dataset_nodes():
    msg = _err(
        registry() | {"prep": {"type": "dataset", "datasets": [{"dataset_name": "iris"}]}, "t": node_train("prp")}
    )
    assert "t.datasets[0].dataset_name" in msg and "did you mean 'prep'" in msg


def test_referencing_an_action_node_is_rejected():
    msg = _err(registry() | {"t": node_train(), "u": node_train("t")})
    assert "'t' is a train node; only datasets can be referenced" in msg


def test_split_reference_only_on_saved_datasets():
    ok = parse_recipe(
        {
            "s": node_test(
                model_path="m",
                datasets=[{"name": "h", "source": "dataset", "path": "p", "filter": {"exclude": {"split": "train"}}}],
            )
        }
    )
    assert ok.nodes["s"].datasets[0].spec.filter.uses_split()
    msg = _err({"t": node_train() | {"datasets": [file_dataset(name="d", filter={"exclude": {"split": "train"}})]}})
    assert "t.datasets[0].filter" in msg and "source: dataset" in msg
    msg = _err(registry() | {"t": node_train(filter={"include": {"split": "test"}})})
    assert "t.filter" in msg and "source: dataset" in msg


def test_save_policy_rejects_overwrite_and_reuse_together():
    msg = _err(
        {
            "datasets": {"iris": file_dataset(save_dataset={"path": "p", "overwrite": True, "reuse": True})},
            "t": node_train(),
        }
    )
    assert "datasets.iris.save_dataset" in msg and "at most one" in msg


def test_test_node_requires_exactly_one_model_source():
    assert "exactly one model source" in _err(registry() | {"s": node_test()})
    assert "exactly one model source" in _err(registry() | {"s": node_test(model_path="x", depends_on=["a"])})


def test_dataset_nodes_have_no_depends_on():
    msg = _err(registry() | {"d": {"type": "dataset", "datasets": [{"dataset_name": "iris"}], "depends_on": ["x"]}})
    assert "d.depends_on" in msg and "unknown field" in msg


def test_eval_candidates_reject_list_params():
    msg = _err(registry() | {"e": node_eval([{"name": "knn", "params": {"n_neighbors": [1, 2]}}])})
    assert "e.candidates[0].params.n_neighbors" in msg and "train node" in msg


def test_datasets_list_must_not_be_empty():
    assert "t.datasets: needs at least 1" in _err(registry() | {"t": node_train(datasets=[])})


def test_old_search_key_is_rejected():
    msg = _err(registry() | {"t": node_train(search={"cv": 3})})
    assert "t.search" in msg and "hyperparam_search" in msg


def test_multiple_errors_reported_together():
    msg = _err(registry() | {"t": node_train(model="nope") | {"seedz": 1}})
    assert "2 problems" in msg


def test_missing_recipe_file():
    with pytest.raises(RecipeError, match="not found"):
        load_recipe("does/not/exist.yaml")
