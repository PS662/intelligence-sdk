"""Every bad recipe must fail at load time with a config path and, where possible, a suggestion."""
from __future__ import annotations

import pytest

from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.recipe import load_recipe, parse_recipe
from tests.conftest import file_dataset, train_block


def _err(data: dict) -> str:
    with pytest.raises(RecipeError) as info:
        parse_recipe(data)
    return str(info.value)


def test_unknown_field_names_path_and_suggests():
    msg = _err({"datasets": {"iris": file_dataset()}, "train_rf": train_block() | {"model": {"name": "knn", "parms": {}}}})
    assert "train_rf.model.parms" in msg and "did you mean 'params'" in msg


def test_unknown_model_suggests_registry_name():
    msg = _err({"datasets": {"iris": file_dataset()}, "t": train_block(model="randon_forest")})
    assert "t.model.name" in msg and "did you mean 'random_forest'" in msg


def test_unknown_section_type_suggests():
    msg = _err({"t": {"type": "trian"}})
    assert "t.type" in msg and "did you mean 'train'" in msg


def test_section_name_must_be_identifier():
    for bad in ("a/b", "../x", "has space", "1abc"):
        msg = _err({bad: train_block()})
        assert "identifiers" in msg and "datasets" in msg


def test_datasets_key_is_reserved_not_a_section():
    recipe = parse_recipe({"datasets": {"iris": file_dataset()}, "t": train_block()})
    assert list(recipe.sections) == ["t"] and "iris" in recipe.datasets


def test_dataset_ref_typo_suggests():
    msg = _err({"datasets": {"iris_1": file_dataset()}, "t": train_block(dataset_name="iris_l")})
    assert "t.datasets[0].dataset_name" in msg and "did you mean 'iris_1'" in msg


def test_test_section_requires_exactly_one_model_source():
    assert "exactly one model source" in _err({"t": {"type": "test"}})
    assert "exactly one model source" in _err({"t": {"type": "test", "model_path": "x", "depends_on": ["a"]}})


def test_multi_dataset_rejected_with_not_supported_message():
    msg = _err({"datasets": {"i": file_dataset()}, "t": train_block() | {"datasets": [{"dataset_name": "i"}, {"dataset_name": "i"}]}})
    assert "not supported yet" in msg


def test_eval_candidates_reject_list_params():
    msg = _err({
        "datasets": {"i": file_dataset()},
        "e": {"type": "eval", "datasets": [{"dataset_name": "i"}], "candidates": [{"name": "knn", "params": {"n_neighbors": [1, 2]}}]},
    })
    assert "e.candidates[0].params.n_neighbors" in msg and "train section" in msg


def test_scoring_restricted_to_known_set():
    msg = _err({"datasets": {"i": file_dataset()}, "t": train_block() | {"search": {"scoring": "acuracy"}}})
    assert "t.search.scoring" in msg and "did you mean 'accuracy'" in msg


def test_multiple_errors_reported_together():
    msg = _err({"datasets": {"iris": file_dataset()}, "t": train_block(model="nope") | {"seedz": 1}})
    assert "2 problems" in msg


def test_grid_params_detects_lists():
    recipe = parse_recipe({"datasets": {"iris": file_dataset()}, "t": train_block(params={"n_estimators": [5, 10], "max_depth": 3})})
    assert recipe.sections["t"].model.grid_params() == {"n_estimators": [5, 10]}


def test_missing_recipe_file():
    with pytest.raises(RecipeError, match="not found"):
        load_recipe("does/not/exist.yaml")


def test_block_style_yaml_round_trip(tmp_path):
    text = """
datasets:
  iris:
    source: file
    path: data/iris.csv
    target: species

train_rf:
  type: train
  datasets:
    - dataset_name: iris
  model:
    name: random_forest
    params:
      n_estimators: [20, 50]
"""
    p = tmp_path / "r.yaml"
    p.write_text(text)
    recipe = load_recipe(p)
    assert recipe.sections["train_rf"].model.params["n_estimators"] == [20, 50]
