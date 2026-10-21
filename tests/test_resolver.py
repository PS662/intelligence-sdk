"""Recursive dataset resolution, consolidation and the save policy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from intelligence_sdk.adapters.loaders.csv import CsvLoader
from intelligence_sdk.adapters.loaders.library import LibraryLoader
from intelligence_sdk.adapters.loaders.saved import SavedDatasetLoader
from intelligence_sdk.adapters.local_artifacts import LocalArtifactStore
from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.recipe import parse_recipe
from intelligence_sdk.core.resolver import DatasetResolver
from tests.conftest import file_dataset, node_train, registry


class CountingCsv(CsvLoader):
    calls = 0

    def load(self, spec, name):
        type(self).calls += 1
        return super().load(spec, name)


def _resolver(data: dict):
    recipe = parse_recipe(data)
    loaders = {"file": CountingCsv(), "library": LibraryLoader(), "dataset": SavedDatasetLoader()}
    return recipe, DatasetResolver(recipe, loaders, LocalArtifactStore())


def _offer(data: dict, node: str):
    recipe, resolver = _resolver(data)
    return resolver.consolidate(node, recipe.nodes[node])


def test_registry_reference_is_resolved_once_per_run():
    CountingCsv.calls = 0
    recipe, resolver = _resolver(registry() | {"a": node_train(), "b": node_train()})
    resolver.consolidate("a", recipe.nodes["a"])
    resolver.consolidate("b", recipe.nodes["b"])
    assert CountingCsv.calls == 1


def test_single_entry_keeps_the_entry_split_and_folds():
    ds = _offer(registry(folds={"k": 5}) | {"t": node_train()}, "t")
    assert ds.split is not None and len(ds.split.test_ids) == 30
    assert ds.folds is not None and ds.folds.k == 5
    assert ds.lineage == ["iris"]


def test_dataset_node_resolves_recursively_and_saves(tmp_path):
    out = tmp_path / "prep"
    data = registry() | {
        "prep": {
            "type": "dataset",
            "datasets": [{"dataset_name": "iris"}],
            "split": {"test_size": 0.5, "stratify": True},
            "recipe_output": {"save_path": str(out)},
        },
        "t": node_train("prep"),
    }
    ds = _offer(data, "t")
    assert len(ds.split.test_ids) == 75  # the node's own split, not the registry's
    assert (out / "dataset.csv").exists() and (out / "train_ids.csv").exists()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["name"] == "prep" and meta["lineage"] == ["iris"] and len(meta["split"]["test_ids"]) == 75


def test_concat_requires_unique_ids_unless_ignored(iris_path, tmp_path):
    extra = tmp_path / "extra.csv"
    frame = pd.read_csv(iris_path).head(10)
    frame.insert(0, "sample_id", range(1000, 1010))
    frame.to_csv(extra, index=False)
    node = {
        "type": "dataset",
        "datasets": [{"dataset_name": "iris"}, {"name": "extra", **file_dataset(extra, id_column="sample_id")}],
    }
    ds = _offer(registry() | {"prep": node, "t": node_train("prep")}, "t")
    assert len(ds.frame) == 160 and ds.lineage == ["prep"] and ds.split is None

    clash = {"type": "dataset", "datasets": [{"dataset_name": "iris"}, {"name": "again", **file_dataset()}]}
    with pytest.raises(RecipeError, match="prep.datasets: 150 duplicate row id"):
        _offer(registry() | {"prep": clash, "t": node_train("prep")}, "t")
    ds = _offer(registry() | {"prep": clash | {"ignore_duplicates": True}, "t": node_train("prep")}, "t")
    assert len(ds.frame) == 150


def test_concat_rejects_different_columns(iris_path, tmp_path):
    narrow = tmp_path / "narrow.csv"
    pd.read_csv(iris_path).drop(columns=["petal_width"]).to_csv(narrow, index=False)
    node = {
        "type": "dataset",
        "datasets": [{"dataset_name": "iris"}, {"name": "n", **file_dataset(narrow)}],
        "ignore_duplicates": True,
    }
    with pytest.raises(RecipeError, match="columns differ on petal_width"):
        _offer(registry() | {"prep": node, "t": node_train("prep")}, "t")


def test_filter_runs_before_split():
    data = {
        "datasets": {"iris": file_dataset(filter={"exclude": list(range(0, 50))}, split={"test_size": 0.2})},
        "t": node_train(),
    }
    ds = _offer(data, "t")
    assert len(ds.frame) == 100 and min(ds.ids) == 50
    assert set(ds.split.train_ids + ds.split.test_ids) == set(ds.ids)


def test_root_options_act_on_the_consolidated_dataset():
    ds = _offer(
        registry()
        | {"t": node_train(filter={"include": list(range(0, 60))}, split={"test_size": 0.5, "stratify": False})},
        "t",
    )
    assert len(ds.frame) == 60 and len(ds.split.test_ids) == 30


def test_save_policy_on_existing_path(tmp_path):
    out = tmp_path / "iris_saved"
    first = registry(save_dataset={"path": str(out)}) | {"t": node_train()}
    _offer(first, "t")
    assert (out / "metadata.json").exists()

    with pytest.raises(RecipeError, match="datasets.iris.save_dataset: .*already holds a dataset; set overwrite"):
        _offer(first, "t")

    CountingCsv.calls = 0
    reused = _offer(registry(save_dataset={"path": str(out), "reuse": True}) | {"t": node_train()}, "t")
    assert CountingCsv.calls == 0 and reused.split is not None and len(reused.frame) == 150

    rebuilt = _offer(
        registry(save_dataset={"path": str(out), "overwrite": True}, filter={"include": list(range(0, 90))})
        | {"t": node_train()},
        "t",
    )
    assert len(rebuilt.frame) == 90
    assert len(pd.read_csv(out / "dataset.csv")) == 90


def test_split_reference_filter_on_saved_dataset(tmp_path):
    out = tmp_path / "saved"
    _offer(registry(save_dataset={"path": str(out)}) | {"t": node_train()}, "t")
    train_ids = pd.read_csv(out / "train_ids.csv")["row_id"].tolist()
    holdout = {"name": "h", "source": "dataset", "path": str(out), "filter": {"exclude": {"split": "train"}}}
    ds = _offer({"s": {"type": "test", "model_path": "m", "datasets": [holdout]}}, "s")
    assert len(ds.frame) == 30 and not set(ds.ids) & set(train_ids)
    assert ds.split is not None and ds.split.train_ids == []  # split follows the restriction


def test_split_reference_needs_a_recorded_split(tmp_path):
    out = tmp_path / "nosplit"
    _offer({"datasets": {"iris": file_dataset(save_dataset={"path": str(out)})}, "t": node_train()}, "t")
    holdout = {"name": "h", "source": "dataset", "path": str(out), "filter": {"exclude": {"split": "train"}}}
    with pytest.raises(RecipeError, match="has no split"):
        _offer({"s": {"type": "test", "model_path": "m", "datasets": [holdout]}}, "s")


def test_resolver_guard_against_unvalidated_cycle():
    data = registry() | {
        "a": {"type": "dataset", "datasets": [{"dataset_name": "b"}]},
        "b": {"type": "dataset", "datasets": [{"dataset_name": "a"}]},
        "t": node_train("a"),
    }
    recipe, resolver = _resolver(data)  # dag validation deliberately skipped
    with pytest.raises(RecipeError, match="circular dataset reference"):
        resolver.consolidate("t", recipe.nodes["t"])


def test_dataset_node_without_output_stays_in_memory(tmp_path):
    data = registry() | {"prep": {"type": "dataset", "datasets": [{"dataset_name": "iris"}]}, "t": node_train("prep")}
    recipe, resolver = _resolver(data)
    resolver.consolidate("t", recipe.nodes["t"])
    assert resolver.executed[0].output_dir is None and not list(Path(tmp_path).iterdir())
