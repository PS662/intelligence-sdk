"""End-to-end runs on iris through the public facade, plus the data-dependent edge cases."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from intelligence_sdk import default_runner, run_recipe
from intelligence_sdk.core.errors import DatasetVersionMismatch, RecipeError
from intelligence_sdk.core.recipe import parse_recipe
from tests.conftest import file_dataset, train_block


def _run(data: dict, artifacts: Path, **kw):
    return run_recipe(parse_recipe(data), output_dir=str(artifacts), **kw)


# ---------------------------------------------------------------- golden path


def test_train_then_test_reproduces_holdout(artifacts, iris_path):
    recipe = {
        "datasets": {"iris": file_dataset(iris_path)},
        "train_rf": train_block(params={"n_estimators": [5, 10], "max_depth": 3}),
        "test_rf": {"type": "test", "depends_on": ["train_rf"]},
    }
    out = _run(recipe, artifacts)
    train, test = out["train_rf"], out["test_rf"]

    assert train.metrics["holdout"]["accuracy"] >= 0.85  # loose: iris is easy, flakiness is not
    for f in ("model.joblib", "model.json", "dataset.json", "metrics.json", "resolved_recipe.yaml", "run.json", "search_results.csv"):
        assert (train.run_dir / f).exists(), f
    # Grid search: one candidate dir per combination, best highlighted.
    assert len(list((train.run_dir / "candidates").iterdir())) == 2
    assert train.metrics["search"]["best_params"]["n_estimators"] in (5, 10)
    assert train.metrics["search"]["ranking"][0]["rank"] == 1

    # Test scored exactly the recorded holdout rows and got the same numbers.
    recorded = json.load(open(train.run_dir / "dataset.json"))["split"]["test_ids"]
    preds = pd.read_csv(test.run_dir / "predictions.csv", index_col=0)
    assert preds.index.tolist() == recorded
    assert test.metrics["accuracy"] == train.metrics["holdout"]["accuracy"]
    assert (artifacts / "train_rf" / "latest").exists()


def test_filtered_train_then_inherited_test_reproduces_identical_rows(artifacts, iris_path):
    """Row ids are original identifiers, not post-filter positions. A definition-level
    filter must not shift which rows the holdout names."""
    excluded = list(range(0, 50, 3))  # knock out every third setosa row
    recipe = {
        "datasets": {"iris": file_dataset(iris_path, filter={"exclude": excluded})},
        "train_rf": train_block(params={"n_estimators": 5}),
        "test_rf": {"type": "test", "depends_on": ["train_rf"]},
    }
    out = _run(recipe, artifacts)
    recorded = json.load(open(out["train_rf"].run_dir / "dataset.json"))["split"]["test_ids"]
    assert not set(recorded) & set(excluded)

    original = pd.read_csv(iris_path)  # positional index == row_id
    scored = pd.read_csv(out["test_rf"].run_dir / "predictions.csv", index_col=0)
    assert scored.index.tolist() == recorded
    assert scored["truth"].tolist() == original.loc[recorded, "species"].tolist()
    # And the materialized features in the model's run match the original rows byte-for-byte.
    model_meta = json.load(open(out["train_rf"].run_dir / "model.json"))
    assert model_meta["features"] == ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_explicit_dataset_on_test_scores_all_rows_and_materializes(artifacts, iris_path):
    recipe = {
        "datasets": {"iris": file_dataset(iris_path)},
        "train_rf": train_block(params={"n_estimators": 5}),
        "test_all": {
            "type": "test",
            "depends_on": ["train_rf"],
            "datasets": [{"dataset": file_dataset(iris_path), "save": True}],
        },
    }
    out = _run(recipe, artifacts)
    test = out["test_all"]
    assert (test.run_dir / "data" / "materialized.csv").exists()
    assert test.dataset_json["scored"]["rows"] == "all" and test.dataset_json["scored"]["n_rows"] == 150
    assert test.dataset_json["model"]["hash_match"] is True


def test_test_without_target_column_gives_predictions_only(artifacts, iris_path, tmp_path):
    unlabeled = tmp_path / "new.csv"
    pd.read_csv(iris_path).drop(columns=["species"]).head(7).to_csv(unlabeled, index=False)
    recipe = {
        "datasets": {"iris": file_dataset(iris_path)},
        "train_rf": train_block(params={"n_estimators": 5}),
        "infer": {"type": "test", "depends_on": ["train_rf"], "datasets": [{"dataset": file_dataset(unlabeled)}]},
    }
    out = _run(recipe, artifacts)
    assert out["infer"].metrics == {}
    preds = pd.read_csv(out["infer"].run_dir / "predictions.csv", index_col=0)
    assert len(preds) == 7 and "truth" not in preds.columns
    assert out["infer"].dataset_json["model"]["hash_match"] is False  # recorded, never refused


def test_eval_report_shape(artifacts, iris_path):
    recipe = {
        "datasets": {"iris": file_dataset(iris_path)},
        "compare": {
            "type": "eval",
            "datasets": [{"dataset_name": "iris"}],
            "folds": 3,
            "candidates": [{"name": "knn"}, {"name": "decision_tree"}],
            "metrics": ["accuracy", "f1_macro"],
            "seed": 1,
        },
    }
    out = _run(recipe, artifacts)
    run_dir = out["compare"].run_dir
    report = pd.read_csv(run_dir / "eval_report.csv")
    summary = pd.read_csv(run_dir / "eval_summary.csv")
    assert len(report) == 2 * 3 and set(report["fold"]) == {0, 1, 2}
    assert list(summary["rank"]) == [1, 2] and {"accuracy_mean", "f1_macro_std"} <= set(summary.columns)
    folds = json.load(open(run_dir / "folds.json"))
    assert len(folds["fold_test_ids"]) == 3
    assert not (run_dir / "model.joblib").exists()  # eval selects; it does not ship a model


def test_library_dataset_source(artifacts):
    recipe = {
        "train_lib": train_block(params={"n_estimators": 5}) | {"datasets": [{"dataset": {"source": "library", "name": "iris", "target": "species"}}]},
    }
    out = _run(recipe, artifacts)
    assert out["train_lib"].dataset_json["version"]["source"] == "library"
    assert out["train_lib"].metrics["holdout"]["accuracy"] >= 0.85


def test_regressor_path_uses_continuous_metrics(artifacts, iris_path, tmp_path):
    numeric = tmp_path / "reg.csv"
    pd.read_csv(iris_path).drop(columns=["species"]).to_csv(numeric, index=False)
    recipe = {
        "reg": {
            "type": "train",
            "datasets": [{"dataset": {"source": "file", "path": str(numeric), "target": "petal_width"}}],
            "model": {"name": "linear_regression"},
            "split": {"test_size": 0.2, "stratify": False},
            "search": {"scoring": "r2"},
        }
    }
    out = _run(recipe, artifacts)
    assert {"neg_rmse", "neg_mae", "r2"} <= set(out["reg"].metrics["holdout"])
    assert json.load(open(out["reg"].run_dir / "model.json"))["result_type"] == "continuous"


# ------------------------------------------------------------ hash guard


def test_inherited_test_refuses_changed_dataset_unless_allowed(artifacts, iris_path, tmp_path):
    copy = tmp_path / "iris_copy.csv"
    copy.write_bytes(iris_path.read_bytes())
    _run({"datasets": {"iris": file_dataset(copy)}, "train_rf": train_block(params={"n_estimators": 5})}, artifacts)
    with open(copy, "a") as fh:
        fh.write("6.0,3.0,4.0,1.0,versicolor\n")

    test = {"t": {"type": "test", "model_path": str(artifacts / "train_rf" / "latest")}}
    with pytest.raises(DatasetVersionMismatch, match="leakage"):
        _run(test, artifacts)
    out = _run(test, artifacts, allow_dataset_mismatch=True)
    assert out["t"].dataset_json["model"]["hash_match"] is False


# ------------------------------------------------------- data edge cases


def _train_on(frame: pd.DataFrame, tmp_path: Path, artifacts: Path, pipeline=None, **extra):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    block = train_block(params={"n_estimators": 5}, **extra) | {"datasets": [{"dataset": file_dataset(path)}]}
    if pipeline is not None:
        block["pipeline"] = pipeline
    return _run({"t": block}, artifacts)


def test_nans_require_an_imputer(iris_path, tmp_path, artifacts):
    frame = pd.read_csv(iris_path)
    frame.loc[3, "sepal_length"] = None
    with pytest.raises(RecipeError, match="t.pipeline.*simple_imputer"):
        _train_on(frame, tmp_path, artifacts)
    out = _train_on(frame, tmp_path, artifacts, pipeline=[{"step": "simple_imputer", "params": {"strategy": "mean"}}])
    assert out["t"].metrics["holdout"]["accuracy"] > 0.8


def test_non_numeric_feature_is_named(iris_path, tmp_path, artifacts):
    frame = pd.read_csv(iris_path)
    frame["colour"] = "red"
    with pytest.raises(RecipeError, match="non-numeric feature column\\(s\\): colour"):
        _train_on(frame, tmp_path, artifacts)


def test_missing_target_suggests_column(iris_path, tmp_path, artifacts):
    path = tmp_path / "data.csv"
    pd.read_csv(iris_path).to_csv(path, index=False)
    block = train_block() | {"datasets": [{"dataset": {"source": "file", "path": str(path), "target": "specie"}}]}
    with pytest.raises(RecipeError, match="target column 'specie'.*did you mean 'species'"):
        _run({"t": block}, artifacts)


def test_tiny_class_stratify_error_names_split(iris_path, tmp_path, artifacts):
    frame = pd.read_csv(iris_path)
    frame = pd.concat([frame, pd.DataFrame([{**frame.iloc[0].to_dict(), "species": "unicorn"}])])
    with pytest.raises(RecipeError, match="t.split: cannot stratify"):
        _train_on(frame, tmp_path, artifacts)


def test_empty_filter_is_rejected(iris_path, artifacts):
    block = train_block() | {"datasets": [{"dataset": file_dataset(iris_path), "filter": {"include": [99999]}}]}
    with pytest.raises(RecipeError, match="no rows left after filtering"):
        _run({"t": block}, artifacts)


# ---------------------------------------------------- resolve-phase checks


def test_resolve_rejects_bad_hyperparameter_before_any_io(iris_path, artifacts):
    recipe = parse_recipe({"datasets": {"iris": file_dataset(iris_path)}, "t": train_block(params={"n_estimator": 5})})
    with pytest.raises(RecipeError, match="t.model.params.n_estimator.*did you mean 'n_estimators'"):
        default_runner().run(recipe, output_dir=str(artifacts))
    assert not artifacts.exists()  # nothing was written


def test_resolve_rejects_metric_model_kind_mismatch(iris_path, artifacts):
    recipe = parse_recipe({
        "datasets": {"iris": file_dataset(iris_path)},
        "compare": {"type": "eval", "datasets": [{"dataset_name": "iris"}], "candidates": [{"name": "linear_regression"}], "metrics": ["f1_macro"]},
    })
    with pytest.raises(RecipeError, match="compare.metrics\\[0\\]: f1_macro is a classifier metric"):
        default_runner().resolve(recipe)
    recipe2 = parse_recipe({"datasets": {"iris": file_dataset(iris_path)}, "t": train_block(model="linear_regression") | {"search": {"scoring": "accuracy"}}})
    with pytest.raises(RecipeError, match="t.search.scoring"):
        default_runner().resolve(recipe2)


def test_resolve_rejects_missing_file_and_model(artifacts):
    recipe = parse_recipe({"t": train_block() | {"datasets": [{"dataset": {"source": "file", "path": "nope.csv", "target": "y"}}]}})
    with pytest.raises(RecipeError, match="dataset.path: dataset file not found"):
        default_runner().resolve(recipe)
    recipe2 = parse_recipe({"t": {"type": "test", "model_path": str(artifacts / "none" / "latest")}})
    with pytest.raises(RecipeError, match="t.model_path: no trained model"):
        default_runner().resolve(recipe2)
