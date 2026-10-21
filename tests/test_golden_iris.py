"""End-to-end runs on iris through the public facade, plus the data-dependent edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from intelligence_sdk import build_runner, run_recipe
from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.recipe import parse_recipe
from tests.conftest import file_dataset, node_eval, node_test, node_train, registry


def _run(data: dict, node: str | None = None):
    return run_recipe(parse_recipe(data), node=node)


def _out(path: Path, name: str, **extra) -> dict:
    return {"recipe_output": {"save_path": str(path / name), "overwrite": True, **extra}}


# ---------------------------------------------------------------- golden path


def test_train_saves_model_and_search_artifacts(artifacts):
    recipe = registry() | {
        "train_rf": node_train(params={"n_estimators": [5, 10], "max_depth": 3}, **_out(artifacts, "train_rf"))
    }
    out = _run(recipe)
    train = out["train_rf"]
    assert train.model_meta["fitted_on"] == {"dataset": "train_rf", "rows": 120, "split": "train"}
    for f in (
        "model.joblib",
        "model.json",
        "metrics.json",
        "search_results.csv",
        "resolved_recipe.yaml",
        "dataset_metadata.json",
        "run.json",
    ):
        assert (train.output_dir / f).exists(), f
    assert len(list((train.output_dir / "candidates").iterdir())) == 2
    assert train.search["best_params"]["n_estimators"] in (5, 10) and train.search["ranking"][0]["rank"] == 1
    meta = json.loads((train.output_dir / "dataset_metadata.json").read_text())
    assert len(meta["split"]["train_ids"]) == 120 and len(meta["split"]["test_ids"]) == 30


def test_test_scores_exactly_the_saved_test_split(artifacts, tmp_path):
    saved = tmp_path / "iris_saved"
    recipe = registry(save_dataset={"path": str(saved)}) | {
        "train_rf": node_train(params={"n_estimators": 5}, **_out(artifacts, "train_rf")),
        "test_rf": node_test(
            depends_on=["train_rf"],
            datasets=[
                {"name": "holdout", "source": "dataset", "path": str(saved), "filter": {"exclude": {"split": "train"}}}
            ],
            **_out(artifacts, "test_rf"),
        ),
    }
    out = _run(recipe)
    test = out["test_rf"]
    test_ids = pd.read_csv(saved / "test_ids.csv")["row_id"].tolist()
    assert test.predictions.index.tolist() == test_ids
    assert test.metrics["accuracy"] >= 0.85  # loose: iris is easy, flakiness is not
    preds = pd.read_csv(test.output_dir / "predictions.csv", index_col=0)
    assert "truth" in preds.columns and len(preds) == 30


def test_test_from_model_path_after_a_separate_train(artifacts, tmp_path):
    saved = tmp_path / "iris_saved"
    _run(
        registry(save_dataset={"path": str(saved)})
        | {"train_rf": node_train(params={"n_estimators": 5}, **_out(artifacts, "train_rf"))}
    )
    recipe = {
        "score": node_test(
            model_path=str(artifacts / "train_rf"), datasets=[{"name": "all", "source": "dataset", "path": str(saved)}]
        )
    }
    out = _run(recipe)
    assert len(out["score"].predictions) == 150 and out["score"].output_dir is None  # no recipe_output: nothing written


def test_test_without_target_column_gives_predictions_only(artifacts, iris_path, tmp_path):
    unlabeled = tmp_path / "new.csv"
    pd.read_csv(iris_path).drop(columns=["species"]).head(7).to_csv(unlabeled, index=False)
    recipe = registry() | {
        "train_rf": node_train(params={"n_estimators": 5}),
        "infer": node_test(
            depends_on=["train_rf"], datasets=[{"name": "new", **file_dataset(unlabeled)}], **_out(artifacts, "infer")
        ),
    }
    out = _run(recipe)
    assert out["infer"].metrics == {} and len(out["infer"].predictions) == 7
    assert "truth" not in out["infer"].predictions.columns


def test_eval_uses_dataset_folds(artifacts):
    recipe = registry(folds={"k": 3}) | {
        "compare": node_eval(
            [{"name": "knn"}, {"name": "decision_tree"}],
            folds=None,
            metrics=["accuracy", "f1_macro"],
            **_out(artifacts, "compare"),
        )
    }
    recipe["compare"].pop("folds")
    out = _run(recipe)
    report, summary = out["compare"].report, out["compare"].summary
    assert len(report) == 2 * 3 and list(summary["rank"]) == [1, 2]
    assert (out["compare"].output_dir / "eval_summary.csv").exists()
    assert not (out["compare"].output_dir / "model.joblib").exists()


def test_eval_without_folds_is_rejected_before_data(artifacts):
    recipe = registry() | {"compare": node_eval()}
    recipe["compare"].pop("folds")
    with pytest.raises(RecipeError, match="compare.folds: eval needs folds"):
        _run(recipe)


def test_dataset_node_combines_and_feeds_train_test_eval(artifacts, iris_path, tmp_path):
    extra = tmp_path / "extra.csv"
    frame = pd.read_csv(iris_path).sample(20, random_state=1)
    frame.insert(0, "sample_id", range(1000, 1020))
    frame.to_csv(extra, index=False)
    prep_dir = tmp_path / "prep"
    recipe = registry() | {
        "prep": {
            "type": "dataset",
            "datasets": [{"dataset_name": "iris"}, {"name": "extra", **file_dataset(extra, id_column="sample_id")}],
            "split": {"test_size": 0.2, "stratify": True},
            "folds": {"k": 3},
            "recipe_output": {"save_path": str(prep_dir), "overwrite": True},
        },
        "train_rf": node_train("prep", params={"n_estimators": 5}, **_out(artifacts, "train_rf")),
        "test_rf": node_test(
            depends_on=["train_rf"],
            datasets=[
                {"name": "h", "source": "dataset", "path": str(prep_dir), "filter": {"exclude": {"split": "train"}}}
            ],
        ),
        "compare": node_eval(datasets=[{"dataset_name": "prep"}], folds=None),
    }
    recipe["compare"].pop("folds")
    out = _run(recipe)
    assert set(out) == {"prep", "train_rf", "test_rf", "compare"}
    assert len(out["prep"].dataset.frame) == 170 and (prep_dir / "resolved_recipe.yaml").exists()
    fitted = out["train_rf"].model_meta["fitted_on"]["rows"]
    assert fitted + len(out["test_rf"].predictions) == 170  # train split + test split of the combined dataset
    assert len(out["compare"].report) == 3  # one candidate x 3 folds from prep


def test_node_mode_runs_only_the_dependency_chain(artifacts):
    recipe = registry() | {
        "train_rf": node_train(params={"n_estimators": 5}),
        "test_rf": node_test(depends_on=["train_rf"]),
        "compare": node_eval(**_out(artifacts, "compare")),
    }
    out = _run(recipe, node="test_rf")
    assert list(out) == ["train_rf", "test_rf"] and not (artifacts / "compare").exists()
    out = _run(recipe, node="compare")
    assert list(out) == ["compare"]


def test_node_mode_on_dataset_node_saves_it(tmp_path):
    prep_dir = tmp_path / "prep"
    recipe = registry() | {
        "prep": {
            "type": "dataset",
            "datasets": [{"dataset_name": "iris"}],
            "recipe_output": {"save_path": str(prep_dir)},
        },
        "t": node_train("prep"),
    }
    out = _run(recipe, node="prep")
    assert list(out) == ["prep"] and (prep_dir / "dataset.csv").exists() and (prep_dir / "run.json").exists()


# ------------------------------------------------------------ save policies


def test_existing_output_needs_overwrite(artifacts):
    recipe = registry() | {
        "train_rf": node_train(params={"n_estimators": 5}, recipe_output={"save_path": str(artifacts / "train_rf")})
    }
    _run(recipe)
    with pytest.raises(RecipeError, match="train_rf.recipe_output.save_path: .*already exists; set overwrite"):
        _run(recipe)


def test_library_dataset_source(artifacts):
    recipe = {
        "train_lib": node_train(params={"n_estimators": 5})
        | {
            "datasets": [{"name": "lib", "source": "library", "dataset": "iris", "target": "species"}],
            "split": {"test_size": 0.2},
        }
    }
    out = _run(recipe)
    assert out["train_lib"].dataset.lineage == ["lib"] and out["train_lib"].model_meta["fitted_on"]["rows"] == 120


def test_regressor_path_uses_continuous_metrics(artifacts, iris_path, tmp_path):
    numeric = tmp_path / "reg.csv"
    pd.read_csv(iris_path).drop(columns=["species"]).to_csv(numeric, index=False)
    recipe = {
        "reg": {
            "type": "train",
            "datasets": [
                {
                    "name": "num",
                    "source": "file",
                    "path": str(numeric),
                    "target": "petal_width",
                    "split": {"test_size": 0.2, "stratify": False},
                }
            ],
            "model": {"name": "linear_regression"},
            "hyperparam_search": {"scoring": "r2"},
            **_out(artifacts, "reg"),
        },
        "score": node_test(
            depends_on=["reg"],
            datasets=[{"name": "num2", "source": "file", "path": str(numeric), "target": "petal_width"}],
        ),
    }
    out = _run(recipe)
    assert out["reg"].model_meta["result_type"] == "continuous"
    assert {"neg_rmse", "neg_mae", "r2"} <= set(out["score"].metrics)


# ------------------------------------------------------- data edge cases


def _train_on(frame: pd.DataFrame, tmp_path: Path, preprocessing=None, **extra):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    block = node_train(params={"n_estimators": 5}, **extra) | {"datasets": [{"name": "d", **file_dataset(path)}]}
    if preprocessing is not None:
        block["preprocessing"] = preprocessing
    return _run({"t": block})


def test_nans_require_an_imputer(iris_path, tmp_path):
    frame = pd.read_csv(iris_path)
    frame.loc[3, "sepal_length"] = None
    with pytest.raises(RecipeError, match="t.preprocessing.*simple_imputer"):
        _train_on(frame, tmp_path)
    out = _train_on(frame, tmp_path, preprocessing=[{"step": "simple_imputer", "params": {"strategy": "mean"}}])
    assert out["t"].model_meta["fitted_on"]["rows"] == 150


def test_non_numeric_feature_is_named(iris_path, tmp_path):
    frame = pd.read_csv(iris_path)
    frame["colour"] = "red"
    with pytest.raises(RecipeError, match="non-numeric feature column\\(s\\): colour"):
        _train_on(frame, tmp_path)


def test_missing_target_suggests_column(iris_path, tmp_path):
    path = tmp_path / "data.csv"
    pd.read_csv(iris_path).to_csv(path, index=False)
    block = node_train() | {"datasets": [{"name": "d", "source": "file", "path": str(path), "target": "specie"}]}
    with pytest.raises(RecipeError, match="target column 'specie'.*did you mean 'species'"):
        _run({"t": block})


def test_tiny_class_stratify_error_names_the_split(iris_path, tmp_path):
    frame = pd.read_csv(iris_path)
    frame = pd.concat([frame, pd.DataFrame([{**frame.iloc[0].to_dict(), "species": "unicorn"}])])
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    block = node_train() | {"datasets": [{"name": "d", **file_dataset(path, split={"test_size": 0.2})}]}
    with pytest.raises(RecipeError, match="t.datasets\\[0\\].split: cannot stratify"):
        _run({"t": block})


# ---------------------------------------------------- resolve-phase checks


def test_resolve_rejects_bad_hyperparameter_before_any_io(artifacts):
    recipe = parse_recipe(registry() | {"t": node_train(params={"n_estimator": 5}, **_out(artifacts, "t"))})
    with pytest.raises(RecipeError, match="t.model.params.n_estimator.*did you mean 'n_estimators'"):
        build_runner(recipe.runtime).run(recipe)
    assert not artifacts.exists()


def test_resolve_rejects_metric_model_kind_mismatch():
    recipe = parse_recipe(registry() | {"compare": node_eval([{"name": "linear_regression"}], metrics=["f1_macro"])})
    with pytest.raises(RecipeError, match="compare.metrics\\[0\\]: f1_macro is a classifier metric"):
        build_runner(recipe.runtime).resolve(recipe)
    recipe2 = parse_recipe(
        registry() | {"t": node_train(model="linear_regression", hyperparam_search={"scoring": "accuracy"})}
    )
    with pytest.raises(RecipeError, match="t.hyperparam_search.scoring"):
        build_runner(recipe2.runtime).resolve(recipe2)


def test_resolve_rejects_missing_paths_unless_produced_in_recipe(tmp_path):
    recipe = parse_recipe(
        {"t": node_train() | {"datasets": [{"name": "d", "source": "file", "path": "nope.csv", "target": "y"}]}}
    )
    with pytest.raises(RecipeError, match="t.datasets\\[0\\].path: dataset file not found"):
        build_runner(recipe.runtime).resolve(recipe)
    recipe2 = parse_recipe(
        {
            "s": node_test(
                model_path=str(tmp_path / "none"),
                datasets=[{"name": "d", "source": "dataset", "path": str(tmp_path / "nodata")}],
            )
        }
    )
    with pytest.raises(RecipeError, match="2 problems"):
        build_runner(recipe2.runtime).resolve(recipe2)
    # produced later in the same recipe: allowed
    produced = tmp_path / "later"
    recipe3 = parse_recipe(
        registry(save_dataset={"path": str(produced)})
        | {
            "t": node_train(),
            "s": node_test(depends_on=["t"], datasets=[{"name": "h", "source": "dataset", "path": str(produced)}]),
        }
    )
    build_runner(recipe3.runtime).resolve(recipe3)


def test_generated_csv_matches_library_loader_columns(iris_path):
    """tools/make_iris_csv.py and the library loader must agree, so file and library
    recipes on iris see identical features."""
    from intelligence_sdk.adapters.loaders.library import LibraryLoader
    from intelligence_sdk.core.recipe import parse_dataset_spec

    lib = LibraryLoader().load(parse_dataset_spec({"source": "library", "dataset": "iris", "target": "species"}), "lib")
    csv = pd.read_csv(iris_path)
    assert list(csv.columns) == list(lib.frame.columns) and len(csv) == len(lib.frame)
