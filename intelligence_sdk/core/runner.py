"""Executes a validated recipe.

Two phases:

1. **resolve** - everything that can be checked without touching data: DAG validity,
   dataset references, component names, hyperparameter names, metric/model
   compatibility, file existence. Any failure raises before a single row is read.
2. **execute** - sections in topological order. Each leaves a run card in
   its run directory and a ``SectionOutput`` for its dependants.

The runner depends on the ports only; ``runtime.build_runner`` supplies the
implementations named by the recipe's ``runtime`` block.
"""

from __future__ import annotations

import itertools
import logging
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intelligence_sdk.core import dag
from intelligence_sdk.core.errors import DatasetVersionMismatch, RecipeError, RecipeValidationError, did_you_mean
from intelligence_sdk.core.filters import apply_filter, filter_to_dict
from intelligence_sdk.core.recipe import (
    DatasetRef,
    EvalSection,
    ModelSpec,
    Recipe,
    StepSpec,
    TestSection,
    TrainSection,
    parse_dataset_def,
)
from intelligence_sdk.core.registry import Preprocessor, Scoring
from intelligence_sdk.core.splits import SplitError, kfold_ids, train_test_split_ids
from intelligence_sdk.core.types import LoadedDataset, SectionOutput, SplitRecord
from intelligence_sdk.interfaces import ArtifactStore, DataLoader, Evaluator, ModelBackend

log = logging.getLogger("intelligence_sdk")

DATASET_JSON = "dataset.json"
MODEL_JSON = "model.json"
DATASET_NOTE = (
    "Dataset identity is a content hash computed by the loader (file bytes or canonical frame); "
    "a loader backed by a dataset registry such as DVC supplies its stored hash instead. "
    "A test run against a different hash is reported as leakage."
)


class Runner:
    """Composition happens outside: pass any objects satisfying the ports."""

    def __init__(
        self,
        loaders: dict[str, DataLoader],
        backend: ModelBackend,
        store: ArtifactStore,
        evaluator: Evaluator,
    ):
        self.loaders = loaders
        self.backend = backend
        self.store = store
        self.evaluator = evaluator

    # ------------------------------------------------------------------ public

    def run(
        self, recipe: Recipe, allow_dataset_mismatch: bool = False, output_dir: str | None = None
    ) -> dict[str, SectionOutput]:
        graph = dag.validated_graph(recipe)
        self.resolve(recipe)
        order = dag.execution_order(graph)
        log.info("recipe %s: %d section(s), order: %s", recipe.source, len(order), " -> ".join(order))

        outputs: dict[str, SectionOutput] = {}
        for name in order:
            section = recipe.sections[name]
            out_dir = output_dir or section.output_dir
            log.info("== [%s] %s", section.type, name)
            started = time.perf_counter()
            if isinstance(section, TrainSection):
                result = self._run_train(name, section, recipe, out_dir)
            elif isinstance(section, TestSection):
                result = self._run_test(name, section, recipe, outputs, out_dir, allow_dataset_mismatch)
            elif isinstance(section, EvalSection):
                result = self._run_eval(name, section, recipe, out_dir)
            else:  # pragma: no cover - schema guarantees the union
                raise RecipeError(name, f"unsupported section type {section.type}")
            self._write_run_json(result, time.perf_counter() - started)
            log.info("   artifacts: %s", result.run_dir)
            outputs[name] = result
        return outputs

    def resolve(self, recipe: Recipe) -> None:
        """Fail fast on everything that does not need data."""
        errors: list[RecipeError] = []
        for name, section in recipe.sections.items():
            if isinstance(section, TrainSection):
                self._check_components(name, section.preprocessing, [section.model], errors, allow_lists=True)
                kind = self.backend.estimator_kind(section.model.name)
                if section.search.scoring.kind != kind:
                    errors.append(
                        _metric_mismatch(f"{name}.search.scoring", section.search.scoring, section.model, kind)
                    )
                self._check_dataset_ref(name, section.datasets[0], recipe, errors)
            elif isinstance(section, EvalSection):
                self._check_components(name, section.preprocessing, section.candidates, errors, allow_lists=False)
                for ci, cand in enumerate(section.candidates):
                    kind = self.backend.estimator_kind(cand.name)
                    for mi, metric in enumerate(section.metrics or []):
                        if metric.kind != kind:
                            errors.append(
                                _metric_mismatch(f"{name}.metrics[{mi}]", metric, cand, kind, f"candidates[{ci}]")
                            )
                self._check_dataset_ref(name, section.datasets[0], recipe, errors)
            elif isinstance(section, TestSection):
                if section.model_path is not None:
                    model_dir = self.store.resolve(section.model_path)
                    if not (model_dir / MODEL_JSON).exists():
                        errors.append(
                            RecipeError(
                                f"{name}.model_path", f"no trained model at {section.model_path} (missing {MODEL_JSON})"
                            )
                        )
                if section.datasets:
                    self._check_dataset_ref(name, section.datasets[0], recipe, errors)
        if errors:
            raise RecipeValidationError(errors)

    # --------------------------------------------------------------- resolve

    def _check_components(
        self,
        name: str,
        preprocessing: list[StepSpec],
        models: list[ModelSpec],
        errors: list[RecipeError],
        allow_lists: bool,
    ) -> None:
        for i, step in enumerate(preprocessing):
            valid = self.backend.step_param_names(step.step)
            for key in step.params:
                if key not in valid:
                    errors.append(
                        RecipeError(
                            f"{name}.preprocessing[{i}].params.{key}",
                            f"not a parameter of {step.step.value}; valid: {', '.join(sorted(valid))}",
                            did_you_mean(key, valid),
                        )
                    )
        for ci, model in enumerate(models):
            valid = self.backend.param_names(model.name)
            prefix = f"{name}.model.params" if len(models) == 1 and allow_lists else f"{name}.candidates[{ci}].params"
            for key, value in model.params.items():
                if key not in valid:
                    errors.append(
                        RecipeError(
                            f"{prefix}.{key}",
                            f"not a hyperparameter of {model.name.value}; valid: {', '.join(sorted(valid))}",
                            did_you_mean(key, valid),
                        )
                    )
                elif isinstance(value, list) and len(value) == 0:
                    errors.append(RecipeError(f"{prefix}.{key}", "search list is empty"))

    def _check_dataset_ref(self, name: str, ref: DatasetRef, recipe: Recipe, errors: list[RecipeError]) -> None:
        path = f"{name}.datasets[0]"
        try:
            definition = recipe.resolve_dataset(ref, path)
        except RecipeError as exc:
            errors.append(exc)
            return
        if definition.source == "file" and not Path(definition.path).exists():
            where = f"datasets.{ref.dataset_name}.path" if ref.dataset_name else f"{path}.dataset.path"
            errors.append(RecipeError(where, f"dataset file not found: {definition.path}"))
        for spec, where in ((definition.filter, f"{path} definition filter"), (ref.filter, f"{path}.filter")):
            if spec is None:
                continue
            for key in ("include", "exclude"):
                value = getattr(spec, key)
                if isinstance(value, str) and not Path(value).exists():
                    errors.append(RecipeError(f"{where}.{key}", f"id file not found: {value}"))

    # --------------------------------------------------------------- shared

    def _load(self, name: str, ref: DatasetRef, recipe: Recipe, run_dir: Path) -> tuple[LoadedDataset, dict[str, Any]]:
        """Shared load path for every section type: resolve → definition filter (loader)
        → reference filter → optional materialization."""
        path = f"{name}.datasets[0]"
        definition = recipe.resolve_dataset(ref, path)
        dataset = self.loaders[definition.source].load(definition)
        dataset.frame = apply_filter(dataset.frame, ref.filter, f"{path}.filter")
        if ref.save:
            self.store.write_csv(run_dir, "data/materialized.csv", dataset.frame)
        record = {
            "version": dataset.version.to_dict(),
            "definition": definition.model_dump(mode="json", exclude_none=True),
            "reference_filter": filter_to_dict(ref.filter),
            "n_rows_after_filters": int(len(dataset.frame)),
        }
        log.info(
            "   dataset %s (%s) hash=%s rows=%d",
            dataset.version.locator,
            dataset.version.source,
            dataset.version.hash[:12],
            len(dataset.frame),
        )
        return dataset, record

    def _check_frame(
        self, name: str, dataset: LoadedDataset, definition_target: str, preprocessing: list[StepSpec]
    ) -> None:
        frame = dataset.frame
        if dataset.target is None:
            raise RecipeError(
                f"{name}.datasets[0]",
                f"target column '{definition_target}' not in dataset; columns: {', '.join(map(str, frame.columns))}",
                did_you_mean(definition_target, map(str, frame.columns)),
            )
        non_numeric = [c for c in dataset.feature_columns if not pd.api.types.is_numeric_dtype(frame[c])]
        if non_numeric:
            raise RecipeError(
                f"{name}.datasets[0]",
                f"non-numeric feature column(s): {', '.join(map(str, non_numeric))}; only numeric features are supported - drop or encode them before loading",
            )
        if frame[dataset.target].isna().any():
            raise RecipeError(f"{name}.datasets[0]", f"target column '{dataset.target}' has missing values")
        nan_cols = [c for c in dataset.feature_columns if frame[c].isna().any()]
        has_imputer = any(step.step == Preprocessor.SIMPLE_IMPUTER for step in preprocessing)
        if nan_cols and not has_imputer:
            raise RecipeError(
                f"{name}.preprocessing",
                f"feature column(s) with missing values: {', '.join(map(str, nan_cols))}; add a 'simple_imputer' step or filter those rows",
            )

    def _fit_predict_score(
        self,
        section,
        model: ModelSpec,
        params: dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        train_ids,
        test_ids,
        metrics: list[Scoring] | None,
    ):
        est = self.backend.build(section.preprocessing, model.name, params, section.seed)
        est = self.backend.fit(est, X.loc[train_ids], y.loc[train_ids])
        pred = self.backend.predict(est, X.loc[test_ids])
        return est, pred, self.evaluator.evaluate(y.loc[test_ids].to_numpy(), pred, metrics)

    # ---------------------------------------------------------------- train

    def _run_train(self, name: str, section: TrainSection, recipe: Recipe, out_dir: str) -> SectionOutput:
        run_dir = self.store.new_run(name, out_dir)
        ref = section.datasets[0]
        dataset, record = self._load(name, ref, recipe, run_dir)
        self._check_frame(name, dataset, recipe.resolve_dataset(ref, name).target, section.preprocessing)
        X, y = dataset.X(), dataset.y()

        try:
            train_ids, test_ids = train_test_split_ids(
                dataset.ids.to_numpy(), y.to_numpy(), section.split.test_size, section.split.stratify, section.seed
            )
        except SplitError as exc:
            raise RecipeError(f"{name}.split", str(exc)) from exc
        split = SplitRecord(train_ids, test_ids, section.split.test_size, section.split.stratify, section.seed)
        log.info(
            "   split: %d train / %d holdout (stratify=%s, seed=%d)",
            len(train_ids),
            len(test_ids),
            section.split.stratify,
            section.seed,
        )

        grid = _expand_grid(section.model.params)
        search_summary: dict[str, Any] = {"n_candidates": len(grid), "scoring": section.search.scoring.value}
        best_params = grid[0]
        if len(grid) > 1:
            best_params, ranked = self._grid_search(name, section, X, y, train_ids, grid, run_dir)
            search_summary.update({"cv": section.search.cv, "best_params": best_params, "ranking": ranked})

        est = self.backend.build(section.preprocessing, section.model.name, best_params, section.seed)
        est = self.backend.fit(est, X.loc[train_ids], y.loc[train_ids])
        pred = self.backend.predict(est, X.loc[test_ids])
        holdout = self.evaluator.evaluate(y.loc[test_ids].to_numpy(), pred, None)
        log.info("   holdout: %s", _fmt_metrics(holdout))

        model_meta = {
            "section": name,
            "model": section.model.name.value,
            "params": best_params,
            "preprocessing": [s.model_dump(mode="json") for s in section.preprocessing],
            "features": dataset.feature_columns,
            "target": dataset.target,
            "result_type": pred.result_type.value,
            "seed": section.seed,
        }
        self.backend.save(est, run_dir, model_meta)
        dataset_json = {
            "_note": DATASET_NOTE,
            **record,
            "id_column": getattr(recipe.resolve_dataset(ref, name), "id_column", None),
            "target": dataset.target,
            "features": dataset.feature_columns,
            "split": split.to_dict(),
        }
        self.store.write_json(run_dir, DATASET_JSON, dataset_json)
        metrics = {"holdout": holdout, "search": search_summary}
        self.store.write_json(run_dir, "metrics.json", metrics)
        self.store.write_csv(run_dir, "holdout_predictions.csv", pred.frame.assign(truth=y.loc[test_ids]))
        self._write_resolved(run_dir, name, section, recipe)
        self.store.set_latest(name, out_dir, run_dir)
        return SectionOutput(
            name=name, type="train", run_dir=run_dir, model_dir=run_dir, dataset_json=dataset_json, metrics=metrics
        )

    def _grid_search(
        self, name, section: TrainSection, X, y, train_ids, grid, run_dir
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scoring = section.search.scoring
        try:
            folds = kfold_ids(
                np.asarray(train_ids),
                y.loc[train_ids].to_numpy(),
                section.search.cv,
                section.split.stratify,
                section.seed,
            )
        except SplitError as exc:
            raise RecipeError(f"{name}.search.cv", str(exc)) from exc
        log.info("   grid search: %d candidates x %d-fold CV on %s", len(grid), len(folds), scoring.value)
        rows = []
        for i, params in enumerate(grid):
            scores = []
            for tr, te in folds:
                _, _, m = self._fit_predict_score(section, section.model, params, X, y, tr, te, [scoring])
                scores.append(m[scoring.value])
            mean, std = float(np.mean(scores)), float(np.std(scores))
            cand_dir = run_dir / "candidates" / f"{i:03d}"
            self.store.write_json(cand_dir, "params.json", params)
            self.store.write_json(
                cand_dir, "metrics.json", {"scoring": scoring.value, "fold_scores": scores, "mean": mean, "std": std}
            )
            rows.append({"candidate": i, "params": params, "mean": mean, "std": std})
            log.info("     [%03d] %s -> %.4f ± %.4f", i, params, mean, std)
        rows.sort(key=lambda r: r["mean"], reverse=scoring.higher_is_better)
        for rank, row in enumerate(rows):
            row["rank"] = rank + 1
        table = pd.DataFrame(
            [
                {
                    "rank": r["rank"],
                    "candidate": r["candidate"],
                    f"mean_{scoring.value}": r["mean"],
                    "std": r["std"],
                    **{f"param_{k}": v for k, v in r["params"].items()},
                }
                for r in rows
            ]
        )
        self.store.write_csv(run_dir, "search_results.csv", table)
        best = rows[0]
        log.info("   BEST: candidate %03d %s (%s=%.4f)", best["candidate"], best["params"], scoring.value, best["mean"])
        return best["params"], rows

    # ----------------------------------------------------------------- test

    def _run_test(
        self,
        name: str,
        section: TestSection,
        recipe: Recipe,
        outputs: dict[str, SectionOutput],
        out_dir: str,
        allow_mismatch: bool,
    ) -> SectionOutput:
        run_dir = self.store.new_run(name, out_dir)
        if section.model_path is not None:
            model_dir = self.store.resolve(section.model_path)
        else:
            dep = dag.explicit_train_dependency(recipe, name)
            model_dir = outputs[dep].model_dir
        model_meta = self.store.read_json(model_dir / MODEL_JSON)
        trained = self.store.read_json(model_dir / DATASET_JSON)
        recorded_hash = trained["version"]["hash"]
        log.info("   model: %s (%s) from %s", model_meta["model"], model_meta["result_type"], model_dir)

        inherited = section.datasets is None
        if inherited:
            definition = parse_dataset_def(trained["definition"], f"{model_dir / DATASET_JSON}:definition")
            dataset = self.loaders[definition.source].load(definition)
            ref_filter = trained.get("reference_filter")
            if ref_filter:
                from intelligence_sdk.core.recipe import FilterSpec

                dataset.frame = apply_filter(dataset.frame, FilterSpec(**ref_filter), f"{name} recorded filter")
            rows = section.rows or "holdout"
            hash_match = dataset.version.hash == recorded_hash
            if not hash_match:
                msg = (
                    f"dataset {dataset.version.locator} hash {dataset.version.hash[:12]} != hash {recorded_hash[:12]} the model was trained on; "
                    "the recorded holdout no longer identifies the same rows (leakage risk)"
                )
                if not allow_mismatch:
                    raise DatasetVersionMismatch(
                        name,
                        msg
                        + ". Re-run with --allow-dataset-mismatch to score anyway, or give the test its own datasets block with rows: all",
                    )
                log.warning("   %s (continuing: --allow-dataset-mismatch)", msg)
            record = {
                "version": dataset.version.to_dict(),
                "definition": definition.model_dump(mode="json", exclude_none=True),
                "reference_filter": ref_filter,
            }
        else:
            dataset, record = self._load(name, section.datasets[0], recipe, run_dir)
            rows = section.rows or "all"
            hash_match = dataset.version.hash == recorded_hash
            if rows == "holdout" and not hash_match:
                raise RecipeError(
                    f"{name}.rows",
                    "holdout is only meaningful on the dataset the model was trained on (hash differs); use rows: all",
                )

        if rows == "holdout":
            wanted = trained["split"]["test_ids"]
            missing = [i for i in wanted if i not in dataset.frame.index]
            if missing:
                raise RecipeError(
                    name,
                    f"{len(missing)} recorded holdout row id(s) are not in the loaded dataset (e.g. {missing[:5]})",
                )
            frame = dataset.frame.loc[wanted]
        else:
            frame = dataset.frame
        features = model_meta["features"]
        absent = [c for c in features if c not in frame.columns]
        if absent:
            raise RecipeError(
                f"{name}.datasets[0]",
                f"dataset lacks feature column(s) the model needs: {', '.join(absent)}",
                did_you_mean(absent[0], map(str, frame.columns)),
            )
        log.info("   scoring %d rows (%s)", len(frame), rows)

        est = self.backend.load(model_dir)
        pred = self.backend.predict(est, frame[features])
        target = model_meta["target"]
        metrics: dict[str, Any] = {}
        if target in frame.columns:
            metrics = self.evaluator.evaluate(frame[target].to_numpy(), pred, None)
            log.info("   metrics: %s", _fmt_metrics(metrics))
            self.store.write_csv(run_dir, "predictions.csv", pred.frame.assign(truth=frame[target]))
        else:
            log.info("   no '%s' column present: predictions only, no metrics", target)
            self.store.write_csv(run_dir, "predictions.csv", pred.frame)
        self.store.write_json(run_dir, "metrics.json", metrics)
        dataset_json = {
            "_note": DATASET_NOTE,
            "scored": {
                **record,
                "rows": rows,
                "n_rows": int(len(frame)),
                "row_ids": frame.index.tolist(),
                "inherited": inherited,
            },
            "model": {"dir": str(model_dir), "trained_on": trained["version"], "hash_match": hash_match},
        }
        self.store.write_json(run_dir, DATASET_JSON, dataset_json)
        self._write_resolved(
            run_dir, name, section, recipe, extra={"resolved_model_dir": str(model_dir), "resolved_rows": rows}
        )
        self.store.set_latest(name, out_dir, run_dir)
        return SectionOutput(
            name=name, type="test", run_dir=run_dir, model_dir=model_dir, dataset_json=dataset_json, metrics=metrics
        )

    # ----------------------------------------------------------------- eval

    def _run_eval(self, name: str, section: EvalSection, recipe: Recipe, out_dir: str) -> SectionOutput:
        run_dir = self.store.new_run(name, out_dir)
        ref = section.datasets[0]
        dataset, record = self._load(name, ref, recipe, run_dir)
        self._check_frame(name, dataset, recipe.resolve_dataset(ref, name).target, section.preprocessing)
        X, y = dataset.X(), dataset.y()
        try:
            folds = kfold_ids(dataset.ids.to_numpy(), y.to_numpy(), section.folds, section.stratify, section.seed)
        except SplitError as exc:
            raise RecipeError(f"{name}.folds", str(exc)) from exc
        log.info(
            "   %d-fold CV (stratify=%s) over %d candidate(s)", len(folds), section.stratify, len(section.candidates)
        )

        rows = []
        for ci, cand in enumerate(section.candidates):
            for fi, (tr, te) in enumerate(folds):
                _, _, m = self._fit_predict_score(section, cand, cand.params, X, y, tr, te, section.metrics)
                scalars = {k: v for k, v in m.items() if isinstance(v, (int, float))}
                rows.append(
                    {
                        "candidate": f"{ci}:{cand.name.value}",
                        "model": cand.name.value,
                        "params": str(cand.params),
                        "fold": fi,
                        "n_train": len(tr),
                        "n_test": len(te),
                        **scalars,
                    }
                )
        report = pd.DataFrame(rows)
        metric_cols = [
            c
            for c in report.columns
            if c not in ("candidate", "model", "params", "fold", "n_train", "n_test", "n_samples")
        ]
        summary = report.groupby("candidate", sort=False)[metric_cols].agg(["mean", "std"])
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary = summary.reset_index()
        primary = metric_cols[0]
        summary = summary.sort_values(f"{primary}_mean", ascending=False).reset_index(drop=True)
        summary.insert(0, "rank", range(1, len(summary) + 1))

        self.store.write_csv(run_dir, "eval_report.csv", report)
        self.store.write_csv(run_dir, "eval_summary.csv", summary)
        self.store.write_json(
            run_dir,
            "folds.json",
            {
                "folds": section.folds,
                "stratify": section.stratify,
                "seed": section.seed,
                "fold_test_ids": [te for _, te in folds],
            },
        )
        self.store.write_json(
            run_dir,
            DATASET_JSON,
            {"_note": DATASET_NOTE, **record, "target": dataset.target, "features": dataset.feature_columns},
        )
        metrics = {"ranked_by": primary, "summary": summary.to_dict(orient="records")}
        self.store.write_json(run_dir, "metrics.json", metrics)
        self._write_resolved(run_dir, name, section, recipe)
        self.store.set_latest(name, out_dir, run_dir)
        log.info("   summary (ranked by %s):\n%s", primary, summary.to_string(index=False))
        return SectionOutput(name=name, type="eval", run_dir=run_dir, dataset_json=record, metrics=metrics)

    # -------------------------------------------------------------- helpers

    def _write_resolved(
        self, run_dir: Path, name: str, section, recipe: Recipe, extra: dict[str, Any] | None = None
    ) -> None:
        block = section.model_dump(mode="json", exclude_none=True)
        refs = getattr(section, "datasets", None) or []
        for i, ref in enumerate(refs):
            block["datasets"][i]["resolved"] = recipe.resolve_dataset(ref, name).model_dump(
                mode="json", exclude_none=True
            )
        payload = {"recipe": recipe.source, "section": name, **(extra or {}), "block": block}
        self.store.write_yaml(run_dir, "resolved_recipe.yaml", payload)

    def _write_run_json(self, result: SectionOutput, seconds: float) -> None:
        import sklearn  # version stamp only; the runner itself never calls into it

        self.store.write_json(
            result.run_dir,
            "run.json",
            {
                "run_id": result.run_dir.name,
                "section": result.name,
                "type": result.type,
                "finished_at": datetime.now(UTC).isoformat(),
                "duration_s": round(seconds, 3),
                "versions": {
                    "python": platform.python_version(),
                    "scikit-learn": sklearn.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                },
            },
        )


def _expand_grid(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Scalar values are fixed; list values are searched (cartesian product)."""
    keys = list(params)
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)] or [{}]


def _metric_mismatch(path: str, metric: Scoring, model: ModelSpec, kind, where: str = "model") -> RecipeError:
    return RecipeError(
        path, f"{metric.value} is a {metric.kind.value} metric; {where} {model.name.value} is a {kind.value}"
    )


def _fmt_metrics(metrics: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
