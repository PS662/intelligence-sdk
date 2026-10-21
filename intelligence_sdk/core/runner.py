"""Executes a validated recipe.

Two phases:

1. **resolve** - everything that can be checked without touching data: both graphs,
   dataset references, component and hyperparameter names, metric/model
   compatibility, file existence. Any failure raises before a row is read.
2. **execute** - the nodes of the execution plan, in order. Each node consolidates
   its datasets through one shared ``DatasetResolver``, acts, and writes its
   outputs when it has a ``recipe_output``.

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
from intelligence_sdk.core.errors import RecipeError, RecipeValidationError, did_you_mean
from intelligence_sdk.core.recipe import (
    AnyDataset,
    DatasetNode,
    EvalNode,
    FilterSpec,
    InlineEntry,
    ModelSpec,
    NodeBase,
    Recipe,
    RefEntry,
    SavedDataset,
    StepSpec,
    TestNode,
    TrainNode,
)
from intelligence_sdk.core.registry import Preprocessor, Scoring
from intelligence_sdk.core.resolver import DatasetResolver
from intelligence_sdk.core.splits import SplitError, kfold_ids
from intelligence_sdk.core.types import (
    Dataset,
    DatasetResult,
    EstimatorKind,
    EvalResult,
    NodeResult,
    TestResult,
    TrainResult,
)
from intelligence_sdk.interfaces import ArtifactStore, DataLoader, Evaluator, ModelBackend

log = logging.getLogger("intelligence_sdk")

MODEL_JSON = "model.json"


class Runner:
    """Composition happens outside: pass any objects implementing the ports."""

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

    def run(self, recipe: Recipe, node: str | None = None) -> dict[str, NodeResult]:
        graph, _ = dag.validate(recipe)
        self.resolve(recipe)
        execution = dag.plan(recipe, graph, node)
        if execution.target:
            log.info("recipe %s: node %s (%s)", recipe.source, execution.target, " -> ".join(execution.ordered))
        else:
            log.info(
                "recipe %s: %s%s",
                recipe.source,
                " -> ".join(execution.ordered) or "(no connected nodes)",
                f"; then isolated: {', '.join(execution.isolated)}" if execution.isolated else "",
            )

        resolver = DatasetResolver(recipe, self.loaders, self.store)
        results: dict[str, NodeResult] = {}
        for name in execution.sequence:
            spec = recipe.nodes[name]
            started = time.perf_counter()
            if isinstance(spec, DatasetNode):
                dataset = resolver.resolve_reference(name, "--node")
                executed = next(e for e in resolver.executed if e.name == name)
                result: NodeResult = DatasetResult(
                    name=name, type="dataset", dataset=dataset, output_dir=executed.output_dir
                )
            else:
                log.info("== [%s] %s", spec.type, name)
                if isinstance(spec, TrainNode):
                    result = self._run_train(name, spec, resolver)
                elif isinstance(spec, TestNode):
                    result = self._run_test(name, spec, recipe, resolver, results)
                elif isinstance(spec, EvalNode):
                    result = self._run_eval(name, spec, resolver)
                else:  # pragma: no cover - schema guarantees the union
                    raise RecipeError(name, f"unsupported node type {spec.type}")
            self._bookkeeping(result, recipe, time.perf_counter() - started)
            results[name] = result

        for executed in resolver.executed:
            if executed.name not in results:
                extra = DatasetResult(
                    name=executed.name, type="dataset", dataset=executed.dataset, output_dir=executed.output_dir
                )
                if not executed.reused:
                    self._bookkeeping(extra, recipe, 0.0)
                results[executed.name] = extra
        return results

    def resolve(self, recipe: Recipe) -> None:
        """Fail fast on everything that does not need data."""
        errors: list[RecipeError] = []
        produced = self._produced_paths(recipe)

        for name, spec in recipe.datasets.items():
            self._check_spec(f"datasets.{name}", spec, produced, errors)
        for name, node in recipe.nodes.items():
            for i, entry in enumerate(node.datasets):
                if isinstance(entry, InlineEntry):
                    self._check_spec(f"{name}.datasets[{i}]", entry.spec, produced, errors)
            self._check_filter_files(f"{name}.filter", node.filter, errors)

            if isinstance(node, TrainNode):
                self._check_components(name, node.preprocessing, [node.model], errors, single=True)
                kind = self.backend.estimator_kind(node.model.name)
                if node.hyperparam_search.scoring.kind != kind:
                    errors.append(
                        _metric_mismatch(
                            f"{name}.hyperparam_search.scoring", node.hyperparam_search.scoring, node.model, kind
                        )
                    )
            elif isinstance(node, EvalNode):
                self._check_components(name, node.preprocessing, node.candidates, errors, single=False)
                for ci, cand in enumerate(node.candidates):
                    kind = self.backend.estimator_kind(cand.name)
                    for mi, metric in enumerate(node.metrics or []):
                        if metric.kind != kind:
                            errors.append(
                                _metric_mismatch(f"{name}.metrics[{mi}]", metric, cand, kind, f"candidates[{ci}]")
                            )
                if not self._folds_declared(recipe, node):
                    errors.append(
                        RecipeError(
                            f"{name}.folds", "eval needs folds; add folds: {k: 5} on this node or on its dataset"
                        )
                    )
            elif isinstance(node, TestNode) and node.model_path is not None:
                model_dir = Path(node.model_path)
                if not (model_dir / MODEL_JSON).exists() and str(model_dir) not in produced:
                    errors.append(
                        RecipeError(f"{name}.model_path", f"no trained model at {model_dir} (missing {MODEL_JSON})")
                    )
        if errors:
            raise RecipeValidationError(errors)

    # --------------------------------------------------------------- resolve

    def _produced_paths(self, recipe: Recipe) -> set[str]:
        """Paths some node or spec in this recipe will write, so a later consumer may name them."""
        out: set[str] = set()
        specs: list[AnyDataset] = list(recipe.datasets.values())
        for node in recipe.nodes.values():
            specs.extend(e.spec for e in node.datasets if isinstance(e, InlineEntry))
            save = getattr(node, "save_dataset", None)
            if save is not None:
                out.add(str(Path(save.path)))
            output = node.recipe_output
            if output is not None:
                out.add(str(Path(output.save_path)))
        for spec in specs:
            if spec.save_dataset is not None:
                out.add(str(Path(spec.save_dataset.path)))
        return out

    def _check_spec(self, path: str, spec: AnyDataset, produced: set[str], errors: list[RecipeError]) -> None:
        if spec.source == "file" and not Path(spec.path).exists():
            errors.append(RecipeError(f"{path}.path", f"dataset file not found: {spec.path}"))
        if isinstance(spec, SavedDataset) and not Path(spec.path).exists() and str(Path(spec.path)) not in produced:
            errors.append(
                RecipeError(
                    f"{path}.path", f"no saved dataset at {spec.path} and nothing in this recipe saves one there"
                )
            )
        self._check_filter_files(f"{path}.filter", spec.filter, errors)

    def _check_filter_files(self, path: str, spec: FilterSpec | None, errors: list[RecipeError]) -> None:
        if spec is None:
            return
        for key in ("include", "exclude"):
            value = getattr(spec, key)
            if isinstance(value, str) and not Path(value).exists():
                errors.append(RecipeError(f"{path}.{key}", f"id file not found: {value}"))

    def _check_components(
        self, name: str, preprocessing: list[StepSpec], models: list[ModelSpec], errors: list[RecipeError], single: bool
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
            prefix = f"{name}.model.params" if single else f"{name}.candidates[{ci}].params"
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

    def _folds_declared(self, recipe: Recipe, node: EvalNode) -> bool:
        """Static answer where possible; a saved dataset may carry folds we cannot see."""
        if node.folds is not None:
            return True
        if len(node.datasets) != 1:
            return False
        entry = node.datasets[0]
        spec = entry.spec if isinstance(entry, InlineEntry) else recipe.lookup(entry.dataset_name, "")
        if isinstance(spec, DatasetNode):
            return spec.folds is not None or self._folds_declared(recipe, spec)  # type: ignore[arg-type]
        return spec.folds is not None or spec.source == "dataset"

    # ---------------------------------------------------------------- shared

    def _offer(
        self, name: str, node: NodeBase, resolver: DatasetResolver, require_target: bool, preprocessing: list[StepSpec]
    ) -> Dataset:
        dataset = resolver.consolidate(name, node)
        frame = dataset.frame
        if dataset.y() is None:
            if require_target:
                raise RecipeError(
                    f"{name}.datasets",
                    f"target column '{dataset.target}' not in dataset; columns: {', '.join(map(str, frame.columns))}",
                    did_you_mean(str(dataset.target), map(str, frame.columns)),
                )
        elif frame[dataset.target].isna().any():
            raise RecipeError(f"{name}.datasets", f"target column '{dataset.target}' has missing values")
        non_numeric = [c for c in dataset.feature_columns if not pd.api.types.is_numeric_dtype(frame[c])]
        if non_numeric:
            raise RecipeError(
                f"{name}.datasets",
                f"non-numeric feature column(s): {', '.join(map(str, non_numeric))}; only numeric features are supported - drop or encode them before loading",
            )
        nan_cols = [c for c in dataset.feature_columns if frame[c].isna().any()]
        if nan_cols and not any(step.step == Preprocessor.SIMPLE_IMPUTER for step in preprocessing):
            raise RecipeError(
                f"{name}.preprocessing",
                f"feature column(s) with missing values: {', '.join(map(str, nan_cols))}; add a 'simple_imputer' step or filter those rows",
            )
        return dataset

    def _fit_predict_score(
        self, preprocessing, model: ModelSpec, params, seed, dataset: Dataset, train_ids, test_ids, metrics
    ):
        X, y = dataset.X(), dataset.y()
        est = self.backend.build(preprocessing, model.name, params, seed)
        est = self.backend.fit(est, X.loc[train_ids], y.loc[train_ids])
        pred = self.backend.predict(est, X.loc[test_ids])
        return est, pred, self.evaluator.evaluate(y.loc[test_ids].to_numpy(), pred, metrics)

    def _output_dir(self, name: str, node) -> Path | None:
        out = node.recipe_output
        if out is None:
            log.info("   no recipe_output: nothing written for %s", name)
            return None
        return self.store.prepare(out.save_path, out.overwrite, f"{name}.recipe_output.save_path")

    # ----------------------------------------------------------------- train

    def _run_train(self, name: str, node: TrainNode, resolver: DatasetResolver) -> TrainResult:
        dataset = self._offer(name, node, resolver, True, node.preprocessing)
        fit_frame = dataset.train_frame()
        fit_ids = fit_frame.index.tolist()
        log.info("   fitting on %d rows (%s)", len(fit_ids), "train split" if dataset.split else "all rows, no split")

        grid = _expand_grid(node.model.params)
        search: dict[str, Any] = {"n_candidates": len(grid), "scoring": node.hyperparam_search.scoring.value}
        best_params, rows = grid[0], []
        if len(grid) > 1:
            best_params, rows = self._grid_search(name, node, dataset, fit_ids, grid)
            search.update({"cv": node.hyperparam_search.cv, "best_params": best_params, "ranking": rows})

        est = self.backend.build(node.preprocessing, node.model.name, best_params, node.seed)
        est = self.backend.fit(est, dataset.X(fit_frame), dataset.y(fit_frame))
        model_meta = {
            "node": name,
            "model": node.model.name.value,
            "params": best_params,
            "preprocessing": [s.model_dump(mode="json") for s in node.preprocessing],
            "features": dataset.feature_columns,
            "target": dataset.target,
            "result_type": self.backend.predict(est, dataset.X(fit_frame).head(1)).result_type.value,
            "seed": node.seed,
            "fitted_on": {"dataset": dataset.name, "rows": len(fit_ids), "split": "train" if dataset.split else "all"},
        }
        output_dir = self._output_dir(name, node)
        if output_dir is not None:
            self.backend.save(est, output_dir, model_meta)
            self.store.write_json(output_dir, "metrics.json", {"search": search})
            if rows:
                table = pd.DataFrame(
                    [
                        {
                            "rank": r["rank"],
                            "candidate": r["candidate"],
                            f"mean_{search['scoring']}": r["mean"],
                            "std": r["std"],
                            **{f"param_{k}": v for k, v in r["params"].items()},
                        }
                        for r in rows
                    ]
                )
                self.store.write_csv(output_dir, "search_results.csv", table)
                for r in rows:
                    cand_dir = output_dir / "candidates" / f"{r['candidate']:03d}"
                    self.store.write_json(cand_dir, "params.json", r["params"])
                    self.store.write_json(
                        cand_dir,
                        "metrics.json",
                        {
                            "scoring": search["scoring"],
                            "fold_scores": r["fold_scores"],
                            "mean": r["mean"],
                            "std": r["std"],
                        },
                    )
        return TrainResult(
            name=name,
            type="train",
            dataset=dataset,
            output_dir=output_dir,
            estimator=est,
            model_meta=model_meta,
            search=search,
        )

    def _grid_search(
        self, name: str, node: TrainNode, dataset: Dataset, fit_ids: list, grid: list[dict]
    ) -> tuple[dict, list[dict]]:
        scoring = node.hyperparam_search.scoring
        stratify = self.backend.estimator_kind(node.model.name) == EstimatorKind.CLASSIFIER
        try:
            folds = kfold_ids(
                np.asarray(fit_ids), dataset.y().loc[fit_ids].to_numpy(), node.hyperparam_search.cv, stratify, node.seed
            )
        except SplitError as exc:
            raise RecipeError(f"{name}.hyperparam_search.cv", str(exc)) from exc
        log.info("   hyperparameter search: %d candidates x %d-fold CV on %s", len(grid), len(folds), scoring.value)
        rows = []
        for i, params in enumerate(grid):
            scores = [
                self._fit_predict_score(node.preprocessing, node.model, params, node.seed, dataset, tr, te, [scoring])[
                    2
                ][scoring.value]
                for tr, te in folds
            ]
            mean, std = float(np.mean(scores)), float(np.std(scores))
            rows.append({"candidate": i, "params": params, "fold_scores": scores, "mean": mean, "std": std})
            log.info("     [%03d] %s -> %.4f ± %.4f", i, params, mean, std)
        rows.sort(key=lambda r: r["mean"], reverse=scoring.higher_is_better)
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        best = rows[0]
        log.info("   BEST: candidate %03d %s (%s=%.4f)", best["candidate"], best["params"], scoring.value, best["mean"])
        return best["params"], rows

    # ------------------------------------------------------------------ test

    def _run_test(
        self, name: str, node: TestNode, recipe: Recipe, resolver: DatasetResolver, results: dict[str, NodeResult]
    ) -> TestResult:
        if node.model_path is not None:
            model_dir = Path(node.model_path)
            model_meta = self.store.read_json(model_dir / MODEL_JSON)
            est = self.backend.load(model_dir)
            log.info("   model: %s (%s) from %s", model_meta["model"], model_meta["result_type"], model_dir)
        else:
            dep = dag.explicit_train_dependency(recipe, name)
            trained = results[dep]
            assert isinstance(trained, TrainResult)
            est, model_meta = trained.estimator, trained.model_meta
            log.info("   model: %s (%s) from node %s", model_meta["model"], model_meta["result_type"], dep)

        dataset = self._offer(name, node, resolver, False, [])
        features = model_meta["features"]
        absent = [c for c in features if c not in dataset.frame.columns]
        if absent:
            raise RecipeError(
                f"{name}.datasets",
                f"dataset lacks feature column(s) the model needs: {', '.join(absent)}",
                did_you_mean(absent[0], map(str, dataset.frame.columns)),
            )
        log.info("   scoring %d rows", len(dataset.frame))

        pred = self.backend.predict(est, dataset.frame[features])
        y = dataset.y()
        metrics: dict[str, Any] = {}
        predictions = pred.frame
        if y is not None:
            metrics = self.evaluator.evaluate(y.to_numpy(), pred, None)
            predictions = predictions.assign(truth=y)
            log.info("   metrics: %s", _fmt_metrics(metrics))
        else:
            log.info("   no '%s' column: predictions only", model_meta["target"])

        output_dir = self._output_dir(name, node)
        if output_dir is not None:
            self.store.write_csv(output_dir, "predictions.csv", predictions)
            self.store.write_json(output_dir, "metrics.json", metrics)
            self.store.write_json(
                output_dir,
                "model.json",
                {
                    **model_meta,
                    "loaded_from": str(node.model_path)
                    if node.model_path
                    else f"node:{dag.explicit_train_dependency(recipe, name)}",
                },
            )
        return TestResult(
            name=name, type="test", dataset=dataset, output_dir=output_dir, predictions=predictions, metrics=metrics
        )

    # ------------------------------------------------------------------ eval

    def _run_eval(self, name: str, node: EvalNode, resolver: DatasetResolver) -> EvalResult:
        dataset = self._offer(name, node, resolver, True, node.preprocessing)
        if dataset.folds is None:
            raise RecipeError(
                f"{name}.folds",
                f"dataset '{dataset.name}' has no folds; add folds: {{k: 5}} on this node or on its dataset",
            )
        pairs = dataset.folds.pairs(dataset.ids)
        log.info("   %d-fold CV over %d candidate(s) on %d rows", len(pairs), len(node.candidates), len(dataset.frame))

        rows = []
        for ci, cand in enumerate(node.candidates):
            for fi, (tr, te) in enumerate(pairs):
                _, _, m = self._fit_predict_score(
                    node.preprocessing, cand, cand.params, node.seed, dataset, tr, te, node.metrics
                )
                scalars = {k: v for k, v in m.items() if isinstance(v, int | float)}
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
        log.info("   summary (ranked by %s):\n%s", primary, summary.to_string(index=False))

        output_dir = self._output_dir(name, node)
        if output_dir is not None:
            self.store.write_csv(output_dir, "eval_report.csv", report)
            self.store.write_csv(output_dir, "eval_summary.csv", summary)
            self.store.write_json(
                output_dir, "metrics.json", {"ranked_by": primary, "summary": summary.to_dict(orient="records")}
            )
        return EvalResult(
            name=name, type="eval", dataset=dataset, output_dir=output_dir, report=report, summary=summary
        )

    # --------------------------------------------------------------- helpers

    def _bookkeeping(self, result: NodeResult, recipe: Recipe, seconds: float) -> None:
        if result.output_dir is None:
            return
        node = recipe.nodes[result.name]
        block = node.model_dump(mode="json", exclude_none=True, exclude={"datasets"})
        block["datasets"] = []
        for entry in node.datasets:
            if isinstance(entry, RefEntry):
                target = recipe.lookup(entry.dataset_name, "")
                resolved = (
                    {"node": entry.dataset_name}
                    if isinstance(target, DatasetNode)
                    else target.model_dump(mode="json", exclude_none=True)
                )
                block["datasets"].append({"dataset_name": entry.dataset_name, "resolved": resolved})
            else:
                block["datasets"].append({"name": entry.name, **entry.spec.model_dump(mode="json", exclude_none=True)})
        self.store.write_yaml(
            result.output_dir, "resolved_recipe.yaml", {"recipe": recipe.source, "node": result.name, "block": block}
        )
        self.store.write_json(result.output_dir, "dataset_metadata.json", result.dataset.to_metadata())
        import sklearn  # version stamp only; the runner itself never calls into it

        self.store.write_json(
            result.output_dir,
            "run.json",
            {
                "node": result.name,
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
        log.info("   outputs: %s", result.output_dir)


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
