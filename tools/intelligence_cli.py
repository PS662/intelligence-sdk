#!/usr/bin/env python
"""intelligence-sdk command line.

This script is a consumer of the SDK, not part of it: it imports only the public
facade (``load_recipe``, ``run_recipe``, ``build_runner``). Install the package first
(``uv pip install -e .``).

    intelligence-sdk run     --recipe configs/train_iris.yaml
    intelligence-sdk run     --recipe configs/iris_dag.yaml --node test_rf
    intelligence-sdk inspect --recipe configs/iris_dag.yaml
    intelligence-sdk list-components --model random_forest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from intelligence_sdk import build_runner, load_recipe, run_recipe
from intelligence_sdk.core import dag
from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.registry import LibraryName, Model, Preprocessor, Scoring
from intelligence_sdk.runtime import available_providers

EPILOG = """\
examples:
  %(prog)s run --recipe configs/train_iris.yaml            train; a list-valued param triggers a search
  %(prog)s run --recipe configs/test_iris.yaml             score a saved model on a saved dataset's test split
  %(prog)s run --recipe configs/eval_iris.yaml             k-fold comparison of candidate models
  %(prog)s run --recipe configs/iris_dag.yaml              dataset node -> train -> test, plus an isolated eval
  %(prog)s run --recipe configs/iris_dag.yaml --node test_rf   only test_rf and what it depends on
  %(prog)s inspect --recipe configs/iris_dag.yaml          validate + show execution order (+ PNG)
  %(prog)s list-components --model random_forest           what you may put in a recipe

recipe skeleton (block-style YAML; every node has a `type`):

  datasets:                      # registry of named datasets; operations live here
    iris:
      source: file               # file | library | dataset (one saved by this tool)
      path: data/iris.csv
      target: species
      split:
        test_size: 0.2
        stratify: true
      save_dataset:
        path: data/derived/iris
        overwrite: true

  train_rf:                      # any identifier
    type: train                  # dataset | train | test | eval
    datasets:
      - dataset_name: iris       # a reference, or an inline dataset with `name:`
    preprocessing:
      - step: standard_scaler
    model:
      name: random_forest
      params:
        n_estimators: [20, 50]   # a list means "search these"
    recipe_output:
      save_path: artifacts/train_rf
      overwrite: true

  test_rf:
    type: test
    depends_on: [train_rf]       # names the model source
    datasets:
      - name: test_split
        source: dataset
        path: data/derived/iris
        filter:
          exclude:
            split: train         # score the recorded test split only
"""


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s" if not verbose else "%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def cmd_run(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    results = run_recipe(args.recipe, node=args.node)
    print()
    print("done.")
    for name, result in results.items():
        print(f"  {name:<20} [{result.type}]  {result.output_dir or '(not saved)'}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    graph, _ = dag.validate(recipe)
    build_runner(recipe.runtime).resolve(recipe)  # component / hyperparameter / metric / path checks, no data
    execution = dag.plan(recipe, graph, args.node)
    rows = dag.describe(recipe, graph, execution)

    print(f"recipe: {args.recipe}")
    print(f"registry datasets: {', '.join(recipe.datasets) or '(none)'}")
    if args.node:
        print(f"node: {args.node} -> runs {' -> '.join(execution.ordered)}")
    print()
    print(f"{'#':<3} {'node':<20} {'type':<8} {'role':<9} {'depends_on':<24} referenced_by")
    for i, row in enumerate(rows, 1):
        deps = ", ".join(row.depends_on) or "-"
        refs = ", ".join(row.referenced_by) or "-"
        print(f"{i:<3} {row.name:<20} {row.type:<8} {row.role:<9} {deps:<24} {refs}")
    print()
    print("valid: both graphs acyclic, all components, hyperparameters, metrics and paths resolve.")
    print("roles: dag = connected by depends_on, isolated = runs after the DAG, dataset = runs when referenced")

    png = args.png or Path(args.recipe).with_suffix(".dag.png")
    try:
        from intelligence_sdk.adapters.dag_render import render_png

        render_png(recipe, graph, Path(png), title=Path(args.recipe).name)
        print(f"DAG image: {png}")
    except ImportError:
        print("DAG image skipped: matplotlib not installed (pip install 'intelligence-sdk[render]')")
    return 0


def cmd_list_components(args: argparse.Namespace) -> int:
    backend = build_runner().backend
    if args.model:
        try:
            model = Model(args.model)
        except ValueError:
            names = [m.value for m in Model]
            print(f"unknown model '{args.model}'. choose one of: {', '.join(names)}", file=sys.stderr)
            return 1
        print(f"{model.value} ({backend.estimator_kind(model).value}) hyperparameters:")
        for p in sorted(backend.param_names(model)):
            print(f"  {p}")
        return 0
    print("preprocessing steps (preprocessing[].step):")
    for p in Preprocessor:
        print(f"  {p.value:<22} params: {', '.join(sorted(backend.step_param_names(p)))}")
    print("\nmodels (model.name / candidates[].name):")
    for m in Model:
        print(f"  {m.value:<22} {backend.estimator_kind(m).value}")
    print("\nscoring / metrics:")
    for s in Scoring:
        print(f"  {s.value:<22} {s.kind.value}")
    print("\nruntime block (optional):")
    for key, names in available_providers().items():
        print(f"  {key:<22} {', '.join(names)}")
    print("\nlibrary datasets (source: library, dataset: <name>):")
    print("  " + ", ".join(n.value for n in LibraryName))
    print("\ntip: list-components --model <name> prints that model's hyperparameters.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build simple ML models from a YAML recipe: datasets, preprocessing, model and what to do (dataset / train / test / eval).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    run = sub.add_parser(
        "run",
        help="validate and execute a recipe",
        description="Validate the whole recipe first (schema, graphs, components, metric compatibility, paths), then execute its nodes in dependency order.",
    )
    run.add_argument("--recipe", required=True, help="path to the recipe YAML")
    run.add_argument(
        "--node", help="run only this node and the nodes it depends on (a dataset node: resolve and save it)"
    )
    run.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    run.set_defaults(func=cmd_run)

    insp = sub.add_parser(
        "inspect",
        help="validate a recipe and show its execution plan without running it",
        description="Validate the recipe and print the execution order with each node's role. Writes a PNG of the graphs next to the recipe when matplotlib is available.",
    )
    insp.add_argument("--recipe", required=True, help="path to the recipe YAML")
    insp.add_argument("--node", help="show the plan for running only this node")
    insp.add_argument("--png", help="where to write the image (default: <recipe>.dag.png)")
    insp.set_defaults(func=cmd_inspect)

    lst = sub.add_parser(
        "list-components", help="show the steps, models, metrics, providers and datasets a recipe may name"
    )
    lst.add_argument("--model", help="print the hyperparameters of one model, e.g. random_forest")
    lst.set_defaults(func=cmd_list_components)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        print("\nerror: choose a command: run, inspect or list-components", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except RecipeError as exc:
        print(f"\nrecipe error\n{exc}", file=sys.stderr)
        print(
            "\nhint: paths are <node>.<field>... as written in the YAML; `list-components` shows valid names.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
