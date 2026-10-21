#!/usr/bin/env python
"""intelligence-sdk command line.

This script is a consumer of the SDK, not part of it: it imports only the public
facade (``load_recipe``, ``run_recipe``, ``build_runner``). Install the package first
(``uv pip install -e .``).

    python tools/intelligence_cli.py run     --recipe configs/train_iris.yaml
    python tools/intelligence_cli.py inspect --recipe configs/iris_dag.yaml
    python tools/intelligence_cli.py list-components --model random_forest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from intelligence_sdk import build_runner, load_recipe, run_recipe
from intelligence_sdk.core.dag import describe, validated_graph
from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.registry import LibraryName, Model, Preprocessor, Scoring
from intelligence_sdk.runtime import available_providers

EPILOG = """\
examples:
  %(prog)s run --recipe configs/train_iris.yaml          train (grid search when a param is a list)
  %(prog)s run --recipe configs/test_iris.yaml           score the recorded holdout of the latest model
  %(prog)s run --recipe configs/eval_iris.yaml           k-fold comparison of candidate models
  %(prog)s run --recipe configs/iris_dag.yaml           multi-section DAG: train -> test, eval
  %(prog)s inspect --recipe configs/iris_dag.yaml       validate + show execution order (+ PNG)
  %(prog)s list-components --model random_forest         what you may put in a recipe

recipe skeleton (block-style YAML; every section has a `type`):

  datasets:                      # registry of named datasets
    iris:
      source: file               # file | library
      path: data/iris.csv
      target: species

  train_rf:                      # any identifier; becomes artifacts/train_rf/<run_id>/
    type: train
    datasets:
      - dataset_name: iris
    split:
      test_size: 0.2
      stratify: true
    preprocessing:
      - step: standard_scaler
    model:
      name: random_forest
      params:
        n_estimators: [20, 50]   # a list means "search these"
    seed: 42

  test_rf:
    type: test
    depends_on: [train_rf]       # names the model source; holdout rows + hash check
"""


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s" if not verbose else "%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def cmd_run(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    outputs = run_recipe(args.recipe, allow_dataset_mismatch=args.allow_dataset_mismatch, output_dir=args.output_dir)
    print()
    print("done. run directories:")
    for name, out in outputs.items():
        print(f"  {name:<20} [{out.type}]  {out.run_dir}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    graph = validated_graph(recipe)
    build_runner(recipe.runtime).resolve(recipe)  # component / hyperparameter / metric checks, no data
    rows = describe(recipe, graph)

    print(f"recipe: {args.recipe}")
    print(f"datasets: {', '.join(recipe.datasets) or '(none in registry)'}")
    print()
    print(f"{'#':<3} {'section':<20} {'type':<6} {'depends_on (explicit)':<28} implicit-after")
    for i, row in enumerate(rows, 1):
        deps = ", ".join(row.explicit_deps) or "-"
        print(f"{i:<3} {row.name:<20} {row.type:<6} {deps:<28} {row.implicit_dep or '-'}")
    print()
    print("valid: DAG ok, all components and hyperparameters resolve.")

    png = args.png or Path(args.recipe).with_suffix(".dag.png")
    try:
        from intelligence_sdk.adapters.dag_render import render_png

        render_png(graph, Path(png), title=f"{Path(args.recipe).name}")
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
    print("\nlibrary datasets (source: library):")
    print("  " + ", ".join(n.value for n in LibraryName))
    print("\ntip: list-components --model <name> prints that model's hyperparameters.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build simple ML models from a YAML recipe: one file describes datasets, preprocessing, model and what to do (train / test / eval).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    run = sub.add_parser(
        "run",
        help="validate and execute a recipe",
        description="Validate the whole recipe first (schema, DAG, components, metric compatibility), then execute its sections in dependency order.",
    )
    run.add_argument("--recipe", required=True, help="path to the recipe YAML")
    run.add_argument(
        "--output-dir",
        help="override every section's output_dir (default: as written in the recipe, usually ./artifacts)",
    )
    run.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="let a test score an inherited dataset whose hash differs from the one the model was trained on (leakage risk; off by default)",
    )
    run.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    run.set_defaults(func=cmd_run)

    insp = sub.add_parser(
        "inspect",
        help="validate a recipe and show its DAG without running it",
        description="Validate the recipe and print the execution order. Writes a PNG of the DAG next to the recipe when matplotlib is available.",
    )
    insp.add_argument("--recipe", required=True, help="path to the recipe YAML")
    insp.add_argument("--png", help="where to write the DAG image (default: <recipe>.dag.png)")
    insp.set_defaults(func=cmd_inspect)

    lst = sub.add_parser("list-components", help="show the steps, models, metrics and datasets a recipe may name")
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
            "\nhint: paths are <section>.<field>... as written in the YAML; `list-components` shows valid names.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
