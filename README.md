# intelligence-sdk

A config-driven way to build simple ML models. One YAML recipe names the datasets, the
preprocessing steps, the model and what to do with it (train, test, eval); the SDK
validates the whole recipe before it reads a row, runs the sections as a DAG, and
leaves a reproducible provenance bundle per run. I kept the scope small on purpose
(numeric tabular data, scikit-learn) and spent the effort on the design: a hexagonal
core, registry-owned component names, and errors that teach.

## Quickstart

```bash
make install-dev                 # creates the venv with uv if needed; editable install with dev extras
make data                        # sample data is not committed; writes data/iris.csv
make run-train                   # train on iris, grid search when a param is a list
make run-test                    # score the latest model on its recorded holdout rows
make run-eval                    # k-fold comparison of candidate models
make run-dag                     # one recipe with train -> test, eval as a DAG
make inspect                     # validate a recipe and draw its DAG without running it
make test lint                   # pytest, ruff
```

From Python the public surface is three names:

```python
from intelligence_sdk import load_recipe, run_recipe, build_runner

outputs = run_recipe("configs/iris_dag.yaml")  # {section_name: SectionOutput}
outputs["test_rf"].metrics, outputs["train_rf"].run_dir
```

A recipe, compressed to its moving parts:

```yaml
datasets:                      # named datasets; source: file | library
  iris:
    source: file
    path: data/iris.csv
    target: species

train_rf:                      # any identifier -> artifacts/train_rf/<run_id>/
  type: train                  # train | test | eval
  datasets:
    - dataset_name: iris
  split:                       # recorded as row ids once; never recomputed downstream
    test_size: 0.2
    stratify: true
  preprocessing:
    - step: standard_scaler
  model:
    name: random_forest
    params:
      n_estimators: [20, 50]   # a list means "search these"; scalars are fixed

test_rf:
  type: test
  depends_on: [train_rf]       # names the model source; holdout rows, dataset hash checked
```

`intelligence-sdk list-components` prints every name a recipe may use.

## Architecture

```
intelligence_sdk/
  __init__.py       facade: load_recipe, run_recipe, build_runner
  interfaces/       ports (ABCs): DataLoader, ModelBackend, ArtifactStore, Evaluator
  core/             domain: recipe schema, registry enums, DAG, splits, filters, runner
  adapters/         loaders/ (csv, library), sklearn_backend, local_artifacts, evaluator, dag_render
  runtime/          provider registry + build_runner: the runtime block -> a wired Runner
tools/              CLI (installed as the `intelligence-sdk` command) and sample-data script; consumers of the facade
configs/ data/ docs/ tests/
```

Imports flow one way: `tools → facade → runtime → {core, adapters} → interfaces → core/types`.
`core` never imports an adapter; `runtime/providers.py` is the only module that names one.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the diagram, the recipe model and
the toolkit this SDK is one layer of.

## Makefile

| target | what it does |
|--------|--------------|
| `venv`, `install`, `install-dev` | create the venv with uv; install the package and the `intelligence-sdk` command; editable install with dev extras |
| `data` | regenerate `data/iris.csv` |
| `test`, `lint`, `format` | pytest; ruff check + format check; apply ruff |
| `docs` | API docs from docstrings into `docs/api/` (pdoc) |
| `inspect`, `run-train`, `run-test`, `run-eval`, `run-dag` | the sample recipes |
| `clean` | artifacts, caches, generated docs |

## Development

Design rules, how to add a backend, a store, a loader or a section type, and the
conventions for recipes and errors are in
[docs/DESIGN_GUIDELINES.md](docs/DESIGN_GUIDELINES.md). The general design is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Licensed under Apache 2.0, see
[LICENSE.md](LICENSE.md).
