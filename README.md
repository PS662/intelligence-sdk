# intelligence-sdk

A config-driven way to build simple ML models. One YAML recipe names the datasets,
the preprocessing steps, the model and what to do with it (dataset, train, test, eval);
the SDK validates the whole recipe before it reads a row, runs the nodes as a DAG, and
writes a run card per node. Datasets are the first-class object: they carry their own
filter, split, folds and save policy, can be combined by a dataset node, and are
resolved by reference wherever a node needs them. I kept the scope small on purpose
(numeric tabular data, scikit-learn) and spent the effort on the design: a hexagonal
core, registry-owned component names, and errors that teach.

## Quickstart

```bash
make install-dev                 # creates the venv with uv if needed; editable install with dev extras
make data                        # sample data is not committed; writes data/iris.csv and data/iris_extra.csv
make run-train                   # train on iris; saves the split dataset and the model
make run-test                    # score the saved model on the saved dataset's test split
make run-eval                    # k-fold comparison of candidate models
make run-dag                     # dataset node -> train -> test, plus an isolated eval
make inspect                     # validate a recipe and show roles and order without running it
make test lint                   # pytest, ruff
```

The same commands without make, through the CLI script or the installed command:

```bash
python tools/make_iris_csv.py                                             # sample data
python tools/intelligence_cli.py run     --recipe configs/train_iris.yaml
python tools/intelligence_cli.py run     --recipe configs/test_iris.yaml
python tools/intelligence_cli.py run     --recipe configs/eval_iris.yaml
python tools/intelligence_cli.py run     --recipe configs/iris_dag.yaml
python tools/intelligence_cli.py run     --recipe configs/iris_dag.yaml --node test_rf   # only test_rf and what it depends on
python tools/intelligence_cli.py inspect --recipe configs/iris_dag.yaml
python tools/intelligence_cli.py list-components --model random_forest
python tools/intelligence_cli.py run     --recipe configs/train_iris_typo.yaml   # a recipe that fails on purpose, to see the errors

intelligence-sdk run --recipe configs/train_iris.yaml                     # after `make install`, same subcommands
```

From Python the public surface is three names:

```python
from intelligence_sdk import load_recipe, run_recipe, build_runner

results = run_recipe("configs/iris_dag.yaml")  # {node_name: NodeResult}
results = run_recipe("configs/iris_dag.yaml", node="test_rf")  # only test_rf and what it depends on
results["test_rf"].metrics, results["train_rf"].output_dir
```

A recipe, compressed to its moving parts:

```yaml
datasets:                      # registry of named datasets; operations live on the dataset
  iris:
    source: file               # file | library | dataset (one saved by this tool)
    path: data/iris.csv
    target: species
    split:                     # recorded as row ids; train fits on the train split
      test_size: 0.2
      stratify: true
    save_dataset:              # optional; dataset.csv + metadata.json + train/test id lists
      path: data/derived/iris
      overwrite: true          # an existing path needs overwrite (rebuild) or reuse (load)

train_rf:                      # any identifier; type: dataset | train | test | eval
  type: train
  datasets:
    - dataset_name: iris       # a reference, or an inline dataset with `name:`
  preprocessing:
    - step: standard_scaler
  model:
    name: random_forest
    params:
      n_estimators: [20, 50]   # a list means "search these"; scalars are fixed
  recipe_output:               # optional; nothing is written without it
    save_path: artifacts/train_rf
    overwrite: true

test_rf:
  type: test
  depends_on: [train_rf]       # explicit only; names the model source
  datasets:
    - name: test_split
      source: dataset
      path: data/derived/iris
      filter:
        exclude:
          split: train         # score only the rows the model never saw
```

`intelligence-sdk list-components` prints every name a recipe may use.

## Architecture

```
intelligence_sdk/
  __init__.py       facade: load_recipe, run_recipe, build_runner
  interfaces/       ports (ABCs): DataLoader, ModelBackend, ArtifactStore, Evaluator
  core/             domain: recipe schema, registry enums, dataset ops, resolver, DAG, splits, runner
  adapters/         loaders/ (csv, library, saved), sklearn_backend, local_artifacts, evaluator, dag_render
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
| `data` | regenerate `data/iris.csv` and `data/iris_extra.csv` |
| `test`, `lint`, `format` | pytest; ruff check + format check; apply ruff |
| `docs` | API docs from docstrings into `docs/api/` (pdoc) |
| `inspect`, `run-train`, `run-test`, `run-eval`, `run-dag` | the sample recipes |
| `clean` | artifacts, derived data, caches, generated docs |

## Development

Design rules, how to add a backend, a store, a loader or a node type, and the
conventions for recipes and errors are in
[docs/DESIGN_GUIDELINES.md](docs/DESIGN_GUIDELINES.md). The general design is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); what is deliberately left out is in
[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md). Licensed under Apache 2.0, see
[LICENSE.md](LICENSE.md).
