# Design guidelines

Rules I follow in this codebase and how to extend it without breaking them.

## Dependency rule

`tools → facade → runtime → {core, adapters} → interfaces → core/types`

- `interfaces/` imports only `core/types`. Ports are ABCs with abstract methods;
  adapters inherit them, so a missing method fails at instantiation, and a test
  asserts every entry in the provider tables subclasses its port.
- `core/` imports `interfaces/`, never `adapters/`.
- `runtime/providers.py` is the only module that names an adapter. Keep
  `adapters/__init__.py` and `adapters/loaders/__init__.py` empty; a convenience
  re-export would create a second import site.
- The facade exports exactly `load_recipe`, `run_recipe`, `build_runner`. The CLI in
  `tools/` uses only those and ships in the wheel as the `intelligence-sdk` command;
  a service would use the same three names.
- Folders exist only where there is plurality. A package with a lone `__init__.py`
  should be a flat module.

## Extending

| to add | touch |
|--------|-------|
| a model backend (torch, the inference engine) | `adapters/<name>_backend.py` subclassing `ModelBackend`; one entry in `runtime/providers.py` `BACKENDS`; one member in `registry.Backend` |
| an artifact store (S3) | `adapters/<name>_artifacts.py` subclassing `ArtifactStore`; `providers.ARTIFACT_STORES`; `registry.ArtifactStoreKind` |
| a data source (a DVC-backed registry, a warehouse) | `adapters/loaders/<name>.py` subclassing `DataLoader`; a new `source` literal in `core/recipe.py`; `providers.LOADERS` |
| a dataset operation | a field on `OpsMixin` in `core/recipe.py`; its function in `core/dataset_ops.py`; its place in `apply_ops` |
| a component (scaler, model) | a member in `registry.Preprocessor` or `registry.Model`; its class in the backend's table |
| a node type (`preprocess`) | a node model in the `Node` union in `core/recipe.py`; a typed `recipe_output`; a handler in `core/runner.py` |
| a metric | a member in `registry.Scoring` with its kind; its function in `adapters/evaluator.py` |

A backend decides what it returns through `ResultType`; the evaluator keys on it.
Do not add a task-type field to the recipe.

## Recipe conventions

- Block-style YAML in anything shipped (configs, docs, CLI help). Never inline
  `{...}` mappings; short scalar lists like `[20, 50]` are fine.
- Config names components by registry enum only. No import paths, ever.
- Names are identifiers (`[A-Za-z_][A-Za-z0-9_-]*`) and share one namespace across
  the registry, node names and inline `name`s. `datasets` and `runtime` are reserved.
- Data operations live on datasets: a spec's `filter`, `split`, `folds`, `seed`,
  `save_dataset`, and the same options on a node's root for the consolidated dataset.
  Order is always filter, split, folds, save.
- `depends_on` is explicit, exists only on train, test and eval, and on a test names
  the model source. Datasets are referenced, never depended on.
- A list-valued hyperparameter means "search"; a scalar means "fixed". Eval
  candidates take scalars only.
- A `filter` by split name is valid only inside a `source: dataset` entry.
- Nothing is written without a `recipe_output`; an existing path needs `overwrite`
  (or `reuse` for datasets). The tool never silently discards what the user wrote.

## Errors

- Fail before IO. Everything checkable without data (schema, both graphs, references,
  component and hyperparameter names, metric/model compatibility, file existence)
  is checked in the resolve phase.
- Every error carries the config path as written in the YAML and, when a close
  match exists, a did-you-mean suggestion. Report all schema problems in one pass.
- User errors are `RecipeError`; the CLI prints them on one line and never shows a
  traceback for them.

## Run cards

A node with a `recipe_output` writes `resolved_recipe.yaml`, `dataset_metadata.json`
and `run.json` next to its outputs. A saved dataset is `dataset.csv` plus
`metadata.json` (and the id lists of its split) and is loadable as `source: dataset`.

## Keeping the docs true

A change that alters the design or makes a document inaccurate updates that document
in the same commit: `README.md` (quickstart, tree, Makefile table),
`docs/ARCHITECTURE.md` (diagram, recipe model, design choices, target state),
`docs/IMPROVEMENTS.md` (what is deliberately out of scope) or this file (rules,
extension table, conventions). Changes that leave the docs true, such as a bug fix
or a wording tweak, do not need a doc edit.

## Tooling

`make lint` runs ruff (line length 120); `make test` runs pytest; `make docs`
renders API docs from docstrings with pdoc. Tests and docs are never de-scoped.
