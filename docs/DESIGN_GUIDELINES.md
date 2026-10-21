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
- The facade exports exactly `load_recipe`, `run_recipe`, `build_runner`. A new
  delivery surface (a service) uses the same three names.
- Folders exist only where there is plurality. A package with a lone `__init__.py`
  should be a flat module.

## Extending

| to add | touch |
|--------|-------|
| a model backend (torch, the inference engine) | `adapters/<name>_backend.py` subclassing `ModelBackend`; one entry in `runtime/providers.py` `BACKENDS`; one member in `registry.Backend` |
| an artifact store (S3) | `adapters/<name>_artifacts.py` subclassing `ArtifactStore`; `providers.ARTIFACT_STORES`; `registry.ArtifactStoreKind` |
| a data source | `adapters/loaders/<name>.py` subclassing `DataLoader`; a new `source` literal in `core/recipe.py`; `providers.LOADERS` |
| a component (scaler, model) | a member in `registry.Preprocessor` or `registry.Model`; its class in the backend's table |
| a section type (`preprocess`) | a new section model in the discriminated union in `core/recipe.py`; a handler in `core/runner.py` |
| a metric | a member in `registry.Scoring` with its kind; its function in `adapters/evaluator.py` |

A backend decides what it returns through `ResultType`; the evaluator keys on it.
Do not add a task-type field to the recipe.

## Recipe conventions

- Block-style YAML in anything shipped (configs, docs, CLI help). Never inline
  `{...}` mappings; short scalar lists like `[20, 50]` are fine.
- Config names components by registry enum only. No import paths, ever.
- Section names are identifiers (`[A-Za-z_][A-Za-z0-9_-]*`); they become directory
  names. `datasets` and `runtime` are reserved.
- A list-valued hyperparameter means "search"; a scalar means "fixed". Eval
  candidates take scalars only.
- Split belongs to train and is recorded as row ids. Nothing downstream re-splits.
- An explicit dependency on a test section names its model source; a test that only
  needs ordering uses listing order.

## Errors

- Fail before IO. Everything checkable without data (schema, DAG, component and
  hyperparameter names, metric/model compatibility, file existence) is checked in
  the resolve phase.
- Every error carries the config path as written in the YAML and, when a close
  match exists, a did-you-mean suggestion. Report all schema problems in one pass.
- User errors are `RecipeError`; the CLI prints them on one line and never shows a
  traceback for them.

## Artifacts

Every run writes `resolved_recipe.yaml`, `dataset.json` and `run.json` next to its
outputs under `<output_dir>/<section>/<run_id>/`, with a `latest` pointer per
section. A trained model must always be re-runnable from its bundle alone.

## Keeping the docs true

Every change, including fine-tuning, updates the three documents in the same commit:
`README.md` (quickstart, tree, Makefile table), `docs/ARCHITECTURE.md` (diagram, recipe
model, design choices, target state) and this file (rules, extension table,
conventions). A diff that changes a module, a recipe key, a make target or a
convention without touching the matching doc is incomplete.

## Tooling

`make lint` runs ruff (line length 120); `make test` runs pytest; `make docs`
renders API docs from docstrings with pdoc. Tests and docs are never de-scoped.
