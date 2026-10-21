# intelligence-sdk
# All targets use the virtualenv in VENV; override with `make VENV=/path/to/venv <target>`.

VENV    ?= $(HOME)/.virtualenvs/intelligence-sdk-venv
PY      := $(VENV)/bin/python
UV      ?= uv
CLI     := $(PY) tools/intelligence_cli.py

.PHONY: help install install-dev data test lint format docs inspect run-train run-test run-eval run-dag clean

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:         ## install the package into the venv
	$(UV) pip install --python $(PY) .

install-dev:     ## editable install with test/lint/docs extras
	$(UV) pip install --python $(PY) -e ".[dev]"

data:            ## regenerate data/iris.csv (sample data is not committed)
	$(PY) tools/make_iris_csv.py

test:            ## run the test suite
	$(PY) -m pytest

lint:            ## ruff lint + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:          ## apply ruff formatting and safe fixes
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

docs:            ## API docs from docstrings into docs/api/
	$(PY) -m pdoc intelligence_sdk intelligence_sdk.core intelligence_sdk.interfaces intelligence_sdk.adapters intelligence_sdk.runtime -o docs/api

inspect:         ## validate configs/iris_dag.yaml and draw its DAG
	$(CLI) inspect --recipe configs/iris_dag.yaml

run-train: data  ## train on iris with grid search
	$(CLI) run --recipe configs/train_iris.yaml

run-test:        ## score the latest trained model on its recorded holdout
	$(CLI) run --recipe configs/test_iris.yaml

run-eval:        ## k-fold comparison of candidate models
	$(CLI) run --recipe configs/eval_iris.yaml

run-dag: data    ## the multi-section recipe end to end
	$(CLI) run --recipe configs/iris_dag.yaml

clean:           ## remove run artifacts, caches and generated docs
	rm -rf artifacts docs/api .pytest_cache .ruff_cache configs/*.dag.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
