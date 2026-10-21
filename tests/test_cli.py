"""The CLI is a consumer of the facade; these tests drive its ``main``."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("intelligence_cli", ROOT / "tools" / "intelligence_cli.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["intelligence_cli"] = module
    spec.loader.exec_module(module)
    return module


def test_no_command_prints_help_and_exits_2(cli, capsys):
    assert cli.main([]) == 2
    out = capsys.readouterr()
    assert "examples:" in out.out and "choose a command" in out.err


def test_inspect_shows_roles_without_running(cli, capsys, tmp_path):
    png = tmp_path / "dag.png"
    assert cli.main(["inspect", "--recipe", str(ROOT / "configs" / "iris_dag.yaml"), "--png", str(png)]) == 0
    out = capsys.readouterr().out
    assert "dag" in out and "isolated" in out and "dataset" in out and "prep" in out
    assert png.exists()


def test_inspect_with_node_shows_the_chain(cli, capsys, tmp_path):
    assert (
        cli.main(
            [
                "inspect",
                "--recipe",
                str(ROOT / "configs" / "iris_dag.yaml"),
                "--node",
                "test_rf",
                "--png",
                str(tmp_path / "n.png"),
            ]
        )
        == 0
    )
    assert "runs train_rf -> test_rf" in capsys.readouterr().out


def test_recipe_error_is_one_message_no_traceback(cli, capsys, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("t:\n  type: trian\n")
    assert cli.main(["run", "--recipe", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "recipe error" in err and "t.type" in err and "Traceback" not in err


def test_run_train_recipe_with_node(cli, capsys, caplog, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    caplog.set_level(logging.INFO, logger="intelligence_sdk")  # pytest owns the root logger
    text = (ROOT / "configs" / "train_iris.yaml").read_text()
    text = text.replace("data/derived/iris", str(tmp_path / "iris")).replace(
        "artifacts/train_rf", str(tmp_path / "train_rf")
    )
    recipe = tmp_path / "train.yaml"
    recipe.write_text(text)
    assert cli.main(["run", "--recipe", str(recipe), "--node", "train_rf"]) == 0
    assert "BEST:" in caplog.text and "done." in capsys.readouterr().out
    assert (tmp_path / "train_rf" / "model.joblib").exists()


def test_list_components(cli, capsys):
    assert cli.main(["list-components", "--model", "knn"]) == 0
    assert "n_neighbors" in capsys.readouterr().out
    assert cli.main(["list-components"]) == 0
    assert "runtime block" in capsys.readouterr().out
    assert cli.main(["list-components", "--model", "nope"]) == 1
