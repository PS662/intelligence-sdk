"""Recursive dataset resolution.

A node's ``datasets`` list is resolved entry by entry: a reference is looked up in
the registry or among dataset nodes (and cached for the run), an inline entry is
materialized in place. The parts are concatenated, the node's root-level
operations are applied, and the result is the single dataset the node acts on.

Saving follows one policy everywhere: a path that already holds a dataset is
rebuilt with ``overwrite``, loaded with ``reuse``, and an error otherwise.
Reference cycles are rejected by ``core.dag`` before execution; the guard here
only protects against a graph that was not validated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from intelligence_sdk.core.dataset_ops import apply_ops, concat
from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.recipe import (
    AnyDataset,
    DatasetNode,
    DatasetOutput,
    InlineEntry,
    NodeBase,
    Recipe,
    RefEntry,
    SavedDataset,
    SaveSpec,
)
from intelligence_sdk.core.types import Dataset
from intelligence_sdk.interfaces import ArtifactStore, DataLoader

log = logging.getLogger("intelligence_sdk")

SAVE_HINT = "set overwrite: true to rebuild it or reuse: true to load it"


@dataclass
class ExecutedDatasetNode:
    name: str
    node: DatasetNode
    dataset: Dataset
    output_dir: Path | None
    reused: bool


class DatasetResolver:
    def __init__(self, recipe: Recipe, loaders: dict[str, DataLoader], store: ArtifactStore):
        self.recipe = recipe
        self.loaders = loaders
        self.store = store
        self.cache: dict[str, Dataset] = {}
        self.executed: list[ExecutedDatasetNode] = []
        self._active: set[str] = set()

    # ---------------------------------------------------------------- public

    def consolidate(self, node_name: str, node: NodeBase) -> Dataset:
        """The single dataset ``node`` acts on."""
        save = getattr(node, "save_dataset", None)
        reused = self._reuse_or_fail(save, f"{node_name}.save_dataset", node_name)
        if reused is not None:
            return reused

        parts = [self._resolve_entry(entry, node_name, i) for i, entry in enumerate(node.datasets)]
        dataset = concat(parts, node_name, node.ignore_duplicates, f"{node_name}.datasets")
        if len(parts) > 1:
            log.info(
                "   combined %d datasets into %d rows; split/folds come from %s's own options",
                len(parts),
                len(dataset.frame),
                node_name,
            )
        dataset = apply_ops(dataset, node, node_name)
        if save is not None:
            self._save(save.path, dataset)
        return dataset

    def run_dataset_node(self, name: str, node: DatasetNode) -> Dataset:
        out: DatasetOutput | None = node.recipe_output
        reused = self._reuse_or_fail(out, f"{name}.recipe_output.save_path", name)
        if reused is not None:
            self.executed.append(ExecutedDatasetNode(name, node, reused, Path(out.save_path), reused=True))
            return reused
        log.info("== [dataset] %s", name)
        dataset = self.consolidate(name, node)
        output_dir = None
        if out is not None:
            output_dir = self._save(out.save_path, dataset)
        else:
            log.info("   no recipe_output: dataset '%s' kept in memory only", name)
        self.executed.append(ExecutedDatasetNode(name, node, dataset, output_dir, reused=False))
        return dataset

    def resolve_reference(self, name: str, path: str) -> Dataset:
        if name in self.cache:
            return self.cache[name]
        if name in self._active:
            raise RecipeError(path, f"circular dataset reference through '{name}'")
        self._active.add(name)
        try:
            target = self.recipe.lookup(name, path)
            if isinstance(target, DatasetNode):
                dataset = self.run_dataset_node(name, target)
            else:
                dataset = self.materialize(target, name, f"datasets.{name}")
        finally:
            self._active.discard(name)
        self.cache[name] = dataset
        return dataset

    def materialize(self, spec: AnyDataset, name: str, path: str) -> Dataset:
        """Load one spec and apply its own operations (and save, if asked)."""
        reused = self._reuse_or_fail(spec.save_dataset, f"{path}.save_dataset", name)
        if reused is not None:
            return reused
        dataset = self.loaders[spec.source].load(spec, name)
        log.info("   dataset %s: %d rows from %s", name, len(dataset.frame), spec.source)
        dataset = apply_ops(dataset, spec, path)
        if spec.save_dataset is not None:
            self._save(spec.save_dataset.path, dataset)
        return dataset

    # --------------------------------------------------------------- private

    def _resolve_entry(self, entry: RefEntry | InlineEntry, node_name: str, index: int) -> Dataset:
        path = f"{node_name}.datasets[{index}]"
        if isinstance(entry, RefEntry):
            return self.resolve_reference(entry.dataset_name, f"{path}.dataset_name")
        return self.materialize(entry.spec, entry.name, path)

    def _reuse_or_fail(self, save: SaveSpec | DatasetOutput | None, path: str, name: str) -> Dataset | None:
        """Apply the save policy to an existing path: load it, allow a rebuild, or fail."""
        if save is None:
            return None
        target = save.path if isinstance(save, SaveSpec) else save.save_path
        if not self.store.exists(target):
            return None
        if save.reuse:
            log.info("   reusing saved dataset at %s for '%s'", target, name)
            return self._load_saved(target, name)
        if save.overwrite:
            return None
        raise RecipeError(path, f"'{target}' already holds a dataset; {SAVE_HINT}")

    def _load_saved(self, target: str, name: str) -> Dataset:
        spec = SavedDataset(source="dataset", path=target)
        return self.loaders["dataset"].load(spec, name)

    def _save(self, target: str, dataset: Dataset) -> Path:
        out = self.store.write_dataset(Path(target), dataset)
        log.info(
            "   saved dataset '%s' (%d rows%s) to %s",
            dataset.name,
            len(dataset.frame),
            ", split" if dataset.split else "",
            out,
        )
        return out
