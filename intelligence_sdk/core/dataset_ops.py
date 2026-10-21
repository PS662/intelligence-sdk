"""Operations on a ``Dataset``: filter, split, folds, concatenation.

Shared by the resolver for registry entries, inline entries, dataset nodes and the
root-level options of every node, so all of them behave identically. Order inside
one spec is always filter -> split -> folds. Ids are never renumbered.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from intelligence_sdk.core.errors import RecipeError
from intelligence_sdk.core.splits import SplitError, kfold_ids, train_test_split_ids
from intelligence_sdk.core.types import Dataset, Folds, Split

if TYPE_CHECKING:
    from intelligence_sdk.core.recipe import FilterSpec, FoldsSpec, OpsMixin, SplitSpec


def _read_ids(spec: Any, dataset: Dataset, path: str) -> list[Any]:
    from intelligence_sdk.core.recipe import SplitRef

    if isinstance(spec, SplitRef):
        if dataset.split is None:
            raise RecipeError(path, f"filters by split '{spec.split}' but dataset '{dataset.name}' has no split")
        return list(dataset.split.train_ids if spec.split == "train" else dataset.split.test_ids)
    if isinstance(spec, list):
        return spec
    file = Path(spec)
    if not file.exists():
        raise RecipeError(path, f"id file not found: {file}")
    frame = pd.read_csv(file)
    if frame.shape[1] != 1:
        raise RecipeError(path, f"id file {file} must have exactly one column, found {frame.shape[1]}")
    return frame.iloc[:, 0].tolist()


def apply_filter(dataset: Dataset, spec: FilterSpec | None, path: str) -> Dataset:
    if spec is None or (spec.include is None and spec.exclude is None):
        return dataset
    before = len(dataset.frame)
    keep = dataset.ids
    if spec.include is not None:
        ids = _read_ids(spec.include, dataset, f"{path}.include")
        keep = keep[keep.isin(ids)]
    if spec.exclude is not None:
        ids = _read_ids(spec.exclude, dataset, f"{path}.exclude")
        keep = keep[~keep.isin(ids)]
    if len(keep) == 0:
        raise RecipeError(path, f"no rows left after filtering (started with {before})")
    out = dataset.restrict(keep.tolist())
    out.filters.append(
        {"at": path, "filter": spec.model_dump(mode="json", exclude_none=True), "rows": int(len(out.frame))}
    )
    return out


def apply_split(dataset: Dataset, spec: SplitSpec, seed: int, path: str) -> Dataset:
    y = dataset.y()
    try:
        train_ids, test_ids = train_test_split_ids(
            dataset.ids.to_numpy(),
            None if y is None else y.to_numpy(),
            spec.test_size,
            spec.stratify and y is not None,
            seed,
        )
    except SplitError as exc:
        raise RecipeError(path, str(exc)) from exc
    dataset.split = Split(spec.test_size, spec.stratify, seed, train_ids, test_ids)
    return dataset


def apply_folds(dataset: Dataset, spec: FoldsSpec, seed: int, path: str) -> Dataset:
    y = dataset.y()
    try:
        pairs = kfold_ids(
            dataset.ids.to_numpy(),
            None if y is None else y.to_numpy(),
            spec.k,
            spec.stratify and y is not None,
            seed,
        )
    except SplitError as exc:
        raise RecipeError(path, str(exc)) from exc
    dataset.folds = Folds(spec.k, spec.stratify, seed, [test for _, test in pairs])
    return dataset


def apply_ops(dataset: Dataset, ops: OpsMixin, path: str) -> Dataset:
    """filter -> split -> folds, as declared on ``ops`` (a spec or a node's root)."""
    dataset = apply_filter(dataset, ops.filter, f"{path}.filter")
    if ops.split is not None:
        dataset = apply_split(dataset, ops.split, ops.seed, f"{path}.split")
    if ops.folds is not None:
        dataset = apply_folds(dataset, ops.folds, ops.seed, f"{path}.folds")
    return dataset


def concat(datasets: list[Dataset], name: str, ignore_duplicates: bool, path: str) -> Dataset:
    """Row-wise concatenation of datasets with the same columns and target.

    Row ids must be unique across inputs: a duplicate is an error unless
    ``ignore_duplicates`` keeps the first occurrence. A single input keeps its split
    and folds; when several inputs are combined their splits and folds do not carry
    over and the consolidated dataset gets its own from the node's root options.
    """
    if len(datasets) == 1:
        single = datasets[0]
        return Dataset(
            name=name,
            frame=single.frame,
            target=single.target,
            split=single.split,
            folds=single.folds,
            lineage=[single.name],
            filters=list(single.filters),
        )

    first = datasets[0]
    columns = list(first.frame.columns)
    for ds in datasets[1:]:
        if set(ds.frame.columns) != set(columns):
            missing = sorted(set(columns) ^ set(ds.frame.columns))
            raise RecipeError(
                path, f"cannot combine '{first.name}' and '{ds.name}': columns differ on {', '.join(map(str, missing))}"
            )
        if ds.target != first.target:
            raise RecipeError(
                path, f"cannot combine '{first.name}' (target {first.target}) and '{ds.name}' (target {ds.target})"
            )

    frame = pd.concat([ds.frame[columns] for ds in datasets], axis=0)
    dup = frame.index.duplicated(keep="first")
    if dup.any():
        if not ignore_duplicates:
            sample = frame.index[dup].unique().tolist()[:5]
            raise RecipeError(
                path,
                f"{int(dup.sum())} duplicate row id(s) across datasets (e.g. {sample}); "
                "set ignore_duplicates: true to keep the first occurrence, or give the inputs distinct ids",
            )
        frame = frame[~dup]
    frame.index.name = first.frame.index.name or "row_id"
    return Dataset(
        name=name,
        frame=frame,
        target=first.target,
        lineage=[ds.name for ds in datasets],
        filters=[f for ds in datasets for f in ds.filters],
    )
