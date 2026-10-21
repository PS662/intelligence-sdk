"""Row-id based splitting. numpy only - splits are domain logic, not a backend concern.

Everything here works on *row identifiers*, never on positions, so what gets
recorded in ``dataset.json`` can be re-selected exactly on a reloaded frame.
"""

from __future__ import annotations

import numpy as np


class SplitError(ValueError):
    """Raised when the requested split cannot be honoured (e.g. a class too small to stratify)."""


def _groups(y: np.ndarray | None, n: int, stratify: bool) -> list[np.ndarray]:
    """Positions grouped by class (stratified) or one group with everything."""
    if not stratify or y is None:
        return [np.arange(n)]
    y = np.asarray(y)
    classes, inverse = np.unique(y, return_inverse=True)
    return [np.flatnonzero(inverse == c) for c in range(len(classes))]


def _class_sizes(y: np.ndarray) -> dict[str, int]:
    classes, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(c): int(k) for c, k in zip(classes, counts, strict=True)}


def train_test_split_ids(
    ids: np.ndarray,
    y: np.ndarray | None,
    test_size: float,
    stratify: bool,
    seed: int,
) -> tuple[list, list]:
    """Return (train_ids, test_ids). Stratified splits keep at least one sample of
    every class on each side, or fail loudly."""
    ids = np.asarray(ids)
    n = len(ids)
    if n < 2:
        raise SplitError(f"need at least 2 rows to split, got {n}")
    rng = np.random.default_rng(seed)

    if stratify and y is not None:
        tiny = {c: k for c, k in _class_sizes(y).items() if k < 2}
        if tiny:
            raise SplitError(
                f"cannot stratify: class(es) with fewer than 2 rows: {tiny}. "
                "Set split.stratify: false or filter those rows out"
            )

    train_pos: list[int] = []
    test_pos: list[int] = []
    for group in _groups(y, n, stratify):
        group = rng.permutation(group)
        k = int(round(len(group) * test_size))
        k = min(max(k, 1), len(group) - 1)  # both sides non-empty within the group
        test_pos.extend(group[:k].tolist())
        train_pos.extend(group[k:].tolist())
    train_pos.sort()
    test_pos.sort()
    return ids[train_pos].tolist(), ids[test_pos].tolist()


def kfold_ids(
    ids: np.ndarray,
    y: np.ndarray | None,
    k: int,
    stratify: bool,
    seed: int,
) -> list[tuple[list, list]]:
    """Return k (train_ids, test_ids) pairs. Stratified folds require every class to
    have at least k rows."""
    ids = np.asarray(ids)
    n = len(ids)
    if k < 2:
        raise SplitError(f"k must be >= 2, got {k}")
    if n < k:
        raise SplitError(f"need at least k={k} rows for {k}-fold, got {n}")
    rng = np.random.default_rng(seed)

    if stratify and y is not None:
        small = {c: s for c, s in _class_sizes(y).items() if s < k}
        if small:
            raise SplitError(
                f"cannot stratify into {k} folds: class(es) with fewer than {k} rows: {small}. "
                "Lower folds, set stratify: false, or filter those rows out"
            )

    fold_of = np.empty(n, dtype=int)
    for group in _groups(y, n, stratify):
        group = rng.permutation(group)
        # Round-robin assignment keeps fold sizes within 1 of each other per class.
        fold_of[group] = np.arange(len(group)) % k

    folds = []
    for f in range(k):
        test_pos = np.flatnonzero(fold_of == f)
        train_pos = np.flatnonzero(fold_of != f)
        folds.append((ids[train_pos].tolist(), ids[test_pos].tolist()))
    return folds
