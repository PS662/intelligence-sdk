"""Splits work on row ids, are deterministic, and fail loudly on tiny classes."""
from __future__ import annotations

import numpy as np
import pytest

from intelligence_sdk.core.splits import SplitError, kfold_ids, train_test_split_ids


def _data(n_per_class=(40, 30, 10), offset=1000):
    y = np.concatenate([[f"c{i}"] * n for i, n in enumerate(n_per_class)])
    ids = np.arange(len(y)) + offset  # ids are not positions
    return ids, y


def test_stratified_split_returns_ids_not_positions():
    ids, y = _data()
    train, test = train_test_split_ids(ids, y, 0.25, True, seed=1)
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == ids.tolist()
    assert min(train + test) >= 1000


def test_stratified_split_keeps_class_proportions():
    ids, y = _data()
    _, test = train_test_split_ids(ids, y, 0.25, True, seed=1)
    lookup = dict(zip(ids.tolist(), y))
    counts = {c: sum(1 for i in test if lookup[i] == c) for c in ("c0", "c1", "c2")}
    assert counts == {"c0": 10, "c1": 8, "c2": 2}


def test_split_is_deterministic_per_seed():
    ids, y = _data()
    a = train_test_split_ids(ids, y, 0.2, True, seed=7)
    b = train_test_split_ids(ids, y, 0.2, True, seed=7)
    c = train_test_split_ids(ids, y, 0.2, True, seed=8)
    assert a == b and a != c


def test_tiny_class_cannot_stratify():
    ids, y = _data((20, 1))
    with pytest.raises(SplitError, match="fewer than 2 rows"):
        train_test_split_ids(ids, y, 0.2, True, seed=0)
    # Unstratified still works.
    train, test = train_test_split_ids(ids, y, 0.2, False, seed=0)
    assert len(train) + len(test) == 21


def test_kfold_covers_every_id_exactly_once():
    ids, y = _data()
    folds = kfold_ids(ids, y, 5, True, seed=3)
    assert len(folds) == 5
    all_test = [i for _, te in folds for i in te]
    assert sorted(all_test) == ids.tolist()
    for tr, te in folds:
        assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(ids)


def test_kfold_requires_k_rows_per_class_when_stratified():
    ids, y = _data((20, 3))
    with pytest.raises(SplitError, match="fewer than 5 rows"):
        kfold_ids(ids, y, 5, True, seed=0)
    assert len(kfold_ids(ids, y, 5, False, seed=0)) == 5
