"""Triple split helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from project.utils.io import ensure_dir


def _compute_counts(size: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    n_val = int(np.floor(size * val_ratio))
    n_test = int(np.floor(size * test_ratio))
    if size >= 10:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    if n_val + n_test >= size:
        n_val = 0
        n_test = 0
    return n_val, n_test


def _move_unseen_entity_triples_to_train(
    triples: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    changed = True
    while changed:
        changed = False
        train_entities = set(int(x) for x in np.concatenate([triples[train_idx, 0], triples[train_idx, 2]]).tolist())
        keep_val = []
        move_val = []
        for idx in val_idx.tolist():
            triple = triples[int(idx)]
            if int(triple[0]) not in train_entities or int(triple[2]) not in train_entities:
                move_val.append(int(idx))
                changed = True
            else:
                keep_val.append(int(idx))
        keep_test = []
        move_test = []
        for idx in test_idx.tolist():
            triple = triples[int(idx)]
            if int(triple[0]) not in train_entities or int(triple[2]) not in train_entities:
                move_test.append(int(idx))
                changed = True
            else:
                keep_test.append(int(idx))
        if move_val or move_test:
            train_idx = np.asarray(np.concatenate([train_idx, np.asarray(move_val + move_test, dtype=np.int64)]), dtype=np.int64)
            val_idx = np.asarray(keep_val, dtype=np.int64)
            test_idx = np.asarray(keep_test, dtype=np.int64)
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def create_or_load_relation_stratified_splits(
    triples: np.ndarray,
    split_path: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rare_relation_min_count: int,
    ensure_entities_in_train: bool = True,
    force_rebuild: bool = False,
) -> Dict[str, np.ndarray]:
    split_path = Path(split_path)
    ensure_dir(split_path.parent)
    if split_path.exists() and not force_rebuild:
        payload = np.load(split_path)
        return {
            "train_idx": payload["train_idx"].astype(np.int64),
            "val_idx": payload["val_idx"].astype(np.int64),
            "test_idx": payload["test_idx"].astype(np.int64),
        }

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []
    rel_ids = np.unique(triples[:, 1]).astype(np.int64)

    for rel_idx in rel_ids.tolist():
        rel_rows = np.where(triples[:, 1] == int(rel_idx))[0].astype(np.int64)
        if rel_rows.size <= int(rare_relation_min_count):
            train_parts.append(rel_rows)
            continue
        rng.shuffle(rel_rows)
        n_val, n_test = _compute_counts(int(rel_rows.size), val_ratio, test_ratio)
        if n_val == 0 and n_test == 0:
            train_parts.append(rel_rows)
            continue
        test_rows = rel_rows[:n_test]
        val_rows = rel_rows[n_test : n_test + n_val]
        train_rows = rel_rows[n_test + n_val :]
        if train_rows.size == 0:
            train_parts.append(rel_rows)
            continue
        train_parts.append(train_rows)
        if val_rows.size > 0:
            val_parts.append(val_rows)
        if test_rows.size > 0:
            test_parts.append(test_rows)

    train_idx = np.concatenate(train_parts, axis=0) if train_parts else np.empty((0,), dtype=np.int64)
    val_idx = np.concatenate(val_parts, axis=0) if val_parts else np.empty((0,), dtype=np.int64)
    test_idx = np.concatenate(test_parts, axis=0) if test_parts else np.empty((0,), dtype=np.int64)

    if ensure_entities_in_train and train_idx.size > 0:
        train_idx, val_idx, test_idx = _move_unseen_entity_triples_to_train(triples, train_idx, val_idx, test_idx)

    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
