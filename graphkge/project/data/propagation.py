"""Helpers for building train-only typed propagation edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TypedPropagationGraph:
    edge_index: np.ndarray
    edge_type: np.ndarray


def _typed_edge_hash(
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    num_entities: int,
    num_relations: int,
) -> np.ndarray:
    src = np.asarray(edge_index[0], dtype=np.int64).astype(np.uint64, copy=False)
    dst = np.asarray(edge_index[1], dtype=np.int64).astype(np.uint64, copy=False)
    rel = np.asarray(edge_type, dtype=np.int64).astype(np.uint64, copy=False)
    return (src * np.uint64(max(1, int(num_relations))) + rel) * np.uint64(max(1, int(num_entities))) + dst


def build_train_propagation_edges(
    triples: np.ndarray,
    train_idx: np.ndarray,
    num_entities: int,
    num_relations: int,
) -> TypedPropagationGraph:
    """Build bidirectional train-only typed propagation edges."""

    if np.asarray(train_idx).size == 0:
        return TypedPropagationGraph(
            edge_index=np.empty((2, 0), dtype=np.int64),
            edge_type=np.empty((0,), dtype=np.int64),
        )
    train_triples = np.asarray(triples[np.asarray(train_idx, dtype=np.int64)], dtype=np.int64)
    src = train_triples[:, 0].astype(np.int64, copy=False)
    rel = train_triples[:, 1].astype(np.int64, copy=False)
    dst = train_triples[:, 2].astype(np.int64, copy=False)
    non_self = src != dst
    directed = np.vstack(
        [
            np.concatenate([src, dst[non_self]], axis=0).astype(np.int64, copy=False),
            np.concatenate([dst, src[non_self]], axis=0).astype(np.int64, copy=False),
        ]
    )
    edge_type = np.concatenate([rel, rel[non_self]], axis=0).astype(np.int64, copy=False)
    if directed.shape[1] <= 1:
        return TypedPropagationGraph(
            edge_index=directed.astype(np.int64, copy=False),
            edge_type=edge_type.astype(np.int64, copy=False),
        )
    hashes = _typed_edge_hash(
        directed,
        edge_type=edge_type,
        num_entities=num_entities,
        num_relations=num_relations,
    )
    _, uniq_idx = np.unique(hashes, return_index=True)
    uniq_idx = np.sort(uniq_idx.astype(np.int64, copy=False))
    return TypedPropagationGraph(
        edge_index=directed[:, uniq_idx].astype(np.int64, copy=False),
        edge_type=edge_type[uniq_idx].astype(np.int64, copy=False),
    )
