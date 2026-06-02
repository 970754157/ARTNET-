"""Evaluation metrics for link prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _compute_auc_aucpr_binary(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    scores = np.asarray(y_score, dtype=np.float32).reshape(-1)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if labels.size == 0:
        return {"auc": 0.5, "aucpr": 0.0}

    pos_total = int(labels.sum())
    neg_total = int(labels.shape[0] - pos_total)
    if pos_total <= 0 or neg_total <= 0:
        return {"auc": 0.5, "aucpr": 0.0}

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    tp = np.cumsum(sorted_labels, dtype=np.int64)
    fp = np.cumsum(1 - sorted_labels, dtype=np.int64)
    distinct_idx = np.flatnonzero(np.r_[np.diff(sorted_scores) != 0, True])

    tp_distinct = tp[distinct_idx].astype(np.float64, copy=False)
    fp_distinct = fp[distinct_idx].astype(np.float64, copy=False)

    tpr = tp_distinct / float(pos_total)
    fpr = fp_distinct / float(neg_total)
    auc = float(np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr]))

    precision = tp_distinct / np.maximum(tp_distinct + fp_distinct, 1.0)
    recall = tp_distinct / float(pos_total)
    aucpr = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    return {"auc": auc, "aucpr": aucpr}


def compute_auc_aucpr(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return _compute_auc_aucpr_binary(y_true, y_score)


def compute_auc_aucpr_from_pos_neg(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> Dict[str, float]:
    pos = np.asarray(pos_scores, dtype=np.float32).reshape(-1)
    neg = np.asarray(neg_scores, dtype=np.float32).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return {"auc": 0.5, "aucpr": 0.0}
    labels = np.concatenate(
        [
            np.ones(pos.shape[0], dtype=np.uint8),
            np.zeros(neg.shape[0], dtype=np.uint8),
        ],
        axis=0,
    )
    scores = np.concatenate([pos, neg], axis=0)
    return _compute_auc_aucpr_binary(labels, scores)


def hit_at_k(pos_scores: np.ndarray, neg_scores: np.ndarray, k: int = 10) -> float:
    """
    pos_scores: [N]
    neg_scores: [N, M]
    """
    pos = np.asarray(pos_scores).reshape(-1, 1)
    neg = np.asarray(neg_scores)
    if pos.shape[0] == 0:
        return 0.0
    all_scores = np.concatenate([pos, neg], axis=1)
    rank = np.sum(all_scores >= pos, axis=1)  # 1 is best
    hits = rank <= int(k)
    return float(np.mean(hits))
