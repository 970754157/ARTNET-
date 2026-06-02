"""Metrics for KGE relation ranking and triple scoring."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auc_aucpr(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    if len(np.unique(y_true)) < 2:
        return {"auc": 0.5, "aucpr": 0.0}
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "aucpr": float(average_precision_score(y_true, y_score)),
    }


def relation_ranking_metrics(ranks: np.ndarray, top1_hits: np.ndarray) -> Dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    top1_hits = np.asarray(top1_hits, dtype=np.float64)
    if ranks.size == 0:
        return {"mrr": 0.0, "hits1": 0.0, "hits3": 0.0, "hits10": 0.0, "top1_acc": 0.0}
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "hits1": float(np.mean(ranks <= 1.0)),
        "hits3": float(np.mean(ranks <= 3.0)),
        "hits10": float(np.mean(ranks <= 10.0)),
        "top1_acc": float(np.mean(top1_hits)),
    }
