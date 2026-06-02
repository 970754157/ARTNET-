"""Training and evaluation loops."""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project.data.sampler import SubgraphData
from project.metrics import compute_auc_aucpr, compute_auc_aucpr_from_pos_neg, hit_at_k
from project.utils.io import CSVLogger, ensure_dir
from project.utils.logging import StepProgressReporter, log_event
from project.utils.plotting import plot_loss_mean_std_var, plot_val_curve
from project.utils.seed import capture_rng_state, restore_rng_state


def _edge_hash(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    return src.astype(np.uint64) * np.uint64(num_nodes) + dst.astype(np.uint64)


def _edge_exists(src: np.ndarray, dst: np.ndarray, sorted_hash: np.ndarray, num_nodes: int) -> np.ndarray:
    h = _edge_hash(src, dst, num_nodes)
    pos = np.searchsorted(sorted_hash, h, side="left")
    ok = pos < sorted_hash.shape[0]
    out = np.zeros(h.shape[0], dtype=bool)
    out[ok] = sorted_hash[pos[ok]] == h[ok]
    return out


def _canonicalize_undirected_pairs(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    src64 = src.astype(np.int64, copy=False)
    dst64 = dst.astype(np.int64, copy=False)
    return (
        np.minimum(src64, dst64).astype(np.int64, copy=False),
        np.maximum(src64, dst64).astype(np.int64, copy=False),
    )


def _undirected_edge_hash(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    lo, hi = _canonicalize_undirected_pairs(src, dst)
    return lo.astype(np.uint64, copy=False) * np.uint64(num_nodes) + hi.astype(np.uint64, copy=False)


def _undirected_edge_exists(
    src: np.ndarray,
    dst: np.ndarray,
    sorted_hash: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    h = _undirected_edge_hash(src, dst, num_nodes)
    pos = np.searchsorted(sorted_hash, h, side="left")
    ok = pos < sorted_hash.shape[0]
    out = np.zeros(h.shape[0], dtype=bool)
    out[ok] = sorted_hash[pos[ok]] == h[ok]
    return out


def sample_negative_edges_undirected(
    pos_edges: np.ndarray,
    num_nodes: int,
    num_neg: int,
    sorted_edge_hash: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    src = np.repeat(pos_edges[0], num_neg).astype(np.int64, copy=False)
    dst = rng.integers(0, num_nodes, size=src.shape[0], dtype=np.int64)
    valid = (src != dst) & (~_undirected_edge_exists(src, dst, sorted_edge_hash, num_nodes))
    max_rounds = 50
    rounds = 0
    while not valid.all() and rounds < max_rounds:
        bad = np.where(~valid)[0]
        dst[bad] = rng.integers(0, num_nodes, size=bad.shape[0], dtype=np.int64)
        valid = (src != dst) & (~_undirected_edge_exists(src, dst, sorted_edge_hash, num_nodes))
        rounds += 1
    neg_src, neg_dst = _canonicalize_undirected_pairs(src, dst)
    return np.vstack([neg_src, neg_dst]).astype(np.int64, copy=False)


def sample_negative_edges_art_subset_undirected(
    pos_edges: np.ndarray,
    num_nodes: int,
    num_neg: int,
    sorted_edge_hash: np.ndarray,
    rng: np.random.Generator,
    art_candidates: np.ndarray,
) -> np.ndarray:
    art_candidates = np.asarray(art_candidates, dtype=np.int64)
    if art_candidates.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    src = np.repeat(pos_edges[0], num_neg).astype(np.int64, copy=False)
    dst = _sample_from_candidates(art_candidates, src.shape[0], rng)
    valid = (src != dst) & (~_undirected_edge_exists(src, dst, sorted_edge_hash, num_nodes))
    max_rounds = 100
    rounds = 0
    while not valid.all() and rounds < max_rounds:
        bad = np.where(~valid)[0]
        if bad.size == 0:
            break
        dst[bad] = _sample_from_candidates(art_candidates, bad.shape[0], rng)
        valid = (src != dst) & (~_undirected_edge_exists(src, dst, sorted_edge_hash, num_nodes))
        rounds += 1
    if not valid.all():
        return np.empty((2, 0), dtype=np.int64)
    neg_src, neg_dst = _canonicalize_undirected_pairs(src, dst)
    return np.vstack([neg_src, neg_dst]).astype(np.int64, copy=False)


def sample_negative_edges_art_any_subset_undirected(
    pos_edges: np.ndarray,
    num_nodes: int,
    num_neg: int,
    sorted_edge_hash: np.ndarray,
    rng: np.random.Generator,
    art_candidates: np.ndarray,
    other_candidates: np.ndarray,
) -> np.ndarray:
    art_candidates = np.asarray(art_candidates, dtype=np.int64)
    other_candidates = np.asarray(other_candidates, dtype=np.int64)
    if art_candidates.size == 0 or other_candidates.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    src = np.repeat(pos_edges[0], num_neg).astype(np.int64, copy=False)
    choose_art_src = rng.integers(0, 2, size=src.shape[0], dtype=np.int64).astype(np.bool_)
    dst = np.empty((src.shape[0],), dtype=np.int64)
    sampled_src = src.copy()
    if choose_art_src.any():
        sampled_src[choose_art_src] = _sample_from_candidates(art_candidates, int(choose_art_src.sum()), rng)
        dst[choose_art_src] = _sample_from_candidates(other_candidates, int(choose_art_src.sum()), rng)
    if (~choose_art_src).any():
        dst[~choose_art_src] = _sample_from_candidates(art_candidates, int((~choose_art_src).sum()), rng)

    valid = (sampled_src != dst) & (~_undirected_edge_exists(sampled_src, dst, sorted_edge_hash, num_nodes))
    max_rounds = 100
    rounds = 0
    while not valid.all() and rounds < max_rounds:
        bad = np.where(~valid)[0]
        if bad.size == 0:
            break
        bad_choose_art_src = choose_art_src[bad]
        if bad_choose_art_src.any():
            sampled_src[bad[bad_choose_art_src]] = _sample_from_candidates(
                art_candidates,
                int(bad_choose_art_src.sum()),
                rng,
            )
            dst[bad[bad_choose_art_src]] = _sample_from_candidates(
                other_candidates,
                int(bad_choose_art_src.sum()),
                rng,
            )
        if (~bad_choose_art_src).any():
            dst[bad[~bad_choose_art_src]] = _sample_from_candidates(
                art_candidates,
                int((~bad_choose_art_src).sum()),
                rng,
            )
        valid = (sampled_src != dst) & (~_undirected_edge_exists(sampled_src, dst, sorted_edge_hash, num_nodes))
        rounds += 1
    if not valid.all():
        return np.empty((2, 0), dtype=np.int64)
    neg_src, neg_dst = _canonicalize_undirected_pairs(sampled_src, dst)
    return np.vstack([neg_src, neg_dst]).astype(np.int64, copy=False)


def sample_negative_edges_from_candidates_undirected(
    pos_edges: np.ndarray,
    num_nodes: int,
    num_neg: int,
    sorted_edge_hash: np.ndarray,
    rng: np.random.Generator,
    candidates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0 or pos_edges.shape[1] == 0 or num_neg <= 0:
        return np.empty((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.bool_)

    src = np.repeat(pos_edges[0], num_neg).astype(np.int64, copy=False)
    dst = _sample_from_candidates(candidates, src.shape[0], rng)
    valid = (src != dst) & (~_undirected_edge_exists(src, dst, sorted_edge_hash, num_nodes))
    max_rounds = 100
    rounds = 0
    while not valid.all() and rounds < max_rounds:
        bad = np.where(~valid)[0]
        if bad.size == 0:
            break
        dst[bad] = _sample_from_candidates(candidates, bad.shape[0], rng)
        valid = (src != dst) & (~_undirected_edge_exists(src, dst, sorted_edge_hash, num_nodes))
        rounds += 1
    neg_src, neg_dst = _canonicalize_undirected_pairs(src, dst)
    neg_edges = np.vstack([neg_src, neg_dst]).astype(np.int64, copy=False)
    return neg_edges, valid.astype(np.bool_, copy=False)


def sample_negative_edges(
    pos_edges: np.ndarray,
    num_nodes: int,
    num_neg: int,
    sorted_edge_hash: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    src = np.repeat(pos_edges[0], num_neg)
    dst = rng.integers(0, num_nodes, size=src.shape[0], dtype=np.int64)
    valid = (src != dst) & (~_edge_exists(src, dst, sorted_edge_hash, num_nodes))
    max_rounds = 50
    rounds = 0
    while not valid.all() and rounds < max_rounds:
        bad = np.where(~valid)[0]
        dst[bad] = rng.integers(0, num_nodes, size=bad.shape[0], dtype=np.int64)
        valid = (src != dst) & (~_edge_exists(src, dst, sorted_edge_hash, num_nodes))
        rounds += 1
    return np.vstack([src, dst]).astype(np.int64)


def sample_negative_edges_center_aware(
    pos_edges: np.ndarray,
    num_nodes: int,
    num_neg: int,
    sorted_edge_hash: np.ndarray,
    rng: np.random.Generator,
    center_mask: np.ndarray,
) -> np.ndarray:
    src = np.repeat(pos_edges[0], num_neg)
    dst = np.repeat(pos_edges[1], num_neg)
    center_mask = np.asarray(center_mask, dtype=np.bool_)
    keep_src = center_mask[src]
    keep_dst = (~keep_src) & center_mask[dst]
    if (~(keep_src | keep_dst)).any():
        raise RuntimeError("chunk_full negative sampling received an edge with no center-chunk endpoint.")

    neg_src = src.copy()
    neg_dst = dst.copy()
    if keep_src.any():
        neg_dst[keep_src] = rng.integers(0, num_nodes, size=int(keep_src.sum()), dtype=np.int64)
    if keep_dst.any():
        neg_src[keep_dst] = rng.integers(0, num_nodes, size=int(keep_dst.sum()), dtype=np.int64)

    valid = (neg_src != neg_dst) & (~_edge_exists(neg_src, neg_dst, sorted_edge_hash, num_nodes))
    max_rounds = 50
    rounds = 0
    while not valid.all() and rounds < max_rounds:
        bad = np.where(~valid)[0]
        if bad.size == 0:
            break
        bad_keep_src = keep_src[bad]
        bad_keep_dst = keep_dst[bad]
        if bad_keep_src.any():
            neg_dst[bad[bad_keep_src]] = rng.integers(
                0,
                num_nodes,
                size=int(bad_keep_src.sum()),
                dtype=np.int64,
            )
        if bad_keep_dst.any():
            neg_src[bad[bad_keep_dst]] = rng.integers(
                0,
                num_nodes,
                size=int(bad_keep_dst.sum()),
                dtype=np.int64,
            )
        valid = (neg_src != neg_dst) & (~_edge_exists(neg_src, neg_dst, sorted_edge_hash, num_nodes))
        rounds += 1
    return np.vstack([neg_src, neg_dst]).astype(np.int64)


def _sample_from_candidates(
    candidates: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if candidates.size == 0 or size <= 0:
        return np.empty((0,), dtype=np.int64)
    idx = rng.integers(0, candidates.shape[0], size=size, dtype=np.int64)
    return candidates[idx].astype(np.int64, copy=False)


def sample_negative_edges_art_subset(
    pos_edges: np.ndarray,
    num_nodes: int,
    num_neg: int,
    sorted_edge_hash: np.ndarray,
    rng: np.random.Generator,
    art_candidates: np.ndarray,
) -> np.ndarray:
    art_candidates = np.asarray(art_candidates, dtype=np.int64)
    if art_candidates.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    src = np.repeat(pos_edges[0], num_neg)
    dst = _sample_from_candidates(art_candidates, src.shape[0], rng)
    valid = (src != dst) & (~_edge_exists(src, dst, sorted_edge_hash, num_nodes))
    max_rounds = 100
    rounds = 0
    while not valid.all() and rounds < max_rounds:
        bad = np.where(~valid)[0]
        if bad.size == 0:
            break
        dst[bad] = _sample_from_candidates(art_candidates, bad.shape[0], rng)
        valid = (src != dst) & (~_edge_exists(src, dst, sorted_edge_hash, num_nodes))
        rounds += 1
    if not valid.all():
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack([src, dst]).astype(np.int64)


def sample_negative_edges_center_aware_art_subset(
    pos_edges: np.ndarray,
    num_nodes: int,
    num_neg: int,
    sorted_edge_hash: np.ndarray,
    rng: np.random.Generator,
    center_mask: np.ndarray,
    art_candidates: np.ndarray,
) -> np.ndarray:
    art_candidates = np.asarray(art_candidates, dtype=np.int64)
    if art_candidates.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    src = np.repeat(pos_edges[0], num_neg)
    dst = np.repeat(pos_edges[1], num_neg)
    center_mask = np.asarray(center_mask, dtype=np.bool_)
    keep_src = center_mask[src]
    keep_dst = (~keep_src) & center_mask[dst]
    if (~(keep_src | keep_dst)).any():
        raise RuntimeError("art-art chunk_full negative sampling received an edge with no center-chunk endpoint.")

    neg_src = src.copy()
    neg_dst = dst.copy()
    if keep_src.any():
        neg_dst[keep_src] = _sample_from_candidates(art_candidates, int(keep_src.sum()), rng)
    if keep_dst.any():
        neg_src[keep_dst] = _sample_from_candidates(art_candidates, int(keep_dst.sum()), rng)

    valid = (neg_src != neg_dst) & (~_edge_exists(neg_src, neg_dst, sorted_edge_hash, num_nodes))
    max_rounds = 100
    rounds = 0
    while not valid.all() and rounds < max_rounds:
        bad = np.where(~valid)[0]
        if bad.size == 0:
            break
        bad_keep_src = keep_src[bad]
        bad_keep_dst = keep_dst[bad]
        if bad_keep_src.any():
            neg_dst[bad[bad_keep_src]] = _sample_from_candidates(
                art_candidates,
                int(bad_keep_src.sum()),
                rng,
            )
        if bad_keep_dst.any():
            neg_src[bad[bad_keep_dst]] = _sample_from_candidates(
                art_candidates,
                int(bad_keep_dst.sum()),
                rng,
            )
        valid = (neg_src != neg_dst) & (~_edge_exists(neg_src, neg_dst, sorted_edge_hash, num_nodes))
        rounds += 1
    if not valid.all():
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack([neg_src, neg_dst]).astype(np.int64)


class ChunkAwareEdgeBatchSampler:
    """Sample positive edges in chunk blocks for better IO locality."""

    def __init__(
        self,
        train_edges: torch.Tensor,
        node_chunk_ids: np.ndarray,
        batch_size: int,
        chunk_block_steps: int,
        seed: int,
    ):
        edges = train_edges.detach().cpu().numpy().astype(np.int64)
        self.edges = edges
        self.batch_size = int(batch_size)
        self.chunk_block_steps = max(1, int(chunk_block_steps))
        self.rng = np.random.default_rng(seed)

        src_chunks = node_chunk_ids[edges[0]]
        self.chunk_ids = np.unique(src_chunks).astype(np.int32)
        self.chunk_to_indices = {
            int(c): np.where(src_chunks == c)[0].astype(np.int64) for c in self.chunk_ids
        }
        counts = np.asarray([self.chunk_to_indices[int(c)].shape[0] for c in self.chunk_ids], dtype=np.float64)
        self.chunk_probs = counts / counts.sum()

    def next_batch(self, global_step: int) -> np.ndarray:
        block_pos = global_step % self.chunk_block_steps
        if block_pos == 0 or not hasattr(self, "_active_chunk"):
            self._active_chunk = int(self.rng.choice(self.chunk_ids, p=self.chunk_probs))
        pool = self.chunk_to_indices[self._active_chunk]
        if pool.shape[0] == 0:
            idx = self.rng.integers(0, self.edges.shape[1], size=self.batch_size, dtype=np.int64)
        else:
            replace = pool.shape[0] < self.batch_size
            idx = self.rng.choice(pool, size=self.batch_size, replace=replace)
        return self.edges[:, idx]

    @property
    def active_chunk(self) -> int:
        return int(getattr(self, "_active_chunk", -1))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "active_chunk": getattr(self, "_active_chunk", None),
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
        if state.get("active_chunk") is not None:
            self._active_chunk = int(state["active_chunk"])


class UniformEdgeBatchSampler:
    """Uniformly sample training positive edges from the global train split."""

    def __init__(self, train_edges: torch.Tensor, batch_size: int, seed: int):
        self.edges = train_edges.detach().cpu().numpy().astype(np.int64, copy=False)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

    def next_batch(self, _global_step: int) -> np.ndarray:
        replace = self.edges.shape[1] < self.batch_size
        idx = self.rng.choice(self.edges.shape[1], size=self.batch_size, replace=replace)
        return self.edges[:, idx]

    @property
    def active_chunk(self) -> int:
        return -1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]


class ChunkFullChunkSampler:
    """Iterate train chunks in shuffled order, one chunk per optimization step."""

    def __init__(self, chunk_ids: List[int], seed: int):
        self.chunk_ids = np.asarray(sorted(int(x) for x in chunk_ids), dtype=np.int32)
        self.rng = np.random.default_rng(seed)
        self._order = np.empty((0,), dtype=np.int32)
        self._pos = 0
        self._active_chunk = -1
        self._refresh_order()

    def _refresh_order(self) -> None:
        if self.chunk_ids.size == 0:
            self._order = np.empty((0,), dtype=np.int32)
            self._pos = 0
            return
        self._order = self.rng.permutation(self.chunk_ids)
        self._pos = 0

    def next_chunk(self) -> int:
        if self._order.size == 0:
            raise RuntimeError("No train chunks available for chunk_full mode.")
        if self._pos >= int(self._order.shape[0]):
            self._refresh_order()
        self._active_chunk = int(self._order[self._pos])
        self._pos += 1
        return self._active_chunk

    @property
    def active_chunk(self) -> int:
        return int(self._active_chunk)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "order": self._order.tolist(),
            "pos": int(self._pos),
            "active_chunk": int(self._active_chunk),
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
        order = state.get("order")
        if order is not None:
            self._order = np.asarray(order, dtype=np.int32)
        self._pos = int(state.get("pos", 0))
        self._active_chunk = int(state.get("active_chunk", -1))


@dataclass
class TrainerContext:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    criterion: nn.Module
    graph: Any
    text_store: Any
    image_provider: Any
    neighbor_sampler: Any
    train_edges: Any
    val_edges: Any
    test_edges: Any
    edge_hash_sorted: Any
    device: torch.device
    run_dir: Path
    cfg: Any
    resolved_paths: Dict[str, Path]
    image_transform: Any
    train_message_edges: Any = None
    train_propagation_edges: Any = None
    effective_max_steps: int = 0


class Trainer:
    def __init__(self, ctx: TrainerContext):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.device = ctx.device
        self.subgraph_mode = str(self.cfg.data.subgraph_mode)
        self.model = ctx.model.to(self.device)
        if self.device.type == "cuda" and hasattr(self.model, "image_encoder"):
            self.model.image_encoder.to(memory_format=torch.channels_last)

        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("-inf")
        self.early_stop_enabled = bool(self.cfg.optim.early_stop_enabled)
        self.early_stop_patience = int(self.cfg.optim.early_stop_patience)
        self.early_stop_min_delta = float(self.cfg.optim.early_stop_min_delta)
        self.early_stop_bad_evals = 0
        self.stopped_early = False
        self.early_stop_reason = ""
        self.loss_history = []
        self.val_steps = []
        self.val_values = []
        self.rng = np.random.default_rng(self.cfg.seed + 2024)
        if self.subgraph_mode != "neighbor_sample":
            raise ValueError(f"Unsupported subgraph mode: {self.subgraph_mode}")
        self.batch_sampler = UniformEdgeBatchSampler(
            train_edges=self.ctx.train_edges,
            batch_size=self.cfg.data.batch_edges,
            seed=self.cfg.seed,
        )

        self.pin_memory = bool(self.cfg.runtime.pin_memory and self.device.type == "cuda")
        self.non_blocking_transfer = bool(
            self.cfg.runtime.non_blocking_transfer and self.device.type == "cuda"
        )
        self.prefetch_batches = max(0, int(self.cfg.runtime.prefetch_batches))
        self.use_amp = bool(self.device.type == "cuda" and self.cfg.runtime.amp_enabled)
        self.amp_dtype = (
            torch.float16 if self.cfg.runtime.amp_dtype == "float16" else torch.bfloat16
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.heartbeat_sec = float(self.cfg.runtime.heartbeat_sec)
        self.log_first_n_steps = int(self.cfg.runtime.log_first_n_steps)
        self.log_every_steps = int(self.cfg.runtime.log_every_steps)
        self.last_heartbeat_ts = time.perf_counter()
        self.first_batch_logged = False
        self.first_forward_logged = False
        self.last_step_stats: Dict[str, float] = {}
        self.effective_batch_edges = int(self.cfg.data.batch_edges)
        self.effective_max_steps = int(self.ctx.effective_max_steps or self.cfg.optim.max_steps)
        self.effective_text_batch_size = int(self.cfg.model.text_batch_size)
        self.effective_image_forward_batch_size = int(self.cfg.model.image_forward_batch_size)
        self.effective_num_workers = int(self.cfg.dataloader.num_workers)
        self.effective_prefetch_batches = int(self.cfg.runtime.prefetch_batches)
        self.prefetch_worker_count = min(max(1, self.effective_prefetch_batches), 4)
        self.global_art_node_indices = np.flatnonzero(
            self.ctx.graph.node_is_art.astype(np.bool_, copy=False)
        ).astype(np.int64, copy=False)

        self.checkpoints_dir = self.ctx.run_dir / "checkpoints"
        self.plots_dir = self.ctx.run_dir / "plots"
        ensure_dir(self.checkpoints_dir)
        ensure_dir(self.plots_dir)

        self.metrics_logger = CSVLogger(
            self.ctx.run_dir / "metrics.csv",
            fieldnames=[
                "step",
                "epoch",
                "train_loss",
                "loss_mean",
                "loss_std",
                "loss_var",
                "lr",
                "active_chunk",
                "active_chunk_name",
                "seed_edges",
                "img_real_hit",
                "img_no_image",
                "img_missing_local",
                "img_real_ratio",
                "text_cache_hit",
                "text_cache_miss",
                "text_cache_hit_rate",
                "text_cache_write_count",
                "prepare_cpu_sec",
                "sub_nodes",
                "sub_nodes_before_trim",
                "sub_nodes_after_trim",
                "seed_nodes",
                "real_image_nodes_before_trim",
                "real_image_nodes_after_trim",
                "trimmed_nodes",
                "trimmed_image_nodes",
                "seed_image_cap_exceeded",
                "masked_supervision_edge_directions",
                "masked_supervision_unique_edges",
                "query_edges_in_message_graph",
                "message_edges",
                "chunk_intra_edges",
                "pos_edges",
                "neg_edges",
                "supervision_batches",
                "supervision_batch_edges",
                "real_image_count",
                "transfer_sec",
                "forward_sec",
                "backward_sec",
                "optimizer_sec",
                "text_encode_sec",
                "image_decode_sec",
                "image_forward_sec",
                "device_type",
                "gpu_mem_alloc_mb",
                "gpu_mem_reserved_mb",
                "effective_batch_edges",
                "effective_max_steps",
                "effective_text_batch_size",
                "effective_image_forward_batch_size",
                "effective_num_workers",
                "effective_prefetch_batches",
                "effective_prefetch_workers",
                "stopped_early",
                "early_stop_bad_evals",
                "val_auc",
                "val_aucpr",
                "val_hit10",
                "val_art_auc",
                "val_art_aucpr",
                "val_art_hit10",
                "val_art_pos_edges",
                "val_art_any_auc",
                "val_art_any_aucpr",
                "val_art_any_hit10",
                "val_art_any_pos_edges",
                "time_sec",
            ],
        )

        self.image_transform = self.ctx.image_transform
        train_prop = self.ctx.train_propagation_edges
        if train_prop is not None and int(train_prop.shape[1]) > 0:
            train_prop_np = train_prop.detach().cpu().numpy().astype(np.int64, copy=False)
            self.train_propagation_hash_sorted = np.sort(
                _undirected_edge_hash(
                    train_prop_np[0],
                    train_prop_np[1],
                    self.ctx.graph.num_nodes,
                )
            )
        else:
            self.train_propagation_hash_sorted = np.empty((0,), dtype=np.uint64)
        self.final_test_message_edges_by_chunk: Dict[int, np.ndarray] = {}
        self._clear_runtime_counters()
        self._validate_device_setup()

    def _clear_runtime_counters(self) -> None:
        if hasattr(self.model, "consume_runtime_stats"):
            self.model.consume_runtime_stats()

    @staticmethod
    def _normalize_chunk_edge_dict(raw: Any) -> Dict[int, np.ndarray]:
        if raw is None:
            return {}
        out: Dict[int, np.ndarray] = {}
        for k, v in raw.items():
            cid = int(k)
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy().astype(np.int64)
            else:
                arr = np.asarray(v, dtype=np.int64)
            out[cid] = arr
        return out

    @staticmethod
    def _normalize_chunk_hash_dict(raw: Any) -> Dict[int, np.ndarray]:
        if raw is None:
            return {}
        out: Dict[int, np.ndarray] = {}
        for k, v in raw.items():
            cid = int(k)
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy().astype(np.uint64)
            else:
                arr = np.asarray(v, dtype=np.uint64)
            out[cid] = arr
        return out

    @staticmethod
    def _device_matches(actual: torch.device, expected: torch.device) -> bool:
        if actual.type != expected.type:
            return False
        if actual.type != "cuda":
            return actual == expected
        if expected.index is None:
            return True
        return actual.index == expected.index

    def _validate_device_setup(self) -> None:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in model.")
        if any(not self._device_matches(p.device, self.device) for p in trainable_params):
            raise RuntimeError("Trainable parameters are not all placed on the resolved device.")
        if self.device.type == "cuda":
            for module_name in ("image_encoder", "predictor"):
                module = getattr(self.model, module_name, None)
                if module is None:
                    continue
                params = list(module.parameters())
                if params and any(not self._device_matches(p.device, self.device) for p in params):
                    raise RuntimeError(f"{module_name} parameters are not on CUDA.")
        log_event(
            "[DEVICE]",
            f"trainer device check passed: device={self.device} trainable_params={len(trainable_params)}",
        )

    def _cuda_memory_stats_mb(self) -> Tuple[float, float]:
        if self.device.type != "cuda":
            return 0.0, 0.0
        allocated = float(torch.cuda.memory_allocated(self.device) / (1024**2))
        reserved = float(torch.cuda.memory_reserved(self.device) / (1024**2))
        return allocated, reserved

    def _should_log_heartbeat(self, now_ts: float) -> bool:
        step = self.global_step
        if step <= self.log_first_n_steps:
            return True
        if step > 0 and step % self.log_every_steps == 0:
            return True
        if now_ts - self.last_heartbeat_ts >= self.heartbeat_sec:
            return True
        return False

    def _emit_heartbeat(
        self,
        loss_val: float,
        lr: float,
        batch: Dict[str, Any],
        runtime_stats: Dict[str, float],
        image_stats: Dict[str, float],
        forward_sec: float,
        backward_sec: float,
        optimizer_sec: float,
    ) -> None:
        alloc_mb, reserved_mb = self._cuda_memory_stats_mb()
        log_event(
            "[HEARTBEAT]",
            (
                f"step={self.global_step} epoch={self.epoch} active_chunk={batch['active_chunk']} "
                f"active_chunk_name={batch.get('active_chunk_name', 'unknown')} "
                f"loss={loss_val:.6f} lr={lr:.6g} "
                f"seed_edges={batch.get('seed_edges', batch['pos_edges'])} "
                f"sub_nodes={batch['sub_nodes']} seed_nodes={batch['seed_nodes']} "
                f"sub_nodes_before_trim={batch.get('sub_nodes_before_trim', batch['sub_nodes'])} "
                f"sub_nodes_after_trim={batch.get('sub_nodes_after_trim', batch['sub_nodes'])} "
                f"real_image_nodes_before_trim={batch.get('real_image_nodes_before_trim', batch['real_image_count'])} "
                f"real_image_nodes_after_trim={batch.get('real_image_nodes_after_trim', batch['real_image_count'])} "
                f"trimmed_nodes={batch.get('trimmed_nodes', 0)} "
                f"trimmed_image_nodes={batch.get('trimmed_image_nodes', 0)} "
                f"masked_supervision_edge_directions={batch.get('masked_supervision_edge_directions', 0)} "
                f"masked_supervision_unique_edges={batch.get('masked_supervision_unique_edges', 0)} "
                f"query_edges_in_message_graph={batch.get('query_edges_in_message_graph', 0)} "
                f"message_edges={batch.get('message_edges', batch.get('chunk_intra_edges', 0))} "
                f"chunk_intra_edges={batch.get('chunk_intra_edges', 0)} "
                f"pos_edges={batch['pos_edges']} neg_edges={batch['neg_edges']} "
                f"supervision_batches={batch.get('supervision_batches', 1)} "
                f"supervision_batch_edges={batch.get('supervision_batch_edges', batch['pos_edges'])} "
                f"real_image_count={batch['real_image_count']} "
                f"prepare_cpu_sec={batch['prepare_cpu_sec']:.2f} transfer_sec={batch['transfer_sec']:.2f} "
                f"forward_sec={forward_sec:.2f} backward_sec={backward_sec:.2f} optimizer_sec={optimizer_sec:.2f} "
                f"text_hit_rate={runtime_stats['text_cache_hit_rate']:.4f} "
                f"text_encode_sec={runtime_stats.get('text_encode_sec', 0.0):.2f} "
                f"image_decode_sec={runtime_stats.get('image_decode_sec', batch['image_decode_sec']):.2f} "
                f"image_forward_sec={runtime_stats.get('image_forward_sec', 0.0):.2f} "
                f"img_real_hit={int(image_stats['img_real_hit'])} img_no_image={int(image_stats['img_no_image'])} "
                f"img_missing_local={int(image_stats['img_missing_local'])} "
                f"gpu_mem_alloc_mb={alloc_mb:.1f} gpu_mem_reserved_mb={reserved_mb:.1f}"
            ),
        )
        self.last_heartbeat_ts = time.perf_counter()

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        if self.device.type != "cuda" or not isinstance(exc, RuntimeError):
            return False
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error: out of memory" in msg

    def _handle_cuda_oom(self, exc: RuntimeError) -> None:
        alloc_mb, reserved_mb = self._cuda_memory_stats_mb()
        log_event(
            "[ERROR]",
            (
                f"CUDA OOM: batch_edges={self.effective_batch_edges} "
                f"text_batch_size={self.effective_text_batch_size} "
                f"image_forward_batch_size={self.effective_image_forward_batch_size} "
                f"num_workers={self.effective_num_workers} "
                f"prefetch_batches={self.effective_prefetch_batches} "
                f"prefetch_workers={self.prefetch_worker_count} "
                f"prepare_cpu_sec={self.last_step_stats.get('prepare_cpu_sec', 0.0):.2f} "
                f"transfer_sec={self.last_step_stats.get('transfer_sec', 0.0):.2f} "
                f"forward_sec={self.last_step_stats.get('forward_sec', 0.0):.2f} "
                f"image_forward_sec={self.last_step_stats.get('image_forward_sec', 0.0):.2f} "
                f"gpu_mem_alloc_mb={alloc_mb:.1f} gpu_mem_reserved_mb={reserved_mb:.1f}"
            ),
        )
        log_event(
            "[ERROR]",
            "Suggested downscale order: 1) image_forward_batch_size 2) prefetch_batches 3) text_batch_size 4) batch_edges",
        )
        raise exc

    def load_checkpoint(self, ckpt_path: Path) -> None:
        log_event("[CHECKPOINT]", f"loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.ctx.optimizer.load_state_dict(ckpt["optimizer"])
        if self.ctx.scheduler is not None and ckpt.get("scheduler") is not None:
            self.ctx.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None and self.use_amp:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = int(ckpt["global_step"])
        self.epoch = int(ckpt["epoch"])
        self.best_metric = float(ckpt["best_metric"])
        self.early_stop_bad_evals = int(ckpt.get("early_stop_bad_evals", 0))
        self.stopped_early = bool(ckpt.get("stopped_early", False))
        self.early_stop_reason = str(ckpt.get("early_stop_reason", ""))
        ckpt_effective_max_steps = int(ckpt.get("effective_max_steps", self.effective_max_steps))
        self.loss_history = list(ckpt.get("loss_history", []))
        if "rng_state" in ckpt:
            restore_rng_state(ckpt["rng_state"])
        trainer_rng_state = ckpt.get("trainer_rng_state")
        if trainer_rng_state is not None:
            self.rng.bit_generator.state = trainer_rng_state
        sampler_state = ckpt.get("batch_sampler_state", ckpt.get("edge_sampler_state"))
        self.batch_sampler.load_state_dict(sampler_state)
        log_event(
            "[CHECKPOINT]",
            (
                f"loaded step={self.global_step} epoch={self.epoch} best_metric={self.best_metric:.6f} "
                f"early_stop_bad_evals={self.early_stop_bad_evals} "
                f"checkpoint_effective_max_steps={ckpt_effective_max_steps} "
                f"active_effective_max_steps={self.effective_max_steps}"
            ),
        )

    def _save_checkpoint(self, name: str) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.ctx.optimizer.state_dict(),
            "scheduler": self.ctx.scheduler.state_dict() if self.ctx.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "early_stop_bad_evals": self.early_stop_bad_evals,
            "stopped_early": self.stopped_early,
            "early_stop_reason": self.early_stop_reason,
            "effective_max_steps": self.effective_max_steps,
            "loss_history": self.loss_history,
            "rng_state": capture_rng_state(),
            "trainer_rng_state": self.rng.bit_generator.state,
            "batch_sampler_state": self.batch_sampler.state_dict(),
            "edge_sampler_state": self.batch_sampler.state_dict(),
        }
        torch.save(payload, self.checkpoints_dir / name)
        log_event("[CHECKPOINT]", f"saved {self.checkpoints_dir / name}")

    def _load_model_weights_only(self, ckpt_path: Path) -> None:
        log_event("[CHECKPOINT]", f"loading model weights for evaluation from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])

    def _select_validation_metric(self, metrics: Dict[str, float]) -> float:
        metric_name = str(self.cfg.logging.best_metric)
        mapping = {
            "val_auc": "auc",
            "val_aucpr": "aucpr",
            "val_hit10": "hit10",
        }
        if metric_name not in mapping:
            raise ValueError(f"Unsupported logging.best_metric for early stopping: {metric_name}")
        return float(metrics[mapping[metric_name]])

    @staticmethod
    def _rolling_stats(values, window: int) -> Tuple[float, float, float]:
        if len(values) == 0:
            return 0.0, 0.0, 0.0
        sub = np.asarray(values[-window:], dtype=np.float64)
        return float(sub.mean()), float(sub.std()), float(sub.var())

    @staticmethod
    def _maybe_pin_tensor(tensor: torch.Tensor, pin_memory: bool) -> torch.Tensor:
        if pin_memory and tensor.device.type == "cpu":
            return tensor.pin_memory()
        return tensor

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _make_step_reporter(self, step_idx: int, active_chunk: int) -> StepProgressReporter:
        return StepProgressReporter(
            step=int(step_idx) + 1,
            epoch=int(self.epoch),
            active_chunk=int(active_chunk),
            log_every_sec=float(self.cfg.runtime.step_progress_sec),
            enable_phase_logs=bool(self.cfg.runtime.step_phase_start_end_logs),
            min_items=int(self.cfg.runtime.step_progress_min_items),
        )

    def _map_edges_to_local(self, edges_global: np.ndarray, global_to_local: Dict[int, int]) -> np.ndarray:
        src = np.asarray([global_to_local[int(x)] for x in edges_global[0]], dtype=np.int64)
        dst = np.asarray([global_to_local[int(x)] for x in edges_global[1]], dtype=np.int64)
        return np.vstack([src, dst])

    @staticmethod
    def _directed_local_edge_hash(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
        return src.astype(np.uint64, copy=False) * np.uint64(max(1, int(num_nodes))) + dst.astype(
            np.uint64,
            copy=False,
        )

    def _mask_supervision_edges_from_message_graph(
        self,
        edge_index_local: np.ndarray,
        pos_edges_local: np.ndarray,
        *,
        remove_edges: bool,
        require_absent: bool,
        context_label: str,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        edge_index_local = np.asarray(edge_index_local, dtype=np.int64)
        pos_edges_local = np.asarray(pos_edges_local, dtype=np.int64)
        stats = {
            "masked_supervision_edge_directions": 0,
            "masked_supervision_unique_edges": 0,
            "query_edges_in_message_graph": 0,
        }
        if edge_index_local.shape[1] == 0 or pos_edges_local.shape[1] == 0:
            return edge_index_local, stats

        num_local_nodes = int(edge_index_local.max()) + 1 if edge_index_local.size > 0 else 1
        forward_src = pos_edges_local[0].astype(np.int64, copy=False)
        forward_dst = pos_edges_local[1].astype(np.int64, copy=False)
        edge_hash = self._directed_local_edge_hash(
            edge_index_local[0],
            edge_index_local[1],
            num_local_nodes,
        )
        forward_hash = self._directed_local_edge_hash(forward_src, forward_dst, num_local_nodes)
        reverse_hash = self._directed_local_edge_hash(forward_dst, forward_src, num_local_nodes)
        forward_present = np.isin(forward_hash, edge_hash, assume_unique=False)
        reverse_present = np.isin(reverse_hash, edge_hash, assume_unique=False)
        stats["masked_supervision_unique_edges"] = int((forward_present | reverse_present).sum())
        non_self = forward_src != forward_dst
        query_src = np.concatenate([forward_src, forward_dst[non_self]], axis=0).astype(np.int64, copy=False)
        query_dst = np.concatenate([forward_dst, forward_src[non_self]], axis=0).astype(np.int64, copy=False)
        query_hash = self._directed_local_edge_hash(query_src, query_dst, num_local_nodes)
        remove_mask = np.isin(edge_hash, query_hash, assume_unique=False)
        stats["query_edges_in_message_graph"] = int(remove_mask.sum())
        if require_absent and stats["query_edges_in_message_graph"] > 0:
            raise RuntimeError(
                f"{context_label} query_edges_in_message_graph={stats['query_edges_in_message_graph']}"
            )
        if not remove_edges or stats["query_edges_in_message_graph"] == 0:
            return edge_index_local, stats
        masked = edge_index_local[:, ~remove_mask].astype(np.int64, copy=False)
        stats["masked_supervision_edge_directions"] = int(remove_mask.sum())
        return masked, stats

    def _deterministic_node_order(self, node_ids: np.ndarray, salt: int) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=np.uint64)
        with np.errstate(over="ignore"):
            x = node_ids ^ np.uint64(self.cfg.seed + salt) ^ np.uint64(0x9E3779B97F4A7C15)
            x = x + np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ np.right_shift(x, 30)) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ np.right_shift(x, 27)) * np.uint64(0x94D049BB133111EB)
            x = x ^ np.right_shift(x, 31)
        return np.lexsort((node_ids.astype(np.int64, copy=False), x))

    def _rebuild_subgraph(self, sub: SubgraphData, keep_local_mask: np.ndarray) -> SubgraphData:
        keep_local_mask = np.asarray(keep_local_mask, dtype=np.bool_)
        keep_idx = np.flatnonzero(keep_local_mask).astype(np.int64, copy=False)
        if keep_idx.shape[0] == int(sub.sub_nodes_global_ids.shape[0]):
            return sub

        old_to_new = np.full(int(sub.sub_nodes_global_ids.shape[0]), -1, dtype=np.int64)
        old_to_new[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int64)

        old_edge_index = sub.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        if old_edge_index.shape[1] > 0:
            edge_keep = keep_local_mask[old_edge_index[0]] & keep_local_mask[old_edge_index[1]]
            edge_src = old_to_new[old_edge_index[0, edge_keep]]
            edge_dst = old_to_new[old_edge_index[1, edge_keep]]
            edge_index = torch.from_numpy(np.vstack([edge_src, edge_dst]).astype(np.int64, copy=False)).long()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        old_seed = sub.seed_local_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        new_seed = old_to_new[old_seed]
        if (new_seed < 0).any():
            raise RuntimeError("Seed nodes were removed while rebuilding sampled subgraph.")
        global_ids = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        node_hops = sub.node_hops.detach().cpu().numpy().astype(np.int64, copy=False)
        kept_global = global_ids[keep_idx]
        kept_hops = node_hops[keep_idx]
        return SubgraphData(
            sub_nodes_global_ids=torch.from_numpy(kept_global.copy()).long(),
            edge_index=edge_index,
            seed_local_indices=torch.from_numpy(new_seed.astype(np.int64, copy=False)).long(),
            node_hops=torch.from_numpy(kept_hops.astype(np.int64, copy=False)).long(),
            global_to_local={int(gid): int(i) for i, gid in enumerate(kept_global.tolist())},
        )

    def _trim_sampled_subgraph_nodes(self, sub: SubgraphData) -> Tuple[SubgraphData, Dict[str, int]]:
        total_nodes = int(sub.sub_nodes_global_ids.shape[0])
        max_nodes = int(self.cfg.data.max_subgraph_nodes)
        stats = {
            "sub_nodes_before_trim": total_nodes,
            "sub_nodes_after_trim": total_nodes,
            "trimmed_nodes": 0,
        }
        if total_nodes <= max_nodes:
            return sub, stats

        node_hops = sub.node_hops.detach().cpu().numpy().astype(np.int64, copy=False)
        node_ids = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        keep = np.zeros(total_nodes, dtype=np.bool_)
        seed_local = sub.seed_local_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        keep[seed_local] = True
        remaining = max_nodes - int(keep.sum())
        if remaining <= 0:
            stats["sub_nodes_after_trim"] = int(keep.sum())
            stats["trimmed_nodes"] = total_nodes - int(keep.sum())
            return self._rebuild_subgraph(sub, keep), stats

        hop1_candidates = np.flatnonzero((node_hops == 1) & (~keep))
        if hop1_candidates.size > 0 and remaining > 0:
            order = self._deterministic_node_order(node_ids[hop1_candidates], salt=11)
            selected = hop1_candidates[order[: min(remaining, hop1_candidates.size)]]
            keep[selected] = True
            remaining = max_nodes - int(keep.sum())

        hop2_candidates = np.flatnonzero((node_hops >= 2) & (~keep))
        if hop2_candidates.size > 0 and remaining > 0:
            order = self._deterministic_node_order(node_ids[hop2_candidates], salt=17)
            selected = hop2_candidates[order[: min(remaining, hop2_candidates.size)]]
            keep[selected] = True

        trimmed = self._rebuild_subgraph(sub, keep)
        stats["sub_nodes_after_trim"] = int(trimmed.sub_nodes_global_ids.shape[0])
        stats["trimmed_nodes"] = total_nodes - stats["sub_nodes_after_trim"]
        return trimmed, stats

    def _trim_real_image_nodes(
        self,
        sub: SubgraphData,
        loaded_mask: np.ndarray,
    ) -> Tuple[SubgraphData, Dict[str, Any]]:
        loaded_mask = np.asarray(loaded_mask, dtype=np.bool_)
        real_count = int(loaded_mask.sum())
        max_images = int(self.cfg.data.max_image_nodes)
        stats: Dict[str, Any] = {
            "real_image_nodes_before_trim": real_count,
            "real_image_nodes_after_trim": real_count,
            "trimmed_image_nodes": 0,
            "seed_image_cap_exceeded": False,
        }
        if real_count <= max_images:
            return sub, stats

        node_ids = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        node_is_art = self.ctx.graph.get_node_is_art(node_ids).astype(np.bool_, copy=False)
        seed_mask = np.zeros(node_ids.shape[0], dtype=np.bool_)
        seed_local = sub.seed_local_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        seed_mask[seed_local] = True

        overflow = real_count - max_images
        drop_local: List[int] = []

        art_candidates = np.flatnonzero(loaded_mask & (~seed_mask) & node_is_art)
        if art_candidates.size > 0 and overflow > 0:
            order = self._deterministic_node_order(node_ids[art_candidates], salt=23)
            chosen = art_candidates[order[: min(overflow, art_candidates.size)]]
            drop_local.extend(chosen.tolist())
            overflow -= int(chosen.shape[0])

        other_candidates = np.flatnonzero(loaded_mask & (~seed_mask) & (~node_is_art))
        if other_candidates.size > 0 and overflow > 0:
            order = self._deterministic_node_order(node_ids[other_candidates], salt=29)
            chosen = other_candidates[order[: min(overflow, other_candidates.size)]]
            drop_local.extend(chosen.tolist())
            overflow -= int(chosen.shape[0])

        if not drop_local:
            stats["seed_image_cap_exceeded"] = real_count > max_images
            return sub, stats

        keep = np.ones(node_ids.shape[0], dtype=np.bool_)
        keep[np.asarray(drop_local, dtype=np.int64)] = False
        trimmed = self._rebuild_subgraph(sub, keep)
        stats["real_image_nodes_after_trim"] = max_images + max(0, overflow)
        stats["trimmed_image_nodes"] = real_count - stats["real_image_nodes_after_trim"]
        stats["seed_image_cap_exceeded"] = overflow > 0
        return trimmed, stats

    @staticmethod
    def _art_edge_mask_global(edges: np.ndarray, node_is_art: np.ndarray) -> np.ndarray:
        if edges.shape[1] == 0:
            return np.zeros((0,), dtype=np.bool_)
        return node_is_art[edges[0]] & node_is_art[edges[1]]

    @staticmethod
    def _art_any_edge_mask_global(edges: np.ndarray, node_is_art: np.ndarray) -> np.ndarray:
        if edges.shape[1] == 0:
            return np.zeros((0,), dtype=np.bool_)
        return node_is_art[edges[0]] | node_is_art[edges[1]]

    @staticmethod
    def _art_edge_mask_local(edges: np.ndarray, node_is_art_local: np.ndarray) -> np.ndarray:
        if edges.shape[1] == 0:
            return np.zeros((0,), dtype=np.bool_)
        return node_is_art_local[edges[0]] & node_is_art_local[edges[1]]

    @staticmethod
    def _art_any_edge_mask_local(edges: np.ndarray, node_is_art_local: np.ndarray) -> np.ndarray:
        if edges.shape[1] == 0:
            return np.zeros((0,), dtype=np.bool_)
        return node_is_art_local[edges[0]] | node_is_art_local[edges[1]]

    def _filter_edges_to_art_global(self, edges: np.ndarray) -> np.ndarray:
        mask = self._art_edge_mask_global(edges, self.ctx.graph.node_is_art)
        return edges[:, mask]

    def _filter_edges_to_art_any_global(self, edges: np.ndarray) -> np.ndarray:
        mask = self._art_any_edge_mask_global(edges, self.ctx.graph.node_is_art)
        return edges[:, mask]

    def _filter_edges_to_art_chunk(
        self,
        edges_by_chunk: Dict[int, np.ndarray],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        filtered: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        for chunk_id, edges in edges_by_chunk.items():
            if edges.shape[1] == 0:
                continue
            sub = self.ctx.graph.load_chunk_subgraph(int(chunk_id))
            mask = self._art_edge_mask_local(edges, sub.node_is_art)
            art_edges = edges[:, mask]
            counts[int(chunk_id)] = int(art_edges.shape[1])
            if art_edges.shape[1] > 0:
                filtered[int(chunk_id)] = art_edges
        return filtered, counts

    def _filter_edges_to_art_any_chunk(
        self,
        edges_by_chunk: Dict[int, np.ndarray],
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        filtered: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        for chunk_id, edges in edges_by_chunk.items():
            if edges.shape[1] == 0:
                continue
            sub = self.ctx.graph.load_chunk_subgraph(int(chunk_id))
            mask = self._art_any_edge_mask_local(edges, sub.node_is_art)
            art_any_edges = edges[:, mask]
            counts[int(chunk_id)] = int(art_any_edges.shape[1])
            if art_any_edges.shape[1] > 0:
                filtered[int(chunk_id)] = art_any_edges
        return filtered, counts

    def _prepare_batch_cpu(
        self,
        pos_edges_global: np.ndarray,
        neg_edges_global: np.ndarray,
        active_chunk: int,
        progress_reporter: Optional[StepProgressReporter] = None,
        *,
        mask_supervision_edges: bool = True,
        require_query_edges_absent: bool = False,
        query_context_label: str = "eval",
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        seeds = np.unique(
            np.concatenate(
                [pos_edges_global[0], pos_edges_global[1], neg_edges_global[0], neg_edges_global[1]]
            )
        ).astype(np.int64)
        if progress_reporter is not None:
            progress_reporter.emit(
                "prepare_cpu",
                "start",
                seed_edges=int(pos_edges_global.shape[1]),
                seed_nodes=int(seeds.shape[0]),
                pos_edges=int(pos_edges_global.shape[1]),
                neg_edges=int(neg_edges_global.shape[1]),
            )
        if progress_reporter is not None:
            with progress_reporter.phase(
                "sample_neighbors",
                seed_nodes=int(seeds.shape[0]),
            ):
                sub = self.ctx.neighbor_sampler.sample(seeds)
        else:
            sub = self.ctx.neighbor_sampler.sample(seeds)

        sub, trim_stats = self._trim_sampled_subgraph_nodes(sub)
        node_idx = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64)
        node_wids = self.ctx.graph.get_node_wids(node_idx)
        node_has_image = self.ctx.graph.get_node_has_image(node_idx).astype(np.bool_)
        node_chunk_ids = self.ctx.graph.node_chunk_ids[node_idx]
        node_chunk_pos = self.ctx.graph.node_chunk_pos[node_idx]
        if progress_reporter is not None:
            with progress_reporter.phase(
                "prepare_images",
                node_count=int(node_idx.shape[0]),
                requested_images=int(node_has_image.sum()),
            ):
                image_batch = self.ctx.image_provider.inspect_batch(
                    node_wids,
                    has_image_mask=node_has_image,
                    pin_memory=self.pin_memory,
                )
        else:
            image_batch = self.ctx.image_provider.inspect_batch(
                node_wids,
                has_image_mask=node_has_image,
                pin_memory=self.pin_memory,
            )

        image_trim_stats = {
            "real_image_nodes_before_trim": int(image_batch["stats_real_hit"]),
            "real_image_nodes_after_trim": int(image_batch["stats_real_hit"]),
            "trimmed_image_nodes": 0,
            "seed_image_cap_exceeded": False,
        }
        if int(image_batch["stats_real_hit"]) > int(self.cfg.data.max_image_nodes):
            sub, image_trim_stats = self._trim_real_image_nodes(
                sub,
                image_batch["loaded_mask_cpu"].detach().cpu().numpy().astype(np.bool_, copy=False),
            )
            node_idx = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64)
            node_wids = self.ctx.graph.get_node_wids(node_idx)
            node_has_image = self.ctx.graph.get_node_has_image(node_idx).astype(np.bool_)
            node_chunk_ids = self.ctx.graph.node_chunk_ids[node_idx]
            node_chunk_pos = self.ctx.graph.node_chunk_pos[node_idx]
            image_batch = self.ctx.image_provider.inspect_batch(
                node_wids,
                has_image_mask=node_has_image,
                pin_memory=self.pin_memory,
            )
            image_trim_stats["real_image_nodes_after_trim"] = int(image_batch["stats_real_hit"])
            image_trim_stats["trimmed_image_nodes"] = (
                image_trim_stats["real_image_nodes_before_trim"]
                - image_trim_stats["real_image_nodes_after_trim"]
            )

        pos_local = self._map_edges_to_local(pos_edges_global, sub.global_to_local)
        neg_local = self._map_edges_to_local(neg_edges_global, sub.global_to_local)
        edge_pairs_local = np.concatenate([pos_local, neg_local], axis=1)

        labels = np.concatenate(
            [
                np.ones(pos_local.shape[1], dtype=np.float32),
                np.zeros(neg_local.shape[1], dtype=np.float32),
            ]
        )

        edge_index_np = sub.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        edge_index_np, leakage_stats = self._mask_supervision_edges_from_message_graph(
            edge_index_np,
            pos_local,
            remove_edges=mask_supervision_edges,
            require_absent=require_query_edges_absent,
            context_label=query_context_label,
        )
        edge_index = torch.from_numpy(edge_index_np).long()
        edge_pairs_t = torch.from_numpy(edge_pairs_local).long()
        labels_t = torch.from_numpy(labels).float()
        has_image_mask_t = image_batch["loaded_mask_cpu"].float()
        edge_index = self._maybe_pin_tensor(edge_index, self.pin_memory)
        edge_pairs_t = self._maybe_pin_tensor(edge_pairs_t, self.pin_memory)
        labels_t = self._maybe_pin_tensor(labels_t, self.pin_memory)
        has_image_mask_t = self._maybe_pin_tensor(has_image_mask_t, self.pin_memory)

        batch = {
            "node_indices_cpu": node_idx,
            "edge_index_cpu": edge_index,
            "edge_pairs_local_cpu": edge_pairs_t,
            "labels_cpu": labels_t,
            "node_wids": node_wids,
            "node_chunk_ids": node_chunk_ids,
            "node_chunk_pos": node_chunk_pos,
            "real_node_image_indices_cpu": image_batch["real_node_indices_cpu"],
            "real_node_image_paths": list(image_batch["real_node_paths"]),
            "node_has_image_mask_cpu": has_image_mask_t,
            "image_stats": {
                "img_real_hit": float(image_batch["stats_real_hit"]),
                "img_no_image": float(image_batch["stats_no_image"]),
                "img_missing_local": float(image_batch["stats_missing_local"]),
            },
            "prepare_cpu_sec": float(time.perf_counter() - start_time),
            "image_decode_sec": float(image_batch["stats_image_resolve_sec"]),
            "active_chunk": int(active_chunk),
            "active_chunk_name": "sampled" if int(active_chunk) < 0 else self.ctx.graph.chunk_name(int(active_chunk)),
            "seed_edges": int(pos_edges_global.shape[1]),
            "sub_nodes": int(node_idx.shape[0]),
            "sub_nodes_before_trim": int(trim_stats["sub_nodes_before_trim"]),
            "sub_nodes_after_trim": int(trim_stats["sub_nodes_after_trim"]),
            "seed_nodes": int(seeds.shape[0]),
            "real_image_nodes_before_trim": int(image_trim_stats["real_image_nodes_before_trim"]),
            "real_image_nodes_after_trim": int(image_trim_stats["real_image_nodes_after_trim"]),
            "trimmed_nodes": int(trim_stats["trimmed_nodes"]),
            "trimmed_image_nodes": int(image_trim_stats["trimmed_image_nodes"]),
            "seed_image_cap_exceeded": bool(image_trim_stats["seed_image_cap_exceeded"]),
            "masked_supervision_edge_directions": int(leakage_stats["masked_supervision_edge_directions"]),
            "masked_supervision_unique_edges": int(leakage_stats["masked_supervision_unique_edges"]),
            "query_edges_in_message_graph": int(leakage_stats["query_edges_in_message_graph"]),
            "pos_edges": int(pos_local.shape[1]),
            "neg_edges": int(neg_local.shape[1]),
            "real_image_count": int(image_batch["real_node_indices_cpu"].shape[0]),
            "chunk_intra_edges": int(edge_index.shape[1]),
            "message_edges": int(edge_index.shape[1]),
        }
        if progress_reporter is not None:
            progress_reporter.emit(
                "prepare_cpu",
                "done",
                seed_edges=int(pos_edges_global.shape[1]),
                sub_nodes=batch["sub_nodes"],
                seed_nodes=batch["seed_nodes"],
                sub_nodes_before_trim=batch["sub_nodes_before_trim"],
                sub_nodes_after_trim=batch["sub_nodes_after_trim"],
                real_image_nodes_before_trim=batch["real_image_nodes_before_trim"],
                real_image_nodes_after_trim=batch["real_image_nodes_after_trim"],
                trimmed_nodes=batch["trimmed_nodes"],
                trimmed_image_nodes=batch["trimmed_image_nodes"],
                seed_image_cap_exceeded=batch["seed_image_cap_exceeded"],
                masked_supervision_edge_directions=batch["masked_supervision_edge_directions"],
                masked_supervision_unique_edges=batch["masked_supervision_unique_edges"],
                query_edges_in_message_graph=batch["query_edges_in_message_graph"],
                edge_pairs=int(edge_pairs_local.shape[1]),
                img_real_hit=int(image_batch["stats_real_hit"]),
                img_no_image=int(image_batch["stats_no_image"]),
                img_missing_local=int(image_batch["stats_missing_local"]),
                prepare_cpu_sec=f"{batch['prepare_cpu_sec']:.2f}",
            )
        return batch

    def _prepare_chunk_batch_cpu(
        self,
        chunk_id: int,
        pos_edges_local: np.ndarray,
        neg_edges_local: Optional[np.ndarray] = None,
        message_edge_index_local: Optional[np.ndarray] = None,
        sub: Optional[Any] = None,
        progress_reporter: Optional[StepProgressReporter] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        if sub is None:
            sub = self.ctx.graph.load_chunk_subgraph(int(chunk_id))
        node_idx = sub.node_indices_global.astype(np.int64, copy=False)
        center_node_idx = sub.center_node_indices_global.astype(np.int64, copy=False)
        edge_index_local = (
            np.asarray(message_edge_index_local, dtype=np.int64)
            if message_edge_index_local is not None
            else sub.message_edge_index_local
        )
        chunk_name = sub.chunk_name
        if progress_reporter is not None:
            progress_reporter.emit(
                "prepare_cpu",
                "start",
                chunk_id=int(chunk_id),
                chunk_name=chunk_name,
                chunk_nodes=int(center_node_idx.shape[0]),
                center_nodes=int(center_node_idx.shape[0]),
                context_nodes=int(node_idx.shape[0]),
                message_edges=int(edge_index_local.shape[1]),
                pos_edges=int(pos_edges_local.shape[1]),
                neg_edges=int(
                    neg_edges_local.shape[1]
                    if neg_edges_local is not None
                    else pos_edges_local.shape[1] * int(self.cfg.data.num_neg)
                ),
                supervision_batches=int(
                    max(1, int(np.ceil(pos_edges_local.shape[1] / max(1, int(self.cfg.data.batch_edges)))))
                ),
                supervision_batch_edges=int(self.cfg.data.batch_edges),
            )

        if neg_edges_local is not None:
            edge_pairs_local = np.concatenate([pos_edges_local, neg_edges_local], axis=1)
            labels = np.concatenate(
                [
                    np.ones(pos_edges_local.shape[1], dtype=np.float32),
                    np.zeros(neg_edges_local.shape[1], dtype=np.float32),
                ]
            )
            edge_pairs_t = torch.from_numpy(edge_pairs_local).long()
            labels_t = torch.from_numpy(labels).float()
            edge_pairs_t = self._maybe_pin_tensor(edge_pairs_t, self.pin_memory)
            labels_t = self._maybe_pin_tensor(labels_t, self.pin_memory)
        else:
            edge_pairs_local = None
            edge_pairs_t = None
            labels_t = None

        if progress_reporter is not None:
            with progress_reporter.phase(
                "prepare_images",
                node_count=int(node_idx.shape[0]),
                requested_images=int(sub.node_has_image.sum()),
            ):
                image_batch = self.ctx.image_provider.inspect_batch(
                    sub.node_wids,
                    has_image_mask=sub.node_has_image,
                    pin_memory=self.pin_memory,
                )
        else:
            image_batch = self.ctx.image_provider.inspect_batch(
                sub.node_wids,
                has_image_mask=sub.node_has_image,
                pin_memory=self.pin_memory,
            )

        edge_index = torch.from_numpy(edge_index_local).long()
        has_image_mask_t = image_batch["loaded_mask_cpu"].float()
        edge_index = self._maybe_pin_tensor(edge_index, self.pin_memory)
        has_image_mask_t = self._maybe_pin_tensor(has_image_mask_t, self.pin_memory)

        batch = {
            "node_indices_cpu": node_idx,
            "edge_index_cpu": edge_index,
            "edge_pairs_local_cpu": edge_pairs_t,
            "labels_cpu": labels_t,
            "pos_edges_local_cpu": torch.from_numpy(pos_edges_local.astype(np.int64, copy=False)).long(),
            "center_mask_local_np": sub.center_mask_local.astype(np.bool_, copy=False),
            "node_wids": sub.node_wids,
            "node_chunk_ids": sub.node_chunk_ids.astype(np.int32, copy=False),
            "node_chunk_pos": sub.node_chunk_pos.astype(np.int32, copy=False),
            "real_node_image_indices_cpu": image_batch["real_node_indices_cpu"],
            "real_node_image_paths": list(image_batch["real_node_paths"]),
            "node_has_image_mask_cpu": has_image_mask_t,
            "image_stats": {
                "img_real_hit": float(image_batch["stats_real_hit"]),
                "img_no_image": float(image_batch["stats_no_image"]),
                "img_missing_local": float(image_batch["stats_missing_local"]),
            },
            "prepare_cpu_sec": float(time.perf_counter() - start_time),
            "image_decode_sec": float(image_batch["stats_image_resolve_sec"]),
            "active_chunk": int(chunk_id),
            "active_chunk_name": chunk_name,
            "sub_nodes": int(node_idx.shape[0]),
            "seed_nodes": int(center_node_idx.shape[0]),
            "center_nodes": int(center_node_idx.shape[0]),
            "context_nodes": int(node_idx.shape[0]),
            "message_edges": int(edge_index_local.shape[1]),
            "pos_edges": int(pos_edges_local.shape[1]),
            "neg_edges": int(
                neg_edges_local.shape[1]
                if neg_edges_local is not None
                else pos_edges_local.shape[1] * int(self.cfg.data.num_neg)
            ),
            "supervision_batches": int(
                max(1, int(np.ceil(pos_edges_local.shape[1] / max(1, int(self.cfg.data.batch_edges)))))
            ),
            "supervision_batch_edges": int(self.cfg.data.batch_edges),
            "real_image_count": int(image_batch["real_node_indices_cpu"].shape[0]),
            "chunk_intra_edges": int(edge_index_local.shape[1]),
        }
        if progress_reporter is not None:
            progress_reporter.emit(
                "prepare_cpu",
                "done",
                chunk_id=int(chunk_id),
                chunk_name=chunk_name,
                chunk_nodes=batch["center_nodes"],
                center_nodes=batch["center_nodes"],
                context_nodes=batch["context_nodes"],
                message_edges=batch["message_edges"],
                chunk_intra_edges=batch["chunk_intra_edges"],
                pos_edges=batch["pos_edges"],
                neg_edges=batch["neg_edges"],
                supervision_batches=batch["supervision_batches"],
                supervision_batch_edges=batch["supervision_batch_edges"],
                img_real_hit=int(image_batch["stats_real_hit"]),
                img_no_image=int(image_batch["stats_no_image"]),
                img_missing_local=int(image_batch["stats_missing_local"]),
                prepare_cpu_sec=f"{batch['prepare_cpu_sec']:.2f}",
            )
        return batch

    def _move_batch_to_device(
        self,
        batch_cpu: Dict[str, Any],
        progress_reporter: Optional[StepProgressReporter] = None,
        transfer_supervision: bool = True,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        if progress_reporter is not None:
            progress_reporter.emit("transfer", "start")
        edge_index = batch_cpu["edge_index_cpu"].to(
            self.device, non_blocking=self.non_blocking_transfer
        )
        edge_pairs_local = None
        labels = None
        if transfer_supervision and batch_cpu.get("edge_pairs_local_cpu") is not None:
            edge_pairs_local = batch_cpu["edge_pairs_local_cpu"].to(
                self.device, non_blocking=self.non_blocking_transfer
            )
            labels = batch_cpu["labels_cpu"].to(
                self.device, non_blocking=self.non_blocking_transfer
            )
        has_image_mask = batch_cpu["node_has_image_mask_cpu"].to(
            self.device, non_blocking=self.non_blocking_transfer
        )
        batch = dict(batch_cpu)
        batch.update(
            {
                "edge_index": edge_index,
                "edge_pairs_local": edge_pairs_local,
                "labels": labels,
                "node_has_image_mask": has_image_mask,
                "transfer_sec": float(time.perf_counter() - start_time),
            }
        )
        if not self.first_batch_logged:
            log_event(
                "[DEVICE]",
                (
                    f"first batch devices edge_index={edge_index.device} "
                    f"labels={(str(labels.device) if labels is not None else 'cpu_lazy')} "
                    f"real_images=lazy_cpu real_image_count={batch['real_image_count']}"
                ),
            )
            self.first_batch_logged = True
        if progress_reporter is not None:
            progress_reporter.emit(
                "transfer",
                "done",
                edge_index_device=str(edge_index.device),
                labels_device=str(labels.device) if labels is not None else "cpu_lazy",
                real_images_device="cpu_lazy",
                transfer_sec=f"{batch['transfer_sec']:.2f}",
            )
        return batch

    def _forward_logits(
        self,
        batch: Dict[str, Any],
        progress_reporter: Optional[StepProgressReporter] = None,
    ) -> torch.Tensor:
        return self.model(
            edge_index=batch["edge_index"],
            edge_pairs_local=batch["edge_pairs_local"],
            node_indices_cpu=batch["node_indices_cpu"],
            node_wids=batch["node_wids"],
            node_chunk_ids=batch["node_chunk_ids"],
            node_chunk_pos=batch["node_chunk_pos"],
            real_node_image_indices_cpu=batch["real_node_image_indices_cpu"],
            real_node_image_paths=batch["real_node_image_paths"],
            node_has_image_mask=batch["node_has_image_mask"],
            text_store=self.ctx.text_store,
            image_provider=self.ctx.image_provider,
            image_transform=self.image_transform,
            progress_reporter=progress_reporter,
        )

    def _encode_node_embeddings(
        self,
        batch: Dict[str, Any],
        progress_reporter: Optional[StepProgressReporter] = None,
    ) -> torch.Tensor:
        return self.model.encode_nodes(
            edge_index=batch["edge_index"],
            node_indices_cpu=batch["node_indices_cpu"],
            node_wids=batch["node_wids"],
            node_chunk_ids=batch["node_chunk_ids"],
            node_chunk_pos=batch["node_chunk_pos"],
            real_node_image_indices_cpu=batch["real_node_image_indices_cpu"],
            real_node_image_paths=batch["real_node_image_paths"],
            node_has_image_mask=batch["node_has_image_mask"],
            text_store=self.ctx.text_store,
            image_provider=self.ctx.image_provider,
            image_transform=self.image_transform,
            progress_reporter=progress_reporter,
        )

    def _chunk_supervision_loss(
        self,
        batch: Dict[str, Any],
        node_embeddings: torch.Tensor,
        progress_reporter: Optional[StepProgressReporter] = None,
    ) -> torch.Tensor:
        pos_edges = batch["pos_edges_local_cpu"].detach().cpu().numpy().astype(np.int64, copy=False)
        chunk_id = int(batch["active_chunk"])
        num_nodes = int(batch["sub_nodes"])
        batch_edges = max(1, int(self.cfg.data.batch_edges))
        supervision_batches = max(1, int(np.ceil(pos_edges.shape[1] / batch_edges)))
        total_examples = max(1, int(pos_edges.shape[1] * (1 + self.cfg.data.num_neg)))
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)

        for batch_idx, start in enumerate(range(0, pos_edges.shape[1], batch_edges), start=1):
            stop = min(start + batch_edges, pos_edges.shape[1])
            pos_batch = pos_edges[:, start:stop]
            neg_batch = sample_negative_edges_center_aware(
                pos_batch,
                num_nodes=num_nodes,
                num_neg=self.cfg.data.num_neg,
                sorted_edge_hash=self.chunk_edge_hashes[chunk_id],
                rng=self.rng,
                center_mask=batch["center_mask_local_np"],
            )
            edge_pairs_np = np.concatenate([pos_batch, neg_batch], axis=1)
            labels_np = np.concatenate(
                [
                    np.ones(pos_batch.shape[1], dtype=np.float32),
                    np.zeros(neg_batch.shape[1], dtype=np.float32),
                ]
            )
            edge_pairs = torch.from_numpy(edge_pairs_np).to(
                self.device,
                non_blocking=self.non_blocking_transfer,
            )
            labels = torch.from_numpy(labels_np).to(
                self.device,
                non_blocking=self.non_blocking_transfer,
            )
            logits = self.model.predict_edges(node_embeddings, edge_pairs)
            loss_sum = loss_sum + F.binary_cross_entropy_with_logits(
                logits.float(),
                labels.float(),
                reduction="sum",
            )
            if progress_reporter is not None:
                progress_reporter.progress(
                    "forward_gnn",
                    supervision_batch=batch_idx,
                    total_supervision_batches=supervision_batches,
                    supervision_batch_edges=int(edge_pairs_np.shape[1]),
                )
        return loss_sum / float(total_examples)

    def _select_chunk_eval_edges(
        self,
        edges_by_chunk: Dict[int, np.ndarray],
        max_edges: int,
    ) -> Dict[int, np.ndarray]:
        counts = {int(cid): int(arr.shape[1]) for cid, arr in edges_by_chunk.items() if int(arr.shape[1]) > 0}
        total = int(sum(counts.values()))
        if total <= 0:
            return {}
        if max_edges <= 0 or total <= max_edges:
            return {int(cid): arr for cid, arr in edges_by_chunk.items() if int(arr.shape[1]) > 0}

        chosen = np.sort(self.rng.choice(total, size=max_edges, replace=False))
        out: Dict[int, np.ndarray] = {}
        offset = 0
        for cid in sorted(counts):
            count = counts[cid]
            start = np.searchsorted(chosen, offset, side="left")
            end = np.searchsorted(chosen, offset + count, side="left")
            if end > start:
                local_idx = chosen[start:end] - offset
                out[int(cid)] = edges_by_chunk[int(cid)][:, local_idx]
            offset += count
        return out

    @staticmethod
    def _resolve_final_cache_dtype(dtype_name: str) -> np.dtype:
        return np.float16 if str(dtype_name).strip().lower() == "float16" else np.float32

    def _final_test_positive_batch_size(self) -> int:
        total_score_batch = max(1, int(self.cfg.eval.final_test_batch_edges))
        negatives = max(1, int(self.cfg.eval.final_test_negatives_per_positive))
        return max(1, total_score_batch // negatives)

    def _final_test_progress_interval_sec(self) -> float:
        return 600.0

    def _get_train_message_edges_for_chunk(
        self,
        chunk_id: int,
        sub: Optional[Any] = None,
    ) -> np.ndarray:
        chunk_id = int(chunk_id)
        cached = self.final_test_message_edges_by_chunk.get(chunk_id)
        if cached is not None:
            return cached

        if sub is None:
            sub = self.ctx.graph.load_chunk_subgraph(chunk_id)
        edge_index_local = np.asarray(sub.message_edge_index_local, dtype=np.int64)
        if edge_index_local.shape[1] == 0 or self.train_propagation_hash_sorted.shape[0] == 0:
            filtered = np.empty((2, 0), dtype=np.int64)
        else:
            src_global = sub.node_indices_global[edge_index_local[0]].astype(np.int64, copy=False)
            dst_global = sub.node_indices_global[edge_index_local[1]].astype(np.int64, copy=False)
            keep = _undirected_edge_exists(
                src_global,
                dst_global,
                self.train_propagation_hash_sorted,
                self.ctx.graph.num_nodes,
            )
            filtered = edge_index_local[:, keep].astype(np.int64, copy=False)
        self.final_test_message_edges_by_chunk[chunk_id] = filtered
        return filtered

    def _build_final_test_node_pool(self) -> Dict[str, np.ndarray]:
        test_edges = self.ctx.test_edges.detach().cpu().numpy().astype(np.int64, copy=False)
        node_pool = np.unique(test_edges.reshape(-1)).astype(np.int64, copy=False)
        art_mask = self.ctx.graph.node_is_art[node_pool].astype(np.bool_, copy=False)
        art_node_pool = node_pool[art_mask].astype(np.int64, copy=False)
        return {
            "test_edges": test_edges,
            "node_pool": node_pool,
            "art_node_pool": art_node_pool,
            "art_any_node_pool": node_pool,
        }

    def _lookup_cached_node_rows(
        self,
        sorted_node_ids: np.ndarray,
        query_node_ids: np.ndarray,
    ) -> np.ndarray:
        query = np.asarray(query_node_ids, dtype=np.int64)
        if query.size == 0:
            return np.empty((0,), dtype=np.int64)
        rows = np.searchsorted(sorted_node_ids, query)
        if (rows >= sorted_node_ids.shape[0]).any():
            raise RuntimeError("final test node cache lookup exceeded cached node pool bounds")
        if not np.array_equal(sorted_node_ids[rows], query):
            raise RuntimeError("final test node cache missing one or more query nodes")
        return rows.astype(np.int64, copy=False)

    @torch.no_grad()
    def _encode_final_test_nodes_by_chunk(
        self,
        node_pool: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        sorted_node_ids = np.asarray(node_pool, dtype=np.int64)
        if sorted_node_ids.size == 0:
            emb_dim = int(self.model.gnn.output_dim)
            return sorted_node_ids, np.empty((0, emb_dim), dtype=np.float16), {
                "node_pool_size": 0,
                "encoded_chunks": 0,
                "embedding_dim": emb_dim,
            }

        cache_dtype = self._resolve_final_cache_dtype(
            self.cfg.eval.final_test_embedding_cache_dtype
        )
        emb_dim = int(self.model.gnn.output_dim)
        embeddings = np.empty((sorted_node_ids.shape[0], emb_dim), dtype=cache_dtype)
        node_chunk_ids = self.ctx.graph.node_chunk_ids[sorted_node_ids].astype(np.int32, copy=False)
        unique_chunks = np.unique(node_chunk_ids).astype(np.int32, copy=False)
        total_chunks = int(unique_chunks.shape[0])
        total_nodes = int(sorted_node_ids.shape[0])
        encoded_nodes = 0
        last_progress_ts = time.perf_counter()
        progress_interval_sec = self._final_test_progress_interval_sec()
        self.model.eval()
        log_event(
            "[STAGE]",
            (
                f"final test node embedding cache start "
                f"node_pool_size={total_nodes} "
                f"chunks={total_chunks}"
            ),
        )

        for idx, chunk_id in enumerate(unique_chunks.tolist(), start=1):
            chunk_mask = node_chunk_ids == int(chunk_id)
            row_idx = np.flatnonzero(chunk_mask).astype(np.int64, copy=False)
            chunk_node_ids = sorted_node_ids[row_idx]
            sub = self.ctx.graph.load_chunk_subgraph(int(chunk_id))
            train_message_edges = self._get_train_message_edges_for_chunk(int(chunk_id), sub=sub)
            batch_cpu = self._prepare_chunk_batch_cpu(
                int(chunk_id),
                np.empty((2, 0), dtype=np.int64),
                None,
                message_edge_index_local=train_message_edges,
                sub=sub,
                progress_reporter=None,
            )
            batch = self._move_batch_to_device(batch_cpu, transfer_supervision=False)
            with self._autocast_context():
                node_embeddings = self._encode_node_embeddings(batch)

            local_rows = np.asarray(
                [sub.global_to_local[int(node_id)] for node_id in chunk_node_ids],
                dtype=np.int64,
            )
            if not sub.center_mask_local[local_rows].all():
                raise RuntimeError(
                    f"final test node cache expected center nodes in chunk {int(chunk_id)}"
                )
            local_rows_t = torch.from_numpy(local_rows).to(
                node_embeddings.device,
                dtype=torch.long,
            )
            chunk_emb = (
                node_embeddings.index_select(0, local_rows_t)
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(cache_dtype, copy=False)
            )
            embeddings[row_idx] = chunk_emb
            encoded_nodes += int(chunk_node_ids.shape[0])

            now_ts = time.perf_counter()
            if (
                idx == total_chunks
                or idx % 25 == 0
                or (now_ts - last_progress_ts) >= progress_interval_sec
            ):
                log_event(
                    "[STAGE]",
                    (
                        f"final test node embedding cache progress chunks={idx}/{total_chunks} "
                        f"encoded_nodes={encoded_nodes}/{total_nodes} "
                        f"progress={(100.0 * float(idx) / float(max(1, total_chunks))):.1f}%"
                    ),
                )
                last_progress_ts = now_ts

        return sorted_node_ids, embeddings, {
            "node_pool_size": total_nodes,
            "encoded_chunks": total_chunks,
            "embedding_dim": emb_dim,
        }

    @torch.no_grad()
    def _score_edges_from_cached_nodes(
        self,
        edges_global: np.ndarray,
        cached_node_ids: np.ndarray,
        cached_embeddings: np.ndarray,
    ) -> np.ndarray:
        edges = np.asarray(edges_global, dtype=np.int64)
        if edges.shape[1] == 0:
            return np.empty((0,), dtype=np.float32)

        score_batch_size = max(1, int(self.cfg.eval.final_test_batch_edges))
        out = np.empty((edges.shape[1],), dtype=np.float32)
        self.model.eval()
        for start in range(0, edges.shape[1], score_batch_size):
            stop = min(start + score_batch_size, edges.shape[1])
            edge_batch = edges[:, start:stop]
            src_rows = self._lookup_cached_node_rows(cached_node_ids, edge_batch[0])
            dst_rows = self._lookup_cached_node_rows(cached_node_ids, edge_batch[1])
            src_np = np.ascontiguousarray(cached_embeddings[src_rows])
            dst_np = np.ascontiguousarray(cached_embeddings[dst_rows])
            src = torch.from_numpy(src_np).to(
                self.device,
                dtype=torch.float32,
                non_blocking=self.non_blocking_transfer,
            )
            dst = torch.from_numpy(dst_np).to(
                self.device,
                dtype=torch.float32,
                non_blocking=self.non_blocking_transfer,
            )
            with self._autocast_context():
                logits = self.model.predictor(src, dst)
            out[start:stop] = logits.detach().float().cpu().numpy().astype(np.float32, copy=False)
        return out

    @torch.no_grad()
    def _evaluate_edge_set_from_cached_nodes(
        self,
        *,
        label: str,
        pos_edges: np.ndarray,
        candidate_nodes: np.ndarray,
        cached_node_ids: np.ndarray,
        cached_embeddings: np.ndarray,
        include_hit10: bool,
    ) -> Dict[str, float]:
        pos = np.asarray(pos_edges, dtype=np.int64)
        candidate_nodes = np.asarray(candidate_nodes, dtype=np.int64)
        if pos.shape[1] == 0 or candidate_nodes.size == 0:
            return {
                "auc": 0.0,
                "aucpr": 0.0,
                "hit10": 0.0,
                "pos_edges_evaluated": int(pos.shape[1]),
                "neg_edges_evaluated": 0,
                "hit10_pos_edges_evaluated": 0,
                "hit10_neg_edges_evaluated": 0,
                "negatives_per_positive": 0.0,
            }

        num_neg = int(self.cfg.eval.final_test_negatives_per_positive)
        pos_batch_size = self._final_test_positive_batch_size()
        pos_score_parts: List[np.ndarray] = []
        neg_score_parts: List[np.ndarray] = []
        total_neg_edges = 0
        hit10_hits = 0
        hit10_pos_edges_evaluated = 0
        hit10_neg_edges_evaluated = 0
        last_progress_ts = time.perf_counter()
        progress_interval_sec = self._final_test_progress_interval_sec()
        total_batches = int(np.ceil(pos.shape[1] / max(1, pos_batch_size)))
        log_event(
            "[STAGE]",
            (
                f"{label} scoring start "
                f"pos_edges={int(pos.shape[1])} "
                f"candidate_nodes={int(candidate_nodes.shape[0])} "
                f"negatives_per_positive={num_neg} "
                f"batch_pos_edges={pos_batch_size} "
                f"total_batches={total_batches} "
                f"hit10_enabled={bool(include_hit10)}"
            ),
        )

        for batch_idx, start in enumerate(range(0, pos.shape[1], pos_batch_size), start=1):
            stop = min(start + pos_batch_size, pos.shape[1])
            pos_batch = pos[:, start:stop]
            neg_batch, neg_valid = sample_negative_edges_from_candidates_undirected(
                pos_batch,
                num_nodes=self.ctx.graph.num_nodes,
                num_neg=num_neg,
                sorted_edge_hash=self.ctx.edge_hash_sorted,
                rng=self.rng,
                candidates=candidate_nodes,
            )
            pos_scores = self._score_edges_from_cached_nodes(
                pos_batch,
                cached_node_ids,
                cached_embeddings,
            )
            pos_score_parts.append(pos_scores)

            if neg_valid.any():
                neg_scores = self._score_edges_from_cached_nodes(
                    neg_batch[:, neg_valid],
                    cached_node_ids,
                    cached_embeddings,
                )
                neg_score_parts.append(neg_scores)
                total_neg_edges += int(neg_scores.shape[0])

            if include_hit10:
                valid_2d = neg_valid.reshape(pos_batch.shape[1], num_neg)
                full_rows = np.flatnonzero(valid_2d.all(axis=1))
                if full_rows.size > 0:
                    neg_grouped = neg_batch.reshape(2, pos_batch.shape[1], num_neg)
                    full_neg_edges = np.ascontiguousarray(
                        neg_grouped[:, full_rows, :].reshape(2, full_rows.size * num_neg)
                    )
                    full_neg_scores = self._score_edges_from_cached_nodes(
                        full_neg_edges,
                        cached_node_ids,
                        cached_embeddings,
                    ).reshape(full_rows.size, num_neg)
                    pos_scores_full = pos_scores[full_rows]
                    ranks = np.sum(full_neg_scores >= pos_scores_full.reshape(-1, 1), axis=1) + 1
                    hit10_hits += int(np.sum(ranks <= 10))
                    hit10_pos_edges_evaluated += int(full_rows.size)
                    hit10_neg_edges_evaluated += int(full_rows.size * num_neg)

            now_ts = time.perf_counter()
            if (
                batch_idx == total_batches
                or batch_idx % 200 == 0
                or (now_ts - last_progress_ts) >= progress_interval_sec
            ):
                log_event(
                    "[STAGE]",
                    (
                        f"{label} scoring progress batches={batch_idx}/{total_batches} "
                        f"pos_edges_done={stop}/{int(pos.shape[1])} "
                        f"neg_edges_done={total_neg_edges} "
                        f"progress={(100.0 * float(stop) / float(max(1, int(pos.shape[1])))):.1f}%"
                    ),
                )
                last_progress_ts = now_ts

        all_pos_scores = np.concatenate(pos_score_parts, axis=0) if pos_score_parts else np.empty((0,), dtype=np.float32)
        all_neg_scores = np.concatenate(neg_score_parts, axis=0) if neg_score_parts else np.empty((0,), dtype=np.float32)
        metrics = compute_auc_aucpr_from_pos_neg(all_pos_scores, all_neg_scores)
        hit10 = (
            float(hit10_hits) / float(hit10_pos_edges_evaluated)
            if include_hit10 and hit10_pos_edges_evaluated > 0
            else 0.0
        )
        return {
            "auc": metrics["auc"],
            "aucpr": metrics["aucpr"],
            "hit10": hit10,
            "pos_edges_evaluated": int(pos.shape[1]),
            "neg_edges_evaluated": int(total_neg_edges),
            "hit10_pos_edges_evaluated": int(hit10_pos_edges_evaluated),
            "hit10_neg_edges_evaluated": int(hit10_neg_edges_evaluated),
            "negatives_per_positive": (
                float(total_neg_edges) / float(max(1, int(pos.shape[1])))
            ),
        }

    @torch.no_grad()
    def _run_final_test(self) -> Dict[str, float]:
        log_event("[STAGE]", "final test start")
        total_start = time.perf_counter()
        pool_info = self._build_final_test_node_pool()
        test_edges = pool_info["test_edges"]
        test_node_pool = pool_info["node_pool"]
        art_test_node_pool = pool_info["art_node_pool"]
        art_any_test_node_pool = pool_info["art_any_node_pool"]
        log_event(
            "[STAGE]",
            (
                f"final test node pool built node_pool_size={int(test_node_pool.shape[0])} "
                f"art_node_pool_size={int(art_test_node_pool.shape[0])} "
                f"art_any_node_pool_size={int(art_any_test_node_pool.shape[0])}"
            ),
        )

        cache_start = time.perf_counter()
        cached_node_ids, cached_embeddings, cache_stats = self._encode_final_test_nodes_by_chunk(
            test_node_pool
        )
        cache_sec = float(time.perf_counter() - cache_start)
        log_event(
            "[STAGE]",
            (
                f"final test node embedding cache built ({cache_sec:.2f}s) "
                f"node_pool_size={cache_stats['node_pool_size']} "
                f"encoded_chunks={cache_stats['encoded_chunks']} "
                f"embedding_dim={cache_stats['embedding_dim']} "
                f"cache_dtype={str(cached_embeddings.dtype)}"
            ),
        )

        out = self._evaluate_edge_set_from_cached_nodes(
            label="final test",
            pos_edges=test_edges,
            candidate_nodes=test_node_pool,
            cached_node_ids=cached_node_ids,
            cached_embeddings=cached_embeddings,
            include_hit10=bool(self.cfg.eval.final_test_enable_hit10),
        )
        out["node_pool_size"] = int(test_node_pool.shape[0])

        if bool(self.cfg.eval.final_test_art_metrics):
            art_test_edges = self._filter_edges_to_art_global(test_edges)
            art_out = self._evaluate_edge_set_from_cached_nodes(
                label="final test art",
                pos_edges=art_test_edges,
                candidate_nodes=art_test_node_pool,
                cached_node_ids=cached_node_ids,
                cached_embeddings=cached_embeddings,
                include_hit10=bool(self.cfg.eval.final_test_enable_hit10),
            )
            art_any_test_edges = self._filter_edges_to_art_any_global(test_edges)
            art_any_out = self._evaluate_edge_set_from_cached_nodes(
                label="final test art_any",
                pos_edges=art_any_test_edges,
                candidate_nodes=art_any_test_node_pool,
                cached_node_ids=cached_node_ids,
                cached_embeddings=cached_embeddings,
                include_hit10=bool(self.cfg.eval.final_test_enable_hit10),
            )
        else:
            art_test_edges = self._filter_edges_to_art_global(test_edges)
            art_any_test_edges = self._filter_edges_to_art_any_global(test_edges)
            art_out = {
                "auc": 0.0,
                "aucpr": 0.0,
                "hit10": 0.0,
                "pos_edges_evaluated": 0,
                "neg_edges_evaluated": 0,
                "hit10_pos_edges_evaluated": 0,
                "hit10_neg_edges_evaluated": 0,
                "negatives_per_positive": 0.0,
            }
            art_any_out = {
                "auc": 0.0,
                "aucpr": 0.0,
                "hit10": 0.0,
                "pos_edges_evaluated": 0,
                "neg_edges_evaluated": 0,
                "hit10_pos_edges_evaluated": 0,
                "hit10_neg_edges_evaluated": 0,
                "negatives_per_positive": 0.0,
            }

        out["art_auc"] = float(art_out["auc"])
        out["art_aucpr"] = float(art_out["aucpr"])
        out["art_hit10"] = float(art_out["hit10"])
        out["art_pos_edges"] = int(art_test_edges.shape[1])
        out["art_pos_edges_evaluated"] = int(art_out["pos_edges_evaluated"])
        out["art_neg_edges_evaluated"] = int(art_out["neg_edges_evaluated"])
        out["art_negatives_per_positive"] = float(art_out["negatives_per_positive"])
        out["art_hit10_pos_edges_evaluated"] = int(art_out["hit10_pos_edges_evaluated"])
        out["art_hit10_neg_edges_evaluated"] = int(art_out["hit10_neg_edges_evaluated"])
        out["art_any_auc"] = float(art_any_out["auc"])
        out["art_any_aucpr"] = float(art_any_out["aucpr"])
        out["art_any_hit10"] = float(art_any_out["hit10"])
        out["art_any_pos_edges"] = int(art_any_test_edges.shape[1])
        out["art_any_pos_edges_evaluated"] = int(art_any_out["pos_edges_evaluated"])
        out["art_any_neg_edges_evaluated"] = int(art_any_out["neg_edges_evaluated"])
        out["art_any_negatives_per_positive"] = float(art_any_out["negatives_per_positive"])
        out["art_any_hit10_pos_edges_evaluated"] = int(art_any_out["hit10_pos_edges_evaluated"])
        out["art_any_hit10_neg_edges_evaluated"] = int(art_any_out["hit10_neg_edges_evaluated"])
        self._clear_runtime_counters()
        log_event("[STAGE]", f"final test done ({time.perf_counter() - total_start:.2f}s) metrics={out}")
        return out

    @torch.no_grad()
    def _evaluate_edges(
        self,
        edges: torch.Tensor,
        max_edges: int,
        num_neg: int,
    ) -> Dict[str, float]:
        pos = edges.detach().cpu().numpy().astype(np.int64)
        if max_edges > 0 and pos.shape[1] > max_edges:
            idx = self.rng.choice(pos.shape[1], size=max_edges, replace=False)
            pos = pos[:, idx]

        y_true = []
        y_score = []
        max_query_edges_in_message_graph = 0
        neg_edges_evaluated = 0
        pos_edges_evaluated = 0
        batch_size = self.cfg.data.batch_edges
        self.model.eval()

        for start in range(0, pos.shape[1], batch_size):
            pos_batch = pos[:, start : start + batch_size]
            neg_batch = sample_negative_edges_undirected(
                pos_batch,
                num_nodes=self.ctx.graph.num_nodes,
                num_neg=num_neg,
                sorted_edge_hash=self.ctx.edge_hash_sorted,
                rng=self.rng,
            )
            neg_edges_evaluated += int(neg_batch.shape[1])
            batch_cpu = self._prepare_batch_cpu(
                pos_batch,
                neg_batch,
                active_chunk=-1,
                mask_supervision_edges=False,
                require_query_edges_absent=True,
                query_context_label="validation/test",
            )
            max_query_edges_in_message_graph = max(
                max_query_edges_in_message_graph,
                int(batch_cpu.get("query_edges_in_message_graph", 0)),
            )
            batch = self._move_batch_to_device(batch_cpu)
            with self._autocast_context():
                logits = self._forward_logits(batch)
            score = logits.detach().float().cpu().numpy()
            n_pos = pos_batch.shape[1]
            n_neg = max(0, int(score.shape[0]) - n_pos)
            y_true.extend([1] * n_pos + [0] * n_neg)
            y_score.extend(score.tolist())

        metrics = compute_auc_aucpr(np.asarray(y_true), np.asarray(y_score))
        return {
            "auc": metrics["auc"],
            "aucpr": metrics["aucpr"],
            "query_edges_in_message_graph": float(max_query_edges_in_message_graph),
            "pos_edges_evaluated": int(pos.shape[1]),
            "neg_edges_evaluated": int(neg_edges_evaluated),
            "negatives_per_positive": int(num_neg),
        }

    @torch.no_grad()
    def _evaluate_edges_art_subset(
        self,
        edges: torch.Tensor,
        max_edges: int,
        num_neg: int,
    ) -> Dict[str, float]:
        pos = edges.detach().cpu().numpy().astype(np.int64)
        pos = self._filter_edges_to_art_global(pos)
        art_pos_edges = int(pos.shape[1])
        if max_edges > 0 and pos.shape[1] > max_edges:
            idx = self.rng.choice(pos.shape[1], size=max_edges, replace=False)
            pos = pos[:, idx]
        if pos.shape[1] == 0:
            return {
                "auc": 0.0,
                "aucpr": 0.0,
                "pos_edges": art_pos_edges,
                "pos_edges_evaluated": 0,
                "neg_edges_evaluated": 0,
                "negatives_per_positive": int(num_neg),
                "query_edges_in_message_graph": 0.0,
            }

        y_true = []
        y_score = []
        max_query_edges_in_message_graph = 0
        neg_edges_evaluated = 0
        pos_edges_evaluated = 0
        batch_size = self.cfg.data.batch_edges
        self.model.eval()

        for start in range(0, pos.shape[1], batch_size):
            pos_batch = pos[:, start : start + batch_size]
            neg_batch = sample_negative_edges_art_subset_undirected(
                pos_batch,
                num_nodes=self.ctx.graph.num_nodes,
                num_neg=num_neg,
                sorted_edge_hash=self.ctx.edge_hash_sorted,
                rng=self.rng,
                art_candidates=self.global_art_node_indices,
            )
            if neg_batch.shape[1] != pos_batch.shape[1] * int(num_neg):
                continue
            pos_edges_evaluated += int(pos_batch.shape[1])
            neg_edges_evaluated += int(neg_batch.shape[1])
            batch_cpu = self._prepare_batch_cpu(
                pos_batch,
                neg_batch,
                active_chunk=-1,
                mask_supervision_edges=False,
                require_query_edges_absent=True,
                query_context_label="validation/test",
            )
            max_query_edges_in_message_graph = max(
                max_query_edges_in_message_graph,
                int(batch_cpu.get("query_edges_in_message_graph", 0)),
            )
            batch = self._move_batch_to_device(batch_cpu)
            with self._autocast_context():
                logits = self._forward_logits(batch)
            score = logits.detach().float().cpu().numpy()
            n_pos = pos_batch.shape[1]
            n_neg = max(0, int(score.shape[0]) - n_pos)
            y_true.extend([1] * n_pos + [0] * n_neg)
            y_score.extend(score.tolist())

        if not y_true:
            return {
                "auc": 0.0,
                "aucpr": 0.0,
                "pos_edges": art_pos_edges,
                "pos_edges_evaluated": int(pos_edges_evaluated),
                "neg_edges_evaluated": int(neg_edges_evaluated),
                "negatives_per_positive": int(num_neg),
                "query_edges_in_message_graph": float(max_query_edges_in_message_graph),
            }
        metrics = compute_auc_aucpr(np.asarray(y_true), np.asarray(y_score))
        return {
            "auc": metrics["auc"],
            "aucpr": metrics["aucpr"],
            "pos_edges": art_pos_edges,
            "query_edges_in_message_graph": float(max_query_edges_in_message_graph),
            "pos_edges_evaluated": int(pos_edges_evaluated),
            "neg_edges_evaluated": int(neg_edges_evaluated),
            "negatives_per_positive": int(num_neg),
        }

    @torch.no_grad()
    def _evaluate_edges_art_any_subset(
        self,
        edges: torch.Tensor,
        max_edges: int,
        num_neg: int,
    ) -> Dict[str, float]:
        pos = edges.detach().cpu().numpy().astype(np.int64)
        pos = self._filter_edges_to_art_any_global(pos)
        art_any_pos_edges = int(pos.shape[1])
        if max_edges > 0 and pos.shape[1] > max_edges:
            idx = self.rng.choice(pos.shape[1], size=max_edges, replace=False)
            pos = pos[:, idx]
        if pos.shape[1] == 0:
            return {
                "auc": 0.0,
                "aucpr": 0.0,
                "pos_edges": art_any_pos_edges,
                "pos_edges_evaluated": 0,
                "neg_edges_evaluated": 0,
                "negatives_per_positive": int(num_neg),
                "query_edges_in_message_graph": 0.0,
            }

        y_true = []
        y_score = []
        max_query_edges_in_message_graph = 0
        neg_edges_evaluated = 0
        pos_edges_evaluated = 0
        batch_size = self.cfg.data.batch_edges
        all_candidates = np.arange(self.ctx.graph.num_nodes, dtype=np.int64)
        self.model.eval()

        for start in range(0, pos.shape[1], batch_size):
            pos_batch = pos[:, start : start + batch_size]
            neg_batch = sample_negative_edges_art_any_subset_undirected(
                pos_batch,
                num_nodes=self.ctx.graph.num_nodes,
                num_neg=num_neg,
                sorted_edge_hash=self.ctx.edge_hash_sorted,
                rng=self.rng,
                art_candidates=self.global_art_node_indices,
                other_candidates=all_candidates,
            )
            if neg_batch.shape[1] != pos_batch.shape[1] * int(num_neg):
                continue
            pos_edges_evaluated += int(pos_batch.shape[1])
            neg_edges_evaluated += int(neg_batch.shape[1])
            batch_cpu = self._prepare_batch_cpu(
                pos_batch,
                neg_batch,
                active_chunk=-1,
                mask_supervision_edges=False,
                require_query_edges_absent=True,
                query_context_label="validation/test",
            )
            max_query_edges_in_message_graph = max(
                max_query_edges_in_message_graph,
                int(batch_cpu.get("query_edges_in_message_graph", 0)),
            )
            batch = self._move_batch_to_device(batch_cpu)
            with self._autocast_context():
                logits = self._forward_logits(batch)
            score = logits.detach().float().cpu().numpy()
            n_pos = pos_batch.shape[1]
            n_neg = max(0, int(score.shape[0]) - n_pos)
            y_true.extend([1] * n_pos + [0] * n_neg)
            y_score.extend(score.tolist())

        if not y_true:
            return {
                "auc": 0.0,
                "aucpr": 0.0,
                "pos_edges": art_any_pos_edges,
                "pos_edges_evaluated": int(pos_edges_evaluated),
                "neg_edges_evaluated": int(neg_edges_evaluated),
                "negatives_per_positive": int(num_neg),
                "query_edges_in_message_graph": float(max_query_edges_in_message_graph),
            }
        metrics = compute_auc_aucpr(np.asarray(y_true), np.asarray(y_score))
        return {
            "auc": metrics["auc"],
            "aucpr": metrics["aucpr"],
            "pos_edges": art_any_pos_edges,
            "query_edges_in_message_graph": float(max_query_edges_in_message_graph),
            "pos_edges_evaluated": int(pos_edges_evaluated),
            "neg_edges_evaluated": int(neg_edges_evaluated),
            "negatives_per_positive": int(num_neg),
        }

    @torch.no_grad()
    def _evaluate_edges_chunk_full(self, edges_by_chunk: Dict[int, np.ndarray], max_edges: int) -> Dict[str, float]:
        selected = self._select_chunk_eval_edges(edges_by_chunk, max_edges)
        if not selected:
            return {"auc": 0.0, "aucpr": 0.0}
        y_true = []
        y_score = []
        self.model.eval()

        for chunk_id in sorted(selected):
            pos_batch = selected[chunk_id]
            if pos_batch.shape[1] == 0:
                continue
            sub = self.ctx.graph.load_chunk_subgraph(int(chunk_id))
            neg_batch = sample_negative_edges_center_aware(
                pos_batch,
                num_nodes=int(sub.node_indices_global.shape[0]),
                num_neg=1,
                sorted_edge_hash=self.chunk_edge_hashes[int(chunk_id)],
                rng=self.rng,
                center_mask=sub.center_mask_local,
            )
            batch_cpu = self._prepare_chunk_batch_cpu(
                int(chunk_id),
                pos_batch,
                neg_batch,
                message_edge_index_local=self.train_message_edges_by_chunk.get(
                    int(chunk_id),
                    np.empty((2, 0), dtype=np.int64),
                ),
                sub=sub,
                progress_reporter=None,
            )
            batch = self._move_batch_to_device(batch_cpu)
            with self._autocast_context():
                logits = self._forward_logits(batch)
            score = logits.detach().float().cpu().numpy()
            n_pos = pos_batch.shape[1]
            y_true.extend([1] * n_pos + [0] * n_pos)
            y_score.extend(score.tolist())

        if not y_true:
            return {"auc": 0.0, "aucpr": 0.0}
        metrics = compute_auc_aucpr(np.asarray(y_true), np.asarray(y_score))
        return {
            "auc": metrics["auc"],
            "aucpr": metrics["aucpr"],
        }

    @torch.no_grad()
    def _evaluate_edges_art_subset_chunk_full(
        self,
        edges_by_chunk: Dict[int, np.ndarray],
        max_edges: int,
    ) -> Dict[str, float]:
        art_edges_by_chunk, counts = self._filter_edges_to_art_chunk(edges_by_chunk)
        art_pos_edges = int(sum(counts.values()))
        selected = self._select_chunk_eval_edges(art_edges_by_chunk, max_edges)
        if not selected:
            return {"auc": 0.0, "aucpr": 0.0, "pos_edges": art_pos_edges}
        y_true = []
        y_score = []
        self.model.eval()

        for chunk_id in sorted(selected):
            pos_batch = selected[chunk_id]
            if pos_batch.shape[1] == 0:
                continue
            sub = self.ctx.graph.load_chunk_subgraph(int(chunk_id))
            art_candidates = np.flatnonzero(sub.node_is_art).astype(np.int64, copy=False)
            neg_batch = sample_negative_edges_center_aware_art_subset(
                pos_batch,
                num_nodes=int(sub.node_indices_global.shape[0]),
                num_neg=1,
                sorted_edge_hash=self.chunk_edge_hashes[int(chunk_id)],
                rng=self.rng,
                center_mask=sub.center_mask_local,
                art_candidates=art_candidates,
            )
            if neg_batch.shape[1] != pos_batch.shape[1]:
                continue
            batch_cpu = self._prepare_chunk_batch_cpu(
                int(chunk_id),
                pos_batch,
                neg_batch,
                message_edge_index_local=self.train_message_edges_by_chunk.get(
                    int(chunk_id),
                    np.empty((2, 0), dtype=np.int64),
                ),
                sub=sub,
                progress_reporter=None,
            )
            batch = self._move_batch_to_device(batch_cpu)
            with self._autocast_context():
                logits = self._forward_logits(batch)
            score = logits.detach().float().cpu().numpy()
            n_pos = pos_batch.shape[1]
            y_true.extend([1] * n_pos + [0] * n_pos)
            y_score.extend(score.tolist())

        if not y_true:
            return {"auc": 0.0, "aucpr": 0.0, "pos_edges": art_pos_edges}
        metrics = compute_auc_aucpr(np.asarray(y_true), np.asarray(y_score))
        return {
            "auc": metrics["auc"],
            "aucpr": metrics["aucpr"],
            "pos_edges": art_pos_edges,
        }

    @torch.no_grad()
    def _evaluate_hit10(self, edges: torch.Tensor, pos_limit: int, negatives: int) -> Dict[str, float]:
        if pos_limit < 0:
            return {"hit10": 0.0, "pos_edges_evaluated": 0, "neg_edges_evaluated": 0}
        pos = edges.detach().cpu().numpy().astype(np.int64)
        if pos_limit > 0 and pos.shape[1] > pos_limit:
            idx = self.rng.choice(pos.shape[1], size=pos_limit, replace=False)
            pos = pos[:, idx]

        pos_scores = []
        neg_scores_all = []
        neg_edges_evaluated = 0
        self.model.eval()
        for i in range(pos.shape[1]):
            pos_edge = pos[:, i : i + 1]
            neg_edges = sample_negative_edges_undirected(
                pos_edge,
                num_nodes=self.ctx.graph.num_nodes,
                num_neg=negatives,
                sorted_edge_hash=self.ctx.edge_hash_sorted,
                rng=self.rng,
            )
            neg_edges_evaluated += int(neg_edges.shape[1])
            batch_cpu = self._prepare_batch_cpu(
                pos_edge,
                neg_edges,
                active_chunk=-1,
                mask_supervision_edges=False,
                require_query_edges_absent=True,
                query_context_label="validation/test",
            )
            batch = self._move_batch_to_device(batch_cpu)
            with self._autocast_context():
                logits = self._forward_logits(batch)
            score = logits.detach().float().cpu().numpy()
            pos_scores.append(float(score[0]))
            neg_scores_all.append(score[1:])

        if not pos_scores:
            return {"hit10": 0.0, "pos_edges_evaluated": 0, "neg_edges_evaluated": 0}
        return {
            "hit10": hit_at_k(np.asarray(pos_scores), np.asarray(neg_scores_all), k=10),
            "pos_edges_evaluated": int(len(pos_scores)),
            "neg_edges_evaluated": int(neg_edges_evaluated),
        }

    @torch.no_grad()
    def _evaluate_hit10_art_subset(self, edges: torch.Tensor, pos_limit: int, negatives: int) -> Dict[str, float]:
        if pos_limit < 0:
            return {"hit10": 0.0, "pos_edges_evaluated": 0, "neg_edges_evaluated": 0}
        pos = edges.detach().cpu().numpy().astype(np.int64)
        pos = self._filter_edges_to_art_global(pos)
        if pos_limit > 0 and pos.shape[1] > pos_limit:
            idx = self.rng.choice(pos.shape[1], size=pos_limit, replace=False)
            pos = pos[:, idx]
        if pos.shape[1] == 0:
            return {"hit10": 0.0, "pos_edges_evaluated": 0, "neg_edges_evaluated": 0}

        pos_scores = []
        neg_scores_all = []
        neg_edges_evaluated = 0
        self.model.eval()
        for i in range(pos.shape[1]):
            pos_edge = pos[:, i : i + 1]
            neg_edges = sample_negative_edges_art_subset_undirected(
                pos_edge,
                num_nodes=self.ctx.graph.num_nodes,
                num_neg=negatives,
                sorted_edge_hash=self.ctx.edge_hash_sorted,
                rng=self.rng,
                art_candidates=self.global_art_node_indices,
            )
            if neg_edges.shape[1] != negatives:
                continue
            neg_edges_evaluated += int(neg_edges.shape[1])
            batch_cpu = self._prepare_batch_cpu(
                pos_edge,
                neg_edges,
                active_chunk=-1,
                mask_supervision_edges=False,
                require_query_edges_absent=True,
                query_context_label="validation/test",
            )
            batch = self._move_batch_to_device(batch_cpu)
            with self._autocast_context():
                logits = self._forward_logits(batch)
            score = logits.detach().float().cpu().numpy()
            pos_scores.append(float(score[0]))
            neg_scores_all.append(score[1:])

        if not pos_scores:
            return {"hit10": 0.0, "pos_edges_evaluated": 0, "neg_edges_evaluated": 0}
        return {
            "hit10": hit_at_k(np.asarray(pos_scores), np.asarray(neg_scores_all), k=10),
            "pos_edges_evaluated": int(len(pos_scores)),
            "neg_edges_evaluated": int(neg_edges_evaluated),
        }

    @torch.no_grad()
    def _evaluate_hit10_art_any_subset(self, edges: torch.Tensor, pos_limit: int, negatives: int) -> Dict[str, float]:
        if pos_limit < 0:
            return {"hit10": 0.0, "pos_edges_evaluated": 0, "neg_edges_evaluated": 0}
        pos = edges.detach().cpu().numpy().astype(np.int64)
        pos = self._filter_edges_to_art_any_global(pos)
        if pos_limit > 0 and pos.shape[1] > pos_limit:
            idx = self.rng.choice(pos.shape[1], size=pos_limit, replace=False)
            pos = pos[:, idx]
        if pos.shape[1] == 0:
            return {"hit10": 0.0, "pos_edges_evaluated": 0, "neg_edges_evaluated": 0}

        pos_scores = []
        neg_scores_all = []
        neg_edges_evaluated = 0
        all_candidates = np.arange(self.ctx.graph.num_nodes, dtype=np.int64)
        self.model.eval()
        for i in range(pos.shape[1]):
            pos_edge = pos[:, i : i + 1]
            neg_edges = sample_negative_edges_art_any_subset_undirected(
                pos_edge,
                num_nodes=self.ctx.graph.num_nodes,
                num_neg=negatives,
                sorted_edge_hash=self.ctx.edge_hash_sorted,
                rng=self.rng,
                art_candidates=self.global_art_node_indices,
                other_candidates=all_candidates,
            )
            if neg_edges.shape[1] != negatives:
                continue
            neg_edges_evaluated += int(neg_edges.shape[1])
            batch_cpu = self._prepare_batch_cpu(
                pos_edge,
                neg_edges,
                active_chunk=-1,
                mask_supervision_edges=False,
                require_query_edges_absent=True,
                query_context_label="validation/test",
            )
            batch = self._move_batch_to_device(batch_cpu)
            with self._autocast_context():
                logits = self._forward_logits(batch)
            score = logits.detach().float().cpu().numpy()
            pos_scores.append(float(score[0]))
            neg_scores_all.append(score[1:])

        if not pos_scores:
            return {"hit10": 0.0, "pos_edges_evaluated": 0, "neg_edges_evaluated": 0}
        return {
            "hit10": hit_at_k(np.asarray(pos_scores), np.asarray(neg_scores_all), k=10),
            "pos_edges_evaluated": int(len(pos_scores)),
            "neg_edges_evaluated": int(neg_edges_evaluated),
        }

    @torch.no_grad()
    def _evaluate_hit10_chunk_full(
        self,
        edges_by_chunk: Dict[int, np.ndarray],
        pos_limit: int,
        negatives: int,
    ) -> float:
        if pos_limit <= 0:
            return 0.0
        selected = self._select_chunk_eval_edges(edges_by_chunk, pos_limit)
        pos_scores = []
        neg_scores_all = []
        self.model.eval()

        for chunk_id in sorted(selected):
            pos_edges = selected[chunk_id]
            sub = self.ctx.graph.load_chunk_subgraph(int(chunk_id))
            for i in range(pos_edges.shape[1]):
                pos_edge = pos_edges[:, i : i + 1]
                neg_edges = sample_negative_edges_center_aware(
                    pos_edge,
                    num_nodes=int(sub.node_indices_global.shape[0]),
                    num_neg=negatives,
                    sorted_edge_hash=self.chunk_edge_hashes[int(chunk_id)],
                    rng=self.rng,
                    center_mask=sub.center_mask_local,
                )
                batch_cpu = self._prepare_chunk_batch_cpu(
                    int(chunk_id),
                    pos_edge,
                    neg_edges,
                    message_edge_index_local=self.train_message_edges_by_chunk.get(
                        int(chunk_id),
                        np.empty((2, 0), dtype=np.int64),
                    ),
                    sub=sub,
                    progress_reporter=None,
                )
                batch = self._move_batch_to_device(batch_cpu)
                with self._autocast_context():
                    logits = self._forward_logits(batch)
                score = logits.detach().float().cpu().numpy()
                pos_scores.append(float(score[0]))
                neg_scores_all.append(score[1:])

        if not pos_scores:
            return 0.0
        return hit_at_k(np.asarray(pos_scores), np.asarray(neg_scores_all), k=10)

    @torch.no_grad()
    def _evaluate_hit10_art_subset_chunk_full(
        self,
        edges_by_chunk: Dict[int, np.ndarray],
        pos_limit: int,
        negatives: int,
    ) -> float:
        if pos_limit <= 0:
            return 0.0
        art_edges_by_chunk, _ = self._filter_edges_to_art_chunk(edges_by_chunk)
        selected = self._select_chunk_eval_edges(art_edges_by_chunk, pos_limit)
        pos_scores = []
        neg_scores_all = []
        self.model.eval()

        for chunk_id in sorted(selected):
            pos_edges = selected[chunk_id]
            sub = self.ctx.graph.load_chunk_subgraph(int(chunk_id))
            art_candidates = np.flatnonzero(sub.node_is_art).astype(np.int64, copy=False)
            for i in range(pos_edges.shape[1]):
                pos_edge = pos_edges[:, i : i + 1]
                neg_edges = sample_negative_edges_center_aware_art_subset(
                    pos_edge,
                    num_nodes=int(sub.node_indices_global.shape[0]),
                    num_neg=negatives,
                    sorted_edge_hash=self.chunk_edge_hashes[int(chunk_id)],
                    rng=self.rng,
                    center_mask=sub.center_mask_local,
                    art_candidates=art_candidates,
                )
                if neg_edges.shape[1] != negatives:
                    continue
                batch_cpu = self._prepare_chunk_batch_cpu(
                    int(chunk_id),
                    pos_edge,
                    neg_edges,
                    message_edge_index_local=self.train_message_edges_by_chunk.get(
                        int(chunk_id),
                        np.empty((2, 0), dtype=np.int64),
                    ),
                    sub=sub,
                    progress_reporter=None,
                )
                batch = self._move_batch_to_device(batch_cpu)
                with self._autocast_context():
                    logits = self._forward_logits(batch)
                score = logits.detach().float().cpu().numpy()
                pos_scores.append(float(score[0]))
                neg_scores_all.append(score[1:])

        if not pos_scores:
            return 0.0
        return hit_at_k(np.asarray(pos_scores), np.asarray(neg_scores_all), k=10)

    def _run_split_evaluation(
        self,
        *,
        split_name: str,
        edges: torch.Tensor,
        max_edges: int,
        num_neg: int,
        hit10_pos_limit: int,
        hit10_negatives: int,
    ) -> Dict[str, float]:
        log_event("[STAGE]", f"{split_name} start")
        start_time = time.perf_counter()
        out = self._evaluate_edges(edges, max_edges=max_edges, num_neg=num_neg)
        log_event(
            "[STAGE]",
            f"{split_name} propagation check query_edges_in_message_graph={int(out.get('query_edges_in_message_graph', 0))}",
        )
        if self.cfg.eval.enable_hit10:
            hit10_out = self._evaluate_hit10(
                edges,
                pos_limit=hit10_pos_limit,
                negatives=hit10_negatives,
            )
            out["hit10"] = hit10_out["hit10"]
            out["hit10_pos_edges_evaluated"] = int(hit10_out["pos_edges_evaluated"])
            out["hit10_neg_edges_evaluated"] = int(hit10_out["neg_edges_evaluated"])
        else:
            out["hit10"] = 0.0
            out["hit10_pos_edges_evaluated"] = 0
            out["hit10_neg_edges_evaluated"] = 0

        art_out = self._evaluate_edges_art_subset(edges, max_edges=max_edges, num_neg=num_neg)
        if self.cfg.eval.enable_hit10:
            art_hit10_out = self._evaluate_hit10_art_subset(
                edges,
                pos_limit=hit10_pos_limit,
                negatives=hit10_negatives,
            )
            art_out["hit10"] = art_hit10_out["hit10"]
            art_out["hit10_pos_edges_evaluated"] = int(art_hit10_out["pos_edges_evaluated"])
            art_out["hit10_neg_edges_evaluated"] = int(art_hit10_out["neg_edges_evaluated"])
        else:
            art_out["hit10"] = 0.0
            art_out["hit10_pos_edges_evaluated"] = 0
            art_out["hit10_neg_edges_evaluated"] = 0

        art_any_out = self._evaluate_edges_art_any_subset(edges, max_edges=max_edges, num_neg=num_neg)
        if self.cfg.eval.enable_hit10:
            art_any_hit10_out = self._evaluate_hit10_art_any_subset(
                edges,
                pos_limit=hit10_pos_limit,
                negatives=hit10_negatives,
            )
            art_any_out["hit10"] = art_any_hit10_out["hit10"]
            art_any_out["hit10_pos_edges_evaluated"] = int(art_any_hit10_out["pos_edges_evaluated"])
            art_any_out["hit10_neg_edges_evaluated"] = int(art_any_hit10_out["neg_edges_evaluated"])
        else:
            art_any_out["hit10"] = 0.0
            art_any_out["hit10_pos_edges_evaluated"] = 0
            art_any_out["hit10_neg_edges_evaluated"] = 0

        out["art_auc"] = art_out["auc"]
        out["art_aucpr"] = art_out["aucpr"]
        out["art_hit10"] = art_out["hit10"]
        out["art_pos_edges"] = int(art_out.get("pos_edges", 0))
        out["art_pos_edges_evaluated"] = int(art_out.get("pos_edges_evaluated", 0))
        out["art_neg_edges_evaluated"] = int(art_out.get("neg_edges_evaluated", 0))
        out["art_negatives_per_positive"] = int(art_out.get("negatives_per_positive", num_neg))
        out["art_hit10_pos_edges_evaluated"] = int(art_out.get("hit10_pos_edges_evaluated", 0))
        out["art_hit10_neg_edges_evaluated"] = int(art_out.get("hit10_neg_edges_evaluated", 0))
        out["art_any_auc"] = art_any_out["auc"]
        out["art_any_aucpr"] = art_any_out["aucpr"]
        out["art_any_hit10"] = art_any_out["hit10"]
        out["art_any_pos_edges"] = int(art_any_out.get("pos_edges", 0))
        out["art_any_pos_edges_evaluated"] = int(art_any_out.get("pos_edges_evaluated", 0))
        out["art_any_neg_edges_evaluated"] = int(art_any_out.get("neg_edges_evaluated", 0))
        out["art_any_negatives_per_positive"] = int(art_any_out.get("negatives_per_positive", num_neg))
        out["art_any_hit10_pos_edges_evaluated"] = int(art_any_out.get("hit10_pos_edges_evaluated", 0))
        out["art_any_hit10_neg_edges_evaluated"] = int(art_any_out.get("hit10_neg_edges_evaluated", 0))
        if out["art_pos_edges"] == 0:
            log_event("[STAGE]", f"{split_name} art subset empty")
        if out["art_any_pos_edges"] == 0:
            log_event("[STAGE]", f"{split_name} art_any subset empty")
        self._clear_runtime_counters()
        log_event("[STAGE]", f"{split_name} done ({time.perf_counter() - start_time:.2f}s) metrics={out}")
        return out

    def _run_validation(self) -> Dict[str, float]:
        return self._run_split_evaluation(
            split_name="validation",
            edges=self.ctx.val_edges,
            max_edges=self.cfg.eval.val_max_edges,
            num_neg=1,
            hit10_pos_limit=self.cfg.eval.hit10_pos_limit,
            hit10_negatives=self.cfg.eval.hit10_negatives,
        )

    def _run_test(self) -> Dict[str, float]:
        return self._run_split_evaluation(
            split_name="test",
            edges=self.ctx.test_edges,
            max_edges=self.cfg.eval.test_max_edges,
            num_neg=1,
            hit10_pos_limit=self.cfg.eval.hit10_pos_limit,
            hit10_negatives=self.cfg.eval.hit10_negatives,
        )

    def _should_schedule_step(self, step_idx: int, steps_per_epoch: int) -> bool:
        return step_idx < self.effective_max_steps and (step_idx // steps_per_epoch) < self.cfg.optim.epochs

    def _fill_prefetch_queue(
        self,
        pending: Deque[Tuple[int, StepProgressReporter, Future]],
        executor: ThreadPoolExecutor,
        next_step_to_schedule: int,
        steps_per_epoch: int,
    ) -> int:
        queue_target = max(1, self.prefetch_batches)
        while len(pending) < queue_target and self._should_schedule_step(next_step_to_schedule, steps_per_epoch):
            pos_batch = self.batch_sampler.next_batch(next_step_to_schedule)
            active_chunk = self.batch_sampler.active_chunk
            reporter = self._make_step_reporter(next_step_to_schedule, active_chunk)
            neg_batch = sample_negative_edges_undirected(
                pos_batch,
                num_nodes=self.ctx.graph.num_nodes,
                num_neg=self.cfg.data.num_neg,
                sorted_edge_hash=self.ctx.edge_hash_sorted,
                rng=self.rng,
            )
            future = executor.submit(
                self._prepare_batch_cpu,
                pos_batch,
                neg_batch,
                active_chunk,
                reporter,
            )
            pending.append((next_step_to_schedule, reporter, future))
            next_step_to_schedule += 1
        return next_step_to_schedule

    def train(self) -> Dict[str, float]:
        steps_per_epoch = max(1, self.ctx.train_edges.shape[1] // self.cfg.data.batch_edges)
        start_time = time.time()
        latest_fast_val = {"auc": 0.0, "aucpr": 0.0, "hit10": 0.0}
        pending: Deque[Tuple[int, StepProgressReporter, Future]] = deque()
        prefetch_executor: Optional[ThreadPoolExecutor] = None
        next_step_to_schedule = self.global_step
        log_event(
            "[STAGE]",
            (
                f"train start mode=neighbor_sample device={self.device} "
                f"batch_edges={self.effective_batch_edges} "
                f"propagation_edges={int(getattr(self.ctx.train_propagation_edges, 'shape', [0, 0])[1])} "
                f"num_hops={int(self.cfg.data.num_hops)} "
                f"num_neighbors={list(self.cfg.data.num_neighbors)} "
                f"max_subgraph_nodes={int(self.cfg.data.max_subgraph_nodes)} "
                f"max_image_nodes={int(self.cfg.data.max_image_nodes)} "
                f"text_batch_size={self.effective_text_batch_size} "
                f"image_forward_batch_size={self.effective_image_forward_batch_size} "
                f"num_workers={self.effective_num_workers} "
                f"prefetch_batches={self.effective_prefetch_batches} "
                f"prefetch_workers={self.prefetch_worker_count} "
                f"effective_max_steps={self.effective_max_steps} "
                f"early_stop_enabled={self.early_stop_enabled} "
                f"early_stop_patience={self.early_stop_patience}"
            ),
        )

        try:
            if self.prefetch_batches > 0:
                prefetch_executor = ThreadPoolExecutor(
                    max_workers=self.prefetch_worker_count,
                    thread_name_prefix="batch-prefetch",
                )
                next_step_to_schedule = self._fill_prefetch_queue(
                    pending,
                    prefetch_executor,
                    next_step_to_schedule,
                    steps_per_epoch,
                )

            while (
                self.global_step < self.effective_max_steps
                and self.epoch < self.cfg.optim.epochs
                and not self.stopped_early
            ):
                if prefetch_executor is not None:
                    _, reporter, future = pending.popleft()
                    batch_cpu = future.result()
                    next_step_to_schedule = self._fill_prefetch_queue(
                        pending,
                        prefetch_executor,
                        next_step_to_schedule,
                        steps_per_epoch,
                    )
                else:
                    pos_batch = self.batch_sampler.next_batch(self.global_step)
                    active_chunk = self.batch_sampler.active_chunk
                    reporter = self._make_step_reporter(self.global_step, active_chunk)
                    neg_batch = sample_negative_edges_undirected(
                        pos_batch,
                        num_nodes=self.ctx.graph.num_nodes,
                        num_neg=self.cfg.data.num_neg,
                        sorted_edge_hash=self.ctx.edge_hash_sorted,
                        rng=self.rng,
                    )
                    batch_cpu = self._prepare_batch_cpu(
                        pos_batch,
                        neg_batch,
                        active_chunk,
                        reporter,
                    )

                try:
                    batch = self._move_batch_to_device(
                        batch_cpu,
                        reporter,
                        transfer_supervision=True,
                    )

                    self.model.train()
                    self.ctx.optimizer.zero_grad(set_to_none=True)

                    forward_start = time.perf_counter()
                    with reporter.phase(
                        "forward_gnn",
                        sub_nodes=int(batch["sub_nodes"]),
                        real_image_count=int(batch["real_image_count"]),
                        supervision_batches=int(batch.get("supervision_batches", 1)),
                        supervision_batch_edges=int(batch.get("supervision_batch_edges", batch["pos_edges"])),
                    ):
                        with self._autocast_context():
                            logits = self._forward_logits(batch, reporter)
                            loss = self.ctx.criterion(logits, batch["labels"])
                    forward_sec = float(time.perf_counter() - forward_start)

                    if self.device.type == "cuda" and not self.first_forward_logged:
                        alloc_mb, reserved_mb = self._cuda_memory_stats_mb()
                        max_alloc_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024**2))
                        log_event(
                            "[DEVICE]",
                            (
                                f"first forward gpu memory allocated_mb={alloc_mb:.1f} "
                                f"reserved_mb={reserved_mb:.1f} max_allocated_mb={max_alloc_mb:.1f}"
                            ),
                        )
                        self.first_forward_logged = True
                        if alloc_mb < 1.0:
                            log_event("[ERROR]", "GPU not actually engaged after first forward.")
                            raise RuntimeError("GPU not actually engaged after first forward.")

                    backward_start = time.perf_counter()
                    with reporter.phase("backward"):
                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    backward_sec = float(time.perf_counter() - backward_start)

                    optimizer_start = time.perf_counter()
                    with reporter.phase("optimizer"):
                        if self.use_amp:
                            self.scaler.step(self.ctx.optimizer)
                            self.scaler.update()
                        else:
                            self.ctx.optimizer.step()
                        if self.ctx.scheduler is not None:
                            self.ctx.scheduler.step()
                    optimizer_sec = float(time.perf_counter() - optimizer_start)
                except RuntimeError as exc:
                    if self._is_cuda_oom(exc):
                        self._handle_cuda_oom(exc)
                    raise

                loss_val = float(loss.detach().float().cpu().item())
                self.loss_history.append(loss_val)
                self.global_step += 1
                self.epoch = self.global_step // steps_per_epoch

                loss_mean, loss_std, loss_var = self._rolling_stats(
                    self.loss_history, self.cfg.logging.loss_window
                )
                runtime_stats = self.model.consume_runtime_stats()
                image_stats = batch["image_stats"]
                lr = float(self.ctx.optimizer.param_groups[0]["lr"])
                gpu_mem_alloc_mb, gpu_mem_reserved_mb = self._cuda_memory_stats_mb()
                self.last_step_stats = {
                    "prepare_cpu_sec": float(batch["prepare_cpu_sec"]),
                    "transfer_sec": float(batch["transfer_sec"]),
                    "forward_sec": float(forward_sec),
                    "image_forward_sec": float(runtime_stats.get("image_forward_sec", 0.0)),
                }

                val_auc = ""
                val_aucpr = ""
                val_hit10 = ""
                val_art_auc = ""
                val_art_aucpr = ""
                val_art_hit10 = ""
                val_art_pos_edges = ""
                if self.global_step % self.cfg.eval.eval_every_steps == 0:
                    latest_fast_val = self._run_validation()
                    val_auc = latest_fast_val["auc"]
                    val_aucpr = latest_fast_val["aucpr"]
                    val_hit10 = latest_fast_val["hit10"]
                    val_art_auc = latest_fast_val["art_auc"]
                    val_art_aucpr = latest_fast_val["art_aucpr"]
                    val_art_hit10 = latest_fast_val["art_hit10"]
                    val_art_pos_edges = latest_fast_val["art_pos_edges"]
                    self.val_steps.append(self.global_step)
                    self.val_values.append(latest_fast_val["aucpr"])

                    metric_name = str(self.cfg.logging.best_metric)
                    current = self._select_validation_metric(latest_fast_val)
                    if current > (self.best_metric + self.early_stop_min_delta):
                        self.best_metric = current
                        self.early_stop_bad_evals = 0
                        self._save_checkpoint("best.pt")
                    else:
                        self.early_stop_bad_evals += 1
                        if self.early_stop_enabled and self.early_stop_bad_evals >= self.early_stop_patience:
                            self.stopped_early = True
                            self.early_stop_reason = (
                                f"{metric_name} did not improve for {self.early_stop_bad_evals} validation checks"
                            )
                            log_event(
                                "[EARLY_STOP]",
                                (
                                    f"triggered step={self.global_step} epoch={self.epoch} "
                                    f"metric={metric_name} current={current:.6f} "
                                    f"best={self.best_metric:.6f} "
                                    f"bad_evals={self.early_stop_bad_evals}"
                                ),
                            )

                row = {
                    "step": self.global_step,
                    "epoch": self.epoch,
                    "train_loss": loss_val,
                    "loss_mean": loss_mean,
                    "loss_std": loss_std,
                    "loss_var": loss_var,
                    "lr": lr,
                    "active_chunk": batch["active_chunk"],
                    "active_chunk_name": batch.get("active_chunk_name", "unknown"),
                    "seed_edges": batch.get("seed_edges", batch["pos_edges"]),
                    "img_real_hit": image_stats["img_real_hit"],
                    "img_no_image": image_stats["img_no_image"],
                    "img_missing_local": image_stats["img_missing_local"],
                    "img_real_ratio": (
                        image_stats["img_real_hit"]
                        / max(1.0, image_stats["img_real_hit"] + image_stats["img_no_image"])
                    ),
                    "text_cache_hit": runtime_stats["text_cache_hit"],
                    "text_cache_miss": runtime_stats["text_cache_miss"],
                    "text_cache_hit_rate": runtime_stats["text_cache_hit_rate"],
                    "text_cache_write_count": runtime_stats["text_cache_write_count"],
                    "prepare_cpu_sec": batch["prepare_cpu_sec"],
                    "sub_nodes": batch["sub_nodes"],
                    "sub_nodes_before_trim": batch.get("sub_nodes_before_trim", batch["sub_nodes"]),
                    "sub_nodes_after_trim": batch.get("sub_nodes_after_trim", batch["sub_nodes"]),
                    "seed_nodes": batch["seed_nodes"],
                    "real_image_nodes_before_trim": batch.get(
                        "real_image_nodes_before_trim",
                        batch["real_image_count"],
                    ),
                    "real_image_nodes_after_trim": batch.get(
                        "real_image_nodes_after_trim",
                        batch["real_image_count"],
                    ),
                    "trimmed_nodes": batch.get("trimmed_nodes", 0),
                    "trimmed_image_nodes": batch.get("trimmed_image_nodes", 0),
                    "seed_image_cap_exceeded": batch.get("seed_image_cap_exceeded", False),
                    "masked_supervision_edge_directions": batch.get("masked_supervision_edge_directions", 0),
                    "masked_supervision_unique_edges": batch.get("masked_supervision_unique_edges", 0),
                    "query_edges_in_message_graph": batch.get("query_edges_in_message_graph", 0),
                    "message_edges": batch.get("message_edges", batch.get("chunk_intra_edges", 0)),
                    "chunk_intra_edges": batch.get("chunk_intra_edges", 0),
                    "pos_edges": batch["pos_edges"],
                    "neg_edges": batch["neg_edges"],
                    "supervision_batches": batch.get("supervision_batches", 1),
                    "supervision_batch_edges": batch.get("supervision_batch_edges", batch["pos_edges"]),
                    "real_image_count": batch["real_image_count"],
                    "transfer_sec": batch["transfer_sec"],
                    "forward_sec": forward_sec,
                    "backward_sec": backward_sec,
                    "optimizer_sec": optimizer_sec,
                    "text_encode_sec": runtime_stats.get("text_encode_sec", 0.0),
                    "image_decode_sec": runtime_stats.get("image_decode_sec", batch["image_decode_sec"]),
                    "image_forward_sec": runtime_stats.get("image_forward_sec", 0.0),
                    "device_type": self.device.type,
                    "gpu_mem_alloc_mb": gpu_mem_alloc_mb,
                    "gpu_mem_reserved_mb": gpu_mem_reserved_mb,
                    "effective_batch_edges": self.effective_batch_edges,
                    "effective_max_steps": self.effective_max_steps,
                    "effective_text_batch_size": self.effective_text_batch_size,
                    "effective_image_forward_batch_size": self.effective_image_forward_batch_size,
                    "effective_num_workers": self.effective_num_workers,
                    "effective_prefetch_batches": self.effective_prefetch_batches,
                    "effective_prefetch_workers": self.prefetch_worker_count,
                    "stopped_early": self.stopped_early,
                    "early_stop_bad_evals": self.early_stop_bad_evals,
                    "val_auc": val_auc,
                    "val_aucpr": val_aucpr,
                    "val_hit10": val_hit10,
                    "val_art_auc": val_art_auc,
                    "val_art_aucpr": val_art_aucpr,
                    "val_art_hit10": val_art_hit10,
                    "val_art_pos_edges": val_art_pos_edges,
                    "val_art_any_auc": latest_fast_val.get("art_any_auc", 0.0),
                    "val_art_any_aucpr": latest_fast_val.get("art_any_aucpr", 0.0),
                    "val_art_any_hit10": latest_fast_val.get("art_any_hit10", 0.0),
                    "val_art_any_pos_edges": latest_fast_val.get("art_any_pos_edges", 0),
                    "time_sec": round(time.time() - start_time, 2),
                }
                self.metrics_logger.log(row)
                now_ts = time.perf_counter()
                if self._should_log_heartbeat(now_ts):
                    self._emit_heartbeat(
                        loss_val=loss_val,
                        lr=lr,
                        batch=batch,
                        runtime_stats=runtime_stats,
                        image_stats=image_stats,
                        forward_sec=forward_sec,
                        backward_sec=backward_sec,
                        optimizer_sec=optimizer_sec,
                    )

                if self.global_step % self.cfg.logging.save_every_steps == 0:
                    self._save_checkpoint("last.pt")
        finally:
            if prefetch_executor is not None:
                prefetch_executor.shutdown(wait=True, cancel_futures=False)

        log_event("[STAGE]", "train loop complete")
        self._save_checkpoint("last.pt")
        plot_loss_mean_std_var(
            self.loss_history,
            output_path=self.plots_dir / "loss_mean_std_var.png",
            window=self.cfg.logging.loss_window,
        )
        if self.val_steps:
            plot_val_curve(
                self.val_steps,
                self.val_values,
                output_path=self.plots_dir / "val_metric.png",
                ylabel=self.cfg.logging.best_metric,
                title="Validation metric curve",
            )

        best_ckpt = self.checkpoints_dir / "best.pt"
        if best_ckpt.exists():
            self._load_model_weights_only(best_ckpt)
        test_metrics = self._run_final_test()
        return {
            "best_metric": self.best_metric,
            "final_test_auc": test_metrics["auc"],
            "final_test_aucpr": test_metrics["aucpr"],
            "final_test_hit10": test_metrics["hit10"],
            "final_test_art_auc": test_metrics.get("art_auc", 0.0),
            "final_test_art_aucpr": test_metrics.get("art_aucpr", 0.0),
            "final_test_art_hit10": test_metrics.get("art_hit10", 0.0),
            "final_test_art_pos_edges": test_metrics.get("art_pos_edges", 0),
            "final_test_art_any_auc": test_metrics.get("art_any_auc", 0.0),
            "final_test_art_any_aucpr": test_metrics.get("art_any_aucpr", 0.0),
            "final_test_art_any_hit10": test_metrics.get("art_any_hit10", 0.0),
            "final_test_art_any_pos_edges": test_metrics.get("art_any_pos_edges", 0),
            "final_test_pos_edges_evaluated": test_metrics.get("pos_edges_evaluated", 0),
            "final_test_neg_edges_evaluated": test_metrics.get("neg_edges_evaluated", 0),
            "final_test_negatives_per_positive": test_metrics.get("negatives_per_positive", 0.0),
            "final_test_node_pool_size": test_metrics.get("node_pool_size", 0),
            "final_test_hit10_pos_edges_evaluated": test_metrics.get("hit10_pos_edges_evaluated", 0),
            "final_test_hit10_neg_edges_evaluated": test_metrics.get("hit10_neg_edges_evaluated", 0),
            "final_test_art_pos_edges_evaluated": test_metrics.get("art_pos_edges_evaluated", 0),
            "final_test_art_neg_edges_evaluated": test_metrics.get("art_neg_edges_evaluated", 0),
            "final_test_art_negatives_per_positive": test_metrics.get("art_negatives_per_positive", 0.0),
            "final_test_art_hit10_pos_edges_evaluated": test_metrics.get("art_hit10_pos_edges_evaluated", 0),
            "final_test_art_hit10_neg_edges_evaluated": test_metrics.get("art_hit10_neg_edges_evaluated", 0),
            "final_test_art_any_pos_edges_evaluated": test_metrics.get("art_any_pos_edges_evaluated", 0),
            "final_test_art_any_neg_edges_evaluated": test_metrics.get("art_any_neg_edges_evaluated", 0),
            "final_test_art_any_negatives_per_positive": test_metrics.get("art_any_negatives_per_positive", 0.0),
            "final_test_art_any_hit10_pos_edges_evaluated": test_metrics.get("art_any_hit10_pos_edges_evaluated", 0),
            "final_test_art_any_hit10_neg_edges_evaluated": test_metrics.get("art_any_hit10_neg_edges_evaluated", 0),
            "val_auc": latest_fast_val.get("auc", 0.0),
            "val_aucpr": latest_fast_val.get("aucpr", 0.0),
            "val_hit10": latest_fast_val.get("hit10", 0.0),
            "val_art_auc": latest_fast_val.get("art_auc", 0.0),
            "val_art_aucpr": latest_fast_val.get("art_aucpr", 0.0),
            "val_art_hit10": latest_fast_val.get("art_hit10", 0.0),
            "val_art_pos_edges": latest_fast_val.get("art_pos_edges", 0),
            "val_art_any_auc": latest_fast_val.get("art_any_auc", 0.0),
            "val_art_any_aucpr": latest_fast_val.get("art_any_aucpr", 0.0),
            "val_art_any_hit10": latest_fast_val.get("art_any_hit10", 0.0),
            "val_art_any_pos_edges": latest_fast_val.get("art_any_pos_edges", 0),
            "val_pos_edges_evaluated": latest_fast_val.get("pos_edges_evaluated", 0),
            "val_neg_edges_evaluated": latest_fast_val.get("neg_edges_evaluated", 0),
            "val_negatives_per_positive": latest_fast_val.get("negatives_per_positive", 0),
            "val_hit10_pos_edges_evaluated": latest_fast_val.get("hit10_pos_edges_evaluated", 0),
            "val_hit10_neg_edges_evaluated": latest_fast_val.get("hit10_neg_edges_evaluated", 0),
            "val_art_pos_edges_evaluated": latest_fast_val.get("art_pos_edges_evaluated", 0),
            "val_art_neg_edges_evaluated": latest_fast_val.get("art_neg_edges_evaluated", 0),
            "val_art_negatives_per_positive": latest_fast_val.get("art_negatives_per_positive", 0),
            "val_art_hit10_pos_edges_evaluated": latest_fast_val.get("art_hit10_pos_edges_evaluated", 0),
            "val_art_hit10_neg_edges_evaluated": latest_fast_val.get("art_hit10_neg_edges_evaluated", 0),
            "val_art_any_pos_edges_evaluated": latest_fast_val.get("art_any_pos_edges_evaluated", 0),
            "val_art_any_neg_edges_evaluated": latest_fast_val.get("art_any_neg_edges_evaluated", 0),
            "val_art_any_negatives_per_positive": latest_fast_val.get("art_any_negatives_per_positive", 0),
            "val_art_any_hit10_pos_edges_evaluated": latest_fast_val.get("art_any_hit10_pos_edges_evaluated", 0),
            "val_art_any_hit10_neg_edges_evaluated": latest_fast_val.get("art_any_hit10_neg_edges_evaluated", 0),
            "stopped_early": self.stopped_early,
            "early_stop_bad_evals": self.early_stop_bad_evals,
            "effective_max_steps": self.effective_max_steps,
        }
