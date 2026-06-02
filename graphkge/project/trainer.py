"""Training and evaluation for multimodal GraphKGE."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from project.data.typed_graph import hash_triples, triple_hash_exists
from project.experiment import ExperimentBundle
from project.graph_batch import build_prepared_graph_batch
from project.metrics import compute_auc_aucpr, relation_ranking_metrics
from project.utils.io import CSVLogger, ensure_dir
from project.utils.logging import log_event
from project.utils.seed import capture_rng_state, restore_rng_state


def sample_negative_triples(
    pos_triples: np.ndarray,
    num_entities: int,
    num_relations: int,
    sorted_true_hashes: np.ndarray,
    rng: np.random.Generator,
    num_neg: int = 1,
    candidate_entity_pool: np.ndarray | None = None,
) -> np.ndarray:
    pos = np.asarray(pos_triples, dtype=np.int64)
    neg = np.repeat(pos, int(num_neg), axis=0).copy()
    replace_head = rng.random(neg.shape[0]) < 0.5
    candidate_pool = None if candidate_entity_pool is None else np.asarray(candidate_entity_pool, dtype=np.int64)

    def _draw_entities(size: int) -> np.ndarray:
        if candidate_pool is None:
            return rng.integers(0, num_entities, size=size, dtype=np.int64)
        if candidate_pool.size == 0:
            raise RuntimeError("candidate_entity_pool is empty")
        pick_idx = rng.integers(0, candidate_pool.shape[0], size=size, dtype=np.int64)
        return candidate_pool[pick_idx]

    max_rounds = 50
    for _ in range(max_rounds):
        bad = neg[:, 0] == neg[:, 2]
        hashes = hash_triples(neg, num_entities=num_entities, num_relations=num_relations)
        bad |= triple_hash_exists(hashes, sorted_true_hashes)
        if not bad.any():
            break
        bad_head = bad & replace_head
        bad_tail = bad & (~replace_head)
        if bad_head.any():
            neg[bad_head, 0] = _draw_entities(int(bad_head.sum()))
        if bad_tail.any():
            neg[bad_tail, 2] = _draw_entities(int(bad_tail.sum()))
    return neg


class Trainer:
    def __init__(self, bundle: ExperimentBundle, run_dir: Path):
        self.bundle = bundle
        self.cfg = bundle.cfg
        self.model = bundle.model
        self.device = bundle.device
        self.run_dir = Path(run_dir)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        ensure_dir(self.checkpoints_dir)
        self.metrics_logger = CSVLogger(
            self.run_dir / "metrics.csv",
            fieldnames=[
                "step",
                "epoch",
                "train_loss",
                "lr",
                "val_mrr",
                "val_hits1",
                "val_hits3",
                "val_hits10",
                "val_top1_acc",
                "val_auc",
                "val_aucpr",
                "val_art_art_mrr",
                "val_art_art_hits1",
                "val_art_art_hits3",
                "val_art_art_hits10",
                "val_art_art_top1_acc",
                "val_art_art_auc",
                "val_art_art_aucpr",
                "val_art_any_mrr",
                "val_art_any_hits1",
                "val_art_any_hits3",
                "val_art_any_hits10",
                "val_art_any_top1_acc",
                "val_art_any_auc",
                "val_art_any_aucpr",
                "time_sec",
            ],
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("-inf")
        self.rng = np.random.default_rng(self.cfg.seed + 2026)
        self.train_idx = np.asarray(bundle.splits["train_idx"], dtype=np.int64)
        self.val_idx = np.asarray(bundle.splits["val_idx"], dtype=np.int64)
        self.test_idx = np.asarray(bundle.splits["test_idx"], dtype=np.int64)
        self.steps_per_epoch = max(1, math.ceil(self.train_idx.shape[0] / self.cfg.data.batch_size))
        self.max_steps = int(self.cfg.optim.max_steps) if int(self.cfg.optim.max_steps) > 0 else int(self.cfg.optim.epochs) * self.steps_per_epoch
        self.early_stopping_patience = int(self.cfg.optim.early_stopping_patience)
        self.early_stopping_min_delta = float(self.cfg.optim.early_stopping_min_delta)
        self.eval_progress_sec = int(self.cfg.runtime.eval_progress_sec)
        self.no_improve_evals = 0
        self.early_stopped = False
        self.art_entity_indices = np.where(self.bundle.catalog.node_is_art)[0].astype(np.int64)
        self.pin_memory = bool(self.cfg.runtime.pin_memory and self.device.type == "cuda")

    @staticmethod
    def _lookup_rows(unique_entities: np.ndarray, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(unique_entities, values.astype(np.int64))

    @staticmethod
    def _rows_from_mapping(global_to_local: Dict[int, int], values: np.ndarray) -> np.ndarray:
        return np.asarray([global_to_local[int(x)] for x in np.asarray(values, dtype=np.int64).tolist()], dtype=np.int64)

    @staticmethod
    def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
        return {f"{prefix}{key}": float(value) for key, value in metrics.items()}

    def _primary_metric_key(self) -> str:
        metric = str(self.cfg.logging.best_metric)
        if metric.startswith("val_"):
            return metric[4:]
        return metric

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_sec = max(0, int(round(float(seconds))))
        hours, rem = divmod(total_sec, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _log_train_progress(self, loss_val: float, lr: float, start_time: float, epoch_step: int) -> None:
        elapsed = max(0.0, time.time() - start_time)
        total_progress = float(self.global_step / max(1, self.max_steps))
        epoch_progress = float(epoch_step / max(1, self.steps_per_epoch))
        avg_step_sec = elapsed / max(1, self.global_step)
        eta_sec = avg_step_sec * max(0, self.max_steps - self.global_step)
        best_metric_text = (
            f"{self.cfg.logging.best_metric}={self.best_metric:.6f}"
            if self.best_metric != float("-inf")
            else f"{self.cfg.logging.best_metric}=NA"
        )
        log_event(
            "[TRAIN]",
            (
                f"step={self.global_step}/{self.max_steps} ({total_progress * 100:.2f}%) "
                f"epoch={self.epoch + 1}/{self.cfg.optim.epochs} "
                f"epoch_step={epoch_step}/{self.steps_per_epoch} ({epoch_progress * 100:.2f}%) "
                f"loss={loss_val:.6f} lr={lr:.6g} "
                f"elapsed={self._format_duration(elapsed)} eta={self._format_duration(eta_sec)} "
                f"{best_metric_text}"
            ),
        )

    def _log_eval_metrics(self, prefix: str, metrics: Dict[str, float]) -> None:
        log_event(
            prefix,
            (
                f"step={self.global_step} epoch={self.epoch} "
                f"mrr={float(metrics.get('mrr', 0.0)):.6f} "
                f"aucpr={float(metrics.get('aucpr', 0.0)):.6f} "
                f"art_art_mrr={float(metrics.get('art_art_mrr', 0.0)):.6f} "
                f"art_art_aucpr={float(metrics.get('art_art_aucpr', 0.0)):.6f} "
                f"art_any_mrr={float(metrics.get('art_any_mrr', 0.0)):.6f} "
                f"art_any_aucpr={float(metrics.get('art_any_aucpr', 0.0)):.6f}"
            ),
        )

    def _log_eval_task_start(self, phase: str, task: str, subset_mode: str, total_triples: int) -> float:
        total_batches = max(1, math.ceil(int(total_triples) / max(1, int(self.cfg.data.eval_batch_size))))
        start_time = time.time()
        log_event(
            "[STAGE]",
            (
                f"{phase} start step={self.global_step} epoch={self.epoch} "
                f"task={task} subset={subset_mode} triples={int(total_triples)} "
                f"batches={total_batches} eval_batch_size={self.cfg.data.eval_batch_size}"
            ),
        )
        return start_time

    def _maybe_log_eval_progress(
        self,
        phase: str,
        task: str,
        subset_mode: str,
        done_batches: int,
        total_batches: int,
        done_triples: int,
        total_triples: int,
        start_time: float,
        last_log_time: float,
    ) -> float:
        if self.eval_progress_sec <= 0:
            return last_log_time
        now = time.time()
        if (now - last_log_time) < self.eval_progress_sec:
            return last_log_time
        log_event(
            "[EVAL]",
            (
                f"phase={phase} step={self.global_step} epoch={self.epoch} "
                f"task={task} subset={subset_mode} "
                f"triples={int(done_triples)}/{int(total_triples)} "
                f"batches={int(done_batches)}/{int(total_batches)} "
                f"elapsed={self._format_duration(now - start_time)}"
            ),
        )
        return now

    def _filter_art_art_split(self, split_idx: np.ndarray) -> np.ndarray:
        triples = self.bundle.triple_store.triples[np.asarray(split_idx, dtype=np.int64)]
        mask = self.bundle.catalog.node_is_art[triples[:, 0]] & self.bundle.catalog.node_is_art[triples[:, 2]]
        return np.asarray(split_idx, dtype=np.int64)[mask]

    def _filter_art_any_split(self, split_idx: np.ndarray) -> np.ndarray:
        triples = self.bundle.triple_store.triples[np.asarray(split_idx, dtype=np.int64)]
        mask = self.bundle.catalog.node_is_art[triples[:, 0]] | self.bundle.catalog.node_is_art[triples[:, 2]]
        return np.asarray(split_idx, dtype=np.int64)[mask]

    def _subset_indices(self, split_idx: np.ndarray, subset_mode: str) -> np.ndarray:
        if subset_mode == "global":
            return np.asarray(split_idx, dtype=np.int64)
        if subset_mode == "art_art":
            return self._filter_art_art_split(split_idx)
        if subset_mode == "art_any":
            return self._filter_art_any_split(split_idx)
        raise ValueError(f"Unknown subset_mode: {subset_mode}")

    def _subset_candidate_entity_pool(self, subset_mode: str) -> np.ndarray | None:
        if subset_mode in {"art_art", "art_any"}:
            return self.art_entity_indices
        if subset_mode == "global":
            return None
        raise ValueError(f"Unknown subset_mode: {subset_mode}")

    @staticmethod
    def _empty_ranking_metrics() -> Dict[str, float]:
        return {"mrr": 0.0, "hits1": 0.0, "hits3": 0.0, "hits10": 0.0, "top1_acc": 0.0}

    @staticmethod
    def _empty_triple_metrics() -> Dict[str, float]:
        return {"auc": 0.5, "aucpr": 0.0}

    def _empty_eval_metrics(self) -> Dict[str, float]:
        return {
            **self._empty_ranking_metrics(),
            **self._empty_triple_metrics(),
            **self._prefix_metrics(self._empty_ranking_metrics(), "art_art_"),
            **self._prefix_metrics(self._empty_triple_metrics(), "art_art_"),
            **self._prefix_metrics(self._empty_ranking_metrics(), "art_any_"),
            **self._prefix_metrics(self._empty_triple_metrics(), "art_any_"),
        }

    @staticmethod
    def _empty_triple_array() -> np.ndarray:
        return np.empty((0, 3), dtype=np.int64)

    def _sample_eval_subset(self, index_array: np.ndarray, max_triples: int) -> np.ndarray:
        if max_triples <= 0 or index_array.shape[0] <= max_triples:
            return index_array
        return index_array[: int(max_triples)]

    def _prepare_eval_subset_triples(
        self,
        split_idx: np.ndarray,
        max_triples: int,
        subset_mode: str,
    ) -> np.ndarray:
        candidate_idx = self._subset_indices(split_idx, subset_mode)
        subset = self._sample_eval_subset(candidate_idx, max_triples)
        if subset.size == 0:
            return self._empty_triple_array()
        return np.asarray(self.bundle.triple_store.triples[subset], dtype=np.int64)

    def _prepare_eval_negative_triples(self, pos_triples: np.ndarray, subset_mode: str) -> np.ndarray:
        if pos_triples.size == 0:
            return self._empty_triple_array()
        batches: List[np.ndarray] = []
        step = max(1, int(self.cfg.data.eval_batch_size))
        for start in range(0, int(pos_triples.shape[0]), step):
            pos = pos_triples[start : start + step]
            batches.append(
                sample_negative_triples(
                    pos,
                    num_entities=self.bundle.catalog.num_entities(),
                    num_relations=len(self.bundle.triple_store.idx_to_relation),
                    sorted_true_hashes=self.bundle.triple_store.triple_hash_sorted,
                    rng=self.rng,
                    num_neg=self.cfg.data.num_neg,
                    candidate_entity_pool=self._subset_candidate_entity_pool(subset_mode),
                )
            )
        if len(batches) == 1:
            return batches[0]
        return np.concatenate(batches, axis=0)

    @staticmethod
    def _collect_eval_entity_ids(triple_groups: List[np.ndarray]) -> np.ndarray:
        parts: List[np.ndarray] = []
        for triples in triple_groups:
            if triples.size == 0:
                continue
            parts.append(np.asarray(triples[:, 0], dtype=np.int64))
            parts.append(np.asarray(triples[:, 2], dtype=np.int64))
        if not parts:
            return np.empty((0,), dtype=np.int64)
        return np.unique(np.concatenate(parts, axis=0))

    def _build_eval_entity_bank(self, entity_ids: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        row_lut = np.full(self.bundle.catalog.num_entities(), -1, dtype=np.int64)
        if entity_ids.size == 0:
            dtype = next(self.model.parameters()).dtype
            return row_lut, torch.empty((0, int(self.model.kge_scorer.output_dim)), dtype=dtype)

        row_lut[entity_ids] = np.arange(entity_ids.shape[0], dtype=np.int64)
        dtype = next(self.model.parameters()).dtype
        bank_kwargs = {}
        if self.pin_memory:
            bank_kwargs["pin_memory"] = True
        entity_bank_cpu = torch.empty(
            (int(entity_ids.shape[0]), int(self.model.kge_scorer.output_dim)),
            dtype=dtype,
            **bank_kwargs,
        )

        step = max(1, int(self.cfg.data.eval_batch_size))
        self.model.eval()
        with torch.no_grad():
            for start in range(0, int(entity_ids.shape[0]), step):
                stop = min(int(entity_ids.shape[0]), start + step)
                chunk = entity_ids[start:stop]
                chunk_emb = self.model.encode_entities(
                    entity_indices=chunk,
                    catalog=self.bundle.catalog,
                    text_store=self.bundle.text_store,
                    image_provider=self.bundle.image_provider,
                    image_transform=self.bundle.image_transform,
                    device=self.device,
                    pin_memory=self.pin_memory,
                )
                entity_bank_cpu[start:stop].copy_(chunk_emb.detach().to(device="cpu"))
        return row_lut, entity_bank_cpu

    @staticmethod
    def _lookup_entity_bank_rows(row_lut: np.ndarray, entity_ids: np.ndarray) -> np.ndarray:
        rows = row_lut[np.asarray(entity_ids, dtype=np.int64)]
        if np.any(rows < 0):
            raise RuntimeError("Missing entity rows in evaluation entity bank")
        return rows.astype(np.int64, copy=False)

    def _gather_eval_embeddings(
        self,
        entity_bank_cpu: torch.Tensor,
        row_lut: np.ndarray,
        entity_ids: np.ndarray,
    ) -> torch.Tensor:
        row_idx = torch.from_numpy(self._lookup_entity_bank_rows(row_lut, entity_ids)).to(dtype=torch.long)
        emb_cpu = entity_bank_cpu.index_select(0, row_idx)
        if self.device.type == "cuda":
            return emb_cpu.to(
                self.device,
                non_blocking=bool(self.cfg.runtime.non_blocking_transfer and self.cfg.runtime.pin_memory),
            )
        return emb_cpu

    def _evaluate_triple_scoring_prepared_nograph(
        self,
        pos_triples: np.ndarray,
        neg_triples: np.ndarray,
        subset_mode: str,
        phase: str,
        entity_bank_cpu: torch.Tensor,
        row_lut: np.ndarray,
    ) -> Dict[str, float]:
        if pos_triples.size == 0:
            return self._empty_triple_metrics()
        total_triples = int(pos_triples.shape[0])
        total_batches = max(1, math.ceil(total_triples / max(1, int(self.cfg.data.eval_batch_size))))
        start_time = self._log_eval_task_start(phase=phase, task="triple_scoring", subset_mode=subset_mode, total_triples=total_triples)
        last_log_time = start_time
        y_true: List[int] = []
        y_score: List[float] = []
        step = max(1, int(self.cfg.data.eval_batch_size))
        num_neg = max(1, int(self.cfg.data.num_neg))
        self.model.eval()
        with torch.no_grad():
            for start in range(0, total_triples, step):
                stop = min(total_triples, start + step)
                pos = pos_triples[start:stop]
                neg_start = start * num_neg
                neg_stop = neg_start + int(pos.shape[0]) * num_neg
                neg = neg_triples[neg_start:neg_stop]
                pos_h = self._gather_eval_embeddings(entity_bank_cpu, row_lut, pos[:, 0])
                pos_t = self._gather_eval_embeddings(entity_bank_cpu, row_lut, pos[:, 2])
                neg_h = self._gather_eval_embeddings(entity_bank_cpu, row_lut, neg[:, 0])
                neg_t = self._gather_eval_embeddings(entity_bank_cpu, row_lut, neg[:, 2])
                pos_r = torch.from_numpy(pos[:, 1]).to(self.device, dtype=torch.long)
                neg_r = torch.from_numpy(neg[:, 1]).to(self.device, dtype=torch.long)
                pos_scores = self.model.score_triples_from_embeddings(pos_h, pos_r, pos_t)
                neg_scores = self.model.score_triples_from_embeddings(neg_h, neg_r, neg_t)
                y_true.extend([1] * pos_scores.shape[0])
                y_true.extend([0] * neg_scores.shape[0])
                y_score.extend(pos_scores.detach().float().cpu().tolist())
                y_score.extend(neg_scores.detach().float().cpu().tolist())
                done_batches = min(total_batches, start // step + 1)
                done_triples = stop
                last_log_time = self._maybe_log_eval_progress(
                    phase=phase,
                    task="triple_scoring",
                    subset_mode=subset_mode,
                    done_batches=done_batches,
                    total_batches=total_batches,
                    done_triples=done_triples,
                    total_triples=total_triples,
                    start_time=start_time,
                    last_log_time=last_log_time,
                )
        return compute_auc_aucpr(np.asarray(y_true), np.asarray(y_score))

    def _evaluate_relation_ranking_prepared_nograph(
        self,
        triples: np.ndarray,
        subset_mode: str,
        phase: str,
        entity_bank_cpu: torch.Tensor,
        row_lut: np.ndarray,
    ) -> Dict[str, float]:
        if triples.size == 0:
            return self._empty_ranking_metrics()
        total_triples = int(triples.shape[0])
        total_batches = max(1, math.ceil(total_triples / max(1, int(self.cfg.data.eval_batch_size))))
        start_time = self._log_eval_task_start(phase=phase, task="relation_ranking", subset_mode=subset_mode, total_triples=total_triples)
        last_log_time = start_time
        relation_ids = np.arange(len(self.bundle.triple_store.idx_to_relation), dtype=np.int64)
        num_entities = self.bundle.catalog.num_entities()
        num_relations = len(self.bundle.triple_store.idx_to_relation)
        all_true_hashes = self.bundle.triple_store.triple_hash_sorted
        ranks: List[float] = []
        top1_hits: List[float] = []
        step = max(1, int(self.cfg.data.eval_batch_size))
        self.model.eval()
        with torch.no_grad():
            for start in range(0, total_triples, step):
                stop = min(total_triples, start + step)
                batch = triples[start:stop]
                head_emb = self._gather_eval_embeddings(entity_bank_cpu, row_lut, batch[:, 0])
                tail_emb = self._gather_eval_embeddings(entity_bank_cpu, row_lut, batch[:, 2])
                score_np = self.model.score_relations_from_embeddings(head_emb, tail_emb).detach().float().cpu().numpy()
                h = batch[:, 0].astype(np.uint64)[:, None]
                t = batch[:, 2].astype(np.uint64)[:, None]
                r = relation_ids.astype(np.uint64)[None, :]
                candidate_hashes = (h * np.uint64(num_relations) + r) * np.uint64(num_entities) + t
                known_true = triple_hash_exists(candidate_hashes, all_true_hashes)
                known_true[np.arange(batch.shape[0]), batch[:, 1]] = False
                score_np[known_true] = -np.inf
                targets = score_np[np.arange(batch.shape[0]), batch[:, 1]]
                batch_ranks = 1.0 + np.sum(score_np > targets[:, None], axis=1)
                ranks.extend(batch_ranks.astype(np.float64).tolist())
                top1_hits.extend((np.argmax(score_np, axis=1) == batch[:, 1]).astype(np.float64).tolist())
                done_batches = min(total_batches, start // step + 1)
                done_triples = stop
                last_log_time = self._maybe_log_eval_progress(
                    phase=phase,
                    task="relation_ranking",
                    subset_mode=subset_mode,
                    done_batches=done_batches,
                    total_batches=total_batches,
                    done_triples=done_triples,
                    total_triples=total_triples,
                    start_time=start_time,
                    last_log_time=last_log_time,
                )
        return relation_ranking_metrics(np.asarray(ranks), np.asarray(top1_hits))

    def _run_eval_phase_nograph(self, split_idx: np.ndarray, max_triples: int, phase: str) -> Dict[str, float]:
        subset_modes = ("global", "art_art", "art_any")
        pos_by_mode = {
            mode: self._prepare_eval_subset_triples(split_idx, max_triples, mode)
            for mode in subset_modes
        }
        log_event(
            "[STAGE]",
            (
                f"{phase} prep subsets "
                f"global={int(pos_by_mode['global'].shape[0])} "
                f"art_art={int(pos_by_mode['art_art'].shape[0])} "
                f"art_any={int(pos_by_mode['art_any'].shape[0])}"
            ),
        )
        if not any(int(pos_by_mode[mode].shape[0]) > 0 for mode in subset_modes):
            return self._empty_eval_metrics()

        neg_by_mode = {
            mode: self._prepare_eval_negative_triples(pos_by_mode[mode], mode)
            for mode in subset_modes
        }
        log_event(
            "[STAGE]",
            (
                f"{phase} prep negatives "
                f"global={int(neg_by_mode['global'].shape[0])} "
                f"art_art={int(neg_by_mode['art_art'].shape[0])} "
                f"art_any={int(neg_by_mode['art_any'].shape[0])}"
            ),
        )

        eval_entity_ids = self._collect_eval_entity_ids(
            [
                pos_by_mode["global"],
                neg_by_mode["global"],
                pos_by_mode["art_art"],
                neg_by_mode["art_art"],
                pos_by_mode["art_any"],
                neg_by_mode["art_any"],
            ]
        )
        log_event("[STAGE]", f"{phase} prep unique_entities={int(eval_entity_ids.shape[0])}")
        row_lut, entity_bank_cpu = self._build_eval_entity_bank(eval_entity_ids)
        log_event(
            "[STAGE]",
            (
                f"{phase} prep entity_bank_ready "
                f"unique_entities={int(eval_entity_ids.shape[0])} "
                f"emb_dim={int(entity_bank_cpu.shape[1])}"
            ),
        )

        return {
            **self._evaluate_relation_ranking_prepared_nograph(
                pos_by_mode["global"],
                subset_mode="global",
                phase=phase,
                entity_bank_cpu=entity_bank_cpu,
                row_lut=row_lut,
            ),
            **self._evaluate_triple_scoring_prepared_nograph(
                pos_by_mode["global"],
                neg_by_mode["global"],
                subset_mode="global",
                phase=phase,
                entity_bank_cpu=entity_bank_cpu,
                row_lut=row_lut,
            ),
            **self._prefix_metrics(
                self._evaluate_relation_ranking_prepared_nograph(
                    pos_by_mode["art_art"],
                    subset_mode="art_art",
                    phase=phase,
                    entity_bank_cpu=entity_bank_cpu,
                    row_lut=row_lut,
                ),
                "art_art_",
            ),
            **self._prefix_metrics(
                self._evaluate_triple_scoring_prepared_nograph(
                    pos_by_mode["art_art"],
                    neg_by_mode["art_art"],
                    subset_mode="art_art",
                    phase=phase,
                    entity_bank_cpu=entity_bank_cpu,
                    row_lut=row_lut,
                ),
                "art_art_",
            ),
            **self._prefix_metrics(
                self._evaluate_relation_ranking_prepared_nograph(
                    pos_by_mode["art_any"],
                    subset_mode="art_any",
                    phase=phase,
                    entity_bank_cpu=entity_bank_cpu,
                    row_lut=row_lut,
                ),
                "art_any_",
            ),
            **self._prefix_metrics(
                self._evaluate_triple_scoring_prepared_nograph(
                    pos_by_mode["art_any"],
                    neg_by_mode["art_any"],
                    subset_mode="art_any",
                    phase=phase,
                    entity_bank_cpu=entity_bank_cpu,
                    row_lut=row_lut,
                ),
                "art_any_",
            ),
        }

    def _encode_for_triples_graph(
        self,
        pos: np.ndarray,
        neg: np.ndarray,
    ) -> tuple[Dict[int, int], torch.Tensor]:
        seed_entities = np.concatenate([pos[:, 0], pos[:, 2], neg[:, 0], neg[:, 2]], axis=0)
        masked_pairs_global = np.vstack([pos[:, 0], pos[:, 2]]).astype(np.int64, copy=False)
        prepared = build_prepared_graph_batch(
            bundle=self.bundle,
            seed_entities=seed_entities,
            masked_entity_pairs_global=masked_pairs_global,
        )
        emb = self.model.encode_subgraph(
            entity_indices=prepared.node_indices,
            edge_index=prepared.edge_index,
            edge_type=prepared.edge_type,
            catalog=self.bundle.catalog,
            text_store=self.bundle.text_store,
            image_provider=self.bundle.image_provider,
            image_transform=self.bundle.image_transform,
            device=self.device,
            pin_memory=self.pin_memory,
            image_batch=prepared.image_batch,
        )
        return prepared.global_to_local, emb

    def _score_triple_batch_graph(self, pos: np.ndarray, neg: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        global_to_local, entity_emb = self._encode_for_triples_graph(pos, neg)
        pos_h = torch.from_numpy(self._rows_from_mapping(global_to_local, pos[:, 0])).to(self.device, dtype=torch.long)
        pos_t = torch.from_numpy(self._rows_from_mapping(global_to_local, pos[:, 2])).to(self.device, dtype=torch.long)
        neg_h = torch.from_numpy(self._rows_from_mapping(global_to_local, neg[:, 0])).to(self.device, dtype=torch.long)
        neg_t = torch.from_numpy(self._rows_from_mapping(global_to_local, neg[:, 2])).to(self.device, dtype=torch.long)
        pos_r = torch.from_numpy(pos[:, 1]).to(self.device, dtype=torch.long)
        neg_r = torch.from_numpy(neg[:, 1]).to(self.device, dtype=torch.long)
        pos_scores = self.model.score_triples_from_embeddings(entity_emb[pos_h], pos_r, entity_emb[pos_t])
        neg_scores = self.model.score_triples_from_embeddings(entity_emb[neg_h], neg_r, entity_emb[neg_t])
        return pos_scores, neg_scores

    def _evaluate_triple_scoring_prepared_graph(
        self,
        pos_triples: np.ndarray,
        neg_triples: np.ndarray,
        subset_mode: str,
        phase: str,
    ) -> Dict[str, float]:
        if pos_triples.size == 0:
            return self._empty_triple_metrics()
        total_triples = int(pos_triples.shape[0])
        total_batches = max(1, math.ceil(total_triples / max(1, int(self.cfg.data.eval_batch_size))))
        start_time = self._log_eval_task_start(phase=phase, task="triple_scoring", subset_mode=subset_mode, total_triples=total_triples)
        last_log_time = start_time
        y_true: List[int] = []
        y_score: List[float] = []
        step = max(1, int(self.cfg.data.eval_batch_size))
        num_neg = max(1, int(self.cfg.data.num_neg))
        self.model.eval()
        with torch.no_grad():
            for start in range(0, total_triples, step):
                stop = min(total_triples, start + step)
                pos = pos_triples[start:stop]
                neg_start = start * num_neg
                neg_stop = neg_start + int(pos.shape[0]) * num_neg
                neg = neg_triples[neg_start:neg_stop]
                pos_scores, neg_scores = self._score_triple_batch_graph(pos, neg)
                y_true.extend([1] * pos_scores.shape[0])
                y_true.extend([0] * neg_scores.shape[0])
                y_score.extend(pos_scores.detach().float().cpu().tolist())
                y_score.extend(neg_scores.detach().float().cpu().tolist())
                done_batches = min(total_batches, start // step + 1)
                done_triples = stop
                last_log_time = self._maybe_log_eval_progress(
                    phase=phase,
                    task="triple_scoring",
                    subset_mode=subset_mode,
                    done_batches=done_batches,
                    total_batches=total_batches,
                    done_triples=done_triples,
                    total_triples=total_triples,
                    start_time=start_time,
                    last_log_time=last_log_time,
                )
        return compute_auc_aucpr(np.asarray(y_true), np.asarray(y_score))

    def _evaluate_relation_ranking_prepared_graph(
        self,
        triples: np.ndarray,
        subset_mode: str,
        phase: str,
    ) -> Dict[str, float]:
        if triples.size == 0:
            return self._empty_ranking_metrics()
        total_triples = int(triples.shape[0])
        total_batches = max(1, math.ceil(total_triples / max(1, int(self.cfg.data.eval_batch_size))))
        start_time = self._log_eval_task_start(phase=phase, task="relation_ranking", subset_mode=subset_mode, total_triples=total_triples)
        last_log_time = start_time
        relation_ids = np.arange(len(self.bundle.triple_store.idx_to_relation), dtype=np.int64)
        num_entities = self.bundle.catalog.num_entities()
        num_relations = len(self.bundle.triple_store.idx_to_relation)
        all_true_hashes = self.bundle.triple_store.triple_hash_sorted
        ranks: List[float] = []
        top1_hits: List[float] = []
        step = max(1, int(self.cfg.data.eval_batch_size))
        self.model.eval()
        with torch.no_grad():
            for start in range(0, total_triples, step):
                stop = min(total_triples, start + step)
                batch = triples[start:stop]
                seed_entities = np.concatenate([batch[:, 0], batch[:, 2]], axis=0)
                masked_pairs_global = np.vstack([batch[:, 0], batch[:, 2]]).astype(np.int64, copy=False)
                prepared = build_prepared_graph_batch(
                    bundle=self.bundle,
                    seed_entities=seed_entities,
                    masked_entity_pairs_global=masked_pairs_global,
                )
                entity_emb = self.model.encode_subgraph(
                    entity_indices=prepared.node_indices,
                    edge_index=prepared.edge_index,
                    edge_type=prepared.edge_type,
                    catalog=self.bundle.catalog,
                    text_store=self.bundle.text_store,
                    image_provider=self.bundle.image_provider,
                    image_transform=self.bundle.image_transform,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    image_batch=prepared.image_batch,
                )
                head_rows = torch.from_numpy(self._rows_from_mapping(prepared.global_to_local, batch[:, 0])).to(self.device, dtype=torch.long)
                tail_rows = torch.from_numpy(self._rows_from_mapping(prepared.global_to_local, batch[:, 2])).to(self.device, dtype=torch.long)
                score_np = self.model.score_relations_from_embeddings(entity_emb[head_rows], entity_emb[tail_rows]).detach().float().cpu().numpy()
                h = batch[:, 0].astype(np.uint64)[:, None]
                t = batch[:, 2].astype(np.uint64)[:, None]
                r = relation_ids.astype(np.uint64)[None, :]
                candidate_hashes = (h * np.uint64(num_relations) + r) * np.uint64(num_entities) + t
                known_true = triple_hash_exists(candidate_hashes, all_true_hashes)
                known_true[np.arange(batch.shape[0]), batch[:, 1]] = False
                score_np[known_true] = -np.inf
                targets = score_np[np.arange(batch.shape[0]), batch[:, 1]]
                batch_ranks = 1.0 + np.sum(score_np > targets[:, None], axis=1)
                ranks.extend(batch_ranks.astype(np.float64).tolist())
                top1_hits.extend((np.argmax(score_np, axis=1) == batch[:, 1]).astype(np.float64).tolist())
                done_batches = min(total_batches, start // step + 1)
                done_triples = stop
                last_log_time = self._maybe_log_eval_progress(
                    phase=phase,
                    task="relation_ranking",
                    subset_mode=subset_mode,
                    done_batches=done_batches,
                    total_batches=total_batches,
                    done_triples=done_triples,
                    total_triples=total_triples,
                    start_time=start_time,
                    last_log_time=last_log_time,
                )
        return relation_ranking_metrics(np.asarray(ranks), np.asarray(top1_hits))

    def _run_eval_phase_graph(self, split_idx: np.ndarray, max_triples: int, phase: str) -> Dict[str, float]:
        subset_modes = ("global", "art_art", "art_any")
        pos_by_mode = {
            mode: self._prepare_eval_subset_triples(split_idx, max_triples, mode)
            for mode in subset_modes
        }
        log_event(
            "[STAGE]",
            (
                f"{phase} prep subsets "
                f"global={int(pos_by_mode['global'].shape[0])} "
                f"art_art={int(pos_by_mode['art_art'].shape[0])} "
                f"art_any={int(pos_by_mode['art_any'].shape[0])}"
            ),
        )
        if not any(int(pos_by_mode[mode].shape[0]) > 0 for mode in subset_modes):
            return self._empty_eval_metrics()

        neg_by_mode = {
            mode: self._prepare_eval_negative_triples(pos_by_mode[mode], mode)
            for mode in subset_modes
        }
        log_event(
            "[STAGE]",
            (
                f"{phase} prep negatives "
                f"global={int(neg_by_mode['global'].shape[0])} "
                f"art_art={int(neg_by_mode['art_art'].shape[0])} "
                f"art_any={int(neg_by_mode['art_any'].shape[0])}"
            ),
        )

        return {
            **self._evaluate_relation_ranking_prepared_graph(
                pos_by_mode["global"],
                subset_mode="global",
                phase=phase,
            ),
            **self._evaluate_triple_scoring_prepared_graph(
                pos_by_mode["global"],
                neg_by_mode["global"],
                subset_mode="global",
                phase=phase,
            ),
            **self._prefix_metrics(
                self._evaluate_relation_ranking_prepared_graph(
                    pos_by_mode["art_art"],
                    subset_mode="art_art",
                    phase=phase,
                ),
                "art_art_",
            ),
            **self._prefix_metrics(
                self._evaluate_triple_scoring_prepared_graph(
                    pos_by_mode["art_art"],
                    neg_by_mode["art_art"],
                    subset_mode="art_art",
                    phase=phase,
                ),
                "art_art_",
            ),
            **self._prefix_metrics(
                self._evaluate_relation_ranking_prepared_graph(
                    pos_by_mode["art_any"],
                    subset_mode="art_any",
                    phase=phase,
                ),
                "art_any_",
            ),
            **self._prefix_metrics(
                self._evaluate_triple_scoring_prepared_graph(
                    pos_by_mode["art_any"],
                    neg_by_mode["art_any"],
                    subset_mode="art_any",
                    phase=phase,
                ),
                "art_any_",
            ),
        }

    def _batch_entity_embeddings_nograph(self, pos: np.ndarray, neg: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        unique_entities = np.unique(np.concatenate([pos[:, 0], pos[:, 2], neg[:, 0], neg[:, 2]], axis=0))
        emb = self.model.encode_entities(
            entity_indices=unique_entities,
            catalog=self.bundle.catalog,
            text_store=self.bundle.text_store,
            image_provider=self.bundle.image_provider,
            image_transform=self.bundle.image_transform,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        return unique_entities, emb

    def _score_triple_batch_nograph(self, pos: np.ndarray, neg: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        unique_entities, entity_emb = self._batch_entity_embeddings_nograph(pos, neg)
        pos_h = torch.from_numpy(self._lookup_rows(unique_entities, pos[:, 0])).to(self.device, dtype=torch.long)
        pos_t = torch.from_numpy(self._lookup_rows(unique_entities, pos[:, 2])).to(self.device, dtype=torch.long)
        neg_h = torch.from_numpy(self._lookup_rows(unique_entities, neg[:, 0])).to(self.device, dtype=torch.long)
        neg_t = torch.from_numpy(self._lookup_rows(unique_entities, neg[:, 2])).to(self.device, dtype=torch.long)
        pos_r = torch.from_numpy(pos[:, 1]).to(self.device, dtype=torch.long)
        neg_r = torch.from_numpy(neg[:, 1]).to(self.device, dtype=torch.long)
        pos_scores = self.model.score_triples_from_embeddings(entity_emb[pos_h], pos_r, entity_emb[pos_t])
        neg_scores = self.model.score_triples_from_embeddings(entity_emb[neg_h], neg_r, entity_emb[neg_t])
        return pos_scores, neg_scores

    def _score_triple_batch(self, pos: np.ndarray, neg: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if bool(self.cfg.model.use_graph):
            return self._score_triple_batch_graph(pos, neg)
        return self._score_triple_batch_nograph(pos, neg)

    def _run_eval_phase(self, split_idx: np.ndarray, max_triples: int, phase: str) -> Dict[str, float]:
        if bool(self.cfg.model.use_graph):
            return self._run_eval_phase_graph(split_idx, max_triples, phase)
        return self._run_eval_phase_nograph(split_idx, max_triples, phase)

    def _run_validation(self) -> Dict[str, float]:
        return self._run_eval_phase(self.val_idx, self.cfg.eval.val_max_triples, phase="val")

    def _run_test(self) -> Dict[str, float]:
        return self._run_eval_phase(self.test_idx, self.cfg.eval.test_max_triples, phase="test")

    def _save_checkpoint(self, name: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.bundle.optimizer.state_dict() if self.bundle.optimizer is not None else None,
                "scheduler": self.bundle.scheduler.state_dict() if self.bundle.scheduler is not None else None,
                "global_step": int(self.global_step),
                "epoch": int(self.epoch),
                "best_metric": float(self.best_metric),
                "no_improve_evals": int(self.no_improve_evals),
                "early_stopped": bool(self.early_stopped),
                "rng_state": self.rng.bit_generator.state,
                "global_rng_state": capture_rng_state(),
                "config": self.bundle.cfg.to_dict(),
                "config_base_dir": str(self.bundle.config_base_dir),
                "sample_chunks": int(self.bundle.sample_chunks),
            },
            self.checkpoints_dir / name,
        )

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if self.bundle.optimizer is not None and ckpt.get("optimizer") is not None:
            self.bundle.optimizer.load_state_dict(ckpt["optimizer"])
        if self.bundle.scheduler is not None and ckpt.get("scheduler") is not None:
            self.bundle.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.epoch = int(ckpt.get("epoch", 0))
        self.best_metric = float(ckpt.get("best_metric", float("-inf")))
        self.no_improve_evals = int(ckpt.get("no_improve_evals", 0))
        self.early_stopped = bool(ckpt.get("early_stopped", False))
        if ckpt.get("rng_state") is not None:
            self.rng.bit_generator.state = ckpt["rng_state"]
        if ckpt.get("global_rng_state") is not None:
            restore_rng_state(ckpt["global_rng_state"])

    def train(self) -> Dict[str, float]:
        if self.bundle.optimizer is None:
            raise RuntimeError("Trainer requires optimizer")
        start_time = time.time()
        final_val = self._empty_eval_metrics()
        while self.global_step < self.max_steps and self.epoch < self.cfg.optim.epochs:
            log_event(
                "[STAGE]",
                (
                    f"epoch start epoch={self.epoch + 1}/{self.cfg.optim.epochs} "
                    f"steps_per_epoch={self.steps_per_epoch} global_step={self.global_step}/{self.max_steps}"
                ),
            )
            train_order = self.train_idx.copy()
            self.rng.shuffle(train_order)
            for start in range(0, train_order.shape[0], self.cfg.data.batch_size):
                if self.global_step >= self.max_steps or self.early_stopped:
                    break
                epoch_step = start // self.cfg.data.batch_size + 1
                batch_rows = train_order[start : start + self.cfg.data.batch_size]
                pos = self.bundle.triple_store.triples[batch_rows]
                neg = sample_negative_triples(
                    pos,
                    num_entities=self.bundle.catalog.num_entities(),
                    num_relations=len(self.bundle.triple_store.idx_to_relation),
                    sorted_true_hashes=self.bundle.triple_store.triple_hash_sorted,
                    rng=self.rng,
                    num_neg=self.cfg.data.num_neg,
                )

                self.model.train()
                self.bundle.optimizer.zero_grad(set_to_none=True)
                pos_scores, neg_scores = self._score_triple_batch(pos, neg)
                scores = torch.cat([pos_scores, neg_scores], dim=0)
                labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
                loss = self.criterion(scores, labels)
                loss.backward()
                self.bundle.optimizer.step()
                if self.bundle.scheduler is not None:
                    self.bundle.scheduler.step()

                self.global_step += 1
                loss_val = float(loss.detach().cpu().item())
                lr = float(self.bundle.optimizer.param_groups[0]["lr"])
                val_metrics = {}
                if self.global_step % self.cfg.eval.eval_every_steps == 0:
                    val_metrics = self._run_validation()
                    final_val = val_metrics
                    self._log_eval_metrics("[VAL]", val_metrics)
                    current = float(val_metrics[self._primary_metric_key()])
                    if current > (self.best_metric + self.early_stopping_min_delta):
                        self.best_metric = current
                        self.no_improve_evals = 0
                        self._save_checkpoint("best.pt")
                    else:
                        self.no_improve_evals += 1
                        if self.early_stopping_patience > 0 and self.no_improve_evals >= self.early_stopping_patience:
                            self.early_stopped = True
                            log_event(
                                "[STAGE]",
                                (
                                    f"early stopping triggered at step={self.global_step} "
                                    f"epoch={self.epoch} no_improve_evals={self.no_improve_evals}"
                                ),
                            )
                if self.global_step % self.cfg.logging.save_every_steps == 0:
                    self._save_checkpoint("last.pt")

                self.metrics_logger.log(
                    {
                        "step": self.global_step,
                        "epoch": self.epoch,
                        "train_loss": loss_val,
                        "lr": lr,
                        "val_mrr": val_metrics.get("mrr", ""),
                        "val_hits1": val_metrics.get("hits1", ""),
                        "val_hits3": val_metrics.get("hits3", ""),
                        "val_hits10": val_metrics.get("hits10", ""),
                        "val_top1_acc": val_metrics.get("top1_acc", ""),
                        "val_auc": val_metrics.get("auc", ""),
                        "val_aucpr": val_metrics.get("aucpr", ""),
                        "val_art_art_mrr": val_metrics.get("art_art_mrr", ""),
                        "val_art_art_hits1": val_metrics.get("art_art_hits1", ""),
                        "val_art_art_hits3": val_metrics.get("art_art_hits3", ""),
                        "val_art_art_hits10": val_metrics.get("art_art_hits10", ""),
                        "val_art_art_top1_acc": val_metrics.get("art_art_top1_acc", ""),
                        "val_art_art_auc": val_metrics.get("art_art_auc", ""),
                        "val_art_art_aucpr": val_metrics.get("art_art_aucpr", ""),
                        "val_art_any_mrr": val_metrics.get("art_any_mrr", ""),
                        "val_art_any_hits1": val_metrics.get("art_any_hits1", ""),
                        "val_art_any_hits3": val_metrics.get("art_any_hits3", ""),
                        "val_art_any_hits10": val_metrics.get("art_any_hits10", ""),
                        "val_art_any_top1_acc": val_metrics.get("art_any_top1_acc", ""),
                        "val_art_any_auc": val_metrics.get("art_any_auc", ""),
                        "val_art_any_aucpr": val_metrics.get("art_any_aucpr", ""),
                        "time_sec": round(time.time() - start_time, 2),
                    }
                )
                if self.global_step <= 5 or self.global_step % 10 == 0:
                    self._log_train_progress(loss_val=loss_val, lr=lr, start_time=start_time, epoch_step=epoch_step)
                if self.early_stopped:
                    break

            self.epoch += 1
            log_event("[STAGE]", f"epoch completed epoch={self.epoch}/{self.cfg.optim.epochs}")
            if self.early_stopped:
                break

        if self.best_metric == float("-inf"):
            final_val = self._run_validation()
            self._log_eval_metrics("[VAL]", final_val)
            self.best_metric = float(final_val[self._primary_metric_key()])
            self._save_checkpoint("best.pt")
        self._save_checkpoint("last.pt")
        test_metrics = self._run_test()
        self._log_eval_metrics("[TEST]", test_metrics)
        summary = {
            "best_metric": self.best_metric,
            "early_stopped": self.early_stopped,
            "stopped_epoch": self.epoch,
            "stopped_step": self.global_step,
        }
        summary.update(self._prefix_metrics(final_val, "val_"))
        summary.update(self._prefix_metrics(test_metrics, "test_"))
        return summary
