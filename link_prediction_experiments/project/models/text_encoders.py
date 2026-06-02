"""Text encoders and chunk-aligned embedding cache."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from project.data.sharded_graph import ShardedGraphDataset
from project.data.text_store import TextStore
from project.models.model_store import load_hf_resource, slugify_model_id
from project.utils.lru import LRUCache
from project.utils.logging import ProgressTicker, StepProgressReporter, log_event


class ChunkAlignedTextCache:
    """Chunk-aligned memmap text embedding cache (no database, no small files)."""

    def __init__(
        self,
        cache_dir: Path,
        graph: ShardedGraphDataset,
        emb_dim: int,
        dtype: str,
        model_name: str,
        max_length: int,
        lru_chunks: int = 8,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.graph = graph
        self.emb_dim = int(emb_dim)
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.model_name = model_name
        self.max_length = int(max_length)
        self.cache = LRUCache[int, Tuple[np.memmap, np.memmap]](lru_chunks)
        self.meta_path = self.cache_dir / "meta.json"
        self._init_meta()

        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _init_meta(self) -> None:
        payload = {
            "format": "chunk_text_emb_v2",
            "emb_dim": self.emb_dim,
            "dtype": "float16" if self.dtype == np.float16 else "float32",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "chunk_count": self.graph.chunk_count(),
        }
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                old = json.load(f)
            for k in ("emb_dim", "dtype", "model_name", "max_length", "chunk_count"):
                if old.get(k) != payload[k]:
                    raise RuntimeError(
                        f"Text cache meta mismatch for {k}: {old.get(k)} != {payload[k]}"
                    )
            return
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _chunk_dir(self, chunk_id: int) -> Path:
        return self.cache_dir / f"chunk_{self.graph.chunk_name(chunk_id)}"

    def _ensure_chunk_files(self, chunk_id: int) -> None:
        cdir = self._chunk_dir(chunk_id)
        cdir.mkdir(parents=True, exist_ok=True)
        emb_path = cdir / ("emb.f16.npy" if self.dtype == np.float16 else "emb.f32.npy")
        valid_path = cdir / "valid.u8.npy"
        n_local = self.graph.chunk_size(chunk_id)

        if not emb_path.exists():
            mm = np.lib.format.open_memmap(
                emb_path, mode="w+", dtype=self.dtype, shape=(n_local, self.emb_dim)
            )
            mm[:] = 0
            mm.flush()
            del mm

        if not valid_path.exists():
            mm = np.lib.format.open_memmap(
                valid_path, mode="w+", dtype=np.uint8, shape=(n_local,)
            )
            mm[:] = 0
            mm.flush()
            del mm

    def _get_chunk_memmap(self, chunk_id: int) -> Tuple[np.memmap, np.memmap]:
        cached = self.cache.get(chunk_id)
        if cached is not None:
            return cached
        self._ensure_chunk_files(chunk_id)
        cdir = self._chunk_dir(chunk_id)
        emb_path = cdir / ("emb.f16.npy" if self.dtype == np.float16 else "emb.f32.npy")
        valid_path = cdir / "valid.u8.npy"
        emb = np.load(emb_path, mmap_mode="r+")
        valid = np.load(valid_path, mmap_mode="r+")
        self.cache.put(chunk_id, (emb, valid))
        return emb, valid

    def get_many(
        self,
        chunk_ids: np.ndarray,
        chunk_pos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = int(chunk_ids.shape[0])
        out = np.zeros((n, self.emb_dim), dtype=np.float32)
        miss = np.zeros(n, dtype=bool)

        uniq = np.unique(chunk_ids)
        for cid in uniq:
            idxs = np.where(chunk_ids == cid)[0]
            emb_mm, valid_mm = self._get_chunk_memmap(int(cid))
            pos = chunk_pos[idxs].astype(np.int64)
            valid = valid_mm[pos] > 0
            if valid.any():
                out[idxs[valid]] = np.asarray(emb_mm[pos[valid]], dtype=np.float32)
            if (~valid).any():
                miss[idxs[~valid]] = True

        self.hits += int((~miss).sum())
        self.misses += int(miss.sum())
        return out, miss

    def set_many(self, chunk_ids: np.ndarray, chunk_pos: np.ndarray, values: np.ndarray) -> None:
        uniq = np.unique(chunk_ids)
        for cid in uniq:
            idxs = np.where(chunk_ids == cid)[0]
            emb_mm, valid_mm = self._get_chunk_memmap(int(cid))
            pos = chunk_pos[idxs].astype(np.int64)
            emb_mm[pos] = values[idxs].astype(self.dtype, copy=False)
            valid_mm[pos] = 1
            emb_mm.flush()
            valid_mm.flush()
            self.writes += int(idxs.shape[0])

    def consume_stats(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = float(self.hits / total) if total > 0 else 0.0
        out = {
            "text_cache_hit": float(self.hits),
            "text_cache_miss": float(self.misses),
            "text_cache_hit_rate": hit_rate,
            "text_cache_write_count": float(self.writes),
        }
        self.hits = 0
        self.misses = 0
        self.writes = 0
        return out


class HFFrozenTextEncoder(nn.Module):
    """Frozen Hugging Face text encoder with chunk-aligned disk cache."""

    def __init__(
        self,
        graph: ShardedGraphDataset,
        cache_dir: Path,
        local_model_root: Path,
        model_name: str,
        max_length: int = 256,
        batch_size: int = 32,
        device: str = "cpu",
        cache_dtype: str = "float16",
        cache_lru_chunks: int = 8,
        progress_sec: float = 30.0,
    ):
        super().__init__()
        self.graph = graph
        self.model_name = str(model_name)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.encode_time_sec = 0.0
        self.progress_sec = float(progress_sec)

        model_slug = slugify_model_id(self.model_name)
        local_model_dir = Path(local_model_root) / "text" / model_slug
        self.tokenizer = load_hf_resource(AutoTokenizer, self.model_name, local_model_dir)
        self.backbone = load_hf_resource(AutoModel, self.model_name, local_model_dir)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.backbone.to(self.device)

        hidden = int(getattr(self.backbone.config, "hidden_size"))
        self.output_dim = hidden
        self.cache = ChunkAlignedTextCache(
            cache_dir=cache_dir,
            graph=graph,
            emb_dim=hidden,
            dtype=cache_dtype,
            model_name=self.model_name,
            max_length=max_length,
            lru_chunks=cache_lru_chunks,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _set_device(self, device: str | torch.device) -> None:
        target = torch.device(device)
        if target == self.device:
            return
        self.device = target
        self.backbone.to(self.device)

    def _pool_outputs(self, outputs) -> torch.Tensor:
        model_type = str(getattr(self.backbone.config, "model_type", "")).lower()
        if model_type == "roberta":
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                raise RuntimeError("RoBERTa output is missing last_hidden_state")
            return last_hidden[:, 0, :]
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise RuntimeError("Unsupported HF text model output: missing pooler_output and last_hidden_state")
        return last_hidden[:, 0, :]

    @torch.no_grad()
    def _encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        progress_reporter: Optional[StepProgressReporter] = None,
    ) -> np.ndarray:
        outputs: List[np.ndarray] = []
        use_batch_size = max(1, int(batch_size or self.batch_size))
        ticker = ProgressTicker("text encode", every_sec=self.progress_sec)
        for i in range(0, len(texts), use_batch_size):
            batch = texts[i : i + use_batch_size]
            start_time = time.perf_counter()
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            out = self.backbone(**tokenized)
            cls = self._pool_outputs(out).detach().cpu().numpy().astype(np.float32)
            outputs.append(cls)
            self.encode_time_sec += float(time.perf_counter() - start_time)
            if progress_reporter is not None and len(texts) >= progress_reporter.min_items:
                progress_reporter.progress(
                    "forward_text",
                    encoded=min(i + len(batch), len(texts)),
                    total=len(texts),
                    batch_size=use_batch_size,
                )
            ticker.maybe(
                f"encoded={min(i + len(batch), len(texts))}/{len(texts)} batch_size={use_batch_size}"
            )
        if not outputs:
            return np.empty((0, self.output_dim), dtype=np.float32)
        ticker.force(f"encoded={len(texts)}/{len(texts)} batch_size={use_batch_size}")
        return np.concatenate(outputs, axis=0)

    def get_node_embeddings(
        self,
        node_indices: np.ndarray,
        node_wids: List[str],
        node_chunk_ids: np.ndarray,
        node_chunk_pos: np.ndarray,
        text_store: TextStore,
        output_device: torch.device,
        progress_reporter: Optional[StepProgressReporter] = None,
    ) -> torch.Tensor:
        start_encode_sec = float(self.encode_time_sec)
        cached, miss_mask = self.cache.get_many(node_chunk_ids, node_chunk_pos)
        hits = int((~miss_mask).sum())
        misses = int(miss_mask.sum())
        if progress_reporter is not None:
            progress_reporter.emit(
                "forward_text",
                "start",
                node_count=int(node_indices.shape[0]),
                text_cache_hit=hits,
                text_cache_miss=misses,
            )
        if miss_mask.any():
            miss_idx = np.where(miss_mask)[0]
            miss_texts = text_store.get_texts([node_wids[i] for i in miss_idx])
            miss_emb = self._encode_texts(
                miss_texts,
                progress_reporter=progress_reporter,
            )
            cached[miss_idx] = miss_emb
            self.cache.set_many(
                node_chunk_ids[miss_idx],
                node_chunk_pos[miss_idx],
                miss_emb,
            )
        if progress_reporter is not None:
            progress_reporter.emit(
                "forward_text",
                "done",
                node_count=int(node_indices.shape[0]),
                text_cache_hit=hits,
                text_cache_miss=misses,
                text_encode_sec=f"{float(self.encode_time_sec - start_encode_sec):.2f}",
            )
        return torch.from_numpy(cached).to(output_device)

    @torch.no_grad()
    def prebuild_cache(
        self,
        graph: Optional[ShardedGraphDataset],
        text_store: TextStore,
        device: Optional[str | torch.device] = None,
        chunk_ids: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, float]:
        if graph is None:
            graph = self.graph
        if device is not None:
            self._set_device(device)

        chunk_ids_iter = chunk_ids if chunk_ids is not None else list(range(graph.chunk_count()))
        iterator = tqdm(chunk_ids_iter, desc="prebuild text cache", unit="chunk")
        total_missing = 0

        for chunk_id in iterator:
            node_indices = graph.chunk_node_indices(int(chunk_id))
            if node_indices.size == 0:
                continue
            node_wids = graph.get_node_wids(node_indices)
            node_chunk_ids = graph.node_chunk_ids[node_indices]
            node_chunk_pos = graph.node_chunk_pos[node_indices]
            _, miss_mask = self.cache.get_many(node_chunk_ids, node_chunk_pos)
            if not miss_mask.any():
                log_event(
                    "[STAGE]",
                    f"text cache chunk {int(chunk_id) + 1}/{graph.chunk_count()} missing=0",
                )
                continue
            miss_idx = np.where(miss_mask)[0]
            miss_texts = text_store.get_texts([node_wids[i] for i in miss_idx])
            miss_emb = self._encode_texts(miss_texts, batch_size=batch_size)
            self.cache.set_many(
                node_chunk_ids[miss_idx],
                node_chunk_pos[miss_idx],
                miss_emb,
            )
            total_missing += int(miss_idx.shape[0])
            log_event(
                "[STAGE]",
                f"text cache chunk {int(chunk_id) + 1}/{graph.chunk_count()} missing={int(miss_idx.shape[0])}",
            )

        stats = self.consume_runtime_stats()
        stats["text_cache_prebuilt"] = float(total_missing)
        return stats

    def consume_runtime_stats(self) -> Dict[str, float]:
        out = self.cache.consume_stats()
        out["text_encode_sec"] = float(self.encode_time_sec)
        self.encode_time_sec = 0.0
        return out

    def consume_cache_stats(self) -> Dict[str, float]:
        return self.consume_runtime_stats()


class BertFrozenTextEncoder(HFFrozenTextEncoder):
    """Backward-compatible alias for the old config name."""


def build_text_encoder(
    name: str,
    graph: ShardedGraphDataset,
    cache_dir: Path,
    local_model_root: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
    device: str,
    cache_dtype: str,
    cache_lru_chunks: int,
    progress_sec: float = 30.0,
    freeze_backbone: bool = True,
) -> nn.Module:
    if not freeze_backbone:
        raise ValueError("Trainable text backbones are not supported in this run.")
    if name in {"bert_frozen", "hf_text_frozen"}:
        return HFFrozenTextEncoder(
            graph=graph,
            cache_dir=cache_dir,
            local_model_root=local_model_root,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            cache_dtype=cache_dtype,
            cache_lru_chunks=cache_lru_chunks,
            progress_sec=progress_sec,
        )
    if name in {"sbert", "custom"}:
        raise NotImplementedError(f"text encoder not implemented: {name}")
    raise ValueError(f"Unknown text encoder: {name}")
