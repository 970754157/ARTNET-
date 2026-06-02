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

from project.data.text_store import TextStore
from project.data.typed_graph import EntityCatalog
from project.models.model_store import load_hf_resource, slugify_model_id
from project.utils.lru import LRUCache
from project.utils.logging import ProgressTicker, log_event


class ChunkAlignedTextCache:
    """Chunk-aligned memmap raw text embedding cache."""

    def __init__(
        self,
        cache_dir: Path,
        catalog: EntityCatalog,
        emb_dim: int,
        dtype: str,
        encoder_name: str,
        model_name: str,
        max_length: int,
        cache_key: str,
        lru_chunks: int = 8,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self.emb_dim = int(emb_dim)
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.encoder_name = str(encoder_name)
        self.model_name = str(model_name)
        self.max_length = int(max_length)
        self.cache_key = str(cache_key)
        self.cache = LRUCache[int, Tuple[np.memmap, np.memmap]](lru_chunks)
        self.meta_path = self.cache_dir / "meta.json"
        self._init_meta()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _init_meta(self) -> None:
        payload = {
            "format": "chunk_text_raw_emb_v2",
            "emb_dim": self.emb_dim,
            "dtype": "float16" if self.dtype == np.float16 else "float32",
            "encoder_name": self.encoder_name,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "cache_key": self.cache_key,
            "chunk_count": self.catalog.chunk_count(),
        }
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                old = json.load(f)
            for key in payload.keys():
                if old.get(key) != payload[key]:
                    raise RuntimeError(f"Text cache meta mismatch for {key}: {old.get(key)} != {payload[key]}")
            return
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _chunk_dir(self, chunk_id: int) -> Path:
        return self.cache_dir / f"chunk_{self.catalog.chunk_name(chunk_id)}"

    def _ensure_chunk_files(self, chunk_id: int) -> None:
        cdir = self._chunk_dir(chunk_id)
        cdir.mkdir(parents=True, exist_ok=True)
        emb_path = cdir / ("emb.f16.npy" if self.dtype == np.float16 else "emb.f32.npy")
        valid_path = cdir / "valid.u8.npy"
        n_local = self.catalog.chunk_size(chunk_id)
        if not emb_path.exists():
            emb_mm = np.lib.format.open_memmap(emb_path, mode="w+", dtype=self.dtype, shape=(n_local, self.emb_dim))
            emb_mm[:] = 0
            emb_mm.flush()
            del emb_mm
        if not valid_path.exists():
            valid_mm = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.uint8, shape=(n_local,))
            valid_mm[:] = 0
            valid_mm.flush()
            del valid_mm

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

    def get_many(self, chunk_ids: np.ndarray, chunk_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(chunk_ids.shape[0])
        out = np.zeros((n, self.emb_dim), dtype=np.float32)
        miss = np.zeros(n, dtype=bool)
        uniq = np.unique(chunk_ids)
        for chunk_id in uniq:
            idxs = np.where(chunk_ids == chunk_id)[0]
            emb_mm, valid_mm = self._get_chunk_memmap(int(chunk_id))
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
        for chunk_id in uniq:
            idxs = np.where(chunk_ids == chunk_id)[0]
            emb_mm, valid_mm = self._get_chunk_memmap(int(chunk_id))
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


class BertFrozenTextEncoder(nn.Module):
    """Frozen HF text encoder with chunk-aligned raw cache."""

    def __init__(
        self,
        catalog: EntityCatalog,
        cache_dir: Path,
        local_model_root: Path,
        encoder_name: str = "bert_frozen",
        model_name: str = "bert-base-uncased",
        max_length: int = 256,
        batch_size: int = 32,
        device: str = "cpu",
        cache_dtype: str = "float16",
        cache_lru_chunks: int = 8,
    ):
        super().__init__()
        self.catalog = catalog
        self.encoder_name = str(encoder_name)
        self.model_name = str(model_name)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.encode_time_sec = 0.0

        model_slug = slugify_model_id(self.model_name)
        local_model_dir = Path(local_model_root) / "text" / model_slug
        self.tokenizer = load_hf_resource(AutoTokenizer, self.model_name, local_model_dir)
        self.bert = load_hf_resource(AutoModel, self.model_name, local_model_dir)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()
        self.bert.to(self.device)
        hidden = int(self.bert.config.hidden_size)
        self.output_dim = hidden
        self.cache = ChunkAlignedTextCache(
            cache_dir=cache_dir,
            catalog=catalog,
            emb_dim=hidden,
            dtype=cache_dtype,
            encoder_name=self.encoder_name,
            model_name=model_name,
            max_length=max_length,
            cache_key=Path(cache_dir).name,
            lru_chunks=cache_lru_chunks,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.bert.eval()
        return self

    def _pool_outputs(self, outputs) -> torch.Tensor:
        model_type = str(getattr(self.bert.config, "model_type", "")).lower()
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
    def _encode_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        outputs: List[np.ndarray] = []
        use_batch_size = max(1, int(batch_size or self.batch_size))
        for start in range(0, len(texts), use_batch_size):
            batch = texts[start : start + use_batch_size]
            t0 = time.perf_counter()
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            out = self.bert(**tokenized)
            cls = self._pool_outputs(out).detach().cpu().numpy().astype(np.float32)
            outputs.append(cls)
            self.encode_time_sec += float(time.perf_counter() - t0)
        if not outputs:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _encode_texts_with_missing(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        out = np.zeros((len(texts), self.output_dim), dtype=np.float32)
        nonempty_idx = [i for i, text in enumerate(texts) if isinstance(text, str) and text.strip()]
        if not nonempty_idx:
            return out
        dense = self._encode_texts([texts[i] for i in nonempty_idx], batch_size=batch_size)
        out[np.asarray(nonempty_idx, dtype=np.int64)] = dense
        return out

    @torch.no_grad()
    def prebuild_cache(self, text_store: TextStore, batch_size: Optional[int] = None) -> Dict[str, float]:
        total_missing = 0
        ticker = ProgressTicker("prebuild text cache", every_sec=30.0)
        for chunk_id in range(self.catalog.chunk_count()):
            idxs = np.where(self.catalog.node_chunk_ids == int(chunk_id))[0]
            if idxs.size == 0:
                continue
            node_wids = self.catalog.get_node_wids(idxs)
            node_chunk_ids = self.catalog.node_chunk_ids[idxs]
            node_chunk_pos = self.catalog.node_chunk_pos[idxs]
            _, miss_mask = self.cache.get_many(node_chunk_ids, node_chunk_pos)
            if not miss_mask.any():
                ticker.maybe(f"chunk={chunk_id + 1}/{self.catalog.chunk_count()} missing=0")
                continue
            miss_idx = np.where(miss_mask)[0]
            miss_texts = text_store.get_texts([node_wids[i] for i in miss_idx.tolist()])
            miss_emb = self._encode_texts_with_missing(miss_texts, batch_size=batch_size)
            self.cache.set_many(node_chunk_ids[miss_idx], node_chunk_pos[miss_idx], miss_emb)
            total_missing += int(miss_idx.shape[0])
            ticker.maybe(
                f"chunk={chunk_id + 1}/{self.catalog.chunk_count()} missing={int(miss_idx.shape[0])}"
            )
        stats = self.consume_runtime_stats()
        stats["text_cache_prebuilt"] = float(total_missing)
        log_event("[STAGE]", f"text cache prebuild done missing={int(total_missing)}")
        return stats

    def get_node_embeddings(
        self,
        node_wids: List[str],
        node_chunk_ids: np.ndarray,
        node_chunk_pos: np.ndarray,
        text_store: TextStore,
        output_device: torch.device,
    ) -> torch.Tensor:
        cached, miss_mask = self.cache.get_many(node_chunk_ids, node_chunk_pos)
        if miss_mask.any():
            miss_idx = np.where(miss_mask)[0]
            miss_texts = text_store.get_texts([node_wids[i] for i in miss_idx.tolist()])
            miss_emb = self._encode_texts_with_missing(miss_texts)
            cached[miss_idx] = miss_emb
            self.cache.set_many(node_chunk_ids[miss_idx], node_chunk_pos[miss_idx], miss_emb)
        return torch.from_numpy(cached).to(output_device)

    def consume_runtime_stats(self) -> Dict[str, float]:
        out = self.cache.consume_stats()
        out["text_encode_sec"] = float(self.encode_time_sec)
        self.encode_time_sec = 0.0
        return out


def build_text_encoder(
    name: str,
    catalog: EntityCatalog,
    cache_dir: Path,
    local_model_root: Path,
    model_name: str,
    max_length: int,
    batch_size: int,
    device: str,
    cache_dtype: str,
    cache_lru_chunks: int,
) -> nn.Module:
    if name in {"bert_frozen", "roberta_frozen", "albert_frozen"}:
        return BertFrozenTextEncoder(
            catalog=catalog,
            cache_dir=cache_dir,
            local_model_root=local_model_root,
            encoder_name=name,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            cache_dtype=cache_dtype,
            cache_lru_chunks=cache_lru_chunks,
        )
    if name in {"sbert", "custom"}:
        raise NotImplementedError(f"text encoder not implemented: {name}")
    raise ValueError(f"Unknown text encoder: {name}")
