"""Chunk-aligned image embedding cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from project.data.typed_graph import EntityCatalog
from project.utils.lru import LRUCache


class ChunkAlignedImageCache:
    """Chunk-aligned memmap raw image embedding cache."""

    def __init__(
        self,
        cache_dir: Path,
        catalog: EntityCatalog,
        emb_dim: int,
        dtype: str,
        encoder_name: str,
        model_name: str,
        image_size: int,
        pretrained: bool,
        freeze_backbone: bool,
        cache_key: str,
        lru_chunks: int = 4,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog
        self.emb_dim = int(emb_dim)
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.encoder_name = str(encoder_name)
        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.pretrained = bool(pretrained)
        self.freeze_backbone = bool(freeze_backbone)
        self.cache_key = str(cache_key)
        self.cache = LRUCache[int, Tuple[np.memmap, np.memmap]](int(lru_chunks))
        self.meta_path = self.cache_dir / "meta.json"
        self._init_meta()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _init_meta(self) -> None:
        payload = {
            "format": "chunk_image_raw_emb_v2",
            "emb_dim": self.emb_dim,
            "dtype": "float16" if self.dtype == np.float16 else "float32",
            "encoder_name": self.encoder_name,
            "model_name": self.model_name,
            "image_size": self.image_size,
            "pretrained": self.pretrained,
            "freeze_backbone": self.freeze_backbone,
            "cache_key": self.cache_key,
            "chunk_count": self.catalog.chunk_count(),
        }
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                old = json.load(f)
            for key in payload.keys():
                if old.get(key) != payload[key]:
                    raise RuntimeError(f"Image cache meta mismatch for {key}: {old.get(key)} != {payload[key]}")
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
        self._ensure_chunk_files(int(chunk_id))
        cdir = self._chunk_dir(int(chunk_id))
        emb_path = cdir / ("emb.f16.npy" if self.dtype == np.float16 else "emb.f32.npy")
        valid_path = cdir / "valid.u8.npy"
        emb = np.load(emb_path, mmap_mode="r+")
        valid = np.load(valid_path, mmap_mode="r+")
        self.cache.put(int(chunk_id), (emb, valid))
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
            "image_cache_hit": float(self.hits),
            "image_cache_miss": float(self.misses),
            "image_cache_hit_rate": hit_rate,
            "image_cache_write_count": float(self.writes),
        }
        self.hits = 0
        self.misses = 0
        self.writes = 0
        return out
