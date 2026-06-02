"""Image provider for wid-based local image lookup."""

from __future__ import annotations

import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from project.utils.lru import LRUCache
from project.utils.logging import ProgressTicker, log_event


class ImageProvider:
    """Base image provider."""

    def get_image(self, wid: str) -> Image.Image:
        raise NotImplementedError

    def get_batch_tensors(
        self,
        wids: List[str],
        has_image_mask: np.ndarray,
        transform,
        pin_memory: bool = False,
    ) -> Dict[str, object]:
        raise NotImplementedError

    def inspect_batch(
        self,
        wids: List[str],
        has_image_mask: np.ndarray,
        pin_memory: bool = False,
    ) -> Dict[str, object]:
        raise NotImplementedError

    def load_resolved_batch_tensors(
        self,
        items: Sequence[Tuple[int, str]],
        transform,
        pin_memory: bool = False,
    ) -> Dict[str, object]:
        raise NotImplementedError

    def close(self) -> None:
        """Release provider-side resources."""


class ByDirectoryImageProvider(ImageProvider):
    """Resolve images from a directory using wid.png or wid_*.png patterns."""

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
        self,
        image_dir: Path,
        cache_dir: Path,
        decode_lru: int = 128,
        verbose: bool = True,
        preindex: bool = True,
        num_workers: int = 0,
        stage_progress_sec: float = 30.0,
    ):
        self.image_dir = Path(image_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.preindex = bool(preindex)
        self.num_workers = max(0, int(num_workers))
        self.stage_progress_sec = float(stage_progress_sec)
        self.decode_cache = LRUCache[str, Image.Image](decode_lru)
        self.wid_to_path: Dict[str, Optional[Path]] = (
            self._load_or_build_index() if self.preindex else {}
        )
        self.fallback = Image.new("RGB", (224, 224), color=(0, 0, 0))
        self._cache_lock = threading.RLock()
        self._executor = (
            ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="img-provider")
            if self.num_workers > 0
            else None
        )

    def _index_path(self) -> Path:
        return self.cache_dir / "image_index.pkl"

    @staticmethod
    def _wid_from_stem(stem: str) -> Optional[str]:
        if not stem.startswith("Q"):
            return None
        if "_" in stem:
            return stem.split("_", 1)[0]
        return stem

    @staticmethod
    def _select_best_path(wid: str, paths: List[Path]) -> Optional[Path]:
        if not paths:
            return None
        paths_sorted = sorted(paths, key=lambda x: x.name)
        exact = [p for p in paths_sorted if p.stem == wid]
        if exact:
            return exact[0]
        wid0 = [p for p in paths_sorted if p.stem == f"{wid}_0"]
        if wid0:
            return wid0[0]
        return paths_sorted[0]

    def _load_or_build_index(self) -> Dict[str, Optional[Path]]:
        idx_path = self._index_path()
        if idx_path.exists():
            with idx_path.open("rb") as f:
                raw = pickle.load(f)
            index = {k: (Path(v) if v else None) for k, v in raw.items()}
            log_event("[STAGE]", f"image index cache loaded: {idx_path} entries={len(index)}")
            return index

        candidates: Dict[str, List[Path]] = {}
        iterator = self.image_dir.rglob("*")
        ticker = ProgressTicker("image index build", every_sec=self.stage_progress_sec)
        files_seen = 0
        if self.verbose:
            iterator = tqdm(iterator, desc="index images", unit="file")
        for p in iterator:
            files_seen += 1
            if not p.is_file():
                ticker.maybe(f"files_seen={files_seen} wid_candidates={len(candidates)}")
                continue
            if p.suffix.lower() not in self.IMAGE_SUFFIXES:
                ticker.maybe(f"files_seen={files_seen} wid_candidates={len(candidates)}")
                continue
            wid = self._wid_from_stem(p.stem)
            if wid is None:
                ticker.maybe(f"files_seen={files_seen} wid_candidates={len(candidates)}")
                continue
            candidates.setdefault(wid, []).append(p)
            ticker.maybe(f"files_seen={files_seen} wid_candidates={len(candidates)}")

        index: Dict[str, Optional[Path]] = {}
        for wid, paths in candidates.items():
            index[wid] = self._select_best_path(wid, paths)

        with idx_path.open("wb") as f:
            pickle.dump(
                {k: (str(v) if v is not None else "") for k, v in index.items()},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        ticker.force(f"completed files_seen={files_seen} indexed_wids={len(index)}")
        log_event("[STAGE]", f"image index cache written: {idx_path}")
        return index

    def _lazy_find_path(self, wid: str) -> Optional[Path]:
        direct_candidates: List[Path] = []
        for suffix in self.IMAGE_SUFFIXES:
            p = self.image_dir / f"{wid}{suffix}"
            if p.exists() and p.is_file():
                direct_candidates.append(p)
            p0 = self.image_dir / f"{wid}_0{suffix}"
            if p0.exists() and p0.is_file():
                direct_candidates.append(p0)

        for p in self.image_dir.glob(f"{wid}_*"):
            if p.is_file() and p.suffix.lower() in self.IMAGE_SUFFIXES:
                direct_candidates.append(p)

        if not direct_candidates:
            for p in self.image_dir.rglob(f"{wid}*"):
                if p.is_file() and p.suffix.lower() in self.IMAGE_SUFFIXES:
                    direct_candidates.append(p)
            for p in self.image_dir.rglob(f"{wid}_*"):
                if p.is_file() and p.suffix.lower() in self.IMAGE_SUFFIXES:
                    direct_candidates.append(p)

        return self._select_best_path(wid, direct_candidates)

    def _resolve_image_path(self, wid: str) -> Optional[Path]:
        with self._cache_lock:
            if wid in self.wid_to_path:
                return self.wid_to_path[wid]
        if self.preindex:
            return None
        found = self._lazy_find_path(wid)
        with self._cache_lock:
            self.wid_to_path[wid] = found
        return found

    def get_image(self, wid: str) -> Image.Image:
        img, loaded = self._get_real_image(wid)
        if loaded:
            return img
        return self.fallback.copy()

    def _get_real_image(self, wid: str) -> tuple[Image.Image, bool]:
        with self._cache_lock:
            cached = self.decode_cache.get(wid)
        if cached is not None:
            return cached.copy(), True

        p = self._resolve_image_path(wid)
        if p is None or not p.exists():
            return self.fallback, False
        try:
            with Image.open(p) as img:
                out = img.convert("RGB")
        except Exception:
            return self.fallback, False
        with self._cache_lock:
            self.decode_cache.put(wid, out)
        return out.copy(), True

    def _load_one_transformed(self, item: tuple[int, str], transform) -> tuple[int, Optional[torch.Tensor], bool]:
        idx, wid = item
        img, loaded = self._get_real_image(wid)
        if not loaded:
            return idx, None, False
        return idx, transform(img), True

    @staticmethod
    def _load_one_resolved_transformed(
        item: tuple[int, str],
        transform,
    ) -> tuple[int, Optional[torch.Tensor], bool]:
        idx, path_text = item
        path = Path(path_text)
        if not path.exists():
            return idx, None, False
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
        except Exception:
            return idx, None, False
        return idx, transform(rgb), True

    @staticmethod
    def _maybe_pin_tensor(tensor: torch.Tensor, pin_memory: bool) -> torch.Tensor:
        if pin_memory and tensor.device.type == "cpu":
            return tensor.pin_memory()
        return tensor

    def get_batch_tensors(
        self,
        wids: List[str],
        has_image_mask: np.ndarray,
        transform,
        pin_memory: bool = False,
    ) -> Dict[str, object]:
        inspected = self.inspect_batch(wids, has_image_mask, pin_memory=pin_memory)
        items = list(
            zip(
                inspected["real_node_indices_cpu"].tolist(),
                inspected["real_node_paths"],
            )
        )
        loaded = self.load_resolved_batch_tensors(items, transform, pin_memory=pin_memory)
        loaded.update(
            {
                "loaded_mask_cpu": inspected["loaded_mask_cpu"],
                "stats_real_hit": inspected["stats_real_hit"],
                "stats_no_image": inspected["stats_no_image"],
                "stats_missing_local": inspected["stats_missing_local"],
                "stats_image_decode_sec": float(
                    inspected["stats_image_resolve_sec"] + loaded["stats_image_decode_sec"]
                ),
            }
        )
        return loaded

    def inspect_batch(
        self,
        wids: List[str],
        has_image_mask: np.ndarray,
        pin_memory: bool = False,
    ) -> Dict[str, object]:
        if len(wids) != int(has_image_mask.shape[0]):
            raise ValueError("wids length must equal has_image_mask length")

        start_time = time.perf_counter()
        requested = [(i, wids[i]) for i in range(len(wids)) if bool(has_image_mask[i])]
        loaded_mask = np.zeros(len(wids), dtype=np.bool_)
        real_indices: List[int] = []
        real_paths: List[str] = []
        for idx, wid in requested:
            path = self._resolve_image_path(wid)
            if path is not None and path.exists():
                loaded_mask[idx] = True
                real_indices.append(idx)
                real_paths.append(str(path))
        real_indices_t = torch.tensor(real_indices, dtype=torch.long)
        loaded_mask_t = torch.from_numpy(loaded_mask)
        real_indices_t = self._maybe_pin_tensor(real_indices_t, pin_memory)
        loaded_mask_t = self._maybe_pin_tensor(loaded_mask_t, pin_memory)

        real_hit_count = int(loaded_mask.sum())
        requested_count = len(requested)
        no_image_count = len(wids) - requested_count
        missing_local_count = requested_count - real_hit_count
        return {
            "real_node_indices_cpu": real_indices_t,
            "real_node_paths": real_paths,
            "loaded_mask_cpu": loaded_mask_t,
            "stats_real_hit": real_hit_count,
            "stats_no_image": no_image_count,
            "stats_missing_local": missing_local_count,
            "stats_image_resolve_sec": float(time.perf_counter() - start_time),
        }

    def load_resolved_batch_tensors(
        self,
        items: Sequence[Tuple[int, str]],
        transform,
        pin_memory: bool = False,
    ) -> Dict[str, object]:
        start_time = time.perf_counter()
        if self._executor is not None and items:
            results = list(
                self._executor.map(
                    lambda item: self._load_one_resolved_transformed(item, transform),
                    items,
                )
            )
        else:
            results = [self._load_one_resolved_transformed(item, transform) for item in items]

        real_tensors: List[torch.Tensor] = []
        real_indices: List[int] = []
        for idx, tensor, loaded in results:
            if loaded and tensor is not None:
                real_indices.append(int(idx))
                real_tensors.append(tensor)

        if real_tensors:
            stacked = torch.stack(real_tensors, dim=0)
        else:
            sample_tensor = transform(self.fallback.copy())
            stacked = sample_tensor.new_empty((0, *sample_tensor.shape))
        real_indices_t = torch.tensor(real_indices, dtype=torch.long)
        stacked = self._maybe_pin_tensor(stacked, pin_memory)
        real_indices_t = self._maybe_pin_tensor(real_indices_t, pin_memory)
        return {
            "real_images_cpu": stacked,
            "real_node_indices_cpu": real_indices_t,
            "stats_image_decode_sec": float(time.perf_counter() - start_time),
        }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
