"""Text store for wid -> assembled text."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from project.utils.lru import LRUCache


class TextStore:
    """Loads sharded text chunks and supports wid-based text assembly."""

    def __init__(
        self,
        text_dir: Path,
        cache_dir: Path,
        lru_chunks: int = 4,
        verbose: bool = True,
        allowed_chunks: Optional[List[str]] = None,
    ):
        self.text_dir = Path(text_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.chunk_cache = LRUCache[str, List[Dict]](lru_chunks)

        self.manifest_path = self.text_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Text manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        all_chunk_names = list(self.manifest.get("chunks", {}).keys())
        if allowed_chunks is None:
            self.chunk_names = all_chunk_names
        else:
            allowed = set(allowed_chunks)
            self.chunk_names = [name for name in all_chunk_names if name in allowed]
        if not self.chunk_names:
            raise RuntimeError("No text chunks found in text manifest")

        self.chunk_to_file = {name: self.text_dir / f"{name}.json" for name in self.chunk_names}
        self.wid_index = self._load_or_build_wid_index()

    def _wid_index_path(self) -> Path:
        return self.cache_dir / "wid_index.pkl"

    def _load_or_build_wid_index(self) -> Dict[str, Tuple[str, int]]:
        idx_path = self._wid_index_path()
        if idx_path.exists():
            with idx_path.open("rb") as f:
                return pickle.load(f)

        wid_index: Dict[str, Tuple[str, int]] = {}
        iterator = self.chunk_names
        if self.verbose:
            iterator = tqdm(iterator, desc="build wid index", unit="chunk")

        for chunk_name in iterator:
            p = self.chunk_to_file[chunk_name]
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                arr = json.load(f)
            for i, rec in enumerate(arr):
                wid = rec.get("wid", "")
                if isinstance(wid, str) and wid:
                    wid_index[wid] = (chunk_name, i)

        with idx_path.open("wb") as f:
            pickle.dump(wid_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        return wid_index

    def _load_chunk(self, chunk_name: str) -> List[Dict]:
        cached = self.chunk_cache.get(chunk_name)
        if cached is not None:
            return cached
        with self.chunk_to_file[chunk_name].open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunk_cache.put(chunk_name, data)
        return data

    @staticmethod
    def assemble_text(record: Dict) -> str:
        brief = record.get("brief", "")
        sections = record.get("sections", {})
        parts: List[str] = []
        if isinstance(brief, str) and brief.strip():
            parts.append(brief.strip())
        if isinstance(sections, dict):
            for _, content in sections.items():
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
        return "\n".join(parts).strip()

    def get_text(self, wid: str) -> str:
        loc = self.wid_index.get(wid)
        if loc is None:
            return ""
        chunk_name, pos = loc
        data = self._load_chunk(chunk_name)
        if pos < 0 or pos >= len(data):
            return ""
        return self.assemble_text(data[pos])

    def get_texts(self, wids: List[str]) -> List[str]:
        return [self.get_text(wid) for wid in wids]

