"""Entity catalog and typed triple store for KGE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from project.utils.io import atomic_write_json, ensure_dir
from project.utils.logging import ProgressTicker


@dataclass
class ChunkInfo:
    chunk_id: int
    name: str
    nodes_path: Path
    edges_path: Path
    node_count: int


class EntityCatalog:
    """Entity metadata extracted from sharded graph export."""

    def __init__(self, graph_dir: Path, cache_dir: Path, sample_chunks: int = 0, verbose: bool = True):
        self.graph_dir = Path(graph_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_chunks = int(sample_chunks)
        self.verbose = verbose
        self.manifest = self._load_manifest()
        self.chunks = self._load_chunks()
        self.meta_dir = self.cache_dir / "entity_meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._load_or_build_node_tables()

    def _load_manifest(self) -> Dict:
        path = self.graph_dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Graph manifest not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_chunks(self) -> List[ChunkInfo]:
        raw_chunks = list(self.manifest.get("chunks", []))
        if self.sample_chunks > 0:
            raw_chunks = raw_chunks[: min(self.sample_chunks, len(raw_chunks))]
        chunks: List[ChunkInfo] = []
        for i, raw in enumerate(raw_chunks):
            chunks.append(
                ChunkInfo(
                    chunk_id=i,
                    name=str(raw["dir_name"]),
                    nodes_path=self.graph_dir / str(raw["nodes_file"]),
                    edges_path=self.graph_dir / str(raw["edges_file"]),
                    node_count=int(raw.get("node_count", 0)),
                )
            )
        if not chunks:
            raise RuntimeError("No graph chunks were found")
        return chunks

    @staticmethod
    def _has_valid_image_url(image_value) -> bool:
        if isinstance(image_value, str):
            value = image_value.strip().lower()
            return value.startswith("http://") or value.startswith("https://")
        if isinstance(image_value, list):
            return any(
                isinstance(item, str) and item.strip().lower().startswith(("http://", "https://"))
                for item in image_value
            )
        return False

    def _tables_exist(self) -> bool:
        required = [
            self.meta_dir / "node_ids.npy",
            self.meta_dir / "node_wids.npy",
            self.meta_dir / "node_has_image.npy",
            self.meta_dir / "node_is_art.npy",
            self.meta_dir / "node_chunk_ids.npy",
            self.meta_dir / "node_chunk_pos.npy",
            self.meta_dir / "id_to_idx.npy",
            self.meta_dir / "wid_to_idx.json",
        ]
        return all(path.exists() for path in required)

    @staticmethod
    def _has_art_label(labels_value) -> bool:
        if isinstance(labels_value, list):
            return any(str(label).strip() == "ART" for label in labels_value)
        if isinstance(labels_value, str):
            return labels_value.strip() == "ART"
        return False

    def _load_or_build_node_tables(self) -> None:
        if self._tables_exist():
            self.node_ids = np.load(self.meta_dir / "node_ids.npy")
            self.node_wids = np.load(self.meta_dir / "node_wids.npy", allow_pickle=True)
            self.node_has_image = np.load(self.meta_dir / "node_has_image.npy")
            self.node_is_art = np.load(self.meta_dir / "node_is_art.npy")
            self.node_chunk_ids = np.load(self.meta_dir / "node_chunk_ids.npy")
            self.node_chunk_pos = np.load(self.meta_dir / "node_chunk_pos.npy")
            self.id_to_idx = np.load(self.meta_dir / "id_to_idx.npy")
            with (self.meta_dir / "wid_to_idx.json").open("r", encoding="utf-8") as f:
                raw = json.load(f)
            self.wid_to_idx = {k: int(v) for k, v in raw.items()}
            return

        node_ids: List[int] = []
        node_wids: List[str] = []
        node_has_image: List[bool] = []
        node_is_art: List[bool] = []
        node_chunk_ids: List[int] = []
        node_chunk_pos: List[int] = []
        wid_to_idx: Dict[str, int] = {}
        next_idx = 0
        iterator = self.chunks
        if self.verbose:
            iterator = tqdm(self.chunks, desc="build entity catalog", unit="chunk")

        for chunk in iterator:
            with chunk.nodes_path.open("r", encoding="utf-8") as f:
                nodes = json.load(f)
            pos = 0
            for node_id_text, payload in nodes.items():
                gid = int(node_id_text)
                props = payload.get("properties", {}) if isinstance(payload, dict) else {}
                labels = payload.get("labels", []) if isinstance(payload, dict) else []
                wid = props.get("wid", "")
                node_ids.append(gid)
                node_wids.append(str(wid) if wid is not None else "")
                node_has_image.append(self._has_valid_image_url(props.get("image")))
                node_is_art.append(self._has_art_label(labels))
                node_chunk_ids.append(chunk.chunk_id)
                node_chunk_pos.append(pos)
                if isinstance(wid, str) and wid and wid not in wid_to_idx:
                    wid_to_idx[wid] = next_idx
                pos += 1
                next_idx += 1

        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.node_wids = np.asarray(node_wids, dtype=object)
        self.node_has_image = np.asarray(node_has_image, dtype=np.bool_)
        self.node_is_art = np.asarray(node_is_art, dtype=np.bool_)
        self.node_chunk_ids = np.asarray(node_chunk_ids, dtype=np.int32)
        self.node_chunk_pos = np.asarray(node_chunk_pos, dtype=np.int32)
        self.id_to_idx = np.full(int(self.node_ids.max()) + 1, -1, dtype=np.int64)
        self.id_to_idx[self.node_ids] = np.arange(self.node_ids.shape[0], dtype=np.int64)
        self.wid_to_idx = wid_to_idx

        np.save(self.meta_dir / "node_ids.npy", self.node_ids)
        np.save(self.meta_dir / "node_wids.npy", self.node_wids, allow_pickle=True)
        np.save(self.meta_dir / "node_has_image.npy", self.node_has_image)
        np.save(self.meta_dir / "node_is_art.npy", self.node_is_art)
        np.save(self.meta_dir / "node_chunk_ids.npy", self.node_chunk_ids)
        np.save(self.meta_dir / "node_chunk_pos.npy", self.node_chunk_pos)
        np.save(self.meta_dir / "id_to_idx.npy", self.id_to_idx)
        atomic_write_json(self.meta_dir / "wid_to_idx.json", {k: int(v) for k, v in wid_to_idx.items()})

    def num_entities(self) -> int:
        return int(self.node_ids.shape[0])

    def chunk_count(self) -> int:
        return len(self.chunks)

    def chunk_name(self, chunk_id: int) -> str:
        return self.chunks[int(chunk_id)].name

    def chunk_size(self, chunk_id: int) -> int:
        return int(self.chunks[int(chunk_id)].node_count)

    def selected_chunk_names(self) -> List[str]:
        return [chunk.name for chunk in self.chunks]

    def get_node_wids(self, node_indices: np.ndarray) -> List[str]:
        idx = np.asarray(node_indices, dtype=np.int64)
        return [str(x) for x in self.node_wids[idx]]

    def get_node_is_art(self, node_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(node_indices, dtype=np.int64)
        return self.node_is_art[idx]

    def resolve_entity_ref(self, ref: str) -> int:
        if ref in self.wid_to_idx:
            return int(self.wid_to_idx[ref])
        gid = int(ref)
        if gid < 0 or gid >= self.id_to_idx.shape[0]:
            raise KeyError(f"Entity global id out of range: {gid}")
        idx = int(self.id_to_idx[gid])
        if idx < 0:
            raise KeyError(f"Entity global id not found: {gid}")
        return idx


@dataclass
class TripleStore:
    triples: np.ndarray
    relation_to_idx: Dict[str, int]
    idx_to_relation: List[str]
    triple_hash_sorted: np.ndarray


def hash_triples(triples: np.ndarray, num_entities: int, num_relations: int) -> np.ndarray:
    arr = np.asarray(triples, dtype=np.int64)
    h = arr[:, 0].astype(np.uint64, copy=False)
    r = arr[:, 1].astype(np.uint64, copy=False)
    t = arr[:, 2].astype(np.uint64, copy=False)
    return (h * np.uint64(num_relations) + r) * np.uint64(num_entities) + t


def triple_hash_exists(hashes: np.ndarray, sorted_hashes: np.ndarray) -> np.ndarray:
    values = np.asarray(hashes, dtype=np.uint64).reshape(-1)
    pos = np.searchsorted(sorted_hashes, values, side="left")
    ok = pos < sorted_hashes.shape[0]
    out = np.zeros(values.shape[0], dtype=np.bool_)
    out[ok] = sorted_hashes[pos[ok]] == values[ok]
    return out.reshape(np.asarray(hashes).shape)


def load_or_build_triple_store(
    graph_dir: Path,
    cache_dir: Path,
    catalog: EntityCatalog,
    force_rebuild: bool = False,
    verbose: bool = True,
    progress_every_sec: float = 30.0,
) -> TripleStore:
    _ = graph_dir
    cache_dir = Path(cache_dir)
    ensure_dir(cache_dir)
    triples_path = cache_dir / "triples.npy"
    rel_vocab_path = cache_dir / "relation_vocab.json"
    hashes_path = cache_dir / "triple_hash_sorted.npy"

    if not force_rebuild and triples_path.exists() and rel_vocab_path.exists() and hashes_path.exists():
        triples = np.load(triples_path, mmap_mode="r")
        with rel_vocab_path.open("r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        relation_to_idx = {k: int(v) for k, v in raw_vocab.items()}
        idx_to_relation = [name for name, _ in sorted(relation_to_idx.items(), key=lambda kv: kv[1])]
        hashes = np.load(hashes_path, mmap_mode="r")
        return TripleStore(
            triples=np.asarray(triples, dtype=np.int64),
            relation_to_idx=relation_to_idx,
            idx_to_relation=idx_to_relation,
            triple_hash_sorted=np.asarray(hashes, dtype=np.uint64),
        )

    relation_to_idx: Dict[str, int] = {}
    head_parts: List[np.ndarray] = []
    rel_parts: List[np.ndarray] = []
    tail_parts: List[np.ndarray] = []
    ticker = ProgressTicker("build triple store", every_sec=progress_every_sec)
    iterator = catalog.chunks
    if verbose:
        iterator = tqdm(catalog.chunks, desc="build triples", unit="chunk")

    for chunk in iterator:
        with chunk.edges_path.open("r", encoding="utf-8") as f:
            edges = json.load(f)
        chunk_heads: List[int] = []
        chunk_rels: List[int] = []
        chunk_tails: List[int] = []
        for edge in edges:
            rel_name = str(edge.get("type", "")).strip()
            if not rel_name:
                continue
            src_gid = int(edge["src"])
            dst_gid = int(edge["dst"])
            if src_gid < 0 or dst_gid < 0:
                continue
            if src_gid >= catalog.id_to_idx.shape[0] or dst_gid >= catalog.id_to_idx.shape[0]:
                continue
            src_idx = int(catalog.id_to_idx[src_gid])
            dst_idx = int(catalog.id_to_idx[dst_gid])
            if src_idx < 0 or dst_idx < 0:
                continue
            if rel_name not in relation_to_idx:
                relation_to_idx[rel_name] = len(relation_to_idx)
            chunk_heads.append(src_idx)
            chunk_rels.append(int(relation_to_idx[rel_name]))
            chunk_tails.append(dst_idx)
        if chunk_heads:
            head_parts.append(np.asarray(chunk_heads, dtype=np.int64))
            rel_parts.append(np.asarray(chunk_rels, dtype=np.int64))
            tail_parts.append(np.asarray(chunk_tails, dtype=np.int64))
        ticker.maybe(f"chunks={chunk.chunk_id + 1}/{catalog.chunk_count()} relations={len(relation_to_idx)}")

    heads = np.concatenate(head_parts, axis=0)
    rels = np.concatenate(rel_parts, axis=0)
    tails = np.concatenate(tail_parts, axis=0)
    triples = np.stack([heads, rels, tails], axis=1)
    hashes = hash_triples(triples, num_entities=catalog.num_entities(), num_relations=len(relation_to_idx))
    _, uniq_idx = np.unique(hashes, return_index=True)
    uniq_idx = np.sort(uniq_idx.astype(np.int64, copy=False))
    triples = triples[uniq_idx]
    hashes = np.sort(hashes[uniq_idx])
    np.save(triples_path, triples)
    np.save(hashes_path, hashes)
    atomic_write_json(rel_vocab_path, relation_to_idx)
    idx_to_relation = [name for name, _ in sorted(relation_to_idx.items(), key=lambda kv: kv[1])]
    return TripleStore(
        triples=triples,
        relation_to_idx=relation_to_idx,
        idx_to_relation=idx_to_relation,
        triple_hash_sorted=hashes,
    )
