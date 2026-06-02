"""Disk adjacency index for train-only typed propagation edges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from project.data.propagation import TypedPropagationGraph
from project.data.typed_graph import EntityCatalog
from project.utils.io import atomic_write_json
from project.utils.lru import LRUCache


class DiskAdjacencyIndex:
    """Chunk-aligned adjacency index backed by .npy files."""

    def __init__(
        self,
        catalog: EntityCatalog,
        index_dir: Path,
        source_graph: TypedPropagationGraph,
        lru_chunks: int = 8,
        verbose: bool = True,
        edge_source_name: str = "train_propagation",
    ):
        self.catalog = catalog
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.index_dir / "index_meta.json"
        self.state_path = self.index_dir / "state.json"
        self.verbose = verbose
        self.cache = LRUCache[int, Tuple[np.ndarray, np.ndarray, np.ndarray]](lru_chunks)
        self.edge_source_name = str(edge_source_name)
        self.source_edges = np.asarray(source_graph.edge_index, dtype=np.int64)
        self.source_edge_types = np.asarray(source_graph.edge_type, dtype=np.int64)
        if self.source_edges.ndim != 2 or self.source_edges.shape[0] != 2:
            raise ValueError("source_edges must have shape [2, num_edges]")
        if self.source_edge_types.ndim != 1 or self.source_edge_types.shape[0] != self.source_edges.shape[1]:
            raise ValueError("source_edge_types must have shape [num_edges]")

        self._chunk_to_nodes: List[np.ndarray] = []
        for chunk_id in range(self.catalog.chunk_count()):
            idx = np.where(self.catalog.node_chunk_ids == chunk_id)[0]
            order = np.argsort(self.catalog.node_chunk_pos[idx], kind="stable")
            self._chunk_to_nodes.append(idx[order].astype(np.int64, copy=False))
        self._prepared_edges = self._prepare_edges_by_source_chunk()

    def _chunk_dir(self, chunk_id: int) -> Path:
        return self.index_dir / f"chunk_{self.catalog.chunk_name(chunk_id)}"

    def _chunk_files_exist(self, chunk_id: int) -> bool:
        cdir = self._chunk_dir(chunk_id)
        return (
            (cdir / "row_ptr.npy").exists()
            and (cdir / "col_idx.npy").exists()
            and (cdir / "rel_type.npy").exists()
            and (cdir / "node_ids.npy").exists()
        )

    def _write_meta(self) -> None:
        payload = {
            "format": "chunk_csr_rel_v2",
            "num_nodes": int(self.catalog.num_entities()),
            "chunk_count": int(self.catalog.chunk_count()),
            "directed": True,
            "typed": True,
            "edge_source": self.edge_source_name,
        }
        atomic_write_json(self.meta_path, payload)

    def _prepare_edges_by_source_chunk(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.source_edges.shape[1] == 0:
            empty = np.empty((0,), dtype=np.int64)
            offsets = np.zeros(self.catalog.chunk_count() + 1, dtype=np.int64)
            return empty, empty, empty, offsets

        src = self.source_edges[0].astype(np.int64, copy=False)
        dst = self.source_edges[1].astype(np.int64, copy=False)
        rel = self.source_edge_types.astype(np.int64, copy=False)
        src_chunk_ids = self.catalog.node_chunk_ids[src].astype(np.int32, copy=False)
        order = np.argsort(src_chunk_ids, kind="stable")
        src = src[order]
        dst = dst[order]
        rel = rel[order]
        src_chunk_ids = src_chunk_ids[order]
        offsets = np.searchsorted(
            src_chunk_ids,
            np.arange(self.catalog.chunk_count() + 1, dtype=np.int32),
            side="left",
        ).astype(np.int64, copy=False)
        return src, dst, rel, offsets

    def _load_state(self) -> Dict:
        if not self.state_path.exists():
            return {"status": "fresh", "completed_chunks": []}
        with self.state_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_state(self, state: Dict) -> None:
        atomic_write_json(self.state_path, state)

    def build_if_needed(self, rebuild: bool = False) -> None:
        if rebuild:
            self._clear_index_dir()
        if self.meta_path.exists() and self.state_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            state = self._load_state()
            if (
                state.get("status") == "completed"
                and meta.get("format") == "chunk_csr_rel_v2"
                and bool(meta.get("typed", False))
                and meta.get("edge_source") == self.edge_source_name
            ):
                return
        self._build()

    def _clear_index_dir(self) -> None:
        for path in self.index_dir.glob("*"):
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                path.rmdir()
            else:
                path.unlink()

    def _build(self) -> None:
        self._write_meta()
        state = self._load_state()
        completed = set(state.get("completed_chunks", []))
        state["status"] = "in_progress"
        self._save_state(state)

        iterator = range(self.catalog.chunk_count())
        if self.verbose:
            iterator = tqdm(iterator, desc="build adjacency index", unit="chunk")

        for chunk_id in iterator:
            chunk_name = self.catalog.chunk_name(chunk_id)
            if chunk_name in completed and self._chunk_files_exist(chunk_id):
                continue
            self._build_one_chunk(chunk_id)
            completed.add(chunk_name)
            state["completed_chunks"] = sorted(completed)
            self._save_state(state)

        state["status"] = "completed"
        state["completed_chunks"] = sorted(completed)
        self._save_state(state)

    def _build_one_chunk(self, chunk_id: int) -> None:
        node_ids = self._chunk_to_nodes[chunk_id]
        n_local = int(node_ids.shape[0])
        src_sorted, dst_sorted, rel_sorted, offsets = self._prepared_edges
        start = int(offsets[chunk_id])
        end = int(offsets[chunk_id + 1])
        if end > start:
            src_slice = src_sorted[start:end]
            dst_slice = dst_sorted[start:end]
            rel_slice = rel_sorted[start:end]
            src_pos = self.catalog.node_chunk_pos[src_slice].astype(np.int64, copy=False)
            valid = (src_pos >= 0) & (src_pos < n_local)
            src_pos = src_pos[valid]
            dst_slice = dst_slice[valid]
            rel_slice = rel_slice[valid]
            if src_pos.shape[0] > 0:
                order = np.argsort(src_pos, kind="stable")
                src_pos = src_pos[order]
                col_idx = dst_slice[order].astype(np.int64, copy=False)
                rel_type = rel_slice[order].astype(np.int64, copy=False)
                counts = np.bincount(src_pos, minlength=n_local).astype(np.int64, copy=False)
            else:
                col_idx = np.empty((0,), dtype=np.int64)
                rel_type = np.empty((0,), dtype=np.int64)
                counts = np.zeros((n_local,), dtype=np.int64)
        else:
            col_idx = np.empty((0,), dtype=np.int64)
            rel_type = np.empty((0,), dtype=np.int64)
            counts = np.zeros((n_local,), dtype=np.int64)

        row_ptr = np.zeros(n_local + 1, dtype=np.int64)
        row_ptr[1:] = np.cumsum(counts)
        cdir = self._chunk_dir(chunk_id)
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(cdir / "row_ptr.npy", row_ptr)
        np.save(cdir / "col_idx.npy", col_idx)
        np.save(cdir / "rel_type.npy", rel_type)
        np.save(cdir / "node_ids.npy", node_ids)

    def _load_chunk_arrays(self, chunk_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self.cache.get(chunk_id)
        if cached is not None:
            return cached
        cdir = self._chunk_dir(chunk_id)
        row_ptr = np.load(cdir / "row_ptr.npy", mmap_mode="r")
        col_idx = np.load(cdir / "col_idx.npy", mmap_mode="r")
        rel_type = np.load(cdir / "rel_type.npy", mmap_mode="r")
        self.cache.put(chunk_id, (row_ptr, col_idx, rel_type))
        return row_ptr, col_idx, rel_type

    def neighbors(self, node_idx: int) -> np.ndarray:
        chunk_id = int(self.catalog.node_chunk_ids[node_idx])
        chunk_pos = int(self.catalog.node_chunk_pos[node_idx])
        row_ptr, col_idx, _ = self._load_chunk_arrays(chunk_id)
        start = int(row_ptr[chunk_pos])
        end = int(row_ptr[chunk_pos + 1])
        return np.asarray(col_idx[start:end], dtype=np.int64)

    def neighbors_with_types(self, node_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        chunk_id = int(self.catalog.node_chunk_ids[node_idx])
        chunk_pos = int(self.catalog.node_chunk_pos[node_idx])
        row_ptr, col_idx, rel_type = self._load_chunk_arrays(chunk_id)
        start = int(row_ptr[chunk_pos])
        end = int(row_ptr[chunk_pos + 1])
        return (
            np.asarray(col_idx[start:end], dtype=np.int64),
            np.asarray(rel_type[start:end], dtype=np.int64),
        )

    def neighbors_many(self, node_indices: np.ndarray) -> List[np.ndarray]:
        nodes = np.asarray(node_indices, dtype=np.int64)
        if nodes.size == 0:
            return []
        out: List[np.ndarray] = [np.empty((0,), dtype=np.int64) for _ in range(int(nodes.shape[0]))]
        chunk_ids = self.catalog.node_chunk_ids[nodes]
        chunk_pos = self.catalog.node_chunk_pos[nodes]
        for chunk_id in np.unique(chunk_ids):
            idxs = np.where(chunk_ids == chunk_id)[0]
            row_ptr, col_idx, _ = self._load_chunk_arrays(int(chunk_id))
            pos = chunk_pos[idxs].astype(np.int64, copy=False)
            starts = row_ptr[pos]
            ends = row_ptr[pos + 1]
            for local_i, start, end in zip(idxs.tolist(), starts.tolist(), ends.tolist()):
                out[local_i] = np.asarray(col_idx[int(start) : int(end)], dtype=np.int64)
        return out

    def neighbors_many_with_types(self, node_indices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        nodes = np.asarray(node_indices, dtype=np.int64)
        if nodes.size == 0:
            return []
        out: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64))
            for _ in range(int(nodes.shape[0]))
        ]
        chunk_ids = self.catalog.node_chunk_ids[nodes]
        chunk_pos = self.catalog.node_chunk_pos[nodes]
        for chunk_id in np.unique(chunk_ids):
            idxs = np.where(chunk_ids == chunk_id)[0]
            row_ptr, col_idx, rel_type = self._load_chunk_arrays(int(chunk_id))
            pos = chunk_pos[idxs].astype(np.int64, copy=False)
            starts = row_ptr[pos]
            ends = row_ptr[pos + 1]
            for local_i, start, end in zip(idxs.tolist(), starts.tolist(), ends.tolist()):
                out[local_i] = (
                    np.asarray(col_idx[int(start) : int(end)], dtype=np.int64),
                    np.asarray(rel_type[int(start) : int(end)], dtype=np.int64),
                )
        return out
