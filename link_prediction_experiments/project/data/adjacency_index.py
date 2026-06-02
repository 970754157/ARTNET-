"""Disk adjacency index with resume and chunk-level LRU cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from project.data.sharded_graph import ShardedGraphDataset
from project.utils.io import atomic_write_json
from project.utils.lru import LRUCache


class DiskAdjacencyIndex:
    """Chunk-aligned adjacency index backed by .npy files."""

    def __init__(
        self,
        graph: ShardedGraphDataset,
        index_dir: Path,
        lru_chunks: int = 8,
        verbose: bool = True,
        source_edges: Optional[np.ndarray] = None,
        edge_source_name: str = "graph",
    ):
        self.graph = graph
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.index_dir / "index_meta.json"
        self.state_path = self.index_dir / "state.json"
        self.verbose = verbose
        self.cache = LRUCache[int, Tuple[np.ndarray, np.ndarray]](lru_chunks)
        self.source_edges = (
            np.asarray(source_edges, dtype=np.int64) if source_edges is not None else None
        )
        self.edge_source_name = str(edge_source_name)
        self._provided_directed_edges: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        self._chunk_to_nodes: List[np.ndarray] = []
        for chunk_id in range(self.graph.chunk_count()):
            idx = np.where(self.graph.node_chunk_ids == chunk_id)[0]
            order = np.argsort(self.graph.node_chunk_pos[idx])
            self._chunk_to_nodes.append(idx[order].astype(np.int64))
        if self.source_edges is not None:
            self._prepare_provided_edges()

    def _chunk_dir(self, chunk_id: int) -> Path:
        return self.index_dir / f"chunk_{self.graph.chunk_name(chunk_id)}"

    def _chunk_files_exist(self, chunk_id: int) -> bool:
        cdir = self._chunk_dir(chunk_id)
        return (
            (cdir / "row_ptr.npy").exists()
            and (cdir / "col_idx.npy").exists()
            and (cdir / "node_ids.npy").exists()
        )

    def _write_meta(self) -> None:
        payload = {
            "format": "chunk_csr_v2",
            "num_nodes": int(self.graph.num_nodes),
            "chunk_count": int(self.graph.chunk_count()),
            "directed": True,
            "edge_source": self.edge_source_name,
        }
        atomic_write_json(self.meta_path, payload)

    def _prepare_provided_edges(self) -> None:
        if self.source_edges is None:
            return
        edges = np.asarray(self.source_edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("source_edges must have shape [2, num_edges]")
        if edges.shape[1] == 0:
            chunk_offsets = np.zeros(self.graph.chunk_count() + 1, dtype=np.int64)
            empty = np.empty((0,), dtype=np.int64)
            self._provided_directed_edges = (empty, empty, chunk_offsets)
            return

        src = edges[0].astype(np.int64, copy=False)
        dst = edges[1].astype(np.int64, copy=False)
        non_self = src != dst
        directed_src = np.concatenate([src, dst[non_self]], axis=0).astype(np.int64, copy=False)
        directed_dst = np.concatenate([dst, src[non_self]], axis=0).astype(np.int64, copy=False)
        src_chunk_ids = self.graph.node_chunk_ids[directed_src].astype(np.int32, copy=False)
        order = np.argsort(src_chunk_ids, kind="stable")
        directed_src = directed_src[order]
        directed_dst = directed_dst[order]
        src_chunk_ids = src_chunk_ids[order]
        chunk_offsets = np.searchsorted(
            src_chunk_ids,
            np.arange(self.graph.chunk_count() + 1, dtype=np.int32),
            side="left",
        ).astype(np.int64, copy=False)
        self._provided_directed_edges = (directed_src, directed_dst, chunk_offsets)

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
                and meta.get("format") == "chunk_csr_v2"
                and meta.get("edge_source") == self.edge_source_name
            ):
                if self.source_edges is not None:
                    self._provided_directed_edges = None
                return
        self._build()

    def _clear_index_dir(self) -> None:
        for p in self.index_dir.glob("*"):
            if p.is_dir():
                for c in p.rglob("*"):
                    if c.is_file():
                        c.unlink()
                for c in sorted(p.rglob("*"), reverse=True):
                    if c.is_dir():
                        c.rmdir()
                p.rmdir()
            else:
                p.unlink()

    def _build(self) -> None:
        self._write_meta()
        state = self._load_state()
        completed = set(state.get("completed_chunks", []))
        state["status"] = "in_progress"
        self._save_state(state)

        iterator = range(self.graph.chunk_count())
        if self.verbose:
            iterator = tqdm(iterator, desc="build adjacency index", unit="chunk")

        for chunk_id in iterator:
            chunk_name = self.graph.chunk_name(chunk_id)
            if chunk_name in completed and self._chunk_files_exist(chunk_id):
                continue
            self._build_one_chunk(chunk_id)
            completed.add(chunk_name)
            state["completed_chunks"] = sorted(completed)
            state["status"] = "in_progress"
            self._save_state(state)

        state["status"] = "completed"
        state["completed_chunks"] = sorted(completed)
        self._save_state(state)
        if self.source_edges is not None:
            self._provided_directed_edges = None
            self.source_edges = None

    def _build_one_chunk(self, chunk_id: int) -> None:
        node_ids = self._chunk_to_nodes[chunk_id]
        n_local = int(node_ids.shape[0])
        if self._provided_directed_edges is not None:
            directed_src, directed_dst, chunk_offsets = self._provided_directed_edges
            start = int(chunk_offsets[chunk_id])
            end = int(chunk_offsets[chunk_id + 1])
            if end > start:
                src_slice = directed_src[start:end]
                dst_slice = directed_dst[start:end]
                src_pos = self.graph.node_chunk_pos[src_slice].astype(np.int64, copy=False)
                valid = (src_pos >= 0) & (src_pos < n_local)
                src_pos = src_pos[valid]
                dst_slice = dst_slice[valid]
                if src_pos.shape[0] > 0:
                    order = np.argsort(src_pos, kind="stable")
                    src_pos = src_pos[order]
                    col_idx = dst_slice[order].astype(np.int64, copy=False)
                    counts = np.bincount(src_pos, minlength=n_local).astype(np.int64, copy=False)
                else:
                    col_idx = np.empty((0,), dtype=np.int64)
                    counts = np.zeros((n_local,), dtype=np.int64)
            else:
                col_idx = np.empty((0,), dtype=np.int64)
                counts = np.zeros((n_local,), dtype=np.int64)
            row_ptr = np.zeros(n_local + 1, dtype=np.int64)
            row_ptr[1:] = np.cumsum(counts)
        else:
            neighbors: List[List[int]] = [[] for _ in range(n_local)]

            for src_idx, dst_idx in self.graph.iter_edges_for_chunk(chunk_id):
                src_chunk = int(self.graph.node_chunk_ids[src_idx])
                if src_chunk != chunk_id:
                    continue
                src_pos = int(self.graph.node_chunk_pos[src_idx])
                if src_pos < 0 or src_pos >= n_local:
                    continue
                neighbors[src_pos].append(int(dst_idx))

            lengths = np.fromiter((len(x) for x in neighbors), dtype=np.int64, count=n_local)
            row_ptr = np.zeros(n_local + 1, dtype=np.int64)
            row_ptr[1:] = np.cumsum(lengths)
            col_idx = np.empty(int(row_ptr[-1]), dtype=np.int64)

            offset = 0
            for lst in neighbors:
                ln = len(lst)
                if ln > 0:
                    col_idx[offset : offset + ln] = np.asarray(lst, dtype=np.int64)
                offset += ln

        cdir = self._chunk_dir(chunk_id)
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(cdir / "row_ptr.npy", row_ptr)
        np.save(cdir / "col_idx.npy", col_idx)
        np.save(cdir / "node_ids.npy", node_ids)

    def _load_chunk_arrays(self, chunk_id: int) -> Tuple[np.ndarray, np.ndarray]:
        cached = self.cache.get(chunk_id)
        if cached is not None:
            return cached
        cdir = self._chunk_dir(chunk_id)
        row_ptr = np.load(cdir / "row_ptr.npy", mmap_mode="r")
        col_idx = np.load(cdir / "col_idx.npy", mmap_mode="r")
        self.cache.put(chunk_id, (row_ptr, col_idx))
        return row_ptr, col_idx

    def neighbors(self, node_idx: int) -> np.ndarray:
        chunk_id = int(self.graph.node_chunk_ids[node_idx])
        chunk_pos = int(self.graph.node_chunk_pos[node_idx])
        row_ptr, col_idx = self._load_chunk_arrays(chunk_id)
        start = int(row_ptr[chunk_pos])
        end = int(row_ptr[chunk_pos + 1])
        return np.asarray(col_idx[start:end], dtype=np.int64)

    def neighbors_many(self, node_indices: np.ndarray) -> List[np.ndarray]:
        nodes = np.asarray(node_indices, dtype=np.int64)
        if nodes.size == 0:
            return []
        out: List[np.ndarray] = [np.empty((0,), dtype=np.int64) for _ in range(int(nodes.shape[0]))]
        chunk_ids = self.graph.node_chunk_ids[nodes]
        chunk_pos = self.graph.node_chunk_pos[nodes]
        for chunk_id in np.unique(chunk_ids):
            idxs = np.where(chunk_ids == chunk_id)[0]
            row_ptr, col_idx = self._load_chunk_arrays(int(chunk_id))
            pos = chunk_pos[idxs].astype(np.int64)
            starts = row_ptr[pos]
            ends = row_ptr[pos + 1]
            for local_i, start, end in zip(idxs.tolist(), starts.tolist(), ends.tolist()):
                out[local_i] = np.asarray(col_idx[int(start) : int(end)], dtype=np.int64)
        return out
