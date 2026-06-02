"""Load sharded graph export and build node/edge mapping tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from project.utils.lru import LRUCache
from project.utils.logging import ProgressTicker
from project.utils.io import atomic_write_json


@dataclass
class ChunkInfo:
    chunk_id: int
    name: str
    nodes_path: Path
    edges_path: Path
    node_count: int


@dataclass
class ChunkSubgraph:
    """Center chunk plus optional cross-chunk one-hop context subgraph."""

    chunk_id: int
    chunk_name: str
    center_node_indices_global: np.ndarray
    node_indices_global: np.ndarray
    center_mask_local: np.ndarray
    message_edge_index_local: np.ndarray
    global_to_local: Dict[int, int]
    node_wids: List[str]
    node_has_image: np.ndarray
    node_is_art: np.ndarray
    node_chunk_ids: np.ndarray
    node_chunk_pos: np.ndarray


class ShardedGraphDataset:
    """Graph reader for full_graph_export sharded format."""

    def __init__(
        self,
        graph_dir: Path,
        cache_dir: Path,
        verbose: bool = True,
        sample_chunks: int = 0,
        chunk_context_mode: str = "intra_only",
        seed: int = 42,
        cross_chunk_neighbors_per_node: int = 3,
    ):
        self.graph_dir = Path(graph_dir)
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose
        self.sample_chunks = int(sample_chunks)
        if self.sample_chunks < 0:
            raise ValueError("sample_chunks must be >= 0")
        self.chunk_context_mode = str(chunk_context_mode)
        if self.chunk_context_mode not in {"intra_only", "cross_chunk_1hop"}:
            raise ValueError("chunk_context_mode must be 'intra_only' or 'cross_chunk_1hop'")
        self.seed = int(seed)
        self.cross_chunk_neighbors_per_node = max(1, int(cross_chunk_neighbors_per_node))
        self.graph_meta_dir = self.cache_dir / "graph_meta"
        self.graph_meta_dir.mkdir(parents=True, exist_ok=True)
        chunk_subgraph_dir_name = (
            "chunk_subgraphs"
            if self.chunk_context_mode == "intra_only"
            else f"chunk_subgraphs_{self.chunk_context_mode}"
        )
        self.chunk_subgraph_dir = self.cache_dir / chunk_subgraph_dir_name
        self.chunk_subgraph_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_subgraph_cache = LRUCache[int, ChunkSubgraph](8)

        self.manifest = self._load_manifest()
        self.chunks: List[ChunkInfo] = self._load_chunks()

        self.node_ids: np.ndarray
        self.node_wids: np.ndarray
        self.node_has_image: np.ndarray
        self.node_is_art: np.ndarray
        self.node_chunk_ids: np.ndarray
        self.node_chunk_pos: np.ndarray
        self.id_to_idx: np.ndarray
        self._load_or_build_node_tables()
        self.num_nodes = int(self.node_ids.shape[0])

    def _load_manifest(self) -> Dict:
        path = self.graph_dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Graph manifest not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_chunks(self) -> List[ChunkInfo]:
        chunks = []
        raw_chunks = list(self.manifest.get("chunks", []))
        if self.sample_chunks > 0:
            raw_chunks = raw_chunks[: min(self.sample_chunks, len(raw_chunks))]

        for i, c in enumerate(raw_chunks):
            name = str(c["dir_name"])
            nodes_file = self.graph_dir / str(c["nodes_file"])
            edges_file = self.graph_dir / str(c["edges_file"])
            node_count = int(c.get("node_count", 0))
            chunks.append(
                ChunkInfo(
                    chunk_id=i,
                    name=name,
                    nodes_path=nodes_file,
                    edges_path=edges_file,
                    node_count=node_count,
                )
            )
        if not chunks:
            raise RuntimeError("No chunks found in graph manifest")
        return chunks

    def _tables_exist(self) -> bool:
        required = [
            self.graph_meta_dir / "node_ids.npy",
            self.graph_meta_dir / "node_wids.npy",
            self.graph_meta_dir / "node_has_image.npy",
            self.graph_meta_dir / "node_is_art.npy",
            self.graph_meta_dir / "node_chunk_ids.npy",
            self.graph_meta_dir / "node_chunk_pos.npy",
            self.graph_meta_dir / "id_to_idx.npy",
        ]
        return all(p.exists() for p in required)

    @staticmethod
    def _has_label(labels_value, target: str) -> bool:
        target_upper = str(target).upper()
        if isinstance(labels_value, list):
            return any(str(item).upper() == target_upper for item in labels_value)
        if labels_value is None:
            return False
        return str(labels_value).upper() == target_upper

    @staticmethod
    def _has_valid_image_url(image_value) -> bool:
        if isinstance(image_value, str):
            v = image_value.strip().lower()
            return v.startswith("http://") or v.startswith("https://")
        if isinstance(image_value, list):
            has_any = False
            for item in image_value:
                if not isinstance(item, str):
                    continue
                v = item.strip().lower()
                if not v:
                    continue
                has_any = True
                if not (v.startswith("http://") or v.startswith("https://")):
                    return False
            return has_any
        return False

    def _load_or_build_node_tables(self) -> None:
        if self._tables_exist():
            self.node_ids = np.load(self.graph_meta_dir / "node_ids.npy")
            self.node_wids = np.load(
                self.graph_meta_dir / "node_wids.npy", allow_pickle=True
            )
            self.node_has_image = np.load(self.graph_meta_dir / "node_has_image.npy")
            self.node_is_art = np.load(self.graph_meta_dir / "node_is_art.npy")
            self.node_chunk_ids = np.load(self.graph_meta_dir / "node_chunk_ids.npy")
            self.node_chunk_pos = np.load(self.graph_meta_dir / "node_chunk_pos.npy")
            self.id_to_idx = np.load(self.graph_meta_dir / "id_to_idx.npy")
            return

        node_ids: List[int] = []
        node_wids: List[str] = []
        node_has_image: List[bool] = []
        node_is_art: List[bool] = []
        node_chunk_ids: List[int] = []
        node_chunk_pos: List[int] = []

        iterator = self.chunks
        if self.verbose:
            iterator = tqdm(self.chunks, desc="build node table", unit="chunk")

        for chunk in iterator:
            with chunk.nodes_path.open("r", encoding="utf-8") as f:
                nodes = json.load(f)
            pos = 0
            for node_id_text, payload in nodes.items():
                gid = int(node_id_text)
                props = payload.get("properties", {}) if isinstance(payload, dict) else {}
                labels = payload.get("labels", []) if isinstance(payload, dict) else []
                wid = props.get("wid", "")
                has_image = self._has_valid_image_url(props.get("image"))
                is_art = self._has_label(labels, "ART")
                node_ids.append(gid)
                node_wids.append(str(wid) if wid is not None else "")
                node_has_image.append(bool(has_image))
                node_is_art.append(bool(is_art))
                node_chunk_ids.append(chunk.chunk_id)
                node_chunk_pos.append(pos)
                pos += 1

        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.node_wids = np.asarray(node_wids, dtype=object)
        self.node_has_image = np.asarray(node_has_image, dtype=np.bool_)
        self.node_is_art = np.asarray(node_is_art, dtype=np.bool_)
        self.node_chunk_ids = np.asarray(node_chunk_ids, dtype=np.int32)
        self.node_chunk_pos = np.asarray(node_chunk_pos, dtype=np.int32)

        max_node_id = int(self.node_ids.max())
        self.id_to_idx = np.full(max_node_id + 1, -1, dtype=np.int64)
        self.id_to_idx[self.node_ids] = np.arange(self.node_ids.shape[0], dtype=np.int64)

        np.save(self.graph_meta_dir / "node_ids.npy", self.node_ids)
        np.save(self.graph_meta_dir / "node_wids.npy", self.node_wids, allow_pickle=True)
        np.save(self.graph_meta_dir / "node_has_image.npy", self.node_has_image)
        np.save(self.graph_meta_dir / "node_is_art.npy", self.node_is_art)
        np.save(self.graph_meta_dir / "node_chunk_ids.npy", self.node_chunk_ids)
        np.save(self.graph_meta_dir / "node_chunk_pos.npy", self.node_chunk_pos)
        np.save(self.graph_meta_dir / "id_to_idx.npy", self.id_to_idx)

    def global_id_to_idx(self, global_ids: np.ndarray) -> np.ndarray:
        gids = np.asarray(global_ids, dtype=np.int64)
        ok = (gids >= 0) & (gids < self.id_to_idx.shape[0])
        out = np.full_like(gids, -1)
        out[ok] = self.id_to_idx[gids[ok]]
        return out

    def get_node_wids(self, node_indices: np.ndarray) -> List[str]:
        idx = np.asarray(node_indices, dtype=np.int64)
        return [str(x) for x in self.node_wids[idx]]

    def get_node_has_image(self, node_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(node_indices, dtype=np.int64)
        return self.node_has_image[idx]

    def get_node_is_art(self, node_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(node_indices, dtype=np.int64)
        return self.node_is_art[idx]

    def iter_edges_for_chunk(self, chunk_id: int) -> Iterator[Tuple[int, int]]:
        chunk = self.chunks[chunk_id]
        with chunk.edges_path.open("r", encoding="utf-8") as f:
            edges = json.load(f)
        for e in edges:
            src_gid = int(e["src"])
            dst_gid = int(e["dst"])
            if src_gid < 0 or dst_gid < 0:
                continue
            if src_gid >= self.id_to_idx.shape[0] or dst_gid >= self.id_to_idx.shape[0]:
                continue
            src_idx = int(self.id_to_idx[src_gid])
            dst_idx = int(self.id_to_idx[dst_gid])
            if src_idx < 0 or dst_idx < 0:
                continue
            yield src_idx, dst_idx

    def load_edges_for_chunk(self, chunk_id: int) -> np.ndarray:
        srcs: List[int] = []
        dsts: List[int] = []
        for src, dst in self.iter_edges_for_chunk(chunk_id):
            srcs.append(src)
            dsts.append(dst)
        if not srcs:
            return np.empty((2, 0), dtype=np.int64)
        return np.vstack(
            [
                np.asarray(srcs, dtype=np.int64),
                np.asarray(dsts, dtype=np.int64),
            ]
        )

    def load_all_edges(
        self,
        progress_every_chunks: int = 5,
        progress_every_sec: float = 30.0,
    ) -> np.ndarray:
        parts: List[np.ndarray] = []
        iterator = range(len(self.chunks))
        ticker = ProgressTicker(
            label="load all edges",
            every_sec=progress_every_sec,
        )
        if self.verbose:
            iterator = tqdm(iterator, desc="load all edges", unit="chunk")
        for chunk_id in iterator:
            arr = self.load_edges_for_chunk(chunk_id)
            if arr.shape[1] > 0:
                parts.append(arr)
            chunk_no = int(chunk_id) + 1
            if progress_every_chunks > 0 and (
                chunk_no == len(self.chunks) or chunk_no % progress_every_chunks == 0
            ):
                total_edges = int(sum(part.shape[1] for part in parts))
                ticker.force(f"chunks={chunk_no}/{len(self.chunks)} edges_so_far={total_edges}")
            else:
                ticker.maybe(f"chunks={chunk_no}/{len(self.chunks)}")
        if not parts:
            return np.empty((2, 0), dtype=np.int64)
        return np.concatenate(parts, axis=1)

    def chunk_name(self, chunk_id: int) -> str:
        return self.chunks[chunk_id].name

    def chunk_size(self, chunk_id: int) -> int:
        return int(self.chunks[chunk_id].node_count)

    def chunk_node_indices(self, chunk_id: int) -> np.ndarray:
        idx = np.where(self.node_chunk_ids == int(chunk_id))[0]
        if idx.size == 0:
            return idx.astype(np.int64)
        order = np.argsort(self.node_chunk_pos[idx])
        return idx[order].astype(np.int64, copy=False)

    def _chunk_subgraph_cache_dir(self, chunk_id: int) -> Path:
        chunk_name = self.chunk_name(chunk_id)
        cdir = self.chunk_subgraph_dir / chunk_name
        cdir.mkdir(parents=True, exist_ok=True)
        return cdir

    def _chunk_subgraph_cache_paths(self, chunk_id: int) -> Dict[str, Path]:
        cdir = self._chunk_subgraph_cache_dir(chunk_id)
        return {
            "center_nodes": cdir / "center_node_indices_global.npy",
            "context_nodes": cdir / "context_node_indices_global.npy",
            "center_mask": cdir / "center_mask_local.npy",
            "message_edges": cdir / "message_edge_index_local.npy",
        }

    def _chunk_subgraph_cache_exists(self, chunk_id: int) -> bool:
        paths = self._chunk_subgraph_cache_paths(chunk_id)
        return all(path.exists() for path in paths.values())

    def _sort_external_nodes(self, node_indices: np.ndarray) -> np.ndarray:
        if node_indices.size == 0:
            return node_indices.astype(np.int64, copy=False)
        chunk_ids = self.node_chunk_ids[node_indices].astype(np.int64, copy=False)
        chunk_pos = self.node_chunk_pos[node_indices].astype(np.int64, copy=False)
        order = np.lexsort((node_indices.astype(np.int64, copy=False), chunk_pos, chunk_ids))
        return node_indices[order].astype(np.int64, copy=False)

    def _stable_neighbor_score(self, center_idx: int, neighbor_idx: int) -> int:
        x = (
            (np.uint64(int(center_idx)) + np.uint64(0x9E3779B97F4A7C15))
            ^ (np.uint64(int(neighbor_idx)) * np.uint64(0xBF58476D1CE4E5B9))
            ^ np.uint64(self.seed)
        )
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        return int(x)

    @staticmethod
    def _update_topk_candidates(
        neighbor_buf: np.ndarray,
        score_buf: np.ndarray,
        row: int,
        neighbor_idx: int,
        score: int,
    ) -> None:
        row_neighbors = neighbor_buf[row]
        row_scores = score_buf[row]
        for col in range(row_neighbors.shape[0]):
            if int(row_neighbors[col]) == int(neighbor_idx):
                if score < int(row_scores[col]):
                    row_scores[col] = np.uint64(score)
                return
        for col in range(row_neighbors.shape[0]):
            if int(row_neighbors[col]) < 0:
                row_neighbors[col] = int(neighbor_idx)
                row_scores[col] = np.uint64(score)
                return
        worst_col = int(np.argmax(row_scores))
        if score < int(row_scores[worst_col]):
            row_neighbors[worst_col] = int(neighbor_idx)
            row_scores[worst_col] = np.uint64(score)

    def _finalize_selected_cross_neighbors(
        self,
        art_neighbors_by_chunk: List[np.ndarray],
        art_scores_by_chunk: List[np.ndarray],
        non_art_neighbors_by_chunk: List[np.ndarray],
        non_art_scores_by_chunk: List[np.ndarray],
    ) -> List[np.ndarray]:
        limit = int(self.cross_chunk_neighbors_per_node)
        selected_by_chunk: List[np.ndarray] = []
        for chunk_id in range(self.chunk_count()):
            center_count = int(art_neighbors_by_chunk[int(chunk_id)].shape[0])
            selected = np.full((center_count, limit), -1, dtype=np.int64)
            art_neighbors = art_neighbors_by_chunk[int(chunk_id)]
            art_scores = art_scores_by_chunk[int(chunk_id)]
            non_neighbors = non_art_neighbors_by_chunk[int(chunk_id)]
            non_scores = non_art_scores_by_chunk[int(chunk_id)]
            for row in range(center_count):
                chosen: List[int] = []
                art_valid = np.where(art_neighbors[row] >= 0)[0]
                if art_valid.size > 0:
                    art_order = art_valid[np.argsort(art_scores[row, art_valid], kind="stable")]
                    for col in art_order.tolist():
                        chosen.append(int(art_neighbors[row, col]))
                        if len(chosen) >= limit:
                            break
                if len(chosen) < limit:
                    non_valid = np.where(non_neighbors[row] >= 0)[0]
                    if non_valid.size > 0:
                        non_order = non_valid[np.argsort(non_scores[row, non_valid], kind="stable")]
                        for col in non_order.tolist():
                            candidate = int(non_neighbors[row, col])
                            if candidate in chosen:
                                continue
                            chosen.append(candidate)
                            if len(chosen) >= limit:
                                break
                if chosen:
                    selected[row, : len(chosen)] = np.asarray(chosen, dtype=np.int64)
            selected_by_chunk.append(selected)
        return selected_by_chunk

    def _save_chunk_subgraph_cache(
        self,
        chunk_id: int,
        center_nodes: np.ndarray,
        context_nodes: np.ndarray,
        center_mask: np.ndarray,
        message_edges: np.ndarray,
    ) -> None:
        paths = self._chunk_subgraph_cache_paths(chunk_id)
        np.save(paths["center_nodes"], np.asarray(center_nodes, dtype=np.int64))
        np.save(paths["context_nodes"], np.asarray(context_nodes, dtype=np.int64))
        np.save(paths["center_mask"], np.asarray(center_mask, dtype=np.bool_))
        np.save(paths["message_edges"], np.asarray(message_edges, dtype=np.int64))

    def _cross_chunk_state_path(self) -> Path:
        return self.chunk_subgraph_dir / "state.json"

    def _cross_chunk_cache_complete(self) -> bool:
        state_path = self._cross_chunk_state_path()
        if not state_path.exists():
            return False
        try:
            with state_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False
        if payload.get("status") != "completed":
            return False
        if payload.get("chunk_context_mode") != self.chunk_context_mode:
            return False
        if payload.get("format") != "chunk_subgraph_cross_chunk_1hop_v2":
            return False
        if int(payload.get("chunk_count", -1)) != int(self.chunk_count()):
            return False
        if int(payload.get("sample_chunks", -1)) != int(self.sample_chunks):
            return False
        if int(payload.get("cross_chunk_neighbors_per_node", -1)) != int(
            self.cross_chunk_neighbors_per_node
        ):
            return False
        if int(payload.get("seed", -1)) != int(self.seed):
            return False
        for chunk_id in range(self.chunk_count()):
            if not self._chunk_subgraph_cache_exists(chunk_id):
                return False
        return True

    def _build_chunk_subgraph_cache_intra_only(self, chunk_id: int) -> None:
        center_nodes = self.chunk_node_indices(chunk_id)
        n_local = int(center_nodes.shape[0])
        src_local: List[int] = []
        dst_local: List[int] = []

        iterator = self.iter_edges_for_chunk(chunk_id)
        for src_idx, dst_idx in iterator:
            if int(self.node_chunk_ids[src_idx]) != int(chunk_id):
                continue
            if int(self.node_chunk_ids[dst_idx]) != int(chunk_id):
                continue
            src_pos = int(self.node_chunk_pos[src_idx])
            dst_pos = int(self.node_chunk_pos[dst_idx])
            if src_pos < 0 or src_pos >= n_local or dst_pos < 0 or dst_pos >= n_local:
                continue
            src_local.append(src_pos)
            dst_local.append(dst_pos)

        if src_local:
            message_edges = np.vstack(
                [
                    np.asarray(src_local, dtype=np.int64),
                    np.asarray(dst_local, dtype=np.int64),
                ]
            )
        else:
            message_edges = np.empty((2, 0), dtype=np.int64)

        center_mask = np.ones(center_nodes.shape[0], dtype=np.bool_)
        self._save_chunk_subgraph_cache(
            chunk_id=chunk_id,
            center_nodes=center_nodes,
            context_nodes=center_nodes,
            center_mask=center_mask,
            message_edges=message_edges,
        )

    def _build_all_cross_chunk_subgraph_caches(
        self,
        progress_every_chunks: int = 5,
        progress_every_sec: float = 30.0,
    ) -> None:
        if self._cross_chunk_cache_complete():
            return

        limit = int(self.cross_chunk_neighbors_per_node)
        art_neighbors_by_chunk: List[np.ndarray] = []
        art_scores_by_chunk: List[np.ndarray] = []
        non_art_neighbors_by_chunk: List[np.ndarray] = []
        non_art_scores_by_chunk: List[np.ndarray] = []
        internal_edge_counts = np.zeros(self.chunk_count(), dtype=np.int64)
        max_score = np.uint64(np.iinfo(np.uint64).max)
        for chunk in self.chunks:
            shape = (int(self.chunk_node_indices(int(chunk.chunk_id)).shape[0]), limit)
            art_neighbors_by_chunk.append(np.full(shape, -1, dtype=np.int64))
            art_scores_by_chunk.append(np.full(shape, max_score, dtype=np.uint64))
            non_art_neighbors_by_chunk.append(np.full(shape, -1, dtype=np.int64))
            non_art_scores_by_chunk.append(np.full(shape, max_score, dtype=np.uint64))

        ticker = ProgressTicker(
            label="build cross-chunk subgraph cache",
            every_sec=progress_every_sec,
        )

        for chunk_id in range(self.chunk_count()):
            for src_idx, dst_idx in self.iter_edges_for_chunk(chunk_id):
                src_chunk = int(self.node_chunk_ids[src_idx])
                dst_chunk = int(self.node_chunk_ids[dst_idx])
                if src_chunk == dst_chunk:
                    internal_edge_counts[src_chunk] += 1
                else:
                    src_row = int(self.node_chunk_pos[src_idx])
                    dst_row = int(self.node_chunk_pos[dst_idx])
                    dst_is_art = bool(self.node_is_art[dst_idx])
                    src_is_art = bool(self.node_is_art[src_idx])
                    dst_score = self._stable_neighbor_score(int(src_idx), int(dst_idx))
                    src_score = self._stable_neighbor_score(int(dst_idx), int(src_idx))
                    if dst_is_art:
                        self._update_topk_candidates(
                            art_neighbors_by_chunk[src_chunk],
                            art_scores_by_chunk[src_chunk],
                            src_row,
                            int(dst_idx),
                            dst_score,
                        )
                    else:
                        self._update_topk_candidates(
                            non_art_neighbors_by_chunk[src_chunk],
                            non_art_scores_by_chunk[src_chunk],
                            src_row,
                            int(dst_idx),
                            dst_score,
                        )
                    if src_is_art:
                        self._update_topk_candidates(
                            art_neighbors_by_chunk[dst_chunk],
                            art_scores_by_chunk[dst_chunk],
                            dst_row,
                            int(src_idx),
                            src_score,
                        )
                    else:
                        self._update_topk_candidates(
                            non_art_neighbors_by_chunk[dst_chunk],
                            non_art_scores_by_chunk[dst_chunk],
                            dst_row,
                            int(src_idx),
                            src_score,
                        )
            chunk_no = int(chunk_id) + 1
            if progress_every_chunks > 0 and (
                chunk_no == self.chunk_count() or chunk_no % progress_every_chunks == 0
            ):
                ticker.force(f"pass=1 chunks={chunk_no}/{self.chunk_count()}")
            else:
                ticker.maybe(f"pass=1 chunks={chunk_no}/{self.chunk_count()}")

        selected_cross_neighbors = self._finalize_selected_cross_neighbors(
            art_neighbors_by_chunk,
            art_scores_by_chunk,
            non_art_neighbors_by_chunk,
            non_art_scores_by_chunk,
        )

        context_nodes_by_chunk: Dict[int, np.ndarray] = {}
        center_nodes_by_chunk: Dict[int, np.ndarray] = {}
        center_mask_by_chunk: Dict[int, np.ndarray] = {}
        message_edge_counts = internal_edge_counts.copy()
        global_to_local_by_chunk: Dict[int, Dict[int, int]] = {}

        for chunk_id in range(self.chunk_count()):
            center_nodes = self.chunk_node_indices(chunk_id)
            ext = selected_cross_neighbors[int(chunk_id)].reshape(-1)
            ext = ext[ext >= 0]
            if ext.size > 0:
                ext = ext[self.node_chunk_ids[ext] != int(chunk_id)]
                ext = np.unique(ext)
                ext = self._sort_external_nodes(ext)
            context_nodes = (
                np.concatenate([center_nodes, ext], axis=0).astype(np.int64, copy=False)
                if ext.size > 0
                else center_nodes.astype(np.int64, copy=False)
            )
            center_mask = np.zeros(context_nodes.shape[0], dtype=np.bool_)
            center_mask[: center_nodes.shape[0]] = True
            center_nodes_by_chunk[int(chunk_id)] = center_nodes.astype(np.int64, copy=False)
            context_nodes_by_chunk[int(chunk_id)] = context_nodes
            center_mask_by_chunk[int(chunk_id)] = center_mask
            global_to_local_by_chunk[int(chunk_id)] = {
                int(gidx): int(local_idx) for local_idx, gidx in enumerate(context_nodes.tolist())
            }

        for chunk_id in range(self.chunk_count()):
            for src_idx, dst_idx in self.iter_edges_for_chunk(chunk_id):
                src_chunk = int(self.node_chunk_ids[src_idx])
                dst_chunk = int(self.node_chunk_ids[dst_idx])
                if src_chunk == dst_chunk:
                    message_edge_counts[src_chunk] += 0
                    continue
                src_selected = selected_cross_neighbors[src_chunk][
                    int(self.node_chunk_pos[src_idx])
                ]
                if np.any(src_selected == int(dst_idx)):
                    message_edge_counts[src_chunk] += 1
                dst_selected = selected_cross_neighbors[dst_chunk][
                    int(self.node_chunk_pos[dst_idx])
                ]
                if np.any(dst_selected == int(src_idx)):
                    message_edge_counts[dst_chunk] += 1
            chunk_no = int(chunk_id) + 1
            if progress_every_chunks > 0 and (
                chunk_no == self.chunk_count() or chunk_no % progress_every_chunks == 0
            ):
                ticker.force(f"pass=2 chunks={chunk_no}/{self.chunk_count()}")
            else:
                ticker.maybe(f"pass=2 chunks={chunk_no}/{self.chunk_count()}")

        edge_buffers: Dict[int, np.ndarray] = {}
        edge_write_ptr = np.zeros(self.chunk_count(), dtype=np.int64)
        for chunk_id in range(self.chunk_count()):
            edge_buffers[int(chunk_id)] = np.empty(
                (2, int(message_edge_counts[int(chunk_id)])),
                dtype=np.int64,
            )

        for chunk_id in range(self.chunk_count()):
            for src_idx, dst_idx in self.iter_edges_for_chunk(chunk_id):
                src_chunk = int(self.node_chunk_ids[src_idx])
                dst_chunk = int(self.node_chunk_ids[dst_idx])
                if src_chunk == dst_chunk:
                    mapping = global_to_local_by_chunk[src_chunk]
                    pos = int(edge_write_ptr[src_chunk])
                    edge_buffers[src_chunk][0, pos] = mapping[int(src_idx)]
                    edge_buffers[src_chunk][1, pos] = mapping[int(dst_idx)]
                    edge_write_ptr[src_chunk] += 1
                    continue

                src_selected = selected_cross_neighbors[src_chunk][
                    int(self.node_chunk_pos[src_idx])
                ]
                if np.any(src_selected == int(dst_idx)):
                    mapping = global_to_local_by_chunk[src_chunk]
                    pos = int(edge_write_ptr[src_chunk])
                    edge_buffers[src_chunk][0, pos] = mapping[int(src_idx)]
                    edge_buffers[src_chunk][1, pos] = mapping[int(dst_idx)]
                    edge_write_ptr[src_chunk] += 1

                dst_selected = selected_cross_neighbors[dst_chunk][
                    int(self.node_chunk_pos[dst_idx])
                ]
                if np.any(dst_selected == int(src_idx)):
                    mapping = global_to_local_by_chunk[dst_chunk]
                    pos = int(edge_write_ptr[dst_chunk])
                    edge_buffers[dst_chunk][0, pos] = mapping[int(src_idx)]
                    edge_buffers[dst_chunk][1, pos] = mapping[int(dst_idx)]
                    edge_write_ptr[dst_chunk] += 1
            chunk_no = int(chunk_id) + 1
            if progress_every_chunks > 0 and (
                chunk_no == self.chunk_count() or chunk_no % progress_every_chunks == 0
            ):
                ticker.force(f"pass=3 chunks={chunk_no}/{self.chunk_count()}")
            else:
                ticker.maybe(f"pass=3 chunks={chunk_no}/{self.chunk_count()}")

        for chunk_id in range(self.chunk_count()):
            count = int(edge_write_ptr[int(chunk_id)])
            edge_index = edge_buffers[int(chunk_id)][:, :count]
            if edge_index.shape[1] > 0:
                hashed = edge_index[0].astype(np.uint64) * np.uint64(
                    context_nodes_by_chunk[int(chunk_id)].shape[0]
                ) + edge_index[1].astype(np.uint64)
                _, uniq_idx = np.unique(hashed, return_index=True)
                uniq_idx = np.sort(uniq_idx.astype(np.int64, copy=False))
                edge_index = edge_index[:, uniq_idx]
            self._save_chunk_subgraph_cache(
                chunk_id=int(chunk_id),
                center_nodes=center_nodes_by_chunk[int(chunk_id)],
                context_nodes=context_nodes_by_chunk[int(chunk_id)],
                center_mask=center_mask_by_chunk[int(chunk_id)],
                message_edges=edge_index,
            )

        atomic_write_json(
            self._cross_chunk_state_path(),
            {
                "status": "completed",
                "format": "chunk_subgraph_cross_chunk_1hop_v2",
                "chunk_context_mode": self.chunk_context_mode,
                "chunk_count": int(self.chunk_count()),
                "sample_chunks": int(self.sample_chunks),
                "seed": int(self.seed),
                "cross_chunk_neighbors_per_node": int(self.cross_chunk_neighbors_per_node),
            },
        )

    def load_chunk_intra_edge_index(self, chunk_id: int) -> np.ndarray:
        return self.load_chunk_subgraph(int(chunk_id)).message_edge_index_local

    def load_chunk_subgraph(self, chunk_id: int) -> ChunkSubgraph:
        cached = self.chunk_subgraph_cache.get(int(chunk_id))
        if cached is not None:
            return cached

        if self.chunk_context_mode == "cross_chunk_1hop":
            if not self._cross_chunk_cache_complete():
                self._build_all_cross_chunk_subgraph_caches()
        elif not self._chunk_subgraph_cache_exists(int(chunk_id)):
            self._build_chunk_subgraph_cache_intra_only(int(chunk_id))

        paths = self._chunk_subgraph_cache_paths(int(chunk_id))
        center_nodes = np.asarray(np.load(paths["center_nodes"], mmap_mode="r"), dtype=np.int64)
        context_nodes = np.asarray(np.load(paths["context_nodes"], mmap_mode="r"), dtype=np.int64)
        center_mask = np.asarray(np.load(paths["center_mask"], mmap_mode="r"), dtype=np.bool_)
        message_edges = np.asarray(np.load(paths["message_edges"], mmap_mode="r"), dtype=np.int64)
        global_to_local = {int(gid): i for i, gid in enumerate(context_nodes.tolist())}
        sub = ChunkSubgraph(
            chunk_id=int(chunk_id),
            chunk_name=self.chunk_name(chunk_id),
            center_node_indices_global=center_nodes,
            node_indices_global=context_nodes,
            center_mask_local=center_mask,
            message_edge_index_local=message_edges,
            global_to_local=global_to_local,
            node_wids=self.get_node_wids(context_nodes),
            node_has_image=self.get_node_has_image(context_nodes).astype(np.bool_),
            node_is_art=self.get_node_is_art(context_nodes).astype(np.bool_),
            node_chunk_ids=self.node_chunk_ids[context_nodes].astype(np.int32, copy=False),
            node_chunk_pos=self.node_chunk_pos[context_nodes].astype(np.int32, copy=False),
        )
        self.chunk_subgraph_cache.put(int(chunk_id), sub)
        return sub

    def node_count(self) -> int:
        return self.num_nodes

    def chunk_count(self) -> int:
        return len(self.chunks)

    def selected_chunk_names(self) -> List[str]:
        return [c.name for c in self.chunks]
