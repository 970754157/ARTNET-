"""Edge split helpers for neighbor-sample and chunk-full training modes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from project.data.sharded_graph import ShardedGraphDataset
from project.utils.logging import log_event


def _edge_hash(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    src64 = src.astype(np.uint64, copy=False)
    dst64 = dst.astype(np.uint64, copy=False)
    return src64 * np.uint64(num_nodes) + dst64


def _canonicalize_undirected_edges(edges: np.ndarray) -> np.ndarray:
    if edges.shape[1] == 0:
        return edges.astype(np.int64, copy=False)
    src = edges[0].astype(np.int64, copy=False)
    dst = edges[1].astype(np.int64, copy=False)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    return np.vstack([lo, hi]).astype(np.int64, copy=False)


def _undirected_edge_hash(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    lo = np.minimum(src.astype(np.int64, copy=False), dst.astype(np.int64, copy=False))
    hi = np.maximum(src.astype(np.int64, copy=False), dst.astype(np.int64, copy=False))
    lo64 = lo.astype(np.uint64, copy=False)
    hi64 = hi.astype(np.uint64, copy=False)
    return lo64 * np.uint64(num_nodes) + hi64


def _deduplicate_undirected_edges(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    canonical = _canonicalize_undirected_edges(edges)
    if canonical.shape[1] <= 1:
        return canonical.astype(np.int64, copy=False)
    hashed = _undirected_edge_hash(canonical[0], canonical[1], num_nodes)
    _, uniq_idx = np.unique(hashed, return_index=True)
    uniq_idx = np.sort(uniq_idx.astype(np.int64, copy=False))
    return canonical[:, uniq_idx].astype(np.int64, copy=False)


def _split_overlap_counts(
    train_edges: np.ndarray,
    val_edges: np.ndarray,
    test_edges: np.ndarray,
    num_nodes: int,
) -> Dict[str, int]:
    train_hash = _undirected_edge_hash(train_edges[0], train_edges[1], num_nodes)
    val_hash = _undirected_edge_hash(val_edges[0], val_edges[1], num_nodes)
    test_hash = _undirected_edge_hash(test_edges[0], test_edges[1], num_nodes)
    return {
        "train_val_overlap": int(np.intersect1d(train_hash, val_hash).shape[0]),
        "train_test_overlap": int(np.intersect1d(train_hash, test_hash).shape[0]),
        "val_test_overlap": int(np.intersect1d(val_hash, test_hash).shape[0]),
    }


def _log_split_stats(split_path: Path, payload: Dict[str, object], loaded: bool) -> None:
    prefix = "split cache loaded" if loaded else "split cache written"
    overlap = payload.get("overlap_counts", {})
    log_event(
        "[STAGE]",
        (
            f"{prefix}: {split_path} "
            f"raw_edges={int(payload.get('raw_edge_count', 0))} "
            f"unique_edges={int(payload.get('unique_edge_count', 0))} "
            f"train_edges={int(payload.get('train_edge_count', 0))} "
            f"val_edges={int(payload.get('val_edge_count', 0))} "
            f"test_edges={int(payload.get('test_edge_count', 0))} "
            f"train_val_overlap={int(overlap.get('train_val_overlap', 0))} "
            f"train_test_overlap={int(overlap.get('train_test_overlap', 0))} "
            f"val_test_overlap={int(overlap.get('val_test_overlap', 0))}"
        ),
    )


def create_or_load_splits(
    graph: ShardedGraphDataset,
    split_path: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    force_rebuild: bool = False,
    progress_every_chunks: int = 5,
    progress_every_sec: float = 30.0,
) -> Dict[str, torch.Tensor]:
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    expected_format = "neighbor_sample_undirected_no_leak_v1"

    if split_path.exists() and not force_rebuild:
        cached = torch.load(split_path, map_location="cpu")
        overlap = cached.get("overlap_counts", {})
        if cached.get("format") == expected_format and not any(int(v) > 0 for v in overlap.values()):
            _log_split_stats(split_path, cached, loaded=True)
            return cached
        log_event("[STAGE]", f"split cache stale, rebuilding: {split_path}")

    log_event("[STAGE]", f"building edge split cache: {split_path}")
    edges = graph.load_all_edges(
        progress_every_chunks=progress_every_chunks,
        progress_every_sec=progress_every_sec,
    )
    raw_edge_count = int(edges.shape[1])
    unique_edges = _deduplicate_undirected_edges(edges, graph.num_nodes)
    num_edges = int(unique_edges.shape[1])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_edges)

    n_train = int(num_edges * train_ratio)
    n_val = int(num_edges * val_ratio)
    n_test = num_edges - n_train - n_val
    if n_test <= 0:
        raise ValueError("Invalid split sizes. Please check ratios.")

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    train_edges = unique_edges[:, train_idx].astype(np.int64, copy=False)
    val_edges = unique_edges[:, val_idx].astype(np.int64, copy=False)
    test_edges = unique_edges[:, test_idx].astype(np.int64, copy=False)
    overlap_counts = _split_overlap_counts(train_edges, val_edges, test_edges, graph.num_nodes)
    if any(count > 0 for count in overlap_counts.values()):
        raise RuntimeError(f"Undirected split overlap detected: {overlap_counts}")

    out = {
        "format": expected_format,
        "raw_edge_count": raw_edge_count,
        "unique_edge_count": int(num_edges),
        "train_edge_count": int(train_edges.shape[1]),
        "val_edge_count": int(val_edges.shape[1]),
        "test_edge_count": int(test_edges.shape[1]),
        "overlap_counts": overlap_counts,
        "train_pos_edges": torch.from_numpy(train_edges),
        "val_pos_edges": torch.from_numpy(val_edges),
        "test_pos_edges": torch.from_numpy(test_edges),
        "train_propagation_edges": torch.from_numpy(train_edges.copy()),
    }
    torch.save(out, split_path)
    _log_split_stats(split_path, out, loaded=False)
    return out


def create_or_load_edge_hash(
    split: Dict[str, torch.Tensor],
    hash_path: Path,
    num_nodes: int,
    force_rebuild: bool = False,
) -> np.ndarray:
    hash_path = Path(hash_path)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    if hash_path.exists() and not force_rebuild:
        log_event("[STAGE]", f"edge hash cache loaded: {hash_path}")
        return np.load(hash_path)

    log_event("[STAGE]", f"building all-positive undirected edge hash cache: {hash_path}")
    hashes = []
    for key in ("train_pos_edges", "val_pos_edges", "test_pos_edges"):
        edges = split[key].detach().cpu().numpy().astype(np.int64, copy=False)
        if edges.shape[1] == 0:
            continue
        hashes.append(_undirected_edge_hash(edges[0], edges[1], num_nodes))
    if hashes:
        hashed = np.concatenate(hashes, axis=0).astype(np.uint64, copy=False)
    else:
        hashed = np.empty((0,), dtype=np.uint64)
    hashed.sort()
    np.save(hash_path, hashed)
    log_event("[STAGE]", f"all-positive undirected edge hash cache written: {hash_path} entries={int(hashed.shape[0])}")
    return hashed


def _to_numpy_edge_dict(raw: Dict[int, torch.Tensor]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for k, v in raw.items():
        cid = int(k)
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy().astype(np.int64)
        else:
            arr = np.asarray(v, dtype=np.int64)
        out[cid] = arr
    return out


def _to_torch_edge_dict(raw: Dict[int, np.ndarray]) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for cid, arr in raw.items():
        out[int(cid)] = torch.from_numpy(np.asarray(arr, dtype=np.int64))
    return out


def _deduplicate_directed_edges(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    if edges.shape[1] <= 1:
        return edges.astype(np.int64, copy=False)
    hashed = _edge_hash(edges[0], edges[1], num_nodes)
    _, uniq_idx = np.unique(hashed, return_index=True)
    uniq_idx = np.sort(uniq_idx.astype(np.int64, copy=False))
    return edges[:, uniq_idx].astype(np.int64, copy=False)


def _chunk_split_format(chunk_context_mode: str, cross_chunk_neighbors_per_node: int) -> str:
    if chunk_context_mode == "intra_only":
        return "chunk_full_v2_intra_only"
    return f"chunk_full_v4_cross_chunk_1hop_k{int(cross_chunk_neighbors_per_node)}"


def _chunk_hash_format(chunk_context_mode: str, cross_chunk_neighbors_per_node: int) -> str:
    if chunk_context_mode == "intra_only":
        return "chunk_edge_hash_v1_intra_only"
    return f"chunk_edge_hash_v2_cross_chunk_1hop_k{int(cross_chunk_neighbors_per_node)}"


def create_or_load_chunk_splits(
    graph: ShardedGraphDataset,
    split_path: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    force_rebuild: bool = False,
    progress_every_chunks: int = 5,
    progress_every_sec: float = 30.0,
) -> Dict[str, Dict[int, np.ndarray]]:
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_context_mode = str(getattr(graph, "chunk_context_mode", "intra_only"))
    expected_format = _chunk_split_format(
        chunk_context_mode,
        int(getattr(graph, "cross_chunk_neighbors_per_node", 3)),
    )

    if split_path.exists() and not force_rebuild:
        cached = torch.load(split_path, map_location="cpu")
        if cached.get("format") == expected_format:
            log_event("[STAGE]", f"chunk-full split cache loaded: {split_path}")
            train_message = cached.get(
                "train_message_edge_index_by_chunk",
                cached["train_pos_edges_by_chunk"],
            )
            return {
                "train_pos_edges_by_chunk": _to_numpy_edge_dict(cached["train_pos_edges_by_chunk"]),
                "val_pos_edges_by_chunk": _to_numpy_edge_dict(cached["val_pos_edges_by_chunk"]),
                "test_pos_edges_by_chunk": _to_numpy_edge_dict(cached["test_pos_edges_by_chunk"]),
                "train_message_edge_index_by_chunk": _to_numpy_edge_dict(train_message),
            }
        log_event("[STAGE]", f"chunk-full split cache stale, rebuilding: {split_path}")

    log_event("[STAGE]", f"building chunk-full split cache: {split_path}")
    if chunk_context_mode == "intra_only":
        chunk_ids_all = []
        src_all = []
        dst_all = []
        total_edges = 0

        for chunk_id in range(graph.chunk_count()):
            edge_index = graph.load_chunk_intra_edge_index(chunk_id)
            if edge_index.shape[1] == 0:
                continue
            ecount = int(edge_index.shape[1])
            chunk_ids_all.append(np.full(ecount, int(chunk_id), dtype=np.int32))
            src_all.append(edge_index[0].astype(np.int32, copy=False))
            dst_all.append(edge_index[1].astype(np.int32, copy=False))
            total_edges += ecount
            chunk_no = int(chunk_id) + 1
            if progress_every_chunks > 0 and (
                chunk_no == graph.chunk_count() or chunk_no % progress_every_chunks == 0
            ):
                log_event(
                    "[STAGE]",
                    f"chunk-full split progress chunks={chunk_no}/{graph.chunk_count()} intra_edges_so_far={total_edges}",
                )

        if total_edges <= 0:
            raise RuntimeError("No intra-chunk edges found for chunk_full mode.")

        chunk_ids = np.concatenate(chunk_ids_all, axis=0)
        src = np.concatenate(src_all, axis=0)
        dst = np.concatenate(dst_all, axis=0)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(total_edges)

        n_train = int(total_edges * train_ratio)
        n_val = int(total_edges * val_ratio)
        n_test = total_edges - n_train - n_val
        if n_test <= 0:
            raise ValueError("Invalid split sizes. Please check ratios.")

        def regroup_intra(indices: np.ndarray) -> Dict[int, torch.Tensor]:
            by_chunk: Dict[int, torch.Tensor] = {}
            if indices.size == 0:
                return by_chunk
            sel_chunk = chunk_ids[indices]
            sel_src = src[indices]
            sel_dst = dst[indices]
            for cid in np.unique(sel_chunk):
                mask = sel_chunk == cid
                arr = np.vstack([sel_src[mask], sel_dst[mask]]).astype(np.int64, copy=False)
                by_chunk[int(cid)] = torch.from_numpy(arr)
            return by_chunk

        train_by_chunk = regroup_intra(perm[:n_train])
        val_by_chunk = regroup_intra(perm[n_train : n_train + n_val])
        test_by_chunk = regroup_intra(perm[n_train + n_val :])
        payload = {
            "format": expected_format,
            "train_pos_edges_by_chunk": train_by_chunk,
            "val_pos_edges_by_chunk": val_by_chunk,
            "test_pos_edges_by_chunk": test_by_chunk,
            "train_message_edge_index_by_chunk": dict(train_by_chunk),
        }
    else:
        edges = graph.load_all_edges(
            progress_every_chunks=progress_every_chunks,
            progress_every_sec=progress_every_sec,
        )
        edges = _deduplicate_directed_edges(edges, graph.num_nodes)
        total_edges = int(edges.shape[1])
        if total_edges <= 0:
            raise RuntimeError("No directed edges found for chunk_full cross_chunk_1hop mode.")

        chunk_src = graph.node_chunk_ids[edges[0]].astype(np.int32, copy=False)
        chunk_dst = graph.node_chunk_ids[edges[1]].astype(np.int32, copy=False)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(total_edges)
        n_train = int(total_edges * train_ratio)
        n_val = int(total_edges * val_ratio)
        n_test = total_edges - n_train - n_val
        if n_test <= 0:
            raise ValueError("Invalid split sizes. Please check ratios.")

        def regroup_cross(indices: np.ndarray, label: str) -> Dict[int, torch.Tensor]:
            out: Dict[int, torch.Tensor] = {}
            if indices.size == 0:
                return out
            sel_src = edges[0, indices]
            sel_dst = edges[1, indices]
            sel_chunk_src = chunk_src[indices]
            sel_chunk_dst = chunk_dst[indices]
            for chunk_id in range(graph.chunk_count()):
                mask = (sel_chunk_src == int(chunk_id)) | (sel_chunk_dst == int(chunk_id))
                count = int(mask.sum())
                if count <= 0:
                    continue
                sub = graph.load_chunk_subgraph(int(chunk_id))
                chunk_src_global = sel_src[mask]
                chunk_dst_global = sel_dst[mask]
                src_local_list = []
                dst_local_list = []
                for src_gid, dst_gid in zip(chunk_src_global.tolist(), chunk_dst_global.tolist()):
                    src_local = sub.global_to_local.get(int(src_gid))
                    dst_local = sub.global_to_local.get(int(dst_gid))
                    if src_local is None or dst_local is None:
                        continue
                    if not (
                        bool(sub.center_mask_local[int(src_local)])
                        or bool(sub.center_mask_local[int(dst_local)])
                    ):
                        continue
                    src_local_list.append(int(src_local))
                    dst_local_list.append(int(dst_local))
                if src_local_list:
                    out[int(chunk_id)] = torch.from_numpy(
                        np.vstack(
                            [
                                np.asarray(src_local_list, dtype=np.int64),
                                np.asarray(dst_local_list, dtype=np.int64),
                            ]
                        )
                    )
                chunk_no = int(chunk_id) + 1
                if progress_every_chunks > 0 and (
                    chunk_no == graph.chunk_count() or chunk_no % progress_every_chunks == 0
                ):
                    log_event(
                        "[STAGE]",
                        f"chunk-full {label} regroup progress chunks={chunk_no}/{graph.chunk_count()}",
                    )
            return out

        train_by_chunk = regroup_cross(perm[:n_train], "train")
        val_by_chunk = regroup_cross(perm[n_train : n_train + n_val], "val")
        test_by_chunk = regroup_cross(perm[n_train + n_val :], "test")
        payload = {
            "format": expected_format,
            "train_pos_edges_by_chunk": train_by_chunk,
            "val_pos_edges_by_chunk": val_by_chunk,
            "test_pos_edges_by_chunk": test_by_chunk,
            "train_message_edge_index_by_chunk": dict(train_by_chunk),
        }
    torch.save(payload, split_path)
    log_event("[STAGE]", f"chunk-full split cache written: {split_path}")
    return {
        "train_pos_edges_by_chunk": _to_numpy_edge_dict(payload["train_pos_edges_by_chunk"]),
        "val_pos_edges_by_chunk": _to_numpy_edge_dict(payload["val_pos_edges_by_chunk"]),
        "test_pos_edges_by_chunk": _to_numpy_edge_dict(payload["test_pos_edges_by_chunk"]),
        "train_message_edge_index_by_chunk": _to_numpy_edge_dict(payload["train_message_edge_index_by_chunk"]),
    }


def create_or_load_chunk_edge_hashes(
    graph: ShardedGraphDataset,
    hash_path: Path,
    force_rebuild: bool = False,
    progress_every_chunks: int = 5,
) -> Dict[int, np.ndarray]:
    hash_path = Path(hash_path)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    expected_format = _chunk_hash_format(
        str(getattr(graph, "chunk_context_mode", "intra_only")),
        int(getattr(graph, "cross_chunk_neighbors_per_node", 3)),
    )
    if hash_path.exists() and not force_rebuild:
        cached = torch.load(hash_path, map_location="cpu")
        meta = cached.get("__meta__") if isinstance(cached, dict) else None
        raw_hashes = cached.get("hashes") if isinstance(cached, dict) and "hashes" in cached else cached
        if isinstance(meta, dict) and meta.get("format") == expected_format:
            log_event("[STAGE]", f"chunk edge hash cache loaded: {hash_path}")
            return {
                int(k): np.asarray(
                    v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v,
                    dtype=np.uint64,
                )
                for k, v in raw_hashes.items()
            }
        log_event("[STAGE]", f"chunk edge hash cache stale, rebuilding: {hash_path}")

    log_event("[STAGE]", f"building chunk edge hash cache: {hash_path}")
    payload: Dict[int, np.ndarray] = {}
    for chunk_id in range(graph.chunk_count()):
        sub = graph.load_chunk_subgraph(chunk_id)
        edge_index = sub.message_edge_index_local
        hashed = _edge_hash(
            edge_index[0],
            edge_index[1],
            int(sub.node_indices_global.shape[0]),
        )
        hashed.sort()
        payload[int(chunk_id)] = hashed.astype(np.uint64, copy=False)
        chunk_no = int(chunk_id) + 1
        if progress_every_chunks > 0 and (
            chunk_no == graph.chunk_count() or chunk_no % progress_every_chunks == 0
        ):
            log_event(
                "[STAGE]",
                f"chunk edge hash progress chunks={chunk_no}/{graph.chunk_count()}",
            )
    torch.save(
        {
            "__meta__": {
                "format": expected_format,
                "chunk_context_mode": str(getattr(graph, "chunk_context_mode", "intra_only")),
                "cross_chunk_neighbors_per_node": int(
                    getattr(graph, "cross_chunk_neighbors_per_node", 3)
                ),
            },
            "hashes": payload,
        },
        hash_path,
    )
    log_event("[STAGE]", f"chunk edge hash cache written: {hash_path}")
    return {int(k): np.asarray(v, dtype=np.uint64) for k, v in payload.items()}


def edge_exists_hash(
    src: np.ndarray,
    dst: np.ndarray,
    sorted_hash: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    h = _edge_hash(src, dst, num_nodes)
    pos = np.searchsorted(sorted_hash, h, side="left")
    ok = pos < sorted_hash.shape[0]
    out = np.zeros(h.shape[0], dtype=bool)
    out[ok] = sorted_hash[pos[ok]] == h[ok]
    return out
