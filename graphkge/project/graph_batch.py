"""Shared sampled-subgraph preparation for GraphKGE training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import torch

from project.data.sampler import SubgraphData

if TYPE_CHECKING:
    from project.experiment import ExperimentBundle


@dataclass
class PreparedGraphBatch:
    node_indices: np.ndarray
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    image_batch: Dict[str, object]
    global_to_local: Dict[int, int]
    stats: Dict[str, Any]


def _seed_list(seed_entities: np.ndarray) -> List[int]:
    seeds = np.asarray(seed_entities, dtype=np.int64).reshape(-1)
    ordered = list(dict.fromkeys(int(x) for x in seeds.tolist()))
    if not ordered:
        raise ValueError("seed_entities must not be empty")
    return ordered


def _make_seed_only_subgraph(seed_entities: np.ndarray) -> SubgraphData:
    seeds = _seed_list(seed_entities)
    node_ids = np.asarray(seeds, dtype=np.int64)
    global_to_local = {gid: i for i, gid in enumerate(seeds)}
    return SubgraphData(
        sub_nodes_global_ids=torch.from_numpy(node_ids).long(),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        seed_local_indices=torch.arange(len(seeds), dtype=torch.long),
        node_hops=torch.zeros((len(seeds),), dtype=torch.long),
        global_to_local=global_to_local,
    )


def _directed_local_edge_hash(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    src64 = np.asarray(src, dtype=np.int64).astype(np.uint64, copy=False)
    dst64 = np.asarray(dst, dtype=np.int64).astype(np.uint64, copy=False)
    return src64 * np.uint64(max(1, int(num_nodes))) + dst64


def _map_entity_pairs_to_local(entity_pairs_global: np.ndarray, global_to_local: Dict[int, int]) -> np.ndarray:
    pairs = np.asarray(entity_pairs_global, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError("entity_pairs_global must have shape [2, num_edges]")
    if pairs.shape[1] == 0:
        return np.empty((2, 0), dtype=np.int64)
    src = np.asarray([global_to_local[int(x)] for x in pairs[0].tolist()], dtype=np.int64)
    dst = np.asarray([global_to_local[int(x)] for x in pairs[1].tolist()], dtype=np.int64)
    return np.vstack([src, dst]).astype(np.int64, copy=False)


def _mask_supervision_edges_from_message_graph(
    edge_index_local: np.ndarray,
    edge_type_local: np.ndarray,
    entity_pairs_local: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    edge_index_local = np.asarray(edge_index_local, dtype=np.int64)
    edge_type_local = np.asarray(edge_type_local, dtype=np.int64)
    entity_pairs_local = np.asarray(entity_pairs_local, dtype=np.int64)
    stats = {
        "masked_supervision_edge_directions": 0,
        "masked_supervision_unique_edges": 0,
        "query_edges_in_message_graph": 0,
    }
    if edge_index_local.shape[1] == 0 or entity_pairs_local.shape[1] == 0:
        return edge_index_local, edge_type_local, stats

    num_local_nodes = int(edge_index_local.max()) + 1 if edge_index_local.size > 0 else 1
    forward_src = entity_pairs_local[0].astype(np.int64, copy=False)
    forward_dst = entity_pairs_local[1].astype(np.int64, copy=False)
    edge_hash = _directed_local_edge_hash(edge_index_local[0], edge_index_local[1], num_local_nodes)
    forward_hash = _directed_local_edge_hash(forward_src, forward_dst, num_local_nodes)
    reverse_hash = _directed_local_edge_hash(forward_dst, forward_src, num_local_nodes)
    forward_present = np.isin(forward_hash, edge_hash, assume_unique=False)
    reverse_present = np.isin(reverse_hash, edge_hash, assume_unique=False)
    stats["masked_supervision_unique_edges"] = int((forward_present | reverse_present).sum())

    non_self = forward_src != forward_dst
    query_src = np.concatenate([forward_src, forward_dst[non_self]], axis=0).astype(np.int64, copy=False)
    query_dst = np.concatenate([forward_dst, forward_src[non_self]], axis=0).astype(np.int64, copy=False)
    query_hash = _directed_local_edge_hash(query_src, query_dst, num_local_nodes)
    remove_mask = np.isin(edge_hash, query_hash, assume_unique=False)
    stats["query_edges_in_message_graph"] = int(remove_mask.sum())
    if stats["query_edges_in_message_graph"] == 0:
        return edge_index_local, edge_type_local, stats
    masked = edge_index_local[:, ~remove_mask].astype(np.int64, copy=False)
    masked_edge_type = edge_type_local[~remove_mask].astype(np.int64, copy=False)
    stats["masked_supervision_edge_directions"] = int(remove_mask.sum())
    return masked, masked_edge_type, stats


def _deterministic_node_order(node_ids: np.ndarray, seed: int, salt: int) -> np.ndarray:
    values = np.asarray(node_ids, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = values ^ np.uint64(seed + salt) ^ np.uint64(0x9E3779B97F4A7C15)
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ np.right_shift(x, 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ np.right_shift(x, 27)) * np.uint64(0x94D049BB133111EB)
        x = x ^ np.right_shift(x, 31)
    return np.lexsort((values.astype(np.int64, copy=False), x))


def _rebuild_subgraph(sub: SubgraphData, keep_local_mask: np.ndarray) -> SubgraphData:
    keep_local_mask = np.asarray(keep_local_mask, dtype=np.bool_)
    keep_idx = np.flatnonzero(keep_local_mask).astype(np.int64, copy=False)
    old_to_new = np.full(int(sub.sub_nodes_global_ids.shape[0]), -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int64)

    old_edge_index = sub.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    old_edge_type = sub.edge_type.detach().cpu().numpy().astype(np.int64, copy=False)
    if old_edge_index.shape[1] > 0:
        edge_keep = keep_local_mask[old_edge_index[0]] & keep_local_mask[old_edge_index[1]]
        edge_src = old_to_new[old_edge_index[0, edge_keep]]
        edge_dst = old_to_new[old_edge_index[1, edge_keep]]
        edge_index = torch.from_numpy(np.vstack([edge_src, edge_dst]).astype(np.int64, copy=False)).long()
        edge_type = torch.from_numpy(old_edge_type[edge_keep].astype(np.int64, copy=False)).long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    old_seed = sub.seed_local_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    new_seed = old_to_new[old_seed]
    if (new_seed < 0).any():
        raise RuntimeError("Seed nodes were removed while rebuilding sampled subgraph.")

    node_ids = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    node_hops = sub.node_hops.detach().cpu().numpy().astype(np.int64, copy=False)
    kept_node_ids = node_ids[keep_idx]
    kept_hops = node_hops[keep_idx]
    return SubgraphData(
        sub_nodes_global_ids=torch.from_numpy(kept_node_ids.astype(np.int64, copy=False)).long(),
        edge_index=edge_index,
        edge_type=edge_type,
        seed_local_indices=torch.from_numpy(new_seed.astype(np.int64, copy=False)).long(),
        node_hops=torch.from_numpy(kept_hops.astype(np.int64, copy=False)).long(),
        global_to_local={int(gid): int(i) for i, gid in enumerate(kept_node_ids.tolist())},
    )


def _trim_sampled_subgraph_nodes(sub: SubgraphData, max_nodes: int, seed: int) -> Tuple[SubgraphData, Dict[str, int]]:
    max_nodes = int(max_nodes)
    total_nodes = int(sub.sub_nodes_global_ids.shape[0])
    stats = {
        "subgraph_nodes_before_trim": total_nodes,
        "subgraph_nodes_after_trim": total_nodes,
        "trimmed_nodes": 0,
    }
    if total_nodes <= max_nodes:
        return sub, stats

    node_hops = sub.node_hops.detach().cpu().numpy().astype(np.int64, copy=False)
    node_ids = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    keep = np.zeros(total_nodes, dtype=np.bool_)
    seed_local = sub.seed_local_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    keep[seed_local] = True

    remaining = max_nodes - int(keep.sum())
    if remaining < 0:
        raise RuntimeError("max_subgraph_nodes is smaller than the number of seed nodes.")
    if remaining == 0:
        trimmed = _rebuild_subgraph(sub, keep)
        stats["subgraph_nodes_after_trim"] = int(trimmed.sub_nodes_global_ids.shape[0])
        stats["trimmed_nodes"] = total_nodes - stats["subgraph_nodes_after_trim"]
        return trimmed, stats

    candidates = np.flatnonzero(~keep)
    if candidates.size > 0:
        order = np.lexsort(
            (
                node_ids[candidates].astype(np.int64, copy=False),
                node_hops[candidates].astype(np.int64, copy=False),
            )
        )
        chosen = candidates[order[:remaining]]
        keep[chosen] = True

    trimmed = _rebuild_subgraph(sub, keep)
    stats["subgraph_nodes_after_trim"] = int(trimmed.sub_nodes_global_ids.shape[0])
    stats["trimmed_nodes"] = total_nodes - stats["subgraph_nodes_after_trim"]
    return trimmed, stats


def _trim_real_image_nodes(
    sub: SubgraphData,
    loaded_mask: np.ndarray,
    node_is_art: np.ndarray,
    max_images: int,
    seed: int,
) -> Tuple[SubgraphData, Dict[str, Any]]:
    loaded_mask = np.asarray(loaded_mask, dtype=np.bool_)
    real_count = int(loaded_mask.sum())
    stats: Dict[str, Any] = {
        "real_image_nodes_before_trim": real_count,
        "real_image_nodes_after_trim": real_count,
        "trimmed_image_nodes": 0,
        "seed_image_cap_exceeded": False,
    }
    if real_count <= int(max_images):
        return sub, stats

    node_ids = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    seed_mask = np.zeros(node_ids.shape[0], dtype=np.bool_)
    seed_local = sub.seed_local_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    seed_mask[seed_local] = True

    overflow = real_count - int(max_images)
    drop_local: List[int] = []

    art_candidates = np.flatnonzero(loaded_mask & (~seed_mask) & node_is_art)
    if art_candidates.size > 0 and overflow > 0:
        order = _deterministic_node_order(node_ids[art_candidates], seed=seed, salt=23)
        chosen = art_candidates[order[: min(overflow, art_candidates.size)]]
        drop_local.extend(chosen.tolist())
        overflow -= int(chosen.shape[0])

    other_candidates = np.flatnonzero(loaded_mask & (~seed_mask) & (~node_is_art))
    if other_candidates.size > 0 and overflow > 0:
        order = _deterministic_node_order(node_ids[other_candidates], seed=seed, salt=29)
        chosen = other_candidates[order[: min(overflow, other_candidates.size)]]
        drop_local.extend(chosen.tolist())
        overflow -= int(chosen.shape[0])

    if not drop_local:
        stats["seed_image_cap_exceeded"] = real_count > int(max_images)
        return sub, stats

    keep = np.ones(node_ids.shape[0], dtype=np.bool_)
    keep[np.asarray(drop_local, dtype=np.int64)] = False
    trimmed = _rebuild_subgraph(sub, keep)
    stats["real_image_nodes_after_trim"] = int(max_images) + max(0, overflow)
    stats["trimmed_image_nodes"] = real_count - stats["real_image_nodes_after_trim"]
    stats["seed_image_cap_exceeded"] = overflow > 0
    return trimmed, stats


def build_prepared_graph_batch(
    bundle: "ExperimentBundle",
    seed_entities: np.ndarray,
    masked_entity_pairs_global: np.ndarray | None = None,
) -> PreparedGraphBatch:
    if bool(bundle.cfg.model.use_graph):
        if bundle.neighbor_sampler is None:
            raise RuntimeError("neighbor_sampler is required when use_graph=true")
        sub = bundle.neighbor_sampler.sample(_seed_list(seed_entities))
    else:
        sub = _make_seed_only_subgraph(seed_entities)

    sub, trim_stats = _trim_sampled_subgraph_nodes(
        sub,
        max_nodes=int(bundle.cfg.data.max_subgraph_nodes),
        seed=int(bundle.cfg.seed),
    )
    node_idx = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    node_wids = bundle.catalog.get_node_wids(node_idx)
    node_has_image = bundle.catalog.node_has_image[node_idx].astype(np.bool_, copy=False)
    image_batch = bundle.image_provider.get_batch_tensors(
        wids=node_wids,
        has_image_mask=node_has_image,
        transform=bundle.image_transform,
        pin_memory=bool(bundle.cfg.runtime.pin_memory and bundle.device.type == "cuda"),
    )
    image_trim_stats = {
        "real_image_nodes_before_trim": int(image_batch["loaded_mask_cpu"].sum().item()),
        "real_image_nodes_after_trim": int(image_batch["loaded_mask_cpu"].sum().item()),
        "trimmed_image_nodes": 0,
        "seed_image_cap_exceeded": False,
    }
    if int(image_batch["loaded_mask_cpu"].sum().item()) > int(bundle.cfg.data.max_image_nodes):
        node_is_art = bundle.catalog.node_is_art[node_idx].astype(np.bool_, copy=False)
        sub, image_trim_stats = _trim_real_image_nodes(
            sub,
            loaded_mask=image_batch["loaded_mask_cpu"].detach().cpu().numpy().astype(np.bool_, copy=False),
            node_is_art=node_is_art,
            max_images=int(bundle.cfg.data.max_image_nodes),
            seed=int(bundle.cfg.seed),
        )
        node_idx = sub.sub_nodes_global_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        node_wids = bundle.catalog.get_node_wids(node_idx)
        node_has_image = bundle.catalog.node_has_image[node_idx].astype(np.bool_, copy=False)
        image_batch = bundle.image_provider.get_batch_tensors(
            wids=node_wids,
            has_image_mask=node_has_image,
            transform=bundle.image_transform,
            pin_memory=bool(bundle.cfg.runtime.pin_memory and bundle.device.type == "cuda"),
        )
        image_trim_stats["real_image_nodes_after_trim"] = int(image_batch["loaded_mask_cpu"].sum().item())
        image_trim_stats["trimmed_image_nodes"] = (
            image_trim_stats["real_image_nodes_before_trim"] - image_trim_stats["real_image_nodes_after_trim"]
        )

    edge_index_np = sub.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    edge_type_np = sub.edge_type.detach().cpu().numpy().astype(np.int64, copy=False)
    leakage_stats = {
        "masked_supervision_edge_directions": 0,
        "masked_supervision_unique_edges": 0,
        "query_edges_in_message_graph": 0,
    }
    if masked_entity_pairs_global is not None and np.asarray(masked_entity_pairs_global).size > 0:
        pairs_local = _map_entity_pairs_to_local(masked_entity_pairs_global, sub.global_to_local)
        edge_index_np, edge_type_np, leakage_stats = _mask_supervision_edges_from_message_graph(
            edge_index_np,
            edge_type_np,
            pairs_local,
        )

    return PreparedGraphBatch(
        node_indices=node_idx,
        edge_index=torch.from_numpy(edge_index_np).long(),
        edge_type=torch.from_numpy(edge_type_np).long(),
        image_batch=image_batch,
        global_to_local=sub.global_to_local,
        stats={**trim_stats, **image_trim_stats, **leakage_stats},
    )
