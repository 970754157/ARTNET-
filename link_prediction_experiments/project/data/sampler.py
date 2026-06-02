"""Neighbor sampler for K-hop subgraph extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from project.data.adjacency_index import DiskAdjacencyIndex


@dataclass
class SubgraphData:
    """Sampled subgraph in local node indexing."""

    sub_nodes_global_ids: torch.Tensor
    edge_index: torch.Tensor
    seed_local_indices: torch.Tensor
    node_hops: torch.Tensor
    global_to_local: Dict[int, int]


class NeighborSampler:
    """CPU-friendly neighbor sampler using disk adjacency index."""

    def __init__(
        self,
        adjacency: DiskAdjacencyIndex,
        num_neighbors: Sequence[int],
        num_hops: int,
        seed: int = 42,
    ):
        self.adjacency = adjacency
        self.num_neighbors = list(num_neighbors)
        self.num_hops = int(num_hops)
        self.rng = np.random.default_rng(seed)

    def _sample_neighbors(self, neigh: np.ndarray, k: int) -> np.ndarray:
        if neigh.size <= k or k <= 0:
            return neigh
        sel = self.rng.choice(neigh.size, size=k, replace=False)
        return neigh[sel]

    def sample(self, seed_nodes: Iterable[int]) -> SubgraphData:
        seed_arr = np.asarray(list(seed_nodes), dtype=np.int64)
        if seed_arr.size == 0:
            raise ValueError("seed_nodes must not be empty")

        visited_order: List[int] = []
        visited_set = set()
        visited_hops: List[int] = []
        hop_by_node: Dict[int, int] = {}

        for x in seed_arr:
            xi = int(x)
            if xi not in visited_set:
                visited_set.add(xi)
                visited_order.append(xi)
                visited_hops.append(0)
                hop_by_node[xi] = 0

        frontier = np.asarray(visited_order, dtype=np.int64)
        sampled_edges: List[Tuple[int, int]] = []

        for hop in range(self.num_hops):
            if frontier.size == 0:
                break
            k = self.num_neighbors[min(hop, len(self.num_neighbors) - 1)]
            next_nodes: List[int] = []
            neighbors_many = self.adjacency.neighbors_many(frontier)
            for src, neigh in zip(frontier.tolist(), neighbors_many):
                chosen = self._sample_neighbors(neigh, int(k))
                for dst in chosen:
                    d = int(dst)
                    sampled_edges.append((int(src), d))
                    if d not in visited_set:
                        visited_set.add(d)
                        visited_order.append(d)
                        visited_hops.append(int(hop) + 1)
                        hop_by_node[d] = int(hop) + 1
                        next_nodes.append(d)
            frontier = np.asarray(next_nodes, dtype=np.int64)

        global_to_local = {gid: i for i, gid in enumerate(visited_order)}

        if sampled_edges:
            src_local = np.asarray([global_to_local[s] for s, _ in sampled_edges], dtype=np.int64)
            dst_local = np.asarray([global_to_local[d] for _, d in sampled_edges], dtype=np.int64)
            edge_index = torch.from_numpy(np.vstack([src_local, dst_local]))
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        seed_local = torch.from_numpy(
            np.asarray([global_to_local[int(s)] for s in seed_arr], dtype=np.int64)
        )
        return SubgraphData(
            sub_nodes_global_ids=torch.from_numpy(np.asarray(visited_order, dtype=np.int64)),
            edge_index=edge_index.long(),
            seed_local_indices=seed_local.long(),
            node_hops=torch.from_numpy(np.asarray(visited_hops, dtype=np.int64)),
            global_to_local=global_to_local,
        )
