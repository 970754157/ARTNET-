"""GNN encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, SAGEConv


class NoHopEncoder(nn.Module):
    """No-hop baseline: skip graph message passing and keep fused node features."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.output_dim = int(in_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = (edge_index, edge_type)
        return x


class GraphSAGEEncoder(nn.Module):
    """Stacked GraphSAGE encoder."""

    def __init__(self, in_dim: int, hidden_dim: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be > 0")
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = edge_type
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.act(h)
                h = self.dropout(h)
        return h


class RGCNEncoder(nn.Module):
    """Stacked RGCN encoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_relations: int,
        layers: int = 2,
        dropout: float = 0.1,
        num_bases: int = 32,
    ):
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be > 0")
        if num_relations <= 0:
            raise ValueError("num_relations must be > 0")
        if num_bases <= 0:
            raise ValueError("num_bases must be > 0")
        effective_bases = min(int(num_bases), int(num_relations))
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations=num_relations, num_bases=effective_bases))
        for _ in range(layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=effective_bases))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_type is None:
            raise RuntimeError("RGCNEncoder requires edge_type")
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_type)
            if i < len(self.convs) - 1:
                h = self.act(h)
                h = self.dropout(h)
        return h


def build_gnn(
    name: str,
    in_dim: int,
    hidden_dim: int,
    layers: int = 2,
    dropout: float = 0.1,
    num_relations: int = 0,
    num_bases: int = 32,
) -> nn.Module:
    if name == "graphsage":
        return GraphSAGEEncoder(in_dim, hidden_dim, layers=layers, dropout=dropout)
    if name == "rgcn":
        return RGCNEncoder(
            in_dim,
            hidden_dim,
            num_relations=num_relations,
            layers=layers,
            dropout=dropout,
            num_bases=num_bases,
        )
    if name == "nohop":
        return NoHopEncoder(in_dim)
    if name in {"hgt", "hgnn"}:
        raise NotImplementedError(f"GNN not implemented: {name}")
    raise ValueError(f"Unknown gnn: {name}")
