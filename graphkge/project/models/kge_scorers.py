"""KGE scoring modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransEScorer(nn.Module):
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.relation_emb = nn.Embedding(int(num_relations), int(dim))
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.output_dim = int(dim)

    def score(self, head: torch.Tensor, rel_idx: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        rel = self.relation_emb(rel_idx)
        return -torch.linalg.norm(head + rel - tail, ord=2, dim=-1)

    def score_all_relations(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        rel = self.relation_emb.weight.unsqueeze(0)
        return -torch.linalg.norm(head.unsqueeze(1) + rel - tail.unsqueeze(1), ord=2, dim=-1)


class ComplExScorer(nn.Module):
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.relation_emb = nn.Embedding(int(num_relations), int(dim) * 2)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.output_dim = int(dim) * 2

    @staticmethod
    def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.chunk(x, 2, dim=-1)

    def score(self, head: torch.Tensor, rel_idx: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        hr, hi = self._split(head)
        tr, ti = self._split(tail)
        rr, ri = self._split(self.relation_emb(rel_idx))
        return torch.sum(hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr, dim=-1)

    def score_all_relations(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        hr, hi = self._split(head)
        tr, ti = self._split(tail)
        rr, ri = self._split(self.relation_emb.weight)
        hr = hr.unsqueeze(1)
        hi = hi.unsqueeze(1)
        tr = tr.unsqueeze(1)
        ti = ti.unsqueeze(1)
        rr = rr.unsqueeze(0)
        ri = ri.unsqueeze(0)
        return torch.sum(hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr, dim=-1)


def build_kge_scorer(name: str, num_relations: int, dim: int) -> nn.Module:
    if name == "transe":
        return TransEScorer(num_relations=num_relations, dim=dim)
    if name == "complex":
        return ComplExScorer(num_relations=num_relations, dim=dim)
    raise ValueError(f"Unknown kge scorer: {name}")

