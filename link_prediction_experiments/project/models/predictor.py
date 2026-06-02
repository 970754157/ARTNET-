"""Link predictors."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    """MLP link predictor returning logits."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_src, h_dst], dim=-1)
        return self.net(x).squeeze(-1)


def build_predictor(name: str, in_dim: int, hidden_dim: int) -> nn.Module:
    if name == "mlp":
        return MLPPredictor(in_dim=in_dim, hidden_dim=hidden_dim)
    if name == "dot":
        raise NotImplementedError("dot predictor is reserved but not implemented in this run")
    raise ValueError(f"Unknown predictor: {name}")

