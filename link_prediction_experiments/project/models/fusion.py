"""Feature fusion modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPConcatFusion(nn.Module):
    """Concatenate text/image embeddings then project with MLP."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.output_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_fusion(
    name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.1,
) -> nn.Module:
    if name == "mlp_concat":
        return MLPConcatFusion(in_dim, hidden_dim, out_dim, dropout=dropout)
    if name in {"gated", "attention_pool"}:
        raise NotImplementedError(f"fusion module not implemented: {name}")
    raise ValueError(f"Unknown fusion module: {name}")

