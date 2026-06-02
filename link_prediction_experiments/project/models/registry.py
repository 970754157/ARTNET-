"""Registry helpers for model components."""

from __future__ import annotations

from project.models.fusion import build_fusion
from project.models.gnn import build_gnn
from project.models.image_encoders import build_image_encoder
from project.models.predictor import build_predictor
from project.models.text_encoders import build_text_encoder

__all__ = [
    "build_text_encoder",
    "build_image_encoder",
    "build_fusion",
    "build_gnn",
    "build_predictor",
]

