"""Local-only pretrained model helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch


def slugify_model_id(model_id: str) -> str:
    value = str(model_id).strip()
    if not value:
        return "default"
    value = value.replace("\\", "/")
    value = value.strip("/")
    value = value.replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return value or "default"


def require_local_dir(local_dir: Path, model_id: str) -> Path:
    local_dir = Path(local_dir)
    if not local_dir.is_dir() or not any(local_dir.iterdir()):
        raise FileNotFoundError(
            f"Local model directory not found or empty for {model_id}: {local_dir}"
        )
    return local_dir


def load_hf_resource(loader: Any, model_id: str, local_dir: Path, **kwargs):
    resolved_dir = require_local_dir(local_dir, model_id)
    return loader.from_pretrained(str(resolved_dir), local_files_only=True, **kwargs)


def load_local_torchvision_state_dict(local_dir: Path, filename: str = "weights.pt") -> dict[str, Any]:
    local_dir = Path(local_dir)
    weight_path = local_dir / filename
    if not weight_path.exists():
        raise FileNotFoundError(f"Local torchvision weights not found: {weight_path}")
    return torch.load(weight_path, map_location="cpu")


def load_state_dict_from_local_dir(local_dir: Path, model_id: str = "") -> dict[str, Any]:
    local_dir = require_local_dir(local_dir, model_id or str(local_dir))
    candidates = [
        local_dir / "model.safetensors",
        local_dir / "pytorch_model.bin",
        local_dir / "weights.pt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except Exception as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "safetensors is required to load model.safetensors from local checkpoints."
                ) from exc
            return load_file(str(path), device="cpu")
        return torch.load(path, map_location="cpu")
    raise FileNotFoundError(f"No supported checkpoint file found in local model dir: {local_dir}")
