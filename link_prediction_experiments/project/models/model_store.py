"""Helpers for local-first pretrained model management."""

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


def is_nonempty_dir(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and any(path.iterdir())


def ensure_local_hf_snapshot(model_id: str, local_dir: Path) -> Path:
    local_dir = Path(local_dir)
    if is_nonempty_dir(local_dir):
        return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "huggingface_hub is required to download pretrained Hugging Face models."
        ) from exc
    snapshot_download(
        repo_id=str(model_id),
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return local_dir


def load_hf_resource(loader: Any, model_id: str, local_dir: Path, **kwargs):
    resolved_dir = ensure_local_hf_snapshot(model_id, local_dir)
    return loader.from_pretrained(str(resolved_dir), local_files_only=True, **kwargs)


def ensure_torchvision_state_dict(url: str, local_dir: Path, filename: str = "weights.pt") -> dict[str, Any]:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    weight_path = local_dir / filename
    if weight_path.exists():
        return torch.load(weight_path, map_location="cpu")
    return torch.hub.load_state_dict_from_url(
        url=str(url),
        model_dir=str(local_dir),
        file_name=filename,
        map_location="cpu",
        progress=True,
    )


def load_state_dict_from_local_dir(local_dir: Path) -> dict[str, Any]:
    local_dir = Path(local_dir)
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
                    "safetensors is required to load local safetensors checkpoints."
                ) from exc
            return load_file(str(path), device="cpu")
        return torch.load(path, map_location="cpu")
    raise FileNotFoundError(f"No supported weight file found under: {local_dir}")
