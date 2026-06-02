"""Configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class PathsConfig:
    graph_dir: str
    text_dir: str
    image_dir: str
    cache_dir: str
    runs_dir: str
    split_file: str
    adj_index_dir: str
    text_cache_dir: str
    local_model_root: str = "../local_models"


@dataclass
class ModelConfig:
    text_encoder: str
    image_encoder: str
    fusion: str
    gnn: str
    predictor: str
    text_max_length: int
    text_batch_size: int
    image_size: int
    fusion_out_dim: int
    fusion_hidden_dim: int
    graphsage_hidden_dim: int
    graphsage_layers: int
    graphsage_dropout: float
    predictor_hidden_dim: int
    bert_model: str = ""
    text_model_name: str = ""
    image_model_name: str = ""
    resnet_pretrained: bool = False
    freeze_text_backbone: bool = True
    freeze_image_backbone: bool = False
    image_pretrained: bool = False
    text_proj_dim: int = 128
    image_proj_dim: int = 128
    image_forward_batch_size: int = 32


@dataclass
class IndexConfig:
    rebuild_index: bool
    adj_lru_chunks: int


@dataclass
class TextCacheConfig:
    dtype: str
    lru_chunks: int


@dataclass
class DataConfig:
    num_hops: int
    num_neighbors: List[int]
    batch_edges: int
    num_neg: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    subgraph_mode: str = "neighbor_sample"
    max_subgraph_nodes: int = 6400
    max_image_nodes: int = 500
    chunk_context_mode: str = "intra_only"
    cross_chunk_neighbors_per_node: int = 3


@dataclass
class OptimConfig:
    lr: float
    weight_decay: float
    epochs: int
    max_steps: int
    scheduler: str
    early_stop_enabled: bool = True
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0


@dataclass
class EvalConfig:
    eval_every_steps: int
    val_max_edges: int
    test_max_edges: int
    enable_hit10: bool
    hit10_negatives: int
    hit10_pos_limit: int
    final_val_max_edges: int = 0
    final_test_max_edges: int = 0
    final_negatives_per_positive: int = 150
    final_hit10_pos_limit: int = 0
    final_hit10_negatives: int = 150
    final_test_negatives_per_positive: int = 150
    final_test_enable_hit10: bool = True
    final_test_art_metrics: bool = True
    final_test_batch_edges: int = 65536
    final_test_embedding_cache_dtype: str = "float16"


@dataclass
class LoggingConfig:
    save_every_steps: int
    best_metric: str
    loss_window: int
    chunk_block_steps: int


@dataclass
class DataloaderConfig:
    num_workers: int


@dataclass
class RuntimeConfig:
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    prefetch_batches: int = 2
    cpu_threads: int = 0
    cudnn_benchmark: bool = True
    auto_tune: bool = True
    heartbeat_sec: int = 60
    log_first_n_steps: int = 5
    log_every_steps: int = 10
    stage_progress_sec: int = 30
    step_progress_sec: int = 30
    step_phase_start_end_logs: bool = True
    step_progress_min_items: int = 1
    max_loader_workers: int = 32
    max_prefetch_batches: int = 8
    max_text_batch_size: int = 256
    max_image_forward_batch_size: int = 256
    max_batch_edges_gpu: int = 3072


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    device: str
    paths: PathsConfig
    model: ModelConfig
    index: IndexConfig
    text_cache: TextCacheConfig
    data: DataConfig
    optim: OptimConfig
    eval: EvalConfig
    logging: LoggingConfig
    dataloader: DataloaderConfig
    runtime: RuntimeConfig

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentConfig":
        raw_paths = dict(raw["paths"])
        raw_paths.setdefault("local_model_root", "../local_models")

        raw_model = dict(raw["model"])
        raw_model.setdefault("bert_model", "")
        if not raw_model.get("text_model_name"):
            raw_model["text_model_name"] = str(
                raw_model.get("bert_model") or "bert-base-uncased"
            )
        raw_model.setdefault("image_model_name", "")
        raw_model.setdefault("resnet_pretrained", False)
        raw_model.setdefault("freeze_text_backbone", True)
        raw_model.setdefault("freeze_image_backbone", False)
        raw_model.setdefault(
            "image_pretrained",
            bool(raw_model.get("resnet_pretrained", False)),
        )
        return cls(
            experiment_name=raw["experiment_name"],
            seed=int(raw["seed"]),
            device=str(raw["device"]),
            paths=PathsConfig(**raw_paths),
            model=ModelConfig(**raw_model),
            index=IndexConfig(**raw["index"]),
            text_cache=TextCacheConfig(**raw["text_cache"]),
            data=DataConfig(**raw["data"]),
            optim=OptimConfig(**raw["optim"]),
            eval=EvalConfig(**raw["eval"]),
            logging=LoggingConfig(**raw["logging"]),
            dataloader=DataloaderConfig(**raw["dataloader"]),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
            "paths": self.paths.__dict__,
            "model": self.model.__dict__,
            "index": self.index.__dict__,
            "text_cache": self.text_cache.__dict__,
            "data": self.data.__dict__,
            "optim": self.optim.__dict__,
            "eval": self.eval.__dict__,
            "logging": self.logging.__dict__,
            "dataloader": self.dataloader.__dict__,
            "runtime": self.runtime.__dict__,
        }

    def resolved_paths(self, base_dir: Path) -> Dict[str, Path]:
        result: Dict[str, Path] = {}
        for key, value in self.paths.__dict__.items():
            path = Path(value)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            result[key] = path
        return result

    def validate(self) -> None:
        total = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train/val/test ratios must sum to 1.0")
        if self.data.subgraph_mode != "neighbor_sample":
            raise ValueError("data.subgraph_mode must be 'neighbor_sample'")
        if self.data.chunk_context_mode not in {"intra_only", "cross_chunk_1hop"}:
            raise ValueError("data.chunk_context_mode must be 'intra_only' or 'cross_chunk_1hop'")
        if self.data.cross_chunk_neighbors_per_node <= 0:
            raise ValueError("data.cross_chunk_neighbors_per_node must be > 0")
        if self.data.num_hops < 0:
            raise ValueError("num_hops must be >= 0")
        if len(self.data.num_neighbors) < self.data.num_hops:
            raise ValueError("num_neighbors length must be >= num_hops")
        if self.data.max_subgraph_nodes <= 0:
            raise ValueError("max_subgraph_nodes must be > 0")
        if self.data.max_image_nodes <= 0:
            raise ValueError("max_image_nodes must be > 0")
        if self.optim.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.optim.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.optim.early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be > 0")
        if self.optim.early_stop_min_delta < 0:
            raise ValueError("early_stop_min_delta must be >= 0")
        if not str(self.model.text_model_name).strip():
            raise ValueError("model.text_model_name must be non-empty")
        if self.model.text_max_length <= 0:
            raise ValueError("text_max_length must be > 0")
        if self.model.text_proj_dim <= 0 or self.model.image_proj_dim <= 0:
            raise ValueError("text_proj_dim and image_proj_dim must be > 0")
        if self.model.image_forward_batch_size <= 0:
            raise ValueError("image_forward_batch_size must be > 0")
        if self.model.image_encoder != "resnet18" and not str(self.model.image_model_name).strip():
            raise ValueError("model.image_model_name must be non-empty for non-resnet18 encoders")
        if self.device not in {"cpu", "auto"} and not str(self.device).startswith("cuda"):
            raise ValueError("device must be 'cpu', 'auto', or start with 'cuda'")
        if self.runtime.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("runtime.amp_dtype must be 'float16' or 'bfloat16'")
        if self.runtime.prefetch_batches < 0:
            raise ValueError("runtime.prefetch_batches must be >= 0")
        if self.runtime.cpu_threads < 0:
            raise ValueError("runtime.cpu_threads must be >= 0")
        if self.runtime.heartbeat_sec <= 0:
            raise ValueError("runtime.heartbeat_sec must be > 0")
        if self.runtime.log_first_n_steps < 0:
            raise ValueError("runtime.log_first_n_steps must be >= 0")
        if self.runtime.log_every_steps <= 0:
            raise ValueError("runtime.log_every_steps must be > 0")
        if self.runtime.stage_progress_sec <= 0:
            raise ValueError("runtime.stage_progress_sec must be > 0")
        if self.runtime.step_progress_sec <= 0:
            raise ValueError("runtime.step_progress_sec must be > 0")
        if self.runtime.step_progress_min_items <= 0:
            raise ValueError("runtime.step_progress_min_items must be > 0")
        if self.runtime.max_loader_workers <= 0:
            raise ValueError("runtime.max_loader_workers must be > 0")
        if self.runtime.max_prefetch_batches <= 0:
            raise ValueError("runtime.max_prefetch_batches must be > 0")
        if self.runtime.max_text_batch_size <= 0:
            raise ValueError("runtime.max_text_batch_size must be > 0")
        if self.runtime.max_image_forward_batch_size <= 0:
            raise ValueError("runtime.max_image_forward_batch_size must be > 0")
        if self.runtime.max_batch_edges_gpu <= 0:
            raise ValueError("runtime.max_batch_edges_gpu must be > 0")
        if self.eval.val_max_edges < 0 or self.eval.test_max_edges < 0:
            raise ValueError("eval.val_max_edges and eval.test_max_edges must be >= 0")
        if self.eval.hit10_negatives <= 0:
            raise ValueError("eval.hit10_negatives must be > 0")
        if self.eval.hit10_pos_limit < 0:
            raise ValueError("eval.hit10_pos_limit must be >= 0")
        if self.eval.final_val_max_edges < 0 or self.eval.final_test_max_edges < 0:
            raise ValueError("eval.final_val_max_edges and eval.final_test_max_edges must be >= 0")
        if self.eval.final_negatives_per_positive <= 0:
            raise ValueError("eval.final_negatives_per_positive must be > 0")
        if self.eval.final_hit10_pos_limit < 0:
            raise ValueError("eval.final_hit10_pos_limit must be >= 0")
        if self.eval.final_hit10_negatives <= 0:
            raise ValueError("eval.final_hit10_negatives must be > 0")
        if self.eval.final_test_negatives_per_positive <= 0:
            raise ValueError("eval.final_test_negatives_per_positive must be > 0")
        if self.eval.final_test_batch_edges <= 0:
            raise ValueError("eval.final_test_batch_edges must be > 0")
        if self.eval.final_test_embedding_cache_dtype not in {"float16", "float32"}:
            raise ValueError(
                "eval.final_test_embedding_cache_dtype must be 'float16' or 'float32'"
            )


def load_config(config_path: Path) -> ExperimentConfig:
    # Use utf-8-sig so configs written by PowerShell (with BOM) can also be loaded.
    with config_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    cfg = ExperimentConfig.from_dict(raw)
    cfg.validate()
    return cfg


def dump_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
