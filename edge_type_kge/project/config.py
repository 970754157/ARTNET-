"""Configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class PathsConfig:
    graph_dir: str
    text_dir: str
    image_dir: str
    cache_dir: str
    runs_dir: str
    local_model_root: str = "../../link_prediction_experiments/local_models"


@dataclass
class ModelConfig:
    use_text: bool
    use_image: bool
    use_graph: bool
    text_encoder: str
    image_encoder: str
    graph_encoder: str
    fusion: str
    kge_scorer: str
    bert_model: str
    image_model: str
    text_max_length: int
    text_batch_size: int
    image_size: int
    image_forward_batch_size: int
    resnet_pretrained: bool
    text_proj_dim: int
    image_proj_dim: int
    fusion_hidden_dim: int
    fusion_out_dim: int
    kge_dim: int
    freeze_image_backbone: bool = True
    dropout: float = 0.1


@dataclass
class TextCacheConfig:
    dtype: str
    lru_chunks: int


@dataclass
class ImageCacheConfig:
    dtype: str
    lru_chunks: int
    enabled: bool = True


@dataclass
class DataConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    batch_size: int
    num_neg: int
    relation_stratified: bool
    rare_relation_min_count: int
    eval_batch_size: int


@dataclass
class OptimConfig:
    lr: float
    weight_decay: float
    epochs: int
    max_steps: int
    scheduler: str
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0


@dataclass
class EvalConfig:
    eval_every_steps: int
    val_max_triples: int
    test_max_triples: int
    relation_topk: int


@dataclass
class LoggingConfig:
    save_every_steps: int
    best_metric: str


@dataclass
class DataloaderConfig:
    image_decode_workers: int
    image_index_lru: int
    verbose_data: bool = True


@dataclass
class RuntimeConfig:
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    stage_progress_sec: int = 30
    eval_progress_sec: int = 1800


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    device: str
    paths: PathsConfig
    model: ModelConfig
    text_cache: TextCacheConfig
    image_cache: ImageCacheConfig
    data: DataConfig
    optim: OptimConfig
    eval: EvalConfig
    logging: LoggingConfig
    dataloader: DataloaderConfig
    runtime: RuntimeConfig

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperimentConfig":
        raw_paths = dict(raw["paths"])
        raw_paths.setdefault("local_model_root", "../../link_prediction_experiments/local_models")
        raw_model = dict(raw["model"])
        raw_model.setdefault("freeze_image_backbone", True)
        raw_model.setdefault("image_model", "")
        raw_image_cache = dict(raw.get("image_cache", {}))
        raw_image_cache.setdefault("dtype", "float16")
        raw_image_cache.setdefault("lru_chunks", 4)
        raw_image_cache.setdefault("enabled", True)
        return cls(
            experiment_name=str(raw["experiment_name"]),
            seed=int(raw["seed"]),
            device=str(raw["device"]),
            paths=PathsConfig(**raw_paths),
            model=ModelConfig(**raw_model),
            text_cache=TextCacheConfig(**raw["text_cache"]),
            image_cache=ImageCacheConfig(**raw_image_cache),
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
            "text_cache": self.text_cache.__dict__,
            "image_cache": self.image_cache.__dict__,
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
        for key, value in self.paths.__dict__.items():
            if "://" in str(value):
                raise ValueError(f"paths.{key} must be a local filesystem path")
            if Path(str(value)).is_absolute():
                raise ValueError(f"paths.{key} must be configured as a relative path")
        if not self.model.use_text and not self.model.use_image:
            raise ValueError("At least one of use_text/use_image must be true")
        if self.model.use_graph and self.model.graph_encoder == "none":
            raise ValueError("use_graph=true requires graph_encoder != none")
        if self.model.graph_encoder not in {"none", "graphsage", "rgcn"}:
            raise ValueError("Unsupported graph_encoder")
        if self.model.text_encoder not in {"bert_frozen", "roberta_frozen", "albert_frozen", "sbert", "custom"}:
            raise ValueError("Unsupported text encoder")
        if self.model.image_encoder not in {
            "resnet18",
            "resnet50",
            "vgg19",
            "dinov2_base",
            "fasterrcnn_resnet50_fpn_v2",
            "inception_resnet_v2",
            "vit",
            "dinov2",
        }:
            raise ValueError("Unsupported image encoder")
        if not self.model.freeze_image_backbone:
            if self.model.image_encoder != "resnet18" or self.model.resnet_pretrained:
                raise ValueError(
                    "Trainable image backbone is only supported for resnet18 scratch "
                    "(image_encoder=resnet18, resnet_pretrained=false)."
                )
            if self.model.text_encoder != "bert_frozen":
                raise ValueError("Trainable image backbone setup requires text_encoder=bert_frozen.")
        if self.model.fusion not in {"mlp_concat", "gated", "attention_pool"}:
            raise ValueError("Unsupported fusion")
        if self.model.kge_scorer not in {"transe", "complex"}:
            raise ValueError("kge_scorer must be transe or complex")
        if self.image_cache.dtype not in {"float16", "float32"}:
            raise ValueError("image_cache.dtype must be float16 or float32")
        if self.image_cache.lru_chunks <= 0:
            raise ValueError("image_cache.lru_chunks must be > 0")
        if self.optim.max_steps < 0:
            raise ValueError("max_steps must be >= 0, where 0 means no explicit step cap")
        if self.optim.scheduler not in {"none", "onecycle"}:
            raise ValueError("scheduler must be none or onecycle")
        if self.optim.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        if self.optim.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        if self.logging.best_metric not in {"val_mrr", "val_auc", "val_aucpr"}:
            raise ValueError("best_metric must be val_mrr, val_auc, or val_aucpr")
        if self.device not in {"cpu", "auto"} and not str(self.device).startswith("cuda"):
            raise ValueError("device must be cpu, auto, or cuda*")
        if self.runtime.eval_progress_sec < 0:
            raise ValueError("runtime.eval_progress_sec must be >= 0")


def load_config(config_path: Path) -> ExperimentConfig:
    with config_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    cfg = ExperimentConfig.from_dict(raw)
    cfg.validate()
    return cfg
