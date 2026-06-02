"""Experiment assembly helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from project.config import ExperimentConfig
from project.data.adjacency_index import DiskAdjacencyIndex
from project.data.image_provider import ByDirectoryImageProvider
from project.data.propagation import TypedPropagationGraph, build_train_propagation_edges
from project.data.sampler import NeighborSampler
from project.data.splits import create_or_load_relation_stratified_splits
from project.data.text_store import TextStore
from project.data.typed_graph import EntityCatalog, TripleStore, load_or_build_triple_store
from project.models.fusion import build_fusion
from project.models.gnn import build_gnn
from project.models.image_cache import ChunkAlignedImageCache
from project.models.image_encoders import build_image_encoder
from project.models.kge_scorers import build_kge_scorer
from project.models.model import MultimodalKGEModel
from project.models.model_store import slugify_model_id
from project.models.text_encoders import build_text_encoder
from project.utils.io import ensure_dir
from project.utils.logging import log_event


@dataclass
class ExperimentBundle:
    cfg: ExperimentConfig
    config_base_dir: Path
    sample_chunks: int
    resolved_paths: Dict[str, Path]
    cache_root: Path
    embed_cache_root: Path
    runs_root: Path
    catalog: EntityCatalog
    triple_store: TripleStore
    splits: Optional[Dict[str, np.ndarray]]
    train_propagation_graph: TypedPropagationGraph
    adjacency_index: Optional[DiskAdjacencyIndex]
    neighbor_sampler: Optional[NeighborSampler]
    text_store: TextStore
    image_provider: ByDirectoryImageProvider
    model: MultimodalKGEModel
    device: torch.device
    image_transform: Any
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]


class NullTextEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)

    def get_node_embeddings(self, node_wids, node_chunk_ids, node_chunk_pos, text_store, output_device):
        _ = (node_wids, node_chunk_ids, node_chunk_pos, text_store)
        return torch.zeros((len(node_wids), self.output_dim), device=output_device)

    def consume_runtime_stats(self) -> Dict[str, float]:
        return {
            "text_cache_hit": 0.0,
            "text_cache_miss": 0.0,
            "text_cache_hit_rate": 0.0,
            "text_cache_write_count": 0.0,
            "text_encode_sec": 0.0,
        }


class NullImageEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], self.output_dim), device=x.device, dtype=x.dtype)


class NullImageProvider:
    def get_batch_tensors(self, wids, has_image_mask, transform, pin_memory=False):
        _ = (wids, has_image_mask, transform, pin_memory)
        return {
            "real_images_cpu": torch.empty((0, 3, 0, 0)),
            "real_node_indices_cpu": torch.empty((0,), dtype=torch.long),
            "loaded_mask_cpu": torch.empty((0,), dtype=torch.bool),
            "stats_image_decode_sec": 0.0,
        }

    def close(self) -> None:
        return


class NullTextStore:
    def get_text(self, wid: str) -> str:
        _ = wid
        return ""

    def get_texts(self, wids):
        return ["" for _ in wids]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is False: {device_name}")
    return torch.device(device_name)


def require_cuda_device(device: torch.device, stage: str) -> None:
    if torch.device(device).type != "cuda":
        raise RuntimeError(f"{stage} requires CUDA, but resolved device is {device}")


def _validate_resolved_local_paths(resolved: Dict[str, Path]) -> None:
    required_dirs = ("graph_dir", "text_dir", "image_dir", "local_model_root")
    for key in required_dirs:
        path = Path(resolved[key])
        if not path.exists():
            raise FileNotFoundError(f"Resolved path does not exist for {key}: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Resolved path is not a directory for {key}: {path}")
    if Path(resolved["graph_dir"]) not in Path(resolved["text_dir"]).parents:
        raise RuntimeError("text_dir must be located under graph_dir to keep graph/text data consistent")
    for subdir in ("text", "image"):
        target = Path(resolved["local_model_root"]) / subdir
        if not target.is_dir():
            raise FileNotFoundError(f"Local model subdirectory not found: {target}")


def _text_raw_cache_key(cfg: ExperimentConfig, text_model_name: str) -> str:
    return slugify_model_id(
        f"{cfg.model.text_encoder}__{text_model_name}__maxlen{int(cfg.model.text_max_length)}"
    )


def _image_raw_cache_key(cfg: ExperimentConfig, image_model_name: str, output_dim: int) -> str:
    return slugify_model_id(
        (
            f"{cfg.model.image_encoder}__{image_model_name}"
            f"__size{int(cfg.model.image_size)}"
            f"__pretrained{int(bool(cfg.model.resnet_pretrained))}"
            f"__frozen{int(bool(cfg.model.freeze_image_backbone))}"
            f"__out{int(output_dim)}"
        )
    )


def build_experiment_bundle(
    cfg: ExperimentConfig,
    config_base_dir: Path,
    sample_chunks: int = 0,
    force_rebuild_data: bool = False,
    rebuild_split: bool = False,
    for_training: bool = False,
    cache_session: str | None = None,
) -> ExperimentBundle:
    resolved = cfg.resolved_paths(config_base_dir)
    _validate_resolved_local_paths(resolved)
    text_model_name = str(cfg.model.bert_model)
    image_model_name = str(cfg.model.image_model or cfg.model.image_encoder)
    text_model_dir = resolved["local_model_root"] / "text" / slugify_model_id(text_model_name)
    image_model_dir = resolved["local_model_root"] / "image" / slugify_model_id(image_model_name)
    log_event(
        "[CONFIG]",
        (
            f"paths graph_dir={resolved['graph_dir']} text_dir={resolved['text_dir']} "
            f"image_dir={resolved['image_dir']} local_model_root={resolved['local_model_root']}"
        ),
    )
    log_event(
        "[CONFIG]",
        (
            f"models text_encoder={cfg.model.text_encoder} text_model={text_model_name} "
            f"text_model_dir={text_model_dir} image_encoder={cfg.model.image_encoder} "
            f"image_model={image_model_name} image_model_dir={image_model_dir} "
            f"graph_encoder={cfg.model.graph_encoder} use_graph={cfg.model.use_graph} "
            f"kge_scorer={cfg.model.kge_scorer}"
        ),
    )

    sample_tag = f"sample_{int(sample_chunks)}" if int(sample_chunks) > 0 else "full"
    cache_root = resolved["cache_dir"] / sample_tag
    cache_session_name = slugify_model_id(str(cache_session).strip()) if cache_session else "default"
    embed_cache_root = cache_root / "shared_raw"
    runs_root = resolved["runs_dir"]
    ensure_dir(cache_root)
    ensure_dir(embed_cache_root)
    ensure_dir(runs_root)
    log_event(
        "[CONFIG]",
        (
            f"cache_root={cache_root} cache_session={cache_session_name} "
            f"raw_cache_root={embed_cache_root}"
        ),
    )

    catalog = EntityCatalog(
        graph_dir=resolved["graph_dir"],
        cache_dir=cache_root / "catalog",
        sample_chunks=sample_chunks,
        verbose=cfg.dataloader.verbose_data,
    )
    triple_store = load_or_build_triple_store(
        graph_dir=resolved["graph_dir"],
        cache_dir=cache_root / "triples",
        catalog=catalog,
        force_rebuild=force_rebuild_data,
        verbose=cfg.dataloader.verbose_data,
        progress_every_sec=cfg.runtime.stage_progress_sec,
    )
    splits = create_or_load_relation_stratified_splits(
        triples=triple_store.triples,
        split_path=cache_root / "splits" / "relation_stratified.npz",
        seed=cfg.seed,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        rare_relation_min_count=cfg.data.rare_relation_min_count,
        ensure_entities_in_train=True,
        force_rebuild=rebuild_split,
    )
    train_propagation_graph = build_train_propagation_edges(
        triples=triple_store.triples,
        train_idx=splits["train_idx"],
        num_entities=catalog.num_entities(),
        num_relations=len(triple_store.idx_to_relation),
    )
    log_event(
        "[DATA]",
        (
            f"catalog entities={catalog.num_entities()} art_entities={int(catalog.node_is_art.sum())} "
            f"chunks={catalog.chunk_count()} triples={int(triple_store.triples.shape[0])} "
            f"relations={len(triple_store.idx_to_relation)}"
        ),
    )
    log_event(
        "[DATA]",
        (
            f"splits train={int(splits['train_idx'].shape[0])} val={int(splits['val_idx'].shape[0])} "
            f"test={int(splits['test_idx'].shape[0])} sample_chunks={int(sample_chunks)} "
            f"train_propagation_edges={int(train_propagation_graph.edge_index.shape[1])}"
        ),
    )
    log_event(
        "[EVAL]",
        (
            f"validation_max_triples={int(cfg.eval.val_max_triples)} "
            f"final_test_max_triples={'FULL' if int(cfg.eval.test_max_triples) <= 0 else int(cfg.eval.test_max_triples)}"
        ),
    )

    adjacency_index: Optional[DiskAdjacencyIndex] = None
    neighbor_sampler: Optional[NeighborSampler] = None
    if bool(cfg.model.use_graph):
        adjacency_index = DiskAdjacencyIndex(
            catalog=catalog,
            index_dir=cache_root / "adj_index",
            source_graph=train_propagation_graph,
            lru_chunks=8,
            verbose=cfg.dataloader.verbose_data,
            edge_source_name="train_propagation",
        )
        adjacency_index.build_if_needed(rebuild=force_rebuild_data)
        neighbor_sampler = NeighborSampler(
            adjacency=adjacency_index,
            num_neighbors=cfg.data.num_neighbors,
            num_hops=cfg.data.num_hops,
            seed=cfg.seed + 17,
        )

    device = resolve_device(cfg.device)
    if cfg.model.use_text:
        text_store = TextStore(
            text_dir=resolved["text_dir"],
            cache_dir=cache_root / "text_store",
            lru_chunks=max(1, int(cfg.text_cache.lru_chunks)),
            verbose=cfg.dataloader.verbose_data,
            allowed_chunks=catalog.selected_chunk_names(),
        )
        text_cache_key = _text_raw_cache_key(cfg, text_model_name)
        text_encoder = build_text_encoder(
            name=cfg.model.text_encoder,
            catalog=catalog,
            cache_dir=embed_cache_root / "text" / text_cache_key,
            local_model_root=resolved["local_model_root"],
            model_name=text_model_name,
            max_length=cfg.model.text_max_length,
            batch_size=cfg.model.text_batch_size,
            device=str(device),
            cache_dtype=cfg.text_cache.dtype,
            cache_lru_chunks=cfg.text_cache.lru_chunks,
        )
    else:
        text_store = NullTextStore()
        text_encoder = NullTextEncoder(output_dim=cfg.model.text_proj_dim)

    if cfg.model.use_image:
        image_provider = ByDirectoryImageProvider(
            image_dir=resolved["image_dir"],
            cache_dir=cache_root / "image_index",
            decode_lru=max(1, int(cfg.dataloader.image_index_lru)),
            verbose=cfg.dataloader.verbose_data,
            preindex=True,
            num_workers=cfg.dataloader.image_decode_workers,
            stage_progress_sec=cfg.runtime.stage_progress_sec,
        )
        image_encoder = build_image_encoder(
            name=cfg.model.image_encoder,
            local_model_root=resolved["local_model_root"],
            model_name=image_model_name,
            pretrained=cfg.model.resnet_pretrained,
            freeze_backbone=cfg.model.freeze_image_backbone,
        )
        use_image_cache = bool(cfg.image_cache.enabled)
        if for_training and not bool(cfg.model.freeze_image_backbone) and use_image_cache:
            use_image_cache = False
            log_event(
                "[CONFIG]",
                (
                    "image cache disabled because image backbone is trainable "
                    f"(image_encoder={cfg.model.image_encoder})"
                ),
            )
        image_cache_key = _image_raw_cache_key(cfg, image_model_name, int(image_encoder.output_dim))
        image_cache = (
            ChunkAlignedImageCache(
                cache_dir=embed_cache_root / "image" / image_cache_key,
                catalog=catalog,
                emb_dim=int(image_encoder.output_dim),
                dtype=cfg.image_cache.dtype,
                encoder_name=cfg.model.image_encoder,
                model_name=image_model_name,
                image_size=cfg.model.image_size,
                pretrained=cfg.model.resnet_pretrained,
                freeze_backbone=cfg.model.freeze_image_backbone,
                cache_key=image_cache_key,
                lru_chunks=cfg.image_cache.lru_chunks,
            )
            if use_image_cache
            else None
        )
    else:
        image_provider = NullImageProvider()
        image_encoder = NullImageEncoder(output_dim=cfg.model.image_proj_dim)
        image_cache = None

    fusion = build_fusion(
        name=cfg.model.fusion,
        in_dim=int(cfg.model.text_proj_dim + cfg.model.image_proj_dim + 1),
        hidden_dim=cfg.model.fusion_hidden_dim,
        out_dim=cfg.model.fusion_out_dim,
        dropout=cfg.model.dropout,
    )
    graph_hidden_dim = int(cfg.model.graphsage_hidden_dim)
    graph_layers = int(cfg.model.graphsage_layers)
    graph_dropout = float(cfg.model.graphsage_dropout)
    graph_num_bases = int(cfg.model.rgcn_num_bases)
    if bool(cfg.model.use_graph) and cfg.model.graph_encoder == "rgcn":
        graph_hidden_dim = int(cfg.model.rgcn_hidden_dim)
        graph_layers = int(cfg.model.rgcn_layers)
        graph_dropout = float(cfg.model.rgcn_dropout)
    graph_encoder = build_gnn(
        name=cfg.model.graph_encoder if bool(cfg.model.use_graph) else "nohop",
        in_dim=int(cfg.model.fusion_out_dim),
        hidden_dim=graph_hidden_dim,
        layers=graph_layers,
        dropout=graph_dropout,
        num_relations=len(triple_store.idx_to_relation),
        num_bases=graph_num_bases,
    )
    kge_scorer = build_kge_scorer(
        name=cfg.model.kge_scorer,
        num_relations=len(triple_store.idx_to_relation),
        dim=cfg.model.kge_dim,
    )
    model = MultimodalKGEModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion=fusion,
        graph_encoder=graph_encoder,
        kge_scorer=kge_scorer,
        use_text=cfg.model.use_text,
        use_image=cfg.model.use_image,
        use_graph=cfg.model.use_graph,
        graph_encoder_name=cfg.model.graph_encoder,
        text_proj_dim=cfg.model.text_proj_dim,
        image_proj_dim=cfg.model.image_proj_dim,
        fusion_out_dim=cfg.model.fusion_out_dim,
        image_forward_batch_size=cfg.model.image_forward_batch_size,
        image_cache=image_cache,
    ).to(device)

    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    if for_training:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        if cfg.optim.scheduler == "onecycle":
            train_steps_per_epoch = max(1, math.ceil(int(splits["train_idx"].shape[0]) / int(cfg.data.batch_size)))
            total_steps = int(cfg.optim.max_steps) if int(cfg.optim.max_steps) > 0 else int(cfg.optim.epochs) * train_steps_per_epoch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.optim.lr,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos",
            )

    if hasattr(image_encoder, "get_image_transform"):
        image_transform = image_encoder.get_image_transform(cfg.model.image_size)
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize((cfg.model.image_size, cfg.model.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return ExperimentBundle(
        cfg=cfg,
        config_base_dir=Path(config_base_dir),
        sample_chunks=int(sample_chunks),
        resolved_paths=resolved,
        cache_root=cache_root,
        embed_cache_root=embed_cache_root,
        runs_root=runs_root,
        catalog=catalog,
        triple_store=triple_store,
        splits=splits,
        train_propagation_graph=train_propagation_graph,
        adjacency_index=adjacency_index,
        neighbor_sampler=neighbor_sampler,
        text_store=text_store,
        image_provider=image_provider,
        model=model,
        device=device,
        image_transform=image_transform,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def prebuild_raw_feature_caches(bundle: ExperimentBundle) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    text_encoder = getattr(bundle.model, "text_encoder", None)
    if hasattr(text_encoder, "prebuild_cache"):
        stats.update(text_encoder.prebuild_cache(bundle.text_store))
    if (
        bool(bundle.cfg.model.use_image)
        and bool(bundle.cfg.model.freeze_image_backbone)
        and getattr(bundle.model, "image_cache", None) is not None
        and hasattr(bundle.model, "prebuild_image_cache")
    ):
        stats.update(
            bundle.model.prebuild_image_cache(
                catalog=bundle.catalog,
                image_provider=bundle.image_provider,
                image_transform=bundle.image_transform,
                device=bundle.device,
                pin_memory=bool(bundle.cfg.runtime.pin_memory and bundle.device.type == "cuda"),
                progress_every_sec=bundle.cfg.runtime.stage_progress_sec,
            )
        )
    return stats


def load_bundle_from_checkpoint(checkpoint_path: Path, device_override: Optional[str] = None) -> ExperimentBundle:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    if device_override:
        cfg.device = device_override
    bundle = build_experiment_bundle(
        cfg=cfg,
        config_base_dir=Path(ckpt["config_base_dir"]),
        sample_chunks=int(ckpt.get("sample_chunks", 0)),
        force_rebuild_data=False,
        rebuild_split=False,
        for_training=False,
        cache_session=Path(checkpoint_path).resolve().parent.parent.name,
    )
    bundle.model.load_state_dict(ckpt["model"])
    bundle.model.to(bundle.device)
    bundle.model.eval()
    return bundle
