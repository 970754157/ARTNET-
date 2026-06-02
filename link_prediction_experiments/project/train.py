"""Main training entrypoint."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Dict


def _sanitize_thread_env() -> None:
    value = str(os.environ.get("OMP_NUM_THREADS", "")).strip()
    if value:
        try:
            if int(value) > 0:
                return
        except Exception:
            pass
    os.environ["OMP_NUM_THREADS"] = "8"


_sanitize_thread_env()

import torch

from project.config import dump_config, load_config
from project.data.adjacency_index import DiskAdjacencyIndex
from project.data.image_provider import ByDirectoryImageProvider
from project.data.sampler import NeighborSampler
from project.data.sharded_graph import ShardedGraphDataset
from project.data.splits import (
    create_or_load_edge_hash,
    create_or_load_splits,
)
from project.data.text_store import TextStore
from project.models.full_model import MultimodalLinkModel
from project.models.registry import (
    build_fusion,
    build_gnn,
    build_image_encoder,
    build_predictor,
    build_text_encoder,
)
from project.models.model_store import slugify_model_id
from project.trainer import Trainer, TrainerContext
from project.utils.io import atomic_write_json, ensure_dir
from project.utils.logging import log_event, log_stage
from project.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (last.pt)")
    parser.add_argument("--build-index-only", action="store_true", help="Only build adjacency index then exit")
    parser.add_argument(
        "--build-text-cache-only",
        action="store_true",
        help="Only prebuild frozen-text cache then exit",
    )
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild adjacency index")
    parser.add_argument("--rebuild-split", action="store_true", help="Force rebuild edge split")
    parser.add_argument(
        "--sample-chunks",
        type=int,
        default=0,
        help="Use first N graph chunks for debug. 0 means full dataset.",
    )
    return parser.parse_args()


def create_run_dir(runs_root: Path, experiment_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{ts}_{experiment_name}"
    ensure_dir(run_dir)
    return run_dir


def apply_sample_mode_paths(resolved: Dict[str, Path], sample_chunks: int) -> Dict[str, Path]:
    if sample_chunks <= 0:
        return resolved
    debug_root = resolved["cache_dir"] / f"debug_chunks_{sample_chunks}"
    out = dict(resolved)
    out["cache_dir"] = debug_root
    out["runs_dir"] = resolved["runs_dir"] / f"debug_chunks_{sample_chunks}"
    out["split_file"] = debug_root / "split_edges.pt"
    out["adj_index_dir"] = debug_root / "adj_index"
    out["text_cache_dir"] = debug_root / "text_emb"
    return out


def resolve_device(device_name: str) -> torch.device:
    name = str(device_name).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested in config, but torch.cuda.is_available() is False")
        return torch.device(device_name)
    return torch.device(device_name)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


def device_summary(cfg, device: torch.device) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "config_device": str(cfg.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "resolved_device": str(device),
        "amp_enabled": bool(device.type == "cuda" and cfg.runtime.amp_enabled),
        "amp_dtype": str(cfg.runtime.amp_dtype),
        "cudnn_benchmark": bool(device.type == "cuda" and cfg.runtime.cudnn_benchmark),
        "visible_cuda_devices": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        summary.update(
            {
                "gpu_name": str(props.name),
                "gpu_total_memory_gb": round(float(props.total_memory) / (1024**3), 2),
                "gpu_index": int(device.index or 0),
            }
        )
    return summary


def apply_runtime_auto_tune(cfg, device: torch.device, sample_chunks: int) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "auto_tune_enabled": bool(cfg.runtime.auto_tune),
        "auto_tune_applied": False,
    }
    if not cfg.runtime.auto_tune or device.type != "cuda" or platform.system().lower() != "linux":
        return summary

    cpu_count = max(1, int(os.cpu_count() or 1))
    props = torch.cuda.get_device_properties(device)
    gpu_mem_gb = float(props.total_memory) / (1024**3)

    cfg.dataloader.num_workers = _clamp(
        max(int(cfg.dataloader.num_workers), max(8, (cpu_count * 3) // 4)),
        8,
        int(cfg.runtime.max_loader_workers),
    )
    cfg.runtime.prefetch_batches = _clamp(
        2,
        1,
        int(cfg.runtime.max_prefetch_batches),
    )
    if gpu_mem_gb >= 24.0:
        text_target = 256
    elif gpu_mem_gb >= 12.0:
        text_target = 128
    else:
        text_target = 64
    cfg.model.text_batch_size = min(
        int(cfg.runtime.max_text_batch_size),
        max(int(cfg.model.text_batch_size), int(text_target)),
    )
    cfg.model.image_forward_batch_size = min(
        int(cfg.runtime.max_image_forward_batch_size),
        32,
    )
    cfg.data.batch_edges = 256

    summary.update(
        {
            "auto_tune_applied": True,
            "cpu_count": cpu_count,
            "gpu_total_memory_gb": round(gpu_mem_gb, 2),
            "effective_num_workers": int(cfg.dataloader.num_workers),
            "effective_prefetch_batches": int(cfg.runtime.prefetch_batches),
            "effective_text_batch_size": int(cfg.model.text_batch_size),
            "effective_image_forward_batch_size": int(cfg.model.image_forward_batch_size),
            "effective_batch_edges": int(cfg.data.batch_edges),
        }
    )
    return summary


def dump_resolved_config(
    cfg,
    resolved: Dict[str, Path],
    run_dir: Path,
    device_info: Dict[str, object],
    auto_summary: Dict[str, object],
    extras: Dict[str, object] | None = None,
) -> None:
    payload = cfg.to_dict()
    payload["resolved_paths"] = {k: str(v) for k, v in resolved.items()}
    payload["device_summary"] = device_info
    payload["auto_tune_summary"] = auto_summary
    if extras:
        payload["runtime_resolved"] = extras
    atomic_write_json(run_dir / "config_resolved.json", payload)


def compute_steps_per_epoch(cfg, train_edges: Any) -> int:
    return max(1, int(train_edges.shape[1]) // int(cfg.data.batch_edges))


def clear_protocol_cache_dir(protocol_cache_dir: Path) -> None:
    if protocol_cache_dir.exists():
        shutil.rmtree(protocol_cache_dir)
    ensure_dir(protocol_cache_dir)


def main() -> None:
    args = parse_args()
    if args.sample_chunks < 0:
        raise ValueError("--sample-chunks must be >= 0")

    with log_stage("load config", prefix="[BOOT]"):
        config_path = Path(args.config).resolve()
        cfg = load_config(config_path)
        original_cfg = copy.deepcopy(cfg)
        base_dir = config_path.parent
        resolved = cfg.resolved_paths(base_dir)
        resolved = apply_sample_mode_paths(resolved, args.sample_chunks)
        protocol_cache_dir = resolved["cache_dir"] / "protocol_leak_safe_v1"
        resolved["protocol_cache_dir"] = protocol_cache_dir
        resolved["split_file"] = protocol_cache_dir / "split_edges.pt"
        resolved["adj_index_dir"] = protocol_cache_dir / "adj_index"
        resolved["chunk_edge_hash_file"] = protocol_cache_dir / "all_positive_undirected_hash.npy"
        text_model_slug = slugify_model_id(cfg.model.text_model_name)
        text_cache_variant = (
            f"{text_model_slug}_ml{int(cfg.model.text_max_length)}"
            f"_tp{int(cfg.model.text_proj_dim)}"
            f"_ip{int(cfg.model.image_proj_dim)}"
        )
        resolved["text_cache_dir"] = resolved["text_cache_dir"] / text_cache_variant

        clear_protocol_cache_dir(protocol_cache_dir)
        log_event("[BOOT]", f"protocol cache cleared: {protocol_cache_dir}")

        for key in ("cache_dir", "runs_dir", "text_cache_dir", "local_model_root"):
            ensure_dir(resolved[key])
        ensure_dir(protocol_cache_dir)

    set_global_seed(cfg.seed)
    device = resolve_device(cfg.device)
    device_info = device_summary(cfg, device)
    if str(cfg.device).strip().lower() == "auto" and torch.cuda.is_available() and device.type != "cuda":
        raise RuntimeError("device=auto but CUDA is available and resolved device is not cuda")
    if str(cfg.device).strip().lower() == "cpu" and torch.cuda.is_available():
        log_event("[DEVICE]", "CUDA is available but config explicitly selected CPU.")
    auto_summary = apply_runtime_auto_tune(cfg, device, args.sample_chunks)
    device_info = device_summary(cfg, device)
    log_event(
        "[DEVICE]",
        (
            f"cfg.device={device_info['config_device']} "
            f"cuda_available={device_info['cuda_available']} "
            f"resolved_device={device_info['resolved_device']} "
            f"visible_cuda_devices={device_info['visible_cuda_devices']}"
        ),
    )
    if device.type == "cuda":
        log_event(
            "[DEVICE]",
            (
                f"gpu_name={device_info['gpu_name']} "
                f"gpu_total_memory_gb={device_info['gpu_total_memory_gb']} "
                f"amp_enabled={device_info['amp_enabled']} "
                f"amp_dtype={device_info['amp_dtype']} "
                f"cudnn_benchmark={device_info['cudnn_benchmark']}"
            ),
        )
    if auto_summary.get("auto_tune_applied"):
        auto_msg = (
            f"workers={auto_summary['effective_num_workers']} "
            f"prefetch={auto_summary['effective_prefetch_batches']} "
            f"text_batch={auto_summary['effective_text_batch_size']} "
            f"image_forward_batch={auto_summary['effective_image_forward_batch_size']} "
            f"batch_edges={auto_summary['effective_batch_edges']} "
            f"max_subgraph_nodes={int(cfg.data.max_subgraph_nodes)} "
            f"max_image_nodes={int(cfg.data.max_image_nodes)}"
        )
        log_event("[AUTO]", auto_msg)

    if cfg.runtime.cpu_threads > 0:
        torch.set_num_threads(int(cfg.runtime.cpu_threads))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.runtime.cudnn_benchmark)

    if args.resume:
        resume_path = Path(args.resume).resolve()
        run_dir = resume_path.parent.parent
        ensure_dir(run_dir)
        if not (run_dir / "config.json").exists():
            dump_config(original_cfg, run_dir / "config.json")
        dump_resolved_config(cfg, resolved, run_dir, device_info, auto_summary)
    elif not args.build_index_only and not args.build_text_cache_only:
        exp_name = cfg.experiment_name
        if args.sample_chunks > 0:
            exp_name = f"{exp_name}_debug_c{args.sample_chunks}"
        run_dir = create_run_dir(resolved["runs_dir"], exp_name)
        dump_config(original_cfg, run_dir / "config.json")
        dump_resolved_config(cfg, resolved, run_dir, device_info, auto_summary)
    else:
        run_dir = resolved["runs_dir"]

    with log_stage("graph init"):
        graph = ShardedGraphDataset(
            graph_dir=resolved["graph_dir"],
            cache_dir=resolved["cache_dir"],
            verbose=True,
            sample_chunks=args.sample_chunks,
            chunk_context_mode=cfg.data.chunk_context_mode,
            seed=cfg.seed,
            cross_chunk_neighbors_per_node=cfg.data.cross_chunk_neighbors_per_node,
        )
    if args.sample_chunks > 0:
        log_event(
            "[BOOT]",
            f"sample-chunks mode enabled: first {args.sample_chunks} chunks (actual loaded={graph.chunk_count()})",
        )
        log_event("[BOOT]", f"isolated cache dir: {resolved['cache_dir']}")
        log_event("[BOOT]", f"isolated runs dir: {resolved['runs_dir']}")

    with log_stage("split load/build"):
        split = create_or_load_splits(
            graph=graph,
            split_path=resolved["split_file"],
            seed=cfg.seed,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            force_rebuild=args.rebuild_split,
            progress_every_chunks=5,
            progress_every_sec=cfg.runtime.stage_progress_sec,
        )
    with log_stage("edge hash load/build"):
        edge_hash = create_or_load_edge_hash(
            split=split,
            hash_path=resolved["chunk_edge_hash_file"],
            num_nodes=graph.num_nodes,
            force_rebuild=False,
        )
    train_edges = split["train_pos_edges"]
    train_propagation_edges = split["train_propagation_edges"]
    val_edges = split["val_pos_edges"]
    test_edges = split["test_pos_edges"]
    propagation_unique_edges = int(train_propagation_edges.shape[1])
    propagation_self_loops = int(
        torch.sum(train_propagation_edges[0] == train_propagation_edges[1]).item()
    )
    propagation_edge_directions = int((propagation_unique_edges * 2) - propagation_self_loops)
    log_event(
        "[BOOT]",
        (
            f"propagation graph prepared: unique_train_edges={propagation_unique_edges} "
            f"directed_edge_directions={propagation_edge_directions}"
        ),
    )

    adj_index = None
    with log_stage("adjacency index"):
        adj_index = DiskAdjacencyIndex(
            graph=graph,
            index_dir=resolved["adj_index_dir"],
            lru_chunks=cfg.index.adj_lru_chunks,
            verbose=True,
            source_edges=train_propagation_edges.detach().cpu().numpy().astype("int64", copy=False),
            edge_source_name="train_propagation_undirected",
        )
        adj_index.build_if_needed(rebuild=(cfg.index.rebuild_index or args.rebuild_index))

    if args.build_index_only:
        log_event("[DONE]", "train-only adjacency index built.")
        return

    with log_stage("text store init"):
        text_store = TextStore(
            text_dir=resolved["text_dir"],
            cache_dir=resolved["cache_dir"] / "text_store",
            lru_chunks=4,
            verbose=True,
            allowed_chunks=graph.selected_chunk_names() if args.sample_chunks > 0 else None,
        )
    neighbor_sampler = NeighborSampler(
        adjacency=adj_index,
        num_neighbors=cfg.data.num_neighbors,
        num_hops=cfg.data.num_hops,
        seed=cfg.seed,
    )

    with log_stage("text encoder init"):
        text_encoder = build_text_encoder(
            name=cfg.model.text_encoder,
            graph=graph,
            cache_dir=resolved["text_cache_dir"],
            local_model_root=resolved["local_model_root"],
            model_name=cfg.model.text_model_name,
            max_length=cfg.model.text_max_length,
            batch_size=cfg.model.text_batch_size,
            device=str(device),
            cache_dtype=cfg.text_cache.dtype,
            cache_lru_chunks=cfg.text_cache.lru_chunks,
            progress_sec=cfg.runtime.stage_progress_sec,
            freeze_backbone=cfg.model.freeze_text_backbone,
        )
    if args.build_text_cache_only:
        with log_stage("text cache prebuild"):
            stats = text_encoder.prebuild_cache(
                graph=graph,
                text_store=text_store,
                device=device,
                chunk_ids=list(range(graph.chunk_count())),
                batch_size=cfg.model.text_batch_size,
            )
        log_event("[DONE]", "text cache prebuilt.")
        log_event("[DONE]", str(stats))
        return

    with log_stage("text cache prewarm"):
        prewarm_stats = text_encoder.prebuild_cache(
            graph=graph,
            text_store=text_store,
            device=device,
            chunk_ids=list(range(graph.chunk_count())),
            batch_size=cfg.model.text_batch_size,
        )
    log_event("[STAGE]", f"text cache prewarm done stats={prewarm_stats}")

    with log_stage("image provider init"):
        image_provider = ByDirectoryImageProvider(
            image_dir=resolved["image_dir"],
            cache_dir=resolved["cache_dir"] / "image_store",
            decode_lru=128,
            verbose=True,
            preindex=(args.sample_chunks <= 0),
            num_workers=cfg.dataloader.num_workers,
            stage_progress_sec=cfg.runtime.stage_progress_sec,
        )
    with log_stage("image encoder init"):
        image_encoder = build_image_encoder(
            name=cfg.model.image_encoder,
            local_model_root=resolved["local_model_root"],
            model_name=cfg.model.image_model_name,
            pretrained=cfg.model.image_pretrained,
            freeze_backbone=cfg.model.freeze_image_backbone,
        )
    image_transform = image_encoder.get_image_transform(cfg.model.image_size)
    text_feat_dim = min(int(text_encoder.output_dim), int(cfg.model.text_proj_dim))
    image_feat_dim = min(int(image_encoder.output_dim), int(cfg.model.image_proj_dim))
    fusion = build_fusion(
        name=cfg.model.fusion,
        in_dim=int(text_feat_dim + image_feat_dim + 1),
        hidden_dim=cfg.model.fusion_hidden_dim,
        out_dim=cfg.model.fusion_out_dim,
        dropout=cfg.model.graphsage_dropout,
    )
    gnn = build_gnn(
        name=cfg.model.gnn,
        in_dim=fusion.output_dim,
        hidden_dim=cfg.model.graphsage_hidden_dim,
        layers=cfg.model.graphsage_layers,
        dropout=cfg.model.graphsage_dropout,
    )
    predictor = build_predictor(
        name=cfg.model.predictor,
        in_dim=gnn.output_dim,
        hidden_dim=cfg.model.predictor_hidden_dim,
    )
    model = MultimodalLinkModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion=fusion,
        gnn=gnn,
        predictor=predictor,
        text_proj_dim=text_feat_dim,
        image_proj_dim=image_feat_dim,
        image_forward_batch_size=cfg.model.image_forward_batch_size,
    )

    steps_per_epoch = compute_steps_per_epoch(cfg, train_edges)
    epoch_step_budget = int(cfg.optim.epochs) * int(steps_per_epoch)
    effective_max_steps = min(int(cfg.optim.max_steps), epoch_step_budget)
    log_event(
        "[BOOT]",
        (
            f"resolved training limits: epochs={cfg.optim.epochs} "
            f"configured_max_steps={cfg.optim.max_steps} "
            f"steps_per_epoch={steps_per_epoch} "
            f"epoch_step_budget={epoch_step_budget} "
            f"effective_max_steps={effective_max_steps} "
            f"early_stop_enabled={cfg.optim.early_stop_enabled} "
            f"early_stop_patience={cfg.optim.early_stop_patience}"
        ),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = None
    if cfg.optim.scheduler.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.optim.lr,
            total_steps=effective_max_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    elif cfg.optim.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=effective_max_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.optim.scheduler}")

    criterion = torch.nn.BCEWithLogitsLoss()

    ctx = TrainerContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        graph=graph,
        text_store=text_store,
        image_provider=image_provider,
        neighbor_sampler=neighbor_sampler,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        edge_hash_sorted=edge_hash,
        device=device,
        run_dir=run_dir,
        cfg=cfg,
        resolved_paths=resolved,
        image_transform=image_transform,
        train_message_edges=None,
        train_propagation_edges=train_propagation_edges,
        effective_max_steps=effective_max_steps,
    )
    with log_stage("trainer init"):
        trainer = Trainer(ctx)

    dump_resolved_config(
        cfg,
        resolved,
        run_dir,
        device_info,
        auto_summary,
        extras={
            "steps_per_epoch": steps_per_epoch,
            "effective_max_steps": effective_max_steps,
            "early_stop_enabled": bool(cfg.optim.early_stop_enabled),
            "early_stop_patience": int(cfg.optim.early_stop_patience),
            "early_stop_min_delta": float(cfg.optim.early_stop_min_delta),
            "text_cache_prewarm_enabled": True,
            "text_cache_prewarm_stats": prewarm_stats,
            "max_subgraph_nodes": int(cfg.data.max_subgraph_nodes),
            "max_image_nodes": int(cfg.data.max_image_nodes),
        },
    )

    if args.resume:
        log_event("[BOOT]", f"resume checkpoint: {Path(args.resume).resolve()}")
        trainer.load_checkpoint(Path(args.resume))

    try:
        log_event("[STAGE]", "train loop starting")
        result = trainer.train()
        log_event("[DONE]", "training complete.")
        log_event("[DONE]", str(result))
    finally:
        if hasattr(image_provider, "close"):
            image_provider.close()


if __name__ == "__main__":
    main()
