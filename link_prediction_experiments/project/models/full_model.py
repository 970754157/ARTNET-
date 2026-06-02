"""Full multimodal link prediction model."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from project.data.text_store import TextStore
from project.utils.logging import log_event
from project.utils.logging import StepProgressReporter


class MultimodalLinkModel(nn.Module):
    """text -> image -> fusion -> graphsage -> link predictor."""

    def __init__(
        self,
        text_encoder: nn.Module,
        image_encoder: nn.Module,
        fusion: nn.Module,
        gnn: nn.Module,
        predictor: nn.Module,
        text_proj_dim: int | None = None,
        image_proj_dim: int | None = None,
        image_forward_batch_size: int = 32,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion = fusion
        self.gnn = gnn
        self.predictor = predictor
        self.image_forward_batch_size = max(1, int(image_forward_batch_size))

        text_in = int(self.text_encoder.output_dim)
        image_in = int(self.image_encoder.output_dim)
        self.text_feat_dim = int(text_proj_dim) if text_proj_dim is not None else text_in
        self.image_feat_dim = int(image_proj_dim) if image_proj_dim is not None else image_in

        self.text_proj = (
            nn.Identity()
            if self.text_feat_dim == text_in
            else nn.Linear(text_in, self.text_feat_dim)
        )
        self.image_proj = (
            nn.Identity()
            if self.image_feat_dim == image_in
            else nn.Linear(image_in, self.image_feat_dim)
        )

        self.img_missing_emb = nn.Parameter(torch.empty(self.image_feat_dim))
        nn.init.normal_(self.img_missing_emb, mean=0.0, std=0.02)
        self.image_forward_time_sec = 0.0
        self.image_decode_time_sec = 0.0

    def encode_nodes(
        self,
        edge_index: torch.Tensor,
        node_indices_cpu: np.ndarray,
        node_wids: List[str],
        node_chunk_ids: np.ndarray,
        node_chunk_pos: np.ndarray,
        real_node_image_indices_cpu: torch.Tensor,
        real_node_image_paths: List[str],
        node_has_image_mask: torch.Tensor,
        text_store: TextStore,
        image_provider: Any,
        image_transform,
        progress_reporter: Optional[StepProgressReporter] = None,
    ) -> torch.Tensor:
        device = edge_index.device
        text_emb = self.text_encoder.get_node_embeddings(
            node_indices=node_indices_cpu,
            node_wids=node_wids,
            node_chunk_ids=node_chunk_ids,
            node_chunk_pos=node_chunk_pos,
            text_store=text_store,
            output_device=device,
            progress_reporter=progress_reporter,
        )
        text_emb = self.text_proj(text_emb)
        has_image = node_has_image_mask.to(device=device).float().view(-1, 1)
        num_nodes = int(node_has_image_mask.shape[0])
        img_emb = torch.zeros(
            (num_nodes, int(self.image_encoder.output_dim)),
            device=device,
            dtype=text_emb.dtype,
        )

        real_count = int(real_node_image_indices_cpu.shape[0])
        if real_count > 0:
            if progress_reporter is not None:
                progress_reporter.emit(
                    "forward_image",
                    "start",
                    real_image_count=real_count,
                    chunk_batch_size=self.image_forward_batch_size,
                )
            real_indices = real_node_image_indices_cpu.detach().cpu().tolist()
            current_batch_size = int(self.image_forward_batch_size)
            start = 0
            while start < real_count:
                stop = min(start + current_batch_size, real_count)
                load_batch = image_provider.load_resolved_batch_tensors(
                    list(zip(real_indices[start:stop], real_node_image_paths[start:stop])),
                    transform=image_transform,
                    pin_memory=(device.type == "cuda"),
                )
                loaded_idx_cpu = load_batch["real_node_indices_cpu"]
                if int(loaded_idx_cpu.shape[0]) <= 0:
                    start = stop
                    continue
                batch_images = None
                batch_idx = None
                real_img_emb = None
                try:
                    batch_images = load_batch["real_images_cpu"].to(device)
                    if device.type == "cuda" and batch_images.ndim == 4:
                        batch_images = batch_images.contiguous(memory_format=torch.channels_last)
                    batch_idx = loaded_idx_cpu.to(device=device, dtype=torch.long)
                    start_time = time.perf_counter()
                    real_img_emb = self.image_encoder(batch_images)
                except torch.cuda.OutOfMemoryError:
                    if device.type != "cuda" or current_batch_size <= 1:
                        raise
                    next_batch_size = max(1, current_batch_size // 2)
                    del batch_images
                    del batch_idx
                    del real_img_emb
                    del load_batch
                    del loaded_idx_cpu
                    torch.cuda.empty_cache()
                    log_event(
                        "[DEVICE]",
                        (
                            f"forward_image cuda oom retry: chunk_batch_size={current_batch_size} "
                            f"-> {next_batch_size} start={start} total_images={real_count}"
                        ),
                    )
                    if progress_reporter is not None:
                        progress_reporter.emit(
                            "forward_image",
                            "oom_retry",
                            failed_batch_size=current_batch_size,
                            next_batch_size=next_batch_size,
                            encoded_images=start,
                            total_images=real_count,
                        )
                    current_batch_size = next_batch_size
                    self.image_forward_batch_size = current_batch_size
                    continue
                self.image_forward_time_sec += float(time.perf_counter() - start_time)
                self.image_decode_time_sec += float(load_batch["stats_image_decode_sec"])
                if real_img_emb.dtype != img_emb.dtype:
                    real_img_emb = real_img_emb.to(dtype=img_emb.dtype)
                img_emb.index_copy_(0, batch_idx, real_img_emb)
                if progress_reporter is not None and real_count >= progress_reporter.min_items:
                    progress_reporter.progress(
                        "forward_image",
                        encoded_images=stop,
                        total_images=real_count,
                        chunk_batch_size=current_batch_size,
                    )
                del batch_images
                del batch_idx
                del real_img_emb
                start = stop
            if progress_reporter is not None:
                progress_reporter.emit(
                    "forward_image",
                    "done",
                    real_image_count=real_count,
                    chunk_batch_size=current_batch_size,
                    image_decode_sec=f"{float(self.image_decode_time_sec):.2f}",
                    image_forward_sec=f"{float(self.image_forward_time_sec):.2f}",
                )
        elif progress_reporter is not None:
            progress_reporter.emit(
                "forward_image",
                "skip",
                reason="no_real_images",
                real_image_count=0,
            )

        img_emb = self.image_proj(img_emb)
        missing_img_emb = self.img_missing_emb.to(device=device, dtype=img_emb.dtype).view(1, -1)
        img_eff = has_image * img_emb + (1.0 - has_image) * missing_img_emb
        fused = self.fusion(torch.cat([text_emb, img_eff, has_image], dim=-1))
        return self.gnn(fused, edge_index.to(device))

    def predict_edges(
        self,
        node_embeddings: torch.Tensor,
        edge_pairs_local: torch.Tensor,
    ) -> torch.Tensor:
        device = node_embeddings.device
        src = edge_pairs_local[0].to(device=device, dtype=torch.long)
        dst = edge_pairs_local[1].to(device=device, dtype=torch.long)
        return self.predictor(node_embeddings[src], node_embeddings[dst])

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_pairs_local: torch.Tensor,
        node_indices_cpu: np.ndarray,
        node_wids: List[str],
        node_chunk_ids: np.ndarray,
        node_chunk_pos: np.ndarray,
        real_node_image_indices_cpu: torch.Tensor,
        real_node_image_paths: List[str],
        node_has_image_mask: torch.Tensor,
        text_store: TextStore,
        image_provider: Any,
        image_transform,
        progress_reporter: Optional[StepProgressReporter] = None,
    ) -> torch.Tensor:
        node_embeddings = self.encode_nodes(
            edge_index=edge_index,
            node_indices_cpu=node_indices_cpu,
            node_wids=node_wids,
            node_chunk_ids=node_chunk_ids,
            node_chunk_pos=node_chunk_pos,
            real_node_image_indices_cpu=real_node_image_indices_cpu,
            real_node_image_paths=real_node_image_paths,
            node_has_image_mask=node_has_image_mask,
            text_store=text_store,
            image_provider=image_provider,
            image_transform=image_transform,
            progress_reporter=progress_reporter,
        )
        return self.predict_edges(node_embeddings, edge_pairs_local)

    def consume_text_cache_stats(self) -> Dict[str, float]:
        return self.consume_runtime_stats()

    def consume_runtime_stats(self) -> Dict[str, float]:
        out = {
            "text_cache_hit": 0.0,
            "text_cache_miss": 0.0,
            "text_cache_hit_rate": 0.0,
            "text_cache_write_count": 0.0,
            "text_encode_sec": 0.0,
            "image_decode_sec": float(self.image_decode_time_sec),
            "image_forward_sec": float(self.image_forward_time_sec),
        }
        self.image_decode_time_sec = 0.0
        self.image_forward_time_sec = 0.0
        if hasattr(self.text_encoder, "consume_cache_stats"):
            out.update(self.text_encoder.consume_cache_stats())
        elif hasattr(self.text_encoder, "consume_runtime_stats"):
            out.update(self.text_encoder.consume_runtime_stats())
        return out
