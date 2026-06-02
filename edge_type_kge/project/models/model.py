"""Multimodal KGE model."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from project.data.image_provider import ByDirectoryImageProvider
from project.data.text_store import TextStore
from project.data.typed_graph import EntityCatalog
from project.models.image_cache import ChunkAlignedImageCache
from project.utils.logging import ProgressTicker, log_event


class MultimodalKGEModel(nn.Module):
    """Text + image -> fusion -> KGE scorer."""

    def __init__(
        self,
        text_encoder: nn.Module,
        image_encoder: nn.Module,
        fusion: nn.Module,
        kge_scorer: nn.Module,
        use_text: bool,
        use_image: bool,
        use_graph: bool,
        graph_encoder_name: str,
        text_proj_dim: int,
        image_proj_dim: int,
        fusion_out_dim: int,
        image_forward_batch_size: int,
        image_cache: ChunkAlignedImageCache | None = None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion = fusion
        self.kge_scorer = kge_scorer
        self.use_text = bool(use_text)
        self.use_image = bool(use_image)
        self.use_graph = bool(use_graph)
        self.graph_encoder_name = str(graph_encoder_name)
        if self.use_graph:
            raise NotImplementedError(f"graph_encoder not implemented: {self.graph_encoder_name}")
        self.image_forward_batch_size = max(1, int(image_forward_batch_size))
        self.image_cache = image_cache

        text_in = int(getattr(self.text_encoder, "output_dim", text_proj_dim))
        image_in = int(getattr(self.image_encoder, "output_dim", image_proj_dim))
        self.image_raw_feat_dim = int(image_in)
        self.text_feat_dim = int(text_proj_dim)
        self.image_feat_dim = int(image_proj_dim)
        self.text_proj = nn.Identity() if self.text_feat_dim == text_in else nn.Linear(text_in, self.text_feat_dim)
        self.image_proj = nn.Identity() if self.image_feat_dim == image_in else nn.Linear(image_in, self.image_feat_dim)
        self.img_missing_emb = nn.Parameter(torch.empty(self.image_feat_dim))
        nn.init.normal_(self.img_missing_emb, mean=0.0, std=0.02)
        self.entity_to_kge = nn.Linear(int(fusion_out_dim), int(kge_scorer.output_dim))
        self.image_forward_time_sec = 0.0
        self.image_decode_time_sec = 0.0

    def _module_dtype(self) -> torch.dtype:
        for param in self.parameters():
            return param.dtype
        return torch.float32

    def _encode_image_raw_features(
        self,
        node_wids: list[str],
        node_chunk_ids: np.ndarray,
        node_chunk_pos: np.ndarray,
        node_has_image: np.ndarray,
        image_provider: ByDirectoryImageProvider,
        image_transform,
        device: torch.device,
        pin_memory: bool = False,
    ) -> tuple[torch.Tensor, np.ndarray]:
        n = int(len(node_wids))
        raw_dtype = self._module_dtype()
        img_raw = torch.zeros((n, self.image_raw_feat_dim), device=device, dtype=raw_dtype)
        effective_has_image = np.zeros(n, dtype=np.bool_)
        need_decode = np.asarray(node_has_image, dtype=np.bool_).copy()
        has_image_rows = np.where(np.asarray(node_has_image, dtype=np.bool_))[0]
        if has_image_rows.size > 0 and self.image_cache is not None:
            cached, miss = self.image_cache.get_many(node_chunk_ids[has_image_rows], node_chunk_pos[has_image_rows])
            cached_rows = has_image_rows[~miss]
            if cached_rows.size > 0:
                img_raw[cached_rows] = torch.from_numpy(cached[~miss]).to(device=device, dtype=raw_dtype)
                effective_has_image[cached_rows] = True
                need_decode[cached_rows] = False

        batch = image_provider.get_batch_tensors(
            wids=node_wids,
            has_image_mask=need_decode,
            transform=image_transform,
            pin_memory=pin_memory,
        )
        self.image_decode_time_sec += float(batch["stats_image_decode_sec"])
        real_idx = batch["real_node_indices_cpu"].to(device=device, dtype=torch.long)
        real_images = batch["real_images_cpu"].to(device=device, non_blocking=False)
        if real_idx.numel() > 0:
            encoded_chunks = []
            encoded_idx = []
            for start in range(0, int(real_idx.shape[0]), self.image_forward_batch_size):
                stop = start + self.image_forward_batch_size
                batch_idx = real_idx[start:stop]
                batch_images = real_images[start:stop]
                t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                if t0 is not None and t1 is not None:
                    t0.record()
                out = self.image_encoder(batch_images)
                if t0 is not None and t1 is not None:
                    t1.record()
                    torch.cuda.synchronize(device)
                    self.image_forward_time_sec += float(t0.elapsed_time(t1) / 1000.0)
                encoded_chunks.append(out.to(dtype=raw_dtype))
                encoded_idx.append(batch_idx)
            encoded = torch.cat(encoded_chunks, dim=0)
            encoded_rows = torch.cat(encoded_idx, dim=0)
            img_raw.index_copy_(0, encoded_rows, encoded)
            loaded_rows = encoded_rows.detach().cpu().numpy()
            effective_has_image[loaded_rows] = True
            if self.image_cache is not None:
                cache_chunk_ids = node_chunk_ids[loaded_rows]
                cache_chunk_pos = node_chunk_pos[loaded_rows]
                cache_vals = encoded.detach().cpu().numpy()
                self.image_cache.set_many(cache_chunk_ids, cache_chunk_pos, cache_vals)
        return img_raw, effective_has_image

    @torch.no_grad()
    def prebuild_image_cache(
        self,
        catalog: EntityCatalog,
        image_provider: ByDirectoryImageProvider,
        image_transform,
        device: torch.device,
        pin_memory: bool = False,
        progress_every_sec: float = 30.0,
    ) -> Dict[str, float]:
        if not self.use_image or self.image_cache is None:
            return {}
        total_missing = 0
        total_loaded = 0
        ticker = ProgressTicker("prebuild image raw cache", every_sec=float(progress_every_sec))
        for chunk_id in range(catalog.chunk_count()):
            idxs = np.where(catalog.node_chunk_ids == int(chunk_id))[0]
            if idxs.size == 0:
                continue
            idxs = idxs[catalog.node_has_image[idxs].astype(np.bool_)]
            if idxs.size == 0:
                ticker.maybe(f"chunk={chunk_id + 1}/{catalog.chunk_count()} missing=0")
                continue
            chunk_ids = catalog.node_chunk_ids[idxs]
            chunk_pos = catalog.node_chunk_pos[idxs]
            _, miss_mask = self.image_cache.get_many(chunk_ids, chunk_pos)
            if not miss_mask.any():
                ticker.maybe(f"chunk={chunk_id + 1}/{catalog.chunk_count()} missing=0")
                continue
            miss_global = idxs[miss_mask]
            total_missing += int(miss_global.shape[0])
            _, loaded_mask = self._encode_image_raw_features(
                node_wids=catalog.get_node_wids(miss_global),
                node_chunk_ids=catalog.node_chunk_ids[miss_global],
                node_chunk_pos=catalog.node_chunk_pos[miss_global],
                node_has_image=np.ones(miss_global.shape[0], dtype=np.bool_),
                image_provider=image_provider,
                image_transform=image_transform,
                device=device,
                pin_memory=pin_memory,
            )
            total_loaded += int(loaded_mask.sum())
            ticker.maybe(
                f"chunk={chunk_id + 1}/{catalog.chunk_count()} missing={int(miss_global.shape[0])}"
            )
        stats = self.consume_runtime_stats()
        stats["image_cache_prebuilt"] = float(total_loaded)
        stats["image_cache_prebuild_missing"] = float(total_missing)
        log_event(
            "[STAGE]",
            f"image raw cache prebuild done missing={int(total_missing)} loaded={int(total_loaded)}",
        )
        return stats

    def encode_entities(
        self,
        entity_indices: np.ndarray,
        catalog: EntityCatalog,
        text_store: TextStore,
        image_provider: ByDirectoryImageProvider,
        image_transform,
        device: torch.device,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        idx = np.asarray(entity_indices, dtype=np.int64)
        node_wids = catalog.get_node_wids(idx)
        node_chunk_ids = catalog.node_chunk_ids[idx]
        node_chunk_pos = catalog.node_chunk_pos[idx]
        node_has_image = catalog.node_has_image[idx].astype(np.bool_)
        n = int(idx.shape[0])

        if self.use_text:
            text_emb = self.text_encoder.get_node_embeddings(
                node_wids=node_wids,
                node_chunk_ids=node_chunk_ids,
                node_chunk_pos=node_chunk_pos,
                text_store=text_store,
                output_device=device,
            )
            text_emb = self.text_proj(text_emb)
        else:
            text_emb = torch.zeros((n, self.text_feat_dim), device=device, dtype=self._module_dtype())

        if self.use_image:
            img_raw, effective_has_image = self._encode_image_raw_features(
                node_wids=node_wids,
                node_chunk_ids=node_chunk_ids,
                node_chunk_pos=node_chunk_pos,
                node_has_image=node_has_image,
                image_provider=image_provider,
                image_transform=image_transform,
                device=device,
                pin_memory=pin_memory,
            )
            img_proj = self.image_proj(img_raw).to(dtype=text_emb.dtype)
            has_image = torch.from_numpy(effective_has_image.astype(np.float32)).to(device).view(-1, 1)
            missing = self.img_missing_emb.to(device=device, dtype=img_proj.dtype).view(1, -1)
            img_eff = has_image * img_proj + (1.0 - has_image) * missing
        else:
            img_eff = torch.zeros((n, self.image_feat_dim), device=device, dtype=text_emb.dtype)
            has_image = torch.zeros((n, 1), device=device, dtype=text_emb.dtype)

        fused = self.fusion(torch.cat([text_emb, img_eff, has_image], dim=-1))
        return self.entity_to_kge(fused)

    def score_triples_from_embeddings(
        self,
        head_emb: torch.Tensor,
        rel_idx: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self.kge_scorer.score(head_emb, rel_idx, tail_emb)

    def score_relations_from_embeddings(self, head_emb: torch.Tensor, tail_emb: torch.Tensor) -> torch.Tensor:
        return self.kge_scorer.score_all_relations(head_emb, tail_emb)

    def consume_runtime_stats(self) -> Dict[str, float]:
        out = {
            "image_forward_sec": float(self.image_forward_time_sec),
            "image_decode_sec": float(self.image_decode_time_sec),
        }
        self.image_forward_time_sec = 0.0
        self.image_decode_time_sec = 0.0
        if hasattr(self.text_encoder, "consume_runtime_stats"):
            out.update(self.text_encoder.consume_runtime_stats())
        if self.image_cache is not None:
            out.update(self.image_cache.consume_stats())
        return out
