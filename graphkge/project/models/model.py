"""Multimodal GraphKGE model."""

from __future__ import annotations

import time
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
    """Raw text/image features -> fusion -> optional graph encoder -> KGE scorer."""

    def __init__(
        self,
        text_encoder: nn.Module,
        image_encoder: nn.Module,
        fusion: nn.Module,
        graph_encoder: nn.Module,
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
        self.graph_encoder = graph_encoder
        self.kge_scorer = kge_scorer
        self.use_text = bool(use_text)
        self.use_image = bool(use_image)
        self.use_graph = bool(use_graph)
        self.graph_encoder_name = str(graph_encoder_name)
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

        graph_out_dim = int(getattr(self.graph_encoder, "output_dim", fusion_out_dim)) if self.use_graph else int(fusion_out_dim)
        self.entity_to_kge = nn.Linear(graph_out_dim, int(kge_scorer.output_dim))
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
        image_batch: Dict[str, object] | None = None,
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

        def _encode_and_store(real_idx_cpu: torch.Tensor, real_images_cpu: torch.Tensor) -> None:
            if int(real_idx_cpu.numel()) == 0:
                return
            for start in range(0, int(real_idx_cpu.shape[0]), self.image_forward_batch_size):
                stop = start + self.image_forward_batch_size
                batch_idx_cpu = real_idx_cpu[start:stop]
                batch_images_cpu = real_images_cpu[start:stop]
                batch_idx = batch_idx_cpu.to(device=device, dtype=torch.long)
                batch_images = batch_images_cpu.to(device=device, non_blocking=False)
                t0 = time.perf_counter()
                encoded = self.image_encoder(batch_images)
                self.image_forward_time_sec += float(time.perf_counter() - t0)
                encoded = encoded.to(dtype=raw_dtype)
                img_raw.index_copy_(0, batch_idx, encoded)
                loaded_rows = batch_idx_cpu.detach().cpu().numpy().astype(np.int64, copy=False)
                effective_has_image[loaded_rows] = True
                if self.image_cache is not None:
                    self.image_cache.set_many(
                        node_chunk_ids[loaded_rows],
                        node_chunk_pos[loaded_rows],
                        encoded.detach().cpu().numpy(),
                    )

        batch = image_batch
        if batch is None:
            for batch_item in image_provider.iter_batch_tensors(
                wids=node_wids,
                has_image_mask=need_decode,
                transform=image_transform,
                batch_size=self.image_forward_batch_size,
                pin_memory=pin_memory,
            ):
                self.image_decode_time_sec += float(batch_item.get("stats_image_decode_sec", 0.0))
                _encode_and_store(
                    real_idx_cpu=batch_item["real_node_indices_cpu"],
                    real_images_cpu=batch_item["real_images_cpu"],
                )
            return img_raw, effective_has_image

        self.image_decode_time_sec += float(batch.get("stats_image_decode_sec", 0.0))
        real_idx_cpu = batch["real_node_indices_cpu"]
        real_images_cpu = batch["real_images_cpu"]
        if int(real_idx_cpu.numel()) > 0:
            real_idx_np = real_idx_cpu.detach().cpu().numpy().astype(np.int64, copy=False)
            keep_decode = need_decode[real_idx_np]
            if bool(keep_decode.any()):
                keep_decode_t = torch.from_numpy(keep_decode)
                selected = torch.nonzero(keep_decode_t, as_tuple=False).flatten()
                _encode_and_store(
                    real_idx_cpu=real_idx_cpu[selected],
                    real_images_cpu=real_images_cpu[selected],
                )
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
        raw_dtype = self._module_dtype()
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
            miss_wids = catalog.get_node_wids(miss_global)
            miss_chunk_ids = catalog.node_chunk_ids[miss_global]
            miss_chunk_pos = catalog.node_chunk_pos[miss_global]
            loaded_in_chunk = 0
            for batch_item in image_provider.iter_batch_tensors(
                wids=miss_wids,
                has_image_mask=np.ones(miss_global.shape[0], dtype=np.bool_),
                transform=image_transform,
                batch_size=self.image_forward_batch_size,
                pin_memory=pin_memory,
            ):
                self.image_decode_time_sec += float(batch_item.get("stats_image_decode_sec", 0.0))
                real_idx_cpu = batch_item["real_node_indices_cpu"]
                real_images_cpu = batch_item["real_images_cpu"]
                if int(real_idx_cpu.numel()) == 0:
                    continue
                batch_images = real_images_cpu.to(device=device, non_blocking=False)
                t0 = time.perf_counter()
                encoded = self.image_encoder(batch_images)
                self.image_forward_time_sec += float(time.perf_counter() - t0)
                encoded = encoded.to(dtype=raw_dtype)
                loaded_rows = real_idx_cpu.detach().cpu().numpy().astype(np.int64, copy=False)
                self.image_cache.set_many(
                    miss_chunk_ids[loaded_rows],
                    miss_chunk_pos[loaded_rows],
                    encoded.detach().cpu().numpy(),
                )
                loaded_in_chunk += int(loaded_rows.shape[0])
            total_loaded += loaded_in_chunk
            ticker.maybe(
                f"chunk={chunk_id + 1}/{catalog.chunk_count()} missing={int(miss_global.shape[0])} loaded={int(loaded_in_chunk)}"
            )
        stats = self.consume_runtime_stats()
        stats["image_cache_prebuilt"] = float(total_loaded)
        stats["image_cache_prebuild_missing"] = float(total_missing)
        log_event(
            "[STAGE]",
            f"image raw cache prebuild done missing={int(total_missing)} loaded={int(total_loaded)}",
        )
        return stats

    def _encode_text_features(
        self,
        node_wids: list[str],
        node_chunk_ids: np.ndarray,
        node_chunk_pos: np.ndarray,
        text_store: TextStore,
        device: torch.device,
    ) -> torch.Tensor:
        n = int(node_chunk_ids.shape[0])
        if not self.use_text:
            return torch.zeros((n, self.text_feat_dim), device=device, dtype=self._module_dtype())
        text_emb = self.text_encoder.get_node_embeddings(
            node_wids=node_wids,
            node_chunk_ids=node_chunk_ids,
            node_chunk_pos=node_chunk_pos,
            text_store=text_store,
            output_device=device,
        )
        return self.text_proj(text_emb)

    def _encode_image_features(
        self,
        node_wids: list[str],
        node_chunk_ids: np.ndarray,
        node_chunk_pos: np.ndarray,
        node_has_image: np.ndarray,
        image_provider: ByDirectoryImageProvider,
        image_transform,
        device: torch.device,
        dtype: torch.dtype,
        pin_memory: bool = False,
        image_batch: Dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(node_chunk_ids.shape[0])
        if not self.use_image:
            return (
                torch.zeros((n, self.image_feat_dim), device=device, dtype=dtype),
                torch.zeros((n, 1), device=device, dtype=dtype),
            )
        img_raw, effective_has_image = self._encode_image_raw_features(
            node_wids=node_wids,
            node_chunk_ids=node_chunk_ids,
            node_chunk_pos=node_chunk_pos,
            node_has_image=node_has_image,
            image_provider=image_provider,
            image_transform=image_transform,
            device=device,
            pin_memory=pin_memory,
            image_batch=image_batch,
        )
        img_proj = self.image_proj(img_raw).to(dtype=dtype)
        has_image = torch.from_numpy(effective_has_image.astype(np.float32)).to(device=device, dtype=dtype).view(-1, 1)
        missing = self.img_missing_emb.to(device=device, dtype=dtype).view(1, -1)
        img_eff = has_image * img_proj + (1.0 - has_image) * missing
        return img_eff, has_image

    def encode_node_features(
        self,
        entity_indices: np.ndarray,
        catalog: EntityCatalog,
        text_store: TextStore,
        image_provider: ByDirectoryImageProvider,
        image_transform,
        device: torch.device,
        pin_memory: bool = False,
        image_batch: Dict[str, object] | None = None,
    ) -> torch.Tensor:
        idx = np.asarray(entity_indices, dtype=np.int64)
        node_wids = catalog.get_node_wids(idx)
        node_chunk_ids = catalog.node_chunk_ids[idx]
        node_chunk_pos = catalog.node_chunk_pos[idx]
        node_has_image = catalog.node_has_image[idx].astype(np.bool_, copy=False)
        text_emb = self._encode_text_features(
            node_wids=node_wids,
            node_chunk_ids=node_chunk_ids,
            node_chunk_pos=node_chunk_pos,
            text_store=text_store,
            device=device,
        )
        img_eff, has_image = self._encode_image_features(
            node_wids=node_wids,
            node_chunk_ids=node_chunk_ids,
            node_chunk_pos=node_chunk_pos,
            node_has_image=node_has_image,
            image_provider=image_provider,
            image_transform=image_transform,
            device=device,
            dtype=text_emb.dtype,
            pin_memory=pin_memory,
            image_batch=image_batch,
        )
        return self.fusion(torch.cat([text_emb, img_eff, has_image], dim=-1))

    def encode_subgraph(
        self,
        entity_indices: np.ndarray,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None,
        catalog: EntityCatalog,
        text_store: TextStore,
        image_provider: ByDirectoryImageProvider,
        image_transform,
        device: torch.device,
        pin_memory: bool = False,
        image_batch: Dict[str, object] | None = None,
    ) -> torch.Tensor:
        fused = self.encode_node_features(
            entity_indices=entity_indices,
            catalog=catalog,
            text_store=text_store,
            image_provider=image_provider,
            image_transform=image_transform,
            device=device,
            pin_memory=pin_memory,
            image_batch=image_batch,
        )
        if self.use_graph:
            graph_repr = self.graph_encoder(
                fused,
                edge_index.to(device=device, dtype=torch.long),
                None if edge_type is None else edge_type.to(device=device, dtype=torch.long),
            )
        else:
            graph_repr = fused
        return self.entity_to_kge(graph_repr)

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
        if self.use_graph:
            return self.encode_subgraph(
                entity_indices=entity_indices,
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_type=torch.empty((0,), dtype=torch.long),
                catalog=catalog,
                text_store=text_store,
                image_provider=image_provider,
                image_transform=image_transform,
                device=device,
                pin_memory=pin_memory,
                image_batch=None,
            )
        fused = self.encode_node_features(
            entity_indices=entity_indices,
            catalog=catalog,
            text_store=text_store,
            image_provider=image_provider,
            image_transform=image_transform,
            device=device,
            pin_memory=pin_memory,
            image_batch=None,
        )
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
