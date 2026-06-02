"""Inference script for triple scoring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from project.experiment import load_bundle_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score triples (h, r, t)")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--head", type=str, default="", help="Head entity ref: wid or global node id")
    parser.add_argument("--relation", type=str, default="", help="Relation name or relation index")
    parser.add_argument("--tail", type=str, default="", help="Tail entity ref: wid or global node id")
    parser.add_argument("--input", type=str, default="", help="Optional csv/jsonl batch input")
    parser.add_argument("--output", type=str, default="", help="Optional output path")
    parser.add_argument("--device", type=str, default="", help="Override device")
    parser.add_argument("--batch-size", type=int, default=1024, help="Number of samples to score per batch")
    parser.add_argument(
        "--entity-batch-size",
        type=int,
        default=4096,
        help="Number of unique entities to encode per sub-batch",
    )
    return parser.parse_args()


def _resolve_entity_cached(bundle, ref: str, cache: Dict[str, int]) -> int:
    cached = cache.get(ref)
    if cached is not None:
        return int(cached)
    value = int(bundle.catalog.resolve_entity_ref(ref))
    cache[ref] = value
    return value


def _resolve_relation(bundle, ref: str) -> int:
    if ref in bundle.triple_store.relation_to_idx:
        return int(bundle.triple_store.relation_to_idx[ref])
    return int(ref)


def _resolve_relation_cached(bundle, ref: str, cache: Dict[str, int]) -> int:
    cached = cache.get(ref)
    if cached is not None:
        return int(cached)
    value = _resolve_relation(bundle, ref)
    cache[ref] = value
    return value


def _load_batch(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _encode_unique_entities(bundle, entity_indices: np.ndarray, entity_batch_size: int) -> torch.Tensor:
    pin_memory = bool(getattr(bundle.cfg.runtime, "pin_memory", False) and bundle.device.type == "cuda")
    chunks: List[torch.Tensor] = []
    step = max(1, int(entity_batch_size))
    for start in range(0, int(entity_indices.shape[0]), step):
        chunk = entity_indices[start : start + step]
        chunks.append(
            bundle.model.encode_entities(
                entity_indices=chunk,
                catalog=bundle.catalog,
                text_store=bundle.text_store,
                image_provider=bundle.image_provider,
                image_transform=bundle.image_transform,
                device=bundle.device,
                pin_memory=pin_memory,
            )
        )
    if not chunks:
        return torch.empty((0, int(bundle.model.kge_scorer.output_dim)), device=bundle.device)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=0)


def _score_batch(
    bundle,
    head_idx: np.ndarray,
    relation_idx: np.ndarray,
    tail_idx: np.ndarray,
    entity_batch_size: int,
) -> List[Dict]:
    all_entities = np.concatenate([head_idx, tail_idx], axis=0)
    unique_entities, inverse = np.unique(all_entities, return_inverse=True)
    batch_size = int(head_idx.shape[0])
    with torch.no_grad():
        entity_emb = _encode_unique_entities(bundle, unique_entities, entity_batch_size=entity_batch_size)
        head_rows = torch.from_numpy(inverse[:batch_size]).to(bundle.device, dtype=torch.long)
        tail_rows = torch.from_numpy(inverse[batch_size:]).to(bundle.device, dtype=torch.long)
        relation_tensor = torch.from_numpy(relation_idx.astype(np.int64, copy=False)).to(bundle.device, dtype=torch.long)
        scores = bundle.model.score_triples_from_embeddings(
            entity_emb[head_rows],
            relation_tensor,
            entity_emb[tail_rows],
        )
    score_np = scores.detach().cpu().numpy()
    relation_names = bundle.triple_store.idx_to_relation
    rows: List[Dict] = []
    for row_idx in range(batch_size):
        rel_idx = int(relation_idx[row_idx])
        rows.append(
            {
                "head": int(head_idx[row_idx]),
                "relation": relation_names[rel_idx],
                "relation_idx": rel_idx,
                "tail": int(tail_idx[row_idx]),
                "score": float(score_np[row_idx]),
            }
        )
    return rows


def _write_jsonl(handle, rows: List[Dict]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(writer: csv.DictWriter, rows: List[Dict]) -> None:
    for row in rows:
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    bundle = load_bundle_from_checkpoint(Path(args.checkpoint).resolve(), device_override=args.device or None)
    bundle.model.eval()
    if args.input:
        raw_rows = _load_batch(Path(args.input).resolve())
        entity_ref_cache: Dict[str, int] = {}
        relation_ref_cache: Dict[str, int] = {}
        head_idx = np.empty(len(raw_rows), dtype=np.int64)
        relation_idx = np.empty(len(raw_rows), dtype=np.int64)
        tail_idx = np.empty(len(raw_rows), dtype=np.int64)
        for i, raw in enumerate(raw_rows):
            head_idx[i] = _resolve_entity_cached(bundle, str(raw["head"]), entity_ref_cache)
            tail_idx[i] = _resolve_entity_cached(bundle, str(raw["tail"]), entity_ref_cache)
            relation_idx[i] = _resolve_relation_cached(bundle, str(raw["relation"]), relation_ref_cache)

        collected: List[Dict] = []
        output_path = Path(args.output).resolve() if args.output else None
        jsonl_handle = None
        csv_handle = None
        csv_writer = None
        try:
            if output_path is not None:
                if output_path.suffix.lower() == ".jsonl":
                    jsonl_handle = output_path.open("w", encoding="utf-8")
                else:
                    csv_handle = output_path.open("w", encoding="utf-8", newline="")
                    csv_writer = csv.DictWriter(csv_handle, fieldnames=["head", "relation", "relation_idx", "tail", "score"])
                    csv_writer.writeheader()

            sample_batch_size = max(1, int(args.batch_size))
            entity_batch_size = max(1, int(args.entity_batch_size))
            for start in range(0, len(raw_rows), sample_batch_size):
                stop = start + sample_batch_size
                rows = _score_batch(
                    bundle,
                    head_idx=head_idx[start:stop],
                    relation_idx=relation_idx[start:stop],
                    tail_idx=tail_idx[start:stop],
                    entity_batch_size=entity_batch_size,
                )
                if output_path is None:
                    collected.extend(rows)
                elif jsonl_handle is not None:
                    _write_jsonl(jsonl_handle, rows)
                else:
                    _write_csv(csv_writer, rows)
        finally:
            if jsonl_handle is not None:
                jsonl_handle.close()
            if csv_handle is not None:
                csv_handle.close()

        if output_path is None:
            print(json.dumps(collected, ensure_ascii=False, indent=2))
    else:
        if not args.head or not args.tail or not args.relation:
            raise ValueError("Single-item mode requires --head --relation --tail")
        entity_ref_cache: Dict[str, int] = {}
        relation_ref_cache: Dict[str, int] = {}
        rows = _score_batch(
            bundle,
            head_idx=np.asarray([_resolve_entity_cached(bundle, args.head, entity_ref_cache)], dtype=np.int64),
            relation_idx=np.asarray([_resolve_relation_cached(bundle, args.relation, relation_ref_cache)], dtype=np.int64),
            tail_idx=np.asarray([_resolve_entity_cached(bundle, args.tail, entity_ref_cache)], dtype=np.int64),
            entity_batch_size=max(1, int(args.entity_batch_size)),
        )
        print(json.dumps(rows[0], ensure_ascii=False, indent=2))
    bundle.image_provider.close()


if __name__ == "__main__":
    main()
