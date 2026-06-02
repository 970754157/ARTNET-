# -*- coding: utf-8 -*-
"""
Export the full Neo4j graph into sharded files with progress bars and auto-resume.

Output format is compatible with graphsql.py:
- nodes: {node_id: {"labels": [...], "properties": {...}}}
- edges: [{"id", "src", "src_labels", "dst", "dst_labels", "type"}, ...]

Default output:
  full_graph_export/
    manifest.json
    chunks/
      chunk_000001_nid_0000001234_0000045678/
        nodes.json
        edges.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm

import config as cfg


TARGET_CHUNK_SIZE = 20_000
MIN_CHUNK_SIZE = 10_000
MAX_CHUNK_SIZE = 30_000
DEFAULT_EDGE_BATCH_SIZE = 2_000
MANIFEST_FILE = "manifest.json"
SCHEMA_VERSION = 1


@dataclass
class ChunkSpec:
    index: int
    target_nodes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export full Neo4j graph into sharded node/edge files."
    )
    parser.add_argument(
        "--output-dir",
        default="full_graph_export",
        help="Output root directory (default: full_graph_export).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing non-empty output directory before exporting.",
    )
    parser.add_argument(
        "--edge-batch-size",
        type=int,
        default=DEFAULT_EDGE_BATCH_SIZE,
        help=f"Batch size for querying edges by src node ids (default: {DEFAULT_EDGE_BATCH_SIZE}).",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, data: Any, indent: Optional[int] = None) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        if indent is None:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(data, handle, ensure_ascii=False, indent=indent)
    tmp_path.replace(path)


def write_manifest_checkpoint(output_dir: Path, manifest: Dict[str, Any]) -> None:
    write_json_atomic(output_dir / MANIFEST_FILE, manifest, indent=2)


def iter_batches(items: Sequence[int], batch_size: int) -> Iterator[List[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def compute_chunk_sizes(total_nodes: int) -> List[int]:
    if total_nodes <= 0:
        return []

    min_chunks = math.ceil(total_nodes / MAX_CHUNK_SIZE)
    max_chunks = max(1, total_nodes // MIN_CHUNK_SIZE)
    ideal_chunks = max(1, round(total_nodes / TARGET_CHUNK_SIZE))
    chunk_count = min(max(ideal_chunks, min_chunks), max_chunks)

    base, remainder = divmod(total_nodes, chunk_count)
    return [base + 1 if i < remainder else base for i in range(chunk_count)]


def chunk_dir_name(chunk_index: int, min_node_id: int, max_node_id: int) -> str:
    return f"chunk_{chunk_index:06d}_nid_{min_node_id:010d}_{max_node_id:010d}"


def load_manifest_if_exists(output_dir: Path) -> Optional[Dict[str, Any]]:
    manifest_path = output_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid manifest file: {manifest_path}") from exc


def prepare_output_dir(
    output_dir: Path, overwrite: bool
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Returns:
      ("fresh", None)
      ("resume", manifest)
      ("completed", manifest)
    """
    if overwrite:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
        return "fresh", None

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
        return "fresh", None

    if not any(output_dir.iterdir()):
        (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
        return "fresh", None

    manifest = load_manifest_if_exists(output_dir)
    if manifest is None:
        raise RuntimeError(
            f"Output directory is not empty but has no {MANIFEST_FILE}: {output_dir}. "
            "Use --overwrite to replace it."
        )

    (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
    status = manifest.get("status")
    if status == "in_progress":
        return "resume", manifest
    if status == "completed":
        return "completed", manifest
    raise RuntimeError(
        f"Unsupported manifest status: {status!r}. Use --overwrite to rebuild."
    )


def export_edges_for_chunk(
    session: Any,
    src_node_ids: Sequence[int],
    edges_path: Path,
    edge_batch_size: int,
    chunk_index: int,
    show_progress: bool = True,
) -> int:
    query = """
    UNWIND $src_ids AS sid
    MATCH (s)-[r]->(t)
    WHERE id(s) = sid
    RETURN
      id(r) AS rid,
      id(s) AS src,
      labels(s) AS src_labels,
      id(t) AS dst,
      labels(t) AS dst_labels,
      type(r) AS rel_type
    """

    tmp_path = edges_path.with_suffix(edges_path.suffix + ".tmp")
    edge_count = 0
    total_batches = math.ceil(len(src_node_ids) / edge_batch_size) if src_node_ids else 0

    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        first = True

        with tqdm(
            total=total_batches,
            desc=f"chunk {chunk_index:06d} edges",
            unit="batch",
            leave=False,
            disable=not show_progress,
        ) as edge_pbar:
            for src_batch in iter_batches(src_node_ids, edge_batch_size):
                records = session.run(query, src_ids=src_batch)
                for record in records:
                    edge = {
                        "id": record["rid"],
                        "src": record["src"],
                        "src_labels": list(record["src_labels"]),
                        "dst": record["dst"],
                        "dst_labels": list(record["dst_labels"]),
                        "type": record["rel_type"],
                    }
                    if not first:
                        handle.write(",\n")
                    json.dump(edge, handle, ensure_ascii=False)
                    first = False
                    edge_count += 1
                edge_pbar.update(1)
                edge_pbar.set_postfix(edges=edge_count)

        handle.write("]")

    tmp_path.replace(edges_path)
    return edge_count


def build_chunk_specs(total_nodes: int) -> List[ChunkSpec]:
    sizes = compute_chunk_sizes(total_nodes)
    return [ChunkSpec(index=i + 1, target_nodes=size) for i, size in enumerate(sizes)]


def build_initial_manifest(
    total_nodes: int,
    total_relationships: int,
    chunk_count_planned: int,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": cfg.NEO4J_DATABASE,
        "total_nodes": total_nodes,
        "total_relationships": total_relationships,
        "chunk_count_planned": chunk_count_planned,
        "chunk_count": chunk_count_planned,
        "chunk_size_target": TARGET_CHUNK_SIZE,
        "chunk_size_min": MIN_CHUNK_SIZE,
        "chunk_size_max": MAX_CHUNK_SIZE,
        "edge_assignment": "by_src_node",
        "completed_nodes": 0,
        "completed_edges": 0,
        "last_src_node_id": None,
        "chunks": [],
    }


def validate_resume_manifest(
    manifest: Dict[str, Any],
    total_nodes: int,
    total_relationships: int,
    chunk_specs: Sequence[ChunkSpec],
    output_dir: Path,
) -> None:
    required_fields = [
        "schema_version",
        "status",
        "database",
        "total_nodes",
        "total_relationships",
        "chunk_count_planned",
        "completed_nodes",
        "completed_edges",
        "last_src_node_id",
        "chunks",
    ]
    missing = [field for field in required_fields if field not in manifest]
    if missing:
        raise RuntimeError(
            f"Manifest missing fields for resume: {missing}. Use --overwrite to rebuild."
        )

    if manifest["database"] != cfg.NEO4J_DATABASE:
        raise RuntimeError(
            "Manifest database mismatch. Use --overwrite to rebuild."
        )
    if int(manifest["total_nodes"]) != int(total_nodes):
        raise RuntimeError("Manifest total_nodes mismatch. Use --overwrite to rebuild.")
    if int(manifest["total_relationships"]) != int(total_relationships):
        raise RuntimeError(
            "Manifest total_relationships mismatch. Use --overwrite to rebuild."
        )
    if int(manifest["chunk_count_planned"]) != len(chunk_specs):
        raise RuntimeError(
            "Manifest chunk_count_planned mismatch. Use --overwrite to rebuild."
        )

    chunks = manifest["chunks"]
    if not isinstance(chunks, list):
        raise RuntimeError("Manifest chunks must be a list.")

    for expected_index, chunk in enumerate(chunks, start=1):
        actual_index = int(chunk.get("chunk_index", -1))
        if actual_index != expected_index:
            raise RuntimeError(
                "Manifest chunk sequence is not continuous. Use --overwrite to rebuild."
            )
        nodes_file = chunk.get("nodes_file")
        edges_file = chunk.get("edges_file")
        if not nodes_file or not edges_file:
            raise RuntimeError("Manifest chunk file paths are incomplete.")
        if not (output_dir / nodes_file).exists() or not (output_dir / edges_file).exists():
            raise RuntimeError(
                "Manifest references missing chunk files. Use --overwrite to rebuild."
            )

    nodes_sum = sum(int(chunk.get("node_count", 0)) for chunk in chunks)
    edges_sum = sum(int(chunk.get("edge_count", 0)) for chunk in chunks)
    if int(manifest["completed_nodes"]) != nodes_sum:
        raise RuntimeError("Manifest completed_nodes mismatch.")
    if int(manifest["completed_edges"]) != edges_sum:
        raise RuntimeError("Manifest completed_edges mismatch.")

    if chunks:
        expected_last = int(chunks[-1]["src_node_id_max"])
        if manifest["last_src_node_id"] is None or int(manifest["last_src_node_id"]) != expected_last:
            raise RuntimeError("Manifest last_src_node_id mismatch.")
    else:
        if manifest["last_src_node_id"] is not None:
            raise RuntimeError("Manifest last_src_node_id should be null when no chunks.")


def export_full_graph_sharded(
    output_dir: Path,
    overwrite: bool,
    edge_batch_size: int,
) -> Dict[str, Any]:
    from graphsql import get_driver

    if edge_batch_size <= 0:
        raise ValueError("edge_batch_size must be > 0")

    run_mode, existing_manifest = prepare_output_dir(output_dir, overwrite=overwrite)
    chunks_dir = output_dir / "chunks"

    driver = get_driver()
    try:
        with driver.session(database=cfg.NEO4J_DATABASE) as count_session:
            total_nodes = int(
                count_session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            )
            total_relationships = int(
                count_session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            )

        chunk_specs = build_chunk_specs(total_nodes)

        if run_mode == "completed":
            if existing_manifest is None:
                raise RuntimeError("Completed mode requires an existing manifest.")
            validate_resume_manifest(
                manifest=existing_manifest,
                total_nodes=total_nodes,
                total_relationships=total_relationships,
                chunk_specs=chunk_specs,
                output_dir=output_dir,
            )
            print("[INFO] Export already completed. Nothing to do.")
            return existing_manifest

        if not chunk_specs:
            manifest = build_initial_manifest(
                total_nodes=0,
                total_relationships=total_relationships,
                chunk_count_planned=0,
            )
            manifest["status"] = "completed"
            manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_manifest_checkpoint(output_dir, manifest)
            return manifest

        if run_mode == "fresh":
            manifest = build_initial_manifest(
                total_nodes=total_nodes,
                total_relationships=total_relationships,
                chunk_count_planned=len(chunk_specs),
            )
            write_manifest_checkpoint(output_dir, manifest)
        else:
            if existing_manifest is None:
                raise RuntimeError("Resume mode requires an existing manifest.")
            validate_resume_manifest(
                manifest=existing_manifest,
                total_nodes=total_nodes,
                total_relationships=total_relationships,
                chunk_specs=chunk_specs,
                output_dir=output_dir,
            )
            manifest = existing_manifest

        completed_chunk_count = len(manifest["chunks"])
        completed_nodes = int(manifest["completed_nodes"])
        completed_edges = int(manifest["completed_edges"])
        last_src_node_id = manifest["last_src_node_id"]

        print(f"[INFO] Total nodes: {total_nodes}")
        print(f"[INFO] Total relationships: {total_relationships}")
        print(f"[INFO] Planned chunks: {len(chunk_specs)}")
        print(f"[INFO] Resume mode: {run_mode}")
        if run_mode == "resume":
            print(f"[INFO] Completed chunks: {completed_chunk_count}")
            print(f"[INFO] Completed nodes : {completed_nodes}")
            print(f"[INFO] Completed edges : {completed_edges}")

        if completed_chunk_count == len(chunk_specs):
            if completed_nodes != total_nodes or completed_edges != total_relationships:
                raise RuntimeError(
                    "Manifest says all chunks done, but totals are not complete."
                )
            manifest["status"] = "completed"
            manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_manifest_checkpoint(output_dir, manifest)
            return manifest

        if run_mode == "resume" and last_src_node_id is not None:
            node_query = """
            MATCH (n)
            WHERE id(n) > $last_src_node_id
            RETURN id(n) AS nid, labels(n) AS labels, properties(n) AS props
            ORDER BY id(n)
            """
            node_params: Dict[str, Any] = {"last_src_node_id": int(last_src_node_id)}
        else:
            node_query = """
            MATCH (n)
            RETURN id(n) AS nid, labels(n) AS labels, properties(n) AS props
            ORDER BY id(n)
            """
            node_params = {}

        spec_idx = completed_chunk_count
        spec = chunk_specs[spec_idx]
        chunk_nodes: Dict[str, Dict[str, Any]] = {}
        chunk_node_ids: List[int] = []

        def flush_current_chunk() -> None:
            nonlocal chunk_nodes, chunk_node_ids
            if not chunk_node_ids:
                return

            min_node_id = min(chunk_node_ids)
            max_node_id = max(chunk_node_ids)
            dir_name = chunk_dir_name(spec.index, min_node_id, max_node_id)
            chunk_dir = chunks_dir / dir_name

            if chunk_dir.exists():
                shutil.rmtree(chunk_dir)
            chunk_dir.mkdir(parents=True, exist_ok=False)

            nodes_path = chunk_dir / "nodes.json"
            edges_path = chunk_dir / "edges.json"

            write_json_atomic(nodes_path, chunk_nodes)
            with driver.session(database=cfg.NEO4J_DATABASE) as edge_session:
                edge_count = export_edges_for_chunk(
                    session=edge_session,
                    src_node_ids=chunk_node_ids,
                    edges_path=edges_path,
                    edge_batch_size=edge_batch_size,
                    chunk_index=spec.index,
                    show_progress=True,
                )

            node_count = len(chunk_node_ids)
            manifest["completed_nodes"] = int(manifest["completed_nodes"]) + node_count
            manifest["completed_edges"] = int(manifest["completed_edges"]) + edge_count
            manifest["last_src_node_id"] = max_node_id
            manifest["chunks"].append(
                {
                    "chunk_index": spec.index,
                    "dir_name": dir_name,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "src_node_id_min": min_node_id,
                    "src_node_id_max": max_node_id,
                    "nodes_file": str(nodes_path.relative_to(output_dir)).replace(
                        "\\", "/"
                    ),
                    "edges_file": str(edges_path.relative_to(output_dir)).replace(
                        "\\", "/"
                    ),
                }
            )
            write_manifest_checkpoint(output_dir, manifest)

            print(f"[INFO] chunk {spec.index:06d}: nodes={node_count}, edges={edge_count}")

            chunk_nodes = {}
            chunk_node_ids = []

        with driver.session(database=cfg.NEO4J_DATABASE) as node_session:
            node_records = node_session.run(node_query, **node_params)

            with tqdm(
                total=total_nodes,
                initial=completed_nodes,
                desc="nodes",
                unit="node",
                leave=True,
            ) as node_pbar:
                for record in node_records:
                    nid = int(record["nid"])
                    chunk_nodes[str(nid)] = {
                        "labels": list(record["labels"]),
                        "properties": record["props"],
                    }
                    chunk_node_ids.append(nid)
                    node_pbar.update(1)

                    if len(chunk_node_ids) >= spec.target_nodes:
                        flush_current_chunk()
                        spec_idx += 1
                        if spec_idx < len(chunk_specs):
                            spec = chunk_specs[spec_idx]

                flush_current_chunk()

        if int(manifest["completed_nodes"]) != total_nodes:
            raise RuntimeError(
                "Node count mismatch after export: "
                f"exported={manifest['completed_nodes']}, expected={total_nodes}"
            )
        if int(manifest["completed_edges"]) != total_relationships:
            raise RuntimeError(
                "Relationship count mismatch after export: "
                f"exported={manifest['completed_edges']}, expected={total_relationships}"
            )

        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["chunk_count"] = len(manifest["chunks"])
        write_manifest_checkpoint(output_dir, manifest)
        return manifest
    finally:
        driver.close()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        manifest = export_full_graph_sharded(
            output_dir=output_dir,
            overwrite=args.overwrite,
            edge_batch_size=args.edge_batch_size,
        )
    except Exception as exc:
        print("[ERROR] Export failed.")
        print(f"        {type(exc).__name__}: {exc}")
        print("")
        print("Check:")
        print("- Neo4j is running and reachable")
        print("- config.py or env vars for NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD/NEO4J_DATABASE")
        print("- output directory permissions")
        print("- if output has stale data, rerun with --overwrite")
        return 1

    print("[DONE] Export completed.")
    print(f"[DONE] status={manifest.get('status')}")
    print(f"[DONE] chunks={manifest.get('chunk_count')}")
    print(f"[DONE] output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
