"""Training entrypoint for multimodal KGE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from project.config import load_config
from project.experiment import build_experiment_bundle, prebuild_raw_feature_caches
from project.trainer import Trainer
from project.utils.io import ensure_dir
from project.utils.logging import log_event, log_stage
from project.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal KGE")
    parser.add_argument("--config", type=str, required=True, help="Path to config json")
    parser.add_argument("--sample-chunks", type=int, default=0, help="Use only the first N graph chunks")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--force-rebuild-data", action="store_true", help="Rebuild catalog/triple caches")
    parser.add_argument("--rebuild-split", action="store_true", help="Rebuild relation-stratified splits")
    parser.add_argument(
        "--prebuild-text-cache",
        action="store_true",
        help="Prebuild text embedding cache before training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    set_global_seed(cfg.seed)

    resolved_paths = cfg.resolved_paths(config_path.parent)
    runs_root = resolved_paths["runs_dir"]
    if args.resume:
        run_dir = Path(args.resume).resolve().parent.parent
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = runs_root / f"{timestamp}_{cfg.experiment_name}"
    ensure_dir(run_dir)

    with log_stage("build experiment"):
        bundle = build_experiment_bundle(
            cfg=cfg,
            config_base_dir=config_path.parent,
            sample_chunks=args.sample_chunks,
            force_rebuild_data=bool(args.force_rebuild_data),
            rebuild_split=bool(args.rebuild_split),
            for_training=True,
            cache_session=run_dir.name,
        )
    log_event("[CONFIG]", f"training run_dir={run_dir} cache_session={run_dir.name}")
    if args.prebuild_text_cache:
        log_event("[CONFIG]", "--prebuild-text-cache is kept for compatibility; raw cache prebuild now runs automatically")
    with log_stage("prebuild raw caches"):
        prebuild_raw_feature_caches(bundle)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    trainer = Trainer(bundle=bundle, run_dir=run_dir)
    if args.resume:
        trainer.load_checkpoint(Path(args.resume).resolve())
        log_event("[STAGE]", f"resumed from checkpoint: {args.resume}")

    with log_stage("train"):
        metrics = trainer.train()
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log_event("[RESULT]", json.dumps(metrics, ensure_ascii=False))
    bundle.image_provider.close()


if __name__ == "__main__":
    main()
