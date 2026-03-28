#!/usr/bin/env python3
"""Train TR-GAT on UrbanAir-500 benchmark.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --device cuda --epochs 150
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from skyflow.config import SkyFlowConfig
from skyflow.data.urbanair500 import UrbanAir500
from skyflow.training.trainer import SkyFlowTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train SkyFlow TR-GAT")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-uavs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced data")
    args = parser.parse_args()

    cfg = SkyFlowConfig.from_yaml(args.config)

    if args.device:
        cfg.training.device = args.device
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.num_uavs:
        cfg.data.num_uavs = args.num_uavs
    if args.seed:
        cfg.training.seed = args.seed

    if args.quick:
        cfg.data.num_uavs = 50
        cfg.training.epochs = 10
        cfg.training.num_seeds = 1

    device = torch.device(
        cfg.training.device if cfg.training.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device: {device}")

    logger.info("Generating UrbanAir-500 benchmark data...")
    sim = UrbanAir500(
        num_uavs=cfg.data.num_uavs,
        seed=cfg.training.seed,
    )

    n_train = 5 if args.quick else 20
    n_val = 2 if args.quick else 5
    n_test = 2 if args.quick else 5
    duration = 10.0 if args.quick else 60.0

    train_data = sim.generate_dataset("train", n_train, duration, device)
    val_data = sim.generate_dataset("val", n_val, duration, device)
    test_data = sim.generate_dataset("test", n_test, duration, device)

    logger.info(f"Train: {len(train_data)} snapshots, Val: {len(val_data)}, Test: {len(test_data)}")

    trainer = SkyFlowTrainer(cfg, device=device)

    if cfg.training.num_seeds > 1:
        summary = trainer.train_multi_seed(train_data, val_data, test_data)
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-SEED RESULTS SUMMARY")
        logger.info("=" * 60)
        for metric, vals in summary.items():
            logger.info(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")
    else:
        trainer.build_model()
        best = trainer.train(train_data, val_data, seed=cfg.training.seed)
        logger.info(f"\nBest validation metrics: {best}")

    logger.info(f"Results saved to {cfg.output_dir}/")


if __name__ == "__main__":
    main()
