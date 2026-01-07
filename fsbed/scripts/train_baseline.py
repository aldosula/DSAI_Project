#!/usr/bin/env python3
"""Training CLI script for baseline model."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fsbed.config import Config
from fsbed.logging_utils import setup_logging
from fsbed.pipelines.train_baseline import train_baseline


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="Override training data directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = Config()
        logger.info("Using default config")

    # Apply overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.train_dir is not None:
        config.paths.train_dir = args.train_dir
    if args.checkpoint_dir is not None:
        config.paths.checkpoint_dir = args.checkpoint_dir

    # Train
    logger.info("Starting baseline model training...")
    logger.info(f"Training dir: {config.paths.train_dir}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Learning rate: {config.training.lr}")

    model = train_baseline(config)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
