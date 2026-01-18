#!/usr/bin/env python3
"""Inference CLI script."""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fsbed.config import Config
from fsbed.logging_utils import setup_logging
from fsbed.data.annotations import write_annotation_csv, Annotation
from fsbed.pipelines.infer_one import infer_single_file
from fsbed.pipelines.infer_batch import infer_batch, load_model_for_inference


def main():
    parser = argparse.ArgumentParser(
        description="Run FSBED inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to single audio file",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default=None,
        help="Path to annotation file (for single file mode)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory for batch mode",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override detection threshold",
    )
    parser.add_argument(
        "--no_adapt",
        action="store_true",
        help="Disable per-file adaptation",
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

    # Validate args
    if args.audio is None and args.input_dir is None:
        parser.error("Either --audio or --input_dir must be specified")

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    # Apply overrides
    if args.threshold is not None:
        config.postprocess.threshold = args.threshold
    if args.no_adapt:
        config.adapt.enabled = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model_for_inference(Path(args.checkpoint), config, device)

    # Run inference
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.audio is not None:
        # Single file mode
        audio_path = Path(args.audio)
        annotation_path = Path(args.annotation) if args.annotation else audio_path.with_suffix(".csv")

        events = infer_single_file(audio_path, annotation_path, model, config, device)

        # Save predictions
        annotations = [
            Annotation(
                audiofilename=audio_path.name,
                starttime=e.start_time,
                endtime=e.end_time,
            )
            for e in events
        ]
        output_csv = output_dir / f"{audio_path.stem}_predictions.csv"
        write_annotation_csv(annotations, output_csv)
        logger.info(f"Saved {len(events)} predictions to {output_csv}")

    else:
        # Batch mode
        results = infer_batch(
            Path(args.input_dir),
            output_dir,
            model,
            config,
            device,
        )
        total_events = sum(len(v) for v in results.values())
        logger.info(f"Processed {len(results)} files, {total_events} total events")


if __name__ == "__main__":
    main()
