#!/usr/bin/env python3
"""Inference CLI script for baseline model."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from fsbed.config import Config
from fsbed.logging_utils import setup_logging
from fsbed.data.dcase_paths import discover_dataset
from fsbed.pipelines.infer_baseline import load_baseline_model, infer_baseline_single_file
from fsbed.data.annotations import write_annotation_csv

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with baseline model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory (Validation or Evaluation set)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
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

    # Load config
    config = Config.from_yaml(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and pretrained classifier
    logger.info(f"Loading model from {args.checkpoint}")
    backbone, pretrained_classifier = load_baseline_model(Path(args.checkpoint), config, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files
    input_dir = Path(args.input_dir)
    file_pairs = discover_dataset(input_dir)
    logger.info(f"Found {len(file_pairs)} files in {input_dir}")
    
    # Run inference
    for audio_path, ann_path in file_pairs:
        try:
            # Run inference with dual-head adaptation
            events = infer_baseline_single_file(
                audio_path,
                ann_path,
                backbone,
                config,
                device,
                pretrained_classifier=pretrained_classifier,
            )
            
            # Save predictions
            # Create a simple object to match what write_annotation_csv expects
            annotated_events = []
            for e in events:
                # Create a mini-class or use SimpleNamespace
                class PredEvent:
                    def __init__(self, start, end):
                        self.audiofilename = audio_path.name
                        self.starttime = start
                        self.endtime = end
                        self.label = "POS"
                
                annotated_events.append(PredEvent(e.start_time, e.end_time))
            
            out_path = output_dir / f"{audio_path.stem}_predictions.csv"
            write_annotation_csv(annotated_events, out_path)
            
        except Exception as e:
            logger.error(f"Failed to process {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    logger.info(f"Inference complete! Predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
