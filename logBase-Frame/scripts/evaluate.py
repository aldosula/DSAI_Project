#!/usr/bin/env python3
"""Evaluation CLI script."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fsbed.config import Config
from fsbed.logging_utils import setup_logging
from fsbed.data.annotations import read_annotation_csv, read_support_events, get_support_end_time
from fsbed.data.dcase_paths import discover_dataset
from fsbed.eval.metrics import event_based_f1, MatchResult
from fsbed.postprocess.events import Event

import pandas as pd

logger = logging.getLogger(__name__)


def load_predictions(pred_dir: Path) -> dict[str, list[Event]]:
    """
    Load predictions from directory.
    
    Returns:
        Dictionary mapping {stem: list[Event]}.
    """
    predictions = {}
    
    # Check for both .csv and _predictions.csv
    files = list(pred_dir.glob("*.csv"))
    
    for csv_path in files:
        if csv_path.name == "evaluation_results.csv":
            continue
            
        # Extract stem (robust to _predictions suffix)
        stem = csv_path.stem.replace("_predictions", "")
        
        try:
            df = pd.read_csv(csv_path)
            events = []
            
            # Normalize columns
            df.columns = [c.strip() for c in df.columns]
            col_map = {c.lower(): c for c in ["Starttime", "Endtime"]}
            df = df.rename(columns=lambda c: col_map.get(c.lower(), c))
            
            for _, row in df.iterrows():
                events.append(Event(
                    start_time=float(row["Starttime"]),
                    end_time=float(row["Endtime"]),
                ))
            predictions[stem] = events
        except pd.errors.EmptyDataError:
            predictions[stem] = []
        except Exception as e:
            logger.warning(f"Failed to load {csv_path}: {e}")
            
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FSBED predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory containing prediction CSVs",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Directory containing ground truth annotations",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--onset_tolerance",
        type=float,
        default=0.2,
        help="Onset tolerance in seconds",
    )
    parser.add_argument(
        "--offset_tolerance",
        type=float,
        default=0.2,
        help="Offset tolerance in seconds",
    )
    parser.add_argument(
        "--min_iou",
        type=float,
        default=0.0,
        help="Minimum IoU for matching (0 = use tolerance)",
    )
    parser.add_argument(
        "--ignore_before_support",
        action="store_true",
        help="Ignore events before support end time",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    # Load predictions (keyed by stem)
    pred_dir = Path(args.pred_dir)
    predictions = load_predictions(pred_dir)
    logger.info(f"Loaded predictions for {len(predictions)} files")

    # Discover ground truth (keyed by stem)
    gt_dir = Path(args.gt_dir)
    gt_pairs = discover_dataset(gt_dir)

    # Evaluate
    total_tp = 0
    total_fp = 0
    total_fn = 0

    results_per_file = []

    for audio_path, annotation_path in gt_pairs:
        stem = audio_path.stem
        
        if stem not in predictions:
            logger.warning(f"No predictions for {stem}")
            continue

        pred_events = predictions[stem]
        
        # Load ground truth
        gt_events = read_annotation_csv(annotation_path, label_filter="POS")

        # Filter by support end time
        if args.ignore_before_support:
            support_events = read_support_events(annotation_path, config.fewshot.n_support)
            support_end = get_support_end_time(support_events)

            # Remove support events and events before support end
            gt_events = [e for e in gt_events if e.starttime >= support_end]
            pred_events = [e for e in pred_events if e.start_time >= support_end]

        # Compute metrics
        from fsbed.eval.metrics import match_events
        result = match_events(
            pred_events,
            gt_events,
            onset_tolerance_s=args.onset_tolerance,
            offset_tolerance_s=args.offset_tolerance,
            min_iou=args.min_iou,
        )
        
        precision = result.precision
        recall = result.recall
        f1 = result.f1

        # Collect results
        results_per_file.append({
            "filename": stem,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": result.true_positives,
            "fp": result.false_positives,
            "fn": result.false_negatives,
            "n_pred": len(pred_events),
            "n_gt": len(gt_events)
        })
        
        logger.info(f"{stem}.wav: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (TP={result.true_positives})")

    # Compute macro average
    if results_per_file:
        df_res = pd.DataFrame(results_per_file)
        macro_f1 = df_res["f1"].mean()
        macro_p = df_res["precision"].mean()
        macro_r = df_res["recall"].mean()
        
        print("\n" + "="*60)
        print("MACRO-AVERAGE RESULTS (per file)")
        print("="*60)
        print(f"Precision: {macro_p:.4f}")
        print(f"Recall:    {macro_r:.4f}")
        print(f"F1-Score:  {macro_f1:.4f}")
        print("="*60)
        
        # Save results
        out_csv = pred_dir / "evaluation_results.csv"
        df_res.to_csv(out_csv, index=False)
        logger.info(f"Saved results to {out_csv}")
    else:
        logger.warning("No files evaluated")


if __name__ == "__main__":
    main()
