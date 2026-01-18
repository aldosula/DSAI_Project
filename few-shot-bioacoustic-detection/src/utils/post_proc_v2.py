"""
Enhanced Post-Processing V2 for Few-Shot Bioacoustic Event Detection.

Improvements over original:
1. Confidence-based filtering (filter low similarity predictions)
2. Non-Maximum Suppression (remove overlapping detections)  
3. Adaptive duration filtering using mean ± std of shot durations
4. Gap-based event merging
"""

import argparse
import csv
import os
from typing import List, Tuple, Dict
import numpy as np


def load_predictions(evaluation_file: str) -> List[Tuple[str, float, float, float]]:
    """Load predictions from CSV, optionally with confidence scores.
    
    Returns list of (audiofile, start, end, confidence) tuples.
    If no confidence column, defaults to 1.0.
    """
    results = []
    with open(evaluation_file, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        header = next(reader, None)
        has_confidence = len(header) >= 4 and "Confidence" in header
        
        for row in reader:
            audiofile = row[0]
            start = float(row[1])
            end = float(row[2])
            confidence = float(row[3]) if has_confidence and len(row) > 3 else 1.0
            results.append((audiofile, start, end, confidence))
    
    return results


def get_shot_durations(val_path: str, n_shots: int = 5) -> Dict[str, List[float]]:
    """Get durations of the first n_shots for each audio file.
    
    Returns dict: {audiofile: [duration1, duration2, ...]}
    """
    dict_durations = {}
    folders = os.listdir(val_path)
    
    for folder in folders:
        folder_path = os.path.join(val_path, folder)
        if not os.path.isdir(folder_path):
            continue
            
        files = os.listdir(folder_path)
        for file in files:
            if file.endswith(".csv"):
                audiofile = file[:-4] + ".wav"
                durations = []
                
                with open(os.path.join(folder_path, file)) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    for row in csv_reader:
                        if len(row) > 0 and row[-1] == "POS" and len(durations) < n_shots:
                            start = float(row[1])
                            end = float(row[2])
                            durations.append(end - start)
                
                if durations:
                    dict_durations[audiofile] = durations
    
    return dict_durations


def filter_by_confidence(
    predictions: List[Tuple[str, float, float, float]],
    min_confidence: float = 0.5
) -> List[Tuple[str, float, float, float]]:
    """Remove predictions with confidence below threshold."""
    return [p for p in predictions if p[3] >= min_confidence]


def filter_by_duration_adaptive(
    predictions: List[Tuple[str, float, float, float]],
    shot_durations: Dict[str, List[float]],
    n_std: float = 1.5
) -> List[Tuple[str, float, float, float]]:
    """Filter by duration using mean ± n_std * std of shot durations.
    
    More robust than using just min duration.
    """
    filtered = []
    
    for pred in predictions:
        audiofile, start, end, conf = pred
        duration = end - start
        
        if audiofile not in shot_durations:
            filtered.append(pred)
            continue
        
        durations = shot_durations[audiofile]
        mean_dur = np.mean(durations)
        std_dur = np.std(durations) if len(durations) > 1 else mean_dur * 0.3
        
        min_dur = max(0.05, mean_dur - n_std * std_dur)
        max_dur = mean_dur + n_std * std_dur * 2  # Allow longer events
        
        if min_dur <= duration <= max_dur:
            filtered.append(pred)
    
    return filtered


def non_maximum_suppression(
    predictions: List[Tuple[str, float, float, float]],
    iou_threshold: float = 0.3
) -> List[Tuple[str, float, float, float]]:
    """Apply NMS to remove overlapping detections, keeping highest confidence."""
    if not predictions:
        return []
    
    # Group by audiofile
    by_file = {}
    for pred in predictions:
        audiofile = pred[0]
        if audiofile not in by_file:
            by_file[audiofile] = []
        by_file[audiofile].append(pred)
    
    result = []
    for audiofile, preds in by_file.items():
        # Sort by confidence (descending)
        preds = sorted(preds, key=lambda x: x[3], reverse=True)
        keep = []
        
        while preds:
            best = preds.pop(0)
            keep.append(best)
            
            # Remove overlapping predictions
            remaining = []
            for p in preds:
                iou = compute_iou(best[1], best[2], p[1], p[2])
                if iou < iou_threshold:
                    remaining.append(p)
            preds = remaining
        
        result.extend(keep)
    
    return result


def compute_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Compute Intersection over Union of two time intervals."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start
    union = (end1 - start1) + (end2 - start2) - intersection
    
    return intersection / union if union > 0 else 0.0


def merge_close_events(
    predictions: List[Tuple[str, float, float, float]],
    max_gap: float = 0.1
) -> List[Tuple[str, float, float, float]]:
    """Merge events that are close together (gap < max_gap seconds)."""
    if not predictions:
        return []
    
    # Group by audiofile
    by_file = {}
    for pred in predictions:
        audiofile = pred[0]
        if audiofile not in by_file:
            by_file[audiofile] = []
        by_file[audiofile].append(pred)
    
    result = []
    for audiofile, preds in by_file.items():
        # Sort by start time
        preds = sorted(preds, key=lambda x: x[1])
        
        merged = [preds[0]]
        for pred in preds[1:]:
            last = merged[-1]
            gap = pred[1] - last[2]  # gap between events
            
            if gap <= max_gap:
                # Merge: extend end time, average confidence
                new_end = max(last[2], pred[2])
                new_conf = (last[3] + pred[3]) / 2
                merged[-1] = (audiofile, last[1], new_end, new_conf)
            else:
                merged.append(pred)
        
        result.extend(merged)
    
    return result


def save_predictions(
    predictions: List[Tuple[str, float, float, float]],
    output_file: str,
    include_confidence: bool = False
):
    """Save predictions to CSV file."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        if include_confidence:
            writer.writerow(["Audiofilename", "Starttime", "Endtime", "Confidence"])
            for pred in predictions:
                writer.writerow([pred[0], pred[1], pred[2], pred[3]])
        else:
            writer.writerow(["Audiofilename", "Starttime", "Endtime"])
            for pred in predictions:
                writer.writerow([pred[0], pred[1], pred[2]])


def post_processing_v2(
    val_path: str,
    evaluation_file: str,
    new_evaluation_file: str,
    n_shots: int = 5,
    min_confidence: float = 0.0,  # Disabled by default (no confidence in input)
    use_nms: bool = True,
    nms_iou_threshold: float = 0.7,  # Lu et al. paper uses IoU=0.7
    use_duration_filter: bool = True,
    duration_n_std: float = 1.5,
    use_merge: bool = False,
    merge_max_gap: float = 0.1,
    min_duration: float = 0.05,  # Absolute minimum duration
):
    """Enhanced post-processing pipeline.
    
    Parameters
    ----------
    val_path: Path to validation set folder
    evaluation_file: Input prediction CSV
    new_evaluation_file: Output prediction CSV
    n_shots: Number of shots
    min_confidence: Minimum confidence threshold
    use_nms: Whether to apply NMS
    nms_iou_threshold: IoU threshold for NMS
    use_duration_filter: Whether to filter by adaptive duration
    duration_n_std: Number of std deviations for duration filter
    use_merge: Whether to merge close events
    merge_max_gap: Maximum gap for merging (seconds)
    min_duration: Absolute minimum duration threshold
    """
    # Load predictions
    predictions = load_predictions(evaluation_file)
    print(f"Loaded {len(predictions)} predictions")
    
    # Get shot durations
    shot_durations = get_shot_durations(val_path, n_shots)
    
    # Apply filters in order
    
    # 1. Confidence filter
    if min_confidence > 0:
        predictions = filter_by_confidence(predictions, min_confidence)
        print(f"After confidence filter: {len(predictions)}")
    
    # 2. Remove very short events (absolute minimum)
    predictions = [p for p in predictions if (p[2] - p[1]) >= min_duration]
    print(f"After min duration filter: {len(predictions)}")
    
    # 3. Adaptive duration filter
    if use_duration_filter:
        predictions = filter_by_duration_adaptive(predictions, shot_durations, duration_n_std)
        print(f"After adaptive duration filter: {len(predictions)}")
    
    # 4. NMS
    if use_nms:
        predictions = non_maximum_suppression(predictions, nms_iou_threshold)
        print(f"After NMS: {len(predictions)}")
    
    # 5. Merge close events
    if use_merge:
        predictions = merge_close_events(predictions, merge_max_gap)
        print(f"After merge: {len(predictions)}")
    
    # Save results
    save_predictions(predictions, new_evaluation_file, include_confidence=False)
    print(f"Saved to {new_evaluation_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced post-processing V2")
    parser.add_argument("-val_path", type=str, required=True)
    parser.add_argument("-evaluation_file", type=str, required=True)
    parser.add_argument("-new_evaluation_file", type=str, required=True)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--use_nms", action="store_true", default=True)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.3)
    parser.add_argument("--use_duration_filter", action="store_true", default=True)
    parser.add_argument("--duration_n_std", type=float, default=1.5)
    parser.add_argument("--use_merge", action="store_true", default=False)
    parser.add_argument("--merge_max_gap", type=float, default=0.1)
    
    args = parser.parse_args()
    
    post_processing_v2(
        args.val_path,
        args.evaluation_file,
        args.new_evaluation_file,
        min_confidence=args.min_confidence,
        use_nms=args.use_nms,
        nms_iou_threshold=args.nms_iou_threshold,
        use_duration_filter=args.use_duration_filter,
        duration_n_std=args.duration_n_std,
        use_merge=args.use_merge,
        merge_max_gap=args.merge_max_gap,
    )
