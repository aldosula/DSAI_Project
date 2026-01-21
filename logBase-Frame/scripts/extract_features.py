#!/usr/bin/env python3
"""Feature extraction script.

Computes and caches Log-Mel and PCEN features for all audio files in given directories.
Output files:
- [filename]_logmel.npy
- [filename]_pcen.npy
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fsbed.config import Config
from fsbed.logging_utils import setup_logging
from fsbed.audio.io import load_audio, load_audio_segment, get_audio_info
from fsbed.audio.features import LogMelExtractor, PCENExtractor
from fsbed.data.dcase_paths import discover_dataset

logger = logging.getLogger(__name__)


def extract_and_save(
    audio_path: Path,
    config: Config,
    device: torch.device,
    logmel_extractor: LogMelExtractor,
    pcen_extractor: PCENExtractor,
    overwrite: bool = False,
    chunk_duration_s: float = 300.0,  # 5 minutes
):
    """Extract and save features for a single file using chunking to save memory."""
    
    # Check if outputs exist
    logmel_path = audio_path.with_suffix('').parent / f"{audio_path.stem}_logmel.npy"
    pcen_path = audio_path.with_suffix('').parent / f"{audio_path.stem}_pcen.npy"
    
    if not overwrite and logmel_path.exists() and pcen_path.exists():
        return
        
    try:
        # Get info without loading
        num_frames, orig_sr = get_audio_info(audio_path)
        total_duration_s = num_frames / orig_sr
        
        logmel_list = []
        pcen_list = []
        
        # Process in chunks using partial loading
        for start_s in np.arange(0, total_duration_s, chunk_duration_s):
            end_s = min(start_s + chunk_duration_s, total_duration_s)
            
            # Load only the segment (CPU)
            chunk, sr = load_audio_segment(
                audio_path,
                start_s=start_s,
                end_s=end_s,
                target_sr=config.audio.sample_rate,
                mono=True,
            )
            
            # Move to GPU
            chunk = chunk.to(device).unsqueeze(0)  # [1, 1, samples]
            
            if chunk.size(-1) < 10:  # Skip empty/tiny chunks
                break
            
            # Extract LogMel
            with torch.no_grad():
                logmel = logmel_extractor(chunk)  # [1, 1, n_mels, time]
                logmel = logmel.squeeze(0).squeeze(0).T  # [time, n_mels]
                logmel_list.append(logmel.cpu().numpy())
            
            # Extract PCEN
            with torch.no_grad():
                pcen = pcen_extractor(chunk)  # [1, 1, n_mels, time]
                pcen = pcen.squeeze(0).squeeze(0).T  # [time, n_mels]
                pcen_list.append(pcen.cpu().numpy())
        
        # Concatenate
        if logmel_list:
            if overwrite or not logmel_path.exists():
                full_logmel = np.concatenate(logmel_list, axis=0)
                np.save(logmel_path, full_logmel)
                
            if overwrite or not pcen_path.exists():
                full_pcen = np.concatenate(pcen_list, axis=0)
                np.save(pcen_path, full_pcen)
            
        # Explicit cleanup
        torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Failed to process {audio_path.name}: {e}")
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache features (LogMel and PCEN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="List of directories to scan for .wav files",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
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
    
    # Initialize extractors
    logger.info("Initializing feature extractors...")
    logmel_extractor = LogMelExtractor.from_config(config.audio).to(device)
    pcen_extractor = PCENExtractor.from_config(config.audio).to(device)
    logmel_extractor.eval()
    pcen_extractor.eval()
    
    # Find all files
    all_files = []
    for d in args.dirs:
        path = Path(d)
        if not path.exists():
            logger.warning(f"Directory not found: {path}")
            continue
            
        # Find .wav files recursively
        files = list(path.rglob("*.wav"))
        logger.info(f"Found {len(files)} .wav files in {path}")
        all_files.extend(files)
        
    if not all_files:
        logger.error("No audio files found!")
        return
        
    logger.info(f"Processing {len(all_files)} total files...")
    
    # Process
    for audio_path in tqdm(all_files, desc="Extracting features"):
        extract_and_save(
            audio_path,
            config,
            device,
            logmel_extractor,
            pcen_extractor,
            args.overwrite,
        )
        
    logger.info("Extraction complete!")


if __name__ == "__main__":
    main()
