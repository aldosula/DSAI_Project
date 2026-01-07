"""Single file inference pipeline with chunked processing."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..config import Config
from ..audio.features import LogMelExtractor
from ..audio.io import load_audio, get_audio_duration
from ..audio.chunking import iter_audio_chunks, time_to_frame_idx
from ..data.annotations import read_support_events, get_support_end_time, Annotation
from ..models.resnet_embed import ResNetEmbedding
from ..models.heads import BinaryHead, PrototypeInit
from ..adapt.finetune import per_file_adaptation
from ..adapt.negatives import get_negative_mask, hard_negative_mining
from ..postprocess.smooth import smooth_probs, kernel_size_from_seconds
from ..postprocess.events import postprocess_predictions, Event

logger = logging.getLogger(__name__)


def _extract_embeddings_chunked(
    waveform: torch.Tensor,
    backbone: nn.Module,
    feature_extractor: LogMelExtractor,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """
    Extract embeddings from audio in chunks to avoid GPU OOM.

    Args:
        waveform: Audio waveform [1, samples].
        backbone: Embedding backbone.
        feature_extractor: Log-mel feature extractor.
        config: Configuration.
        device: Compute device.

    Returns:
        Tuple of (embeddings [time, embed_dim], total_frames).
    """
    chunk_samples = int(config.inference.chunk_duration_s * config.audio.sample_rate)
    overlap_samples = int(config.inference.chunk_overlap_s * config.audio.sample_rate)
    hop_length = config.audio.hop_length
    
    # Calculate overlap in frames for stitching
    overlap_frames = overlap_samples // hop_length
    
    all_embeddings = []
    total_frames = 0
    
    backbone.eval()
    feature_extractor.eval()
    
    for chunk_idx, (chunk, start_sample) in enumerate(
        iter_audio_chunks(waveform, chunk_samples, overlap_samples)
    ):
        # Move chunk to device and extract features
        chunk = chunk.unsqueeze(0).to(device)  # [1, 1, samples]
        
        with torch.no_grad():
            # Extract mel features
            features = feature_extractor(chunk)  # [1, 1, n_mels, time]
            
            # Get embeddings
            embeddings = backbone(features)  # [1, time, embed_dim]
            embeddings = embeddings.squeeze(0)  # [time, embed_dim]
        
        chunk_frames = embeddings.shape[0]
        
        if chunk_idx == 0:
            # First chunk: keep all frames
            all_embeddings.append(embeddings.cpu())
            total_frames += chunk_frames
        else:
            # Subsequent chunks: discard overlap frames from the beginning
            # to avoid duplicating embeddings
            if overlap_frames > 0 and chunk_frames > overlap_frames:
                embeddings = embeddings[overlap_frames:]
            all_embeddings.append(embeddings.cpu())
            total_frames += embeddings.shape[0]
        
        # Clear GPU cache periodically
        if chunk_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)  # [total_frames, embed_dim]
    
    return embeddings.to(device), embeddings.shape[0]


def infer_single_file(
    audio_path: Path,
    annotation_path: Path,
    backbone: nn.Module,
    config: Config,
    device: torch.device,
) -> list[Event]:
    """
    Run inference on a single audio file with chunked processing.

    Args:
        audio_path: Path to audio file.
        annotation_path: Path to annotation CSV.
        backbone: Trained embedding backbone.
        config: Configuration.
        device: Compute device.

    Returns:
        List of detected events.
    """
    logger.info(f"Processing {audio_path.name}")

    # Read support events
    support_events = read_support_events(annotation_path, config.fewshot.n_support)
    if len(support_events) < config.fewshot.n_support:
        logger.warning(f"Only {len(support_events)} support events found")

    support_end_time = get_support_end_time(support_events)
    logger.info(f"Support end time: {support_end_time:.2f}s")

    # Load audio
    waveform, sr = load_audio(
        audio_path,
        target_sr=config.audio.sample_rate,
        mono=True,
    )
    
    # Determine if we should use chunked processing
    duration_s = waveform.shape[-1] / config.audio.sample_rate
    use_chunked = duration_s > config.inference.chunk_duration_s
    
    # Create feature extractor on device
    feature_extractor = LogMelExtractor.from_config(config.audio).to(device)
    
    if use_chunked:
        logger.debug(f"Using chunked processing for {duration_s:.1f}s audio")
        embeddings, num_frames = _extract_embeddings_chunked(
            waveform, backbone, feature_extractor, config, device
        )
    else:
        # Short file: process all at once
        logger.debug(f"Processing {duration_s:.1f}s audio in single pass")
        waveform = waveform.unsqueeze(0).to(device)  # [1, 1, samples]
        
        backbone.eval()
        with torch.no_grad():
            features = feature_extractor(waveform)  # [1, 1, n_mels, time]
            embeddings = backbone(features)  # [1, time, embed_dim]
        embeddings = embeddings.squeeze(0)  # [time, embed_dim]
        num_frames = embeddings.shape[0]

    # Get support frame indices
    support_frames = []
    for event in support_events:
        center_frame = time_to_frame_idx(
            event.center,
            config.audio.hop_length,
            config.audio.sample_rate,
        )
        # Clamp to valid range
        center_frame = min(center_frame, num_frames - 1)
        support_frames.append(center_frame)
    support_frames = torch.tensor(support_frames, device=device)

    # Create binary head
    head = BinaryHead(config.model.embed_dim).to(device)

    # Per-file adaptation
    if config.adapt.enabled:
        head = per_file_adaptation(
            backbone,
            head,
            embeddings,
            support_frames,
            num_frames,
            config.adapt,
            config.audio,
            device,
        )
    else:
        # Simple prototype initialization
        pos_emb = embeddings[support_frames]
        neg_mask = get_negative_mask(
            num_frames,
            support_events,
            config.audio.hop_length,
            config.audio.sample_rate,
        )
        neg_emb, _ = hard_negative_mining(embeddings, neg_mask.to(device), pos_emb.mean(0))
        PrototypeInit.compute_and_init(head, pos_emb, neg_emb)

    # Inference
    head.eval()
    with torch.no_grad():
        probs = head.get_probs(embeddings)  # [time]

    # Smoothing
    kernel_size = kernel_size_from_seconds(
        config.postprocess.smooth_kernel_s,
        config.audio.hop_length,
        config.audio.sample_rate,
    )
    probs = smooth_probs(probs, method=config.postprocess.smooth_method, kernel_size=kernel_size)

    # Post-processing
    frame_duration = config.audio.hop_length / config.audio.sample_rate
    min_start_time = support_end_time if config.postprocess.output_after_support_end else None

    events = postprocess_predictions(
        probs,
        frame_duration,
        threshold=config.postprocess.threshold,
        merge_gap_s=config.postprocess.merge_gap_s,
        min_event_s=config.postprocess.min_event_s,
        max_event_s=config.postprocess.max_event_s,
        min_start_time=min_start_time,
    )

    logger.info(f"Detected {len(events)} events")
    return events
