"""Inference pipeline for baseline model"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..config import Config
from ..audio.features import LogMelExtractor
from ..audio.io import load_audio
from ..audio.chunking import iter_audio_chunks, time_to_frame_idx
from ..data.annotations import read_support_events, get_support_end_time
from ..models.baseline_embed import BaselineEmbedding
from ..adapt.finetune_baseline import two_step_finetune
from ..postprocess.smooth import smooth_probs, kernel_size_from_seconds
from ..postprocess.events import postprocess_predictions, Event

logger = logging.getLogger(__name__)


def load_baseline_model(
    checkpoint_path: Path,
    config: Config,
    device: torch.device,
) -> tuple[BaselineEmbedding, Optional[nn.Linear]]:
    """
    Load trained baseline backbone and pretrained classifier.
    
    Returns:
        Tuple of (backbone, pretrained_classifier).
        pretrained_classifier is the 20-class head from pretraining.
    """
    backbone = BaselineEmbedding(
        n_mels=config.audio.n_mels,
        embed_dim=config.model.embed_dim,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained_classifier = None
    
    # Try to load backbone weights and classifier
    if 'backbone_state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
    elif 'model_state_dict' in checkpoint:
        # Extract backbone and classifier weights from full model
        state_dict = checkpoint['model_state_dict']
        backbone_dict = {}
        classifier_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                backbone_dict[k.replace('backbone.', '')] = v
            elif k.startswith('classifier.'):
                classifier_dict[k.replace('classifier.', '')] = v
        
        if backbone_dict:
            backbone.load_state_dict(backbone_dict)
        else:
            backbone.load_state_dict(state_dict)
        
        # Try to reconstruct the classifier
        if classifier_dict:
            # Infer dimensions from weights
            if 'weight' in classifier_dict:
                out_features, in_features = classifier_dict['weight'].shape
                pretrained_classifier = nn.Linear(in_features, out_features)
                pretrained_classifier.load_state_dict(classifier_dict)
                pretrained_classifier = pretrained_classifier.to(device)
                logger.info(f"Loaded pretrained {out_features}-class classifier")
    else:
        backbone.load_state_dict(checkpoint)
    
    backbone = backbone.to(device)
    backbone.eval()
    
    logger.info(f"Loaded baseline model from {checkpoint_path}")
    return backbone, pretrained_classifier


def infer_baseline_single_file(
    audio_path: Path,
    annotation_path: Path,
    backbone: nn.Module,
    config: Config,
    device: torch.device,
    pretrained_classifier: Optional[nn.Linear] = None,
) -> list[Event]:
    """
    Run baseline inference on a single audio file.
    
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
    
    # Extract features and embeddings
    feature_extractor = LogMelExtractor.from_config(config.audio).to(device)
    
    # Check if chunked processing needed
    duration_s = waveform.shape[-1] / config.audio.sample_rate
    chunk_samples = int(config.inference.chunk_duration_s * config.audio.sample_rate)
    overlap_samples = int(config.inference.chunk_overlap_s * config.audio.sample_rate)
    overlap_frames = overlap_samples // config.audio.hop_length
    
    if duration_s > config.inference.chunk_duration_s:
        # Chunked processing
        all_embeddings = []
        backbone.eval()
        
        for chunk_idx, (chunk, start_sample) in enumerate(
            iter_audio_chunks(waveform, chunk_samples, overlap_samples)
        ):
            chunk = chunk.unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = feature_extractor(chunk)
                # Normalize features (same as training)
                mean = features.mean()
                std = features.std() + 1e-8
                features = (features - mean) / std
                embeddings = backbone(features).squeeze(0)
            
            if chunk_idx == 0:
                all_embeddings.append(embeddings.cpu())
            else:
                if overlap_frames > 0 and embeddings.shape[0] > overlap_frames:
                    embeddings = embeddings[overlap_frames:]
                all_embeddings.append(embeddings.cpu())
            
            if chunk_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        embeddings = torch.cat(all_embeddings, dim=0).to(device)
        num_frames = embeddings.shape[0]
    else:
        # Single-pass
        waveform = waveform.unsqueeze(0).to(device)
        backbone.eval()
        with torch.no_grad():
            features = feature_extractor(waveform)
            # Normalize features (same as training)
            mean = features.mean()
            std = features.std() + 1e-8
            features = (features - mean) / std
            embeddings = backbone(features).squeeze(0)
        num_frames = embeddings.shape[0]
    
    # Get support frame indices
    support_frames = []
    for event in support_events:
        center_frame = time_to_frame_idx(
            event.center,
            config.audio.hop_length,
            config.audio.sample_rate,
        )
        center_frame = min(center_frame, num_frames - 1)
        support_frames.append(center_frame)
    support_frames = torch.tensor(support_frames, device=device)
    
    # Two-step fine-tuning with dual heads 
    adapter, probs = two_step_finetune(
        backbone,
        embeddings,
        support_frames,
        num_frames,
        config,
        device,
        pretrained_classifier=pretrained_classifier,
    )
    
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
