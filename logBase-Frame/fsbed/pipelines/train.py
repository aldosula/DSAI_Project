"""Training pipeline for backbone network."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import Config
from ..seed import set_seed
from ..data.dataset_train import TrainingDataset
from ..models.resnet_embed import ResNetEmbedding, FullModel
from ..models.heads import BinaryHead
from ..models.losses import CombinedLoss

logger = logging.getLogger(__name__)


def train_backbone(config: Config) -> nn.Module:
    """
    Train embedding backbone on training data.

    Args:
        config: Full configuration.

    Returns:
        Trained model.
    """
    # Setup
    set_seed(config.seed, config.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(config.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # Dataset
    logger.info("Creating dataset...")
    train_dataset = TrainingDataset(
        root_dir=config.paths.train_dir,
        audio_config=config.audio,
        segment_duration_s=0.2,
        samples_per_epoch=config.training.batch_size * 500,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # Model
    logger.info("Creating model...")
    backbone = ResNetEmbedding.from_config(config.model, n_mels=config.audio.n_mels)
    head = BinaryHead(config.model.embed_dim)
    model = FullModel(backbone, head).to(device)

    # Loss
    class_weights = train_dataset.get_class_weights().to(device) if config.training.class_balance else None
    criterion = CombinedLoss(class_weights=class_weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Scheduler
    if config.training.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.lr / 100,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.training.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs}")
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)

            # Handle variable time dimension
            if logits.dim() == 3:
                # Pool over time
                logits = logits.mean(dim=1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100*correct/total:.2f}%"
            })

        # Epoch stats
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, LR={lr:.6f}")
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/accuracy", accuracy, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        scheduler.step()

        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, checkpoint_dir / "best.pt")
            logger.info("Saved best checkpoint")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.training.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    writer.close()
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")

    # Load best model
    checkpoint = torch.load(checkpoint_dir / "best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
