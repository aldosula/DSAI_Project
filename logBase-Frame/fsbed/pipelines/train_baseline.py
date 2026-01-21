"""Training pipeline for baseline model with multi-class pretraining.

Implements the Liu et al. training procedure:
- 20-class species classification
- Adam optimizer, lr=0.0003
- StepLR scheduler, gamma=0.5, step_size=10
- 80/20 train/val split
- Early stopping after 10 epochs
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..config import Config
from ..seed import set_seed
from ..data.dataset_multiclass import MultiClassDataset
from ..data.dataset_cached import CachedMultiClassDataset
from ..models.baseline_embed import BaselineEmbedding, BaselineFullModel
from ..audio.augmentation import mixup_batch

logger = logging.getLogger(__name__)


def train_baseline(config: Config) -> nn.Module:
    """
    Train baseline model with multi-class pretraining.
    
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
    
    # WandB logging
    if WANDB_AVAILABLE:
        wandb.init(
            project="fsbed",
            name=f"fsbed-{config.training.epochs}ep-lr{config.training.lr}",
            config={
                "epochs": config.training.epochs,
                "batch_size": config.training.batch_size,
                "lr": config.training.lr,
                "embed_dim": config.model.embed_dim,
                "feature_type": getattr(config.training, 'feature_type', 'logmel'),
                "use_mixup": getattr(config.training, 'use_mixup', True),
                "mixup_alpha": getattr(config.training, 'mixup_alpha', 0.4),
            }
        )
        logger.info("WandB logging enabled")
    
    # Feature type: logmel (default) or pcen
    feature_type = getattr(config.training, 'feature_type', 'logmel')
    feature_suffix = f"_{feature_type}.npy"
    logger.info(f"Creating dataset (using cached {feature_type} features)...")
    
    full_dataset = CachedMultiClassDataset(
        root_dir=config.paths.train_dir,
        audio_config=config.audio,
        segment_frames=17,
        augment=True,
        freq_mask_param=getattr(config.training, 'freq_mask_param', 27),
        time_mask_param=getattr(config.training, 'time_mask_param', 100),
        feature_suffix=feature_suffix,
    )
    
    num_classes = full_dataset.get_num_classes()
    logger.info(f"Number of classes: {num_classes}")
    
    # 80/20 train/val split with proper separation (no data leakage!)
    train_dataset, val_dataset = full_dataset.create_train_val_split(
        val_ratio=0.2, 
        seed=config.seed
    )
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    
    # Model
    logger.info("Creating model...")
    backbone = BaselineEmbedding(
        n_mels=config.audio.n_mels,
        embed_dim=config.model.embed_dim,
    )
    model = BaselineFullModel(backbone, num_classes=num_classes).to(device)
    
    # Loss with class weights
    class_weights = full_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer 
    lr = getattr(config.training, 'lr', 0.0003)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler 
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=getattr(config.training, 'scheduler_step', 10),
        gamma=getattr(config.training, 'scheduler_gamma', 0.5),
    )
    
    # Mixup settings
    use_mixup = getattr(config.training, 'use_mixup', True)
    mixup_alpha = getattr(config.training, 'mixup_alpha', 0.4)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = getattr(config.training, 'early_stop_patience', 10)
    
    for epoch in range(config.training.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs}")
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(device)
            labels = labels.to(device)
            
            # Apply mixup
            if use_mixup and mixup_alpha > 0:
                # Convert labels to one-hot
                one_hot = torch.zeros(labels.size(0), num_classes, device=device)
                one_hot.scatter_(1, labels.unsqueeze(1), 1)
                
                features, soft_labels = mixup_batch(features, one_hot, mixup_alpha)
            else:
                soft_labels = None
            
            optimizer.zero_grad()
            logits = model(features)
            
            # Loss
            if soft_labels is not None:
                # Soft cross-entropy for mixup
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = -(soft_labels * log_probs).sum(dim=-1).mean()
            else:
                loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                logits = model(features)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Epoch stats
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        lr_current = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss_avg:.4f}, Train Acc={train_acc:.2f}%, "
            f"Val Loss={val_loss_avg:.4f}, Val Acc={val_acc:.2f}%, "
            f"LR={lr_current:.6f}"
        )
        
        writer.add_scalar('train/loss', train_loss_avg, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss_avg, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('train/lr', lr_current, epoch)
        
        # WandB logging
        if WANDB_AVAILABLE:
            wandb.log({
                "train/loss": train_loss_avg,
                "train/accuracy": train_acc,
                "val/loss": val_loss_avg,
                "val/accuracy": val_acc,
                "lr": lr_current,
                "epoch": epoch + 1,
            })
        
        scheduler.step()
        
        # Save best model by accuracy 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'backbone_state_dict': backbone.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss_avg,
                'num_classes': num_classes,
                'classes': full_dataset.classes,
            }, checkpoint_dir / 'best.pt')
            logger.info(f"Saved best checkpoint (Val Acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    if WANDB_AVAILABLE:
        wandb.finish()
    logger.info(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
