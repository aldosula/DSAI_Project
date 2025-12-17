import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import BioacousticDataset
from src.features import FeatureExtractor
from src.model import PrototypicalNetwork
import argparse
import random
import numpy as np
from tqdm import tqdm

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Feature Extractor
    transform = FeatureExtractor(use_pcen=args.use_pcen).to(device)
    
    # Dataset
    # We need to split classes manually since Dataset loads all.
    # Actually, better to modify Dataset to accept class list or split ratio.
    # For now, let's instantiate one dataset, get classes, and split.
    
    full_dataset = BioacousticDataset(
        root_dir=args.data_dir,
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        episodes=args.episodes_per_epoch
    )
    
    all_classes = full_dataset.all_classes
    random.shuffle(all_classes)
    
    split_idx = int(len(all_classes) * 0.8)
    train_classes = all_classes[:split_idx]
    val_classes = all_classes[split_idx:]
    
    print(f"Total classes: {len(all_classes)}. Train: {len(train_classes)}, Val: {len(val_classes)}")
    
    # We need to hack the dataset to restrict classes.
    # A cleaner way would be to pass allowed_classes to __init__.
    # Let's just modify the instance for now.
    
    train_dataset = BioacousticDataset(
        root_dir=args.data_dir,
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        episodes=args.episodes_per_epoch
    )
    train_dataset.all_classes = train_classes
    
    val_dataset = BioacousticDataset(
        root_dir=args.data_dir,
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        episodes=args.episodes_per_epoch // 5
    )
    val_dataset.all_classes = val_classes
    
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    
    # Model
    model = PrototypicalNetwork(backbone=args.backbone).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        
        # Wrap training loop with tqdm progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", 
                          bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        
        for i, (support, support_labels, query, query_labels) in enumerate(train_pbar):
            # Shapes: (1, N*K, ...), (1, N*K), ...
            support = support.squeeze(0).to(device)
            query = query.squeeze(0).to(device)
            support_labels = support_labels.squeeze(0).to(device)
            query_labels = query_labels.squeeze(0).to(device)
            
            # Extract features
            # Note: transform is on device, but inputs might be raw audio?
            # Dataset returns raw audio if transform is None.
            # We passed transform=None to Dataset, so we do it here.
            # Wait, Dataset code: if self.transform: wav = self.transform(wav)
            # We didn't pass transform to Dataset constructor above.
            # So we do it here.
            
            with torch.no_grad():
                support = transform(support)
                query = transform(query)
                
            optimizer.zero_grad()
            
            # Forward
            # logits: (N*Q, N) - distances
            logits = model(support, query, args.n_way, args.n_support, args.n_query)
            
            # Labels for query are 0..N-1
            loss = criterion(logits, query_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Acc
            _, preds = torch.max(logits, 1)
            acc = (preds == query_labels).float().mean()
            total_acc += acc.item()
            
            # Update progress bar with current metrics
            current_loss = total_loss / (i + 1)
            current_acc = total_acc / (i + 1)
            train_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
            
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Wrap validation dataloader with tqdm
        val_pbar = tqdm(val_loader, desc=f"Validation", 
                        bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        
        with torch.no_grad():
            for i, (support, support_labels, query, query_labels) in enumerate(val_pbar):
                support = support.squeeze(0).to(device) # Added squeeze(0) back
                support_labels = support_labels.squeeze(0).to(device) # Added squeeze(0) back
                query = query.squeeze(0).to(device) # Added squeeze(0) back
                query_labels = query_labels.squeeze(0).to(device) # Added squeeze(0) back
                
                # Apply feature extraction, as in training loop
                support = transform(support)
                query = transform(query)
                
                logits = model(support, query, args.n_way, args.n_support, args.n_query)
                logits = logits.view(args.n_way * args.n_query, args.n_way)
                
                loss = criterion(logits, query_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += query_labels.size(0)
                val_correct += (predicted == query_labels).sum().item()
                
                # Update progress bar
                current_val_loss = val_loss / (i + 1)
                current_val_acc = 100 * val_correct / val_total
                val_pbar.set_postfix({
                    'loss': f'{current_val_loss:.4f}',
                    'acc': f'{current_val_acc:.2f}%'
                })
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total  # Fixed: use val_correct/val_total instead of val_acc
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f} Acc: {avg_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}")
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/msc-proj/Training_Set")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--n_support", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=5)
    parser.add_argument("--episodes_per_epoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--use_pcen", action="store_true")
    
    args = parser.parse_args()
    train(args)
