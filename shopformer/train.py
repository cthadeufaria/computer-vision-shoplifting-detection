"""
Shopformer Training Script (Enhanced)

Two-stage training with:
- Learning rate scheduling
- Early stopping
- Gradient clipping
- Weight decay regularization
- Data augmentation
- Comprehensive logging

Usage:
    python train.py --data_dir ./data/PoseLift
    python train.py --use_synthetic --stage1_epochs 20 --stage2_epochs 50
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from models import Shopformer
from data import PoseLiftDataModule
from data.poselift_dataset import SyntheticPoseLiftDataset
from utils.metrics import compute_auc_roc, compute_metrics


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class PoseAugmentation:
    """Data augmentation for pose sequences."""

    def __init__(self,
                 jitter_std: float = 0.02,
                 scale_range: tuple = (0.9, 1.1),
                 rotation_range: float = 0.1,
                 temporal_dropout: float = 0.1):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.temporal_dropout = temporal_dropout

    def __call__(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to pose batch.
        Args:
            poses: (batch, 2, seq_len, num_keypoints)
        """
        batch_size = poses.size(0)

        # Random jitter
        if self.jitter_std > 0:
            noise = torch.randn_like(poses) * self.jitter_std
            poses = poses + noise

        # Random scaling per sample
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scales = torch.empty(batch_size, 1, 1, 1).uniform_(*self.scale_range)
            scales = scales.to(poses.device)
            poses = poses * scales

        # Random rotation (small angle)
        if self.rotation_range > 0:
            angles = torch.empty(batch_size).uniform_(-self.rotation_range, self.rotation_range)
            cos_a = torch.cos(angles).view(batch_size, 1, 1, 1).to(poses.device)
            sin_a = torch.sin(angles).view(batch_size, 1, 1, 1).to(poses.device)

            x = poses[:, 0:1, :, :]
            y = poses[:, 1:2, :, :]

            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a

            poses = torch.cat([new_x, new_y], dim=1)

        # Temporal dropout (zero out random frames)
        if self.temporal_dropout > 0:
            seq_len = poses.size(2)
            mask = torch.rand(batch_size, 1, seq_len, 1) > self.temporal_dropout
            mask = mask.to(poses.device).float()
            poses = poses * mask

        return poses


def train_stage1(
    model: Shopformer,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    augmentation: PoseAugmentation = None
) -> float:
    """Train GCAE tokenizer (Stage 1)."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch}")
    for batch_idx, (poses, _) in enumerate(pbar):
        poses = poses.to(device)

        # Apply augmentation
        if augmentation is not None:
            poses = augmentation(poses)

        optimizer.zero_grad()

        # Forward through GCAE
        reconstructed, tokens = model.gcae(poses)

        # Reconstruction loss
        loss = nn.functional.mse_loss(reconstructed, poses)

        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{current_lr:.2e}'})

    # Step scheduler (if per-epoch)
    if scheduler is not None and isinstance(scheduler, CosineAnnealingWarmRestarts):
        scheduler.step()

    return total_loss / len(train_loader)


def train_stage2(
    model: Shopformer,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    augmentation: PoseAugmentation = None
) -> float:
    """Train transformer with frozen GCAE encoder (Stage 2)."""
    model.train()

    # Freeze GCAE encoder
    for param in model.gcae.encoder.parameters():
        param.requires_grad = False

    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch}")
    for batch_idx, (poses, _) in enumerate(pbar):
        poses = poses.to(device)

        # Apply augmentation
        if augmentation is not None:
            poses = augmentation(poses)

        optimizer.zero_grad()

        # Get tokens from frozen encoder
        with torch.no_grad():
            tokens = model.tokenize(poses)

        # Reconstruct tokens with transformer
        reconstructed_tokens = model.reconstruct_tokens(tokens)

        # Add positional encoding to tokens for loss
        tokens_pe = model.pos_encoder.pe[:, :tokens.size(1), :].expand(
            tokens.size(0), -1, -1
        ).to(device)
        tokens_with_pe = tokens + tokens_pe

        # Reconstruction loss
        loss = nn.functional.mse_loss(reconstructed_tokens, tokens_with_pe)

        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{current_lr:.2e}'})

    return total_loss / len(train_loader)


def evaluate(
    model: Shopformer,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate model on test set."""
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for poses, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            poses = poses.to(device)

            output = model(poses)
            scores = output['normality_score'].cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_labels, all_scores)
    auc, fpr, tpr = compute_auc_roc(all_labels, all_scores)

    return {
        **metrics,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Train Shopformer (Enhanced)")
    parser.add_argument('--data_dir', type=str, default='./data/PoseLift',
                        help='Path to PoseLift dataset')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')

    # Model config
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--num_keypoints', type=int, default=17)
    parser.add_argument('--num_tokens', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--latent_channels', type=int, default=8)
    parser.add_argument('--transformer_heads', type=int, default=2)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--transformer_ff_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training config
    parser.add_argument('--stage1_epochs', type=int, default=30)
    parser.add_argument('--stage2_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='auto')

    # Scheduler config
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'])
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=15)

    # Augmentation
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--jitter_std', type=float, default=0.02)
    parser.add_argument('--scale_range', type=float, nargs=2, default=[0.95, 1.05])
    parser.add_argument('--rotation_range', type=float, default=0.05)
    parser.add_argument('--temporal_dropout', type=float, default=0.05)

    # Logging
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup augmentation
    augmentation = None
    if args.augment:
        augmentation = PoseAugmentation(
            jitter_std=args.jitter_std,
            scale_range=tuple(args.scale_range),
            rotation_range=args.rotation_range,
            temporal_dropout=args.temporal_dropout
        )
        print("Data augmentation enabled")

    # Setup data
    print("Setting up data...")
    data_module = PoseLiftDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        use_synthetic=args.use_synthetic,
        synthetic_samples=2000 if args.use_synthetic else 0
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Test samples: {len(data_module.test_dataset)}")

    # Create model
    print("Creating model...")
    model = Shopformer(
        in_channels=2,
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        num_keypoints=args.num_keypoints,
        seq_len=args.seq_len,
        num_tokens=args.num_tokens,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Save config
    config = vars(args)
    config['timestamp'] = timestamp
    config['total_params'] = total_params
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training history
    history = {
        'stage1_losses': [],
        'stage2_losses': [],
        'stage2_aucs': [],
        'stage1_lrs': [],
        'stage2_lrs': []
    }

    # ========== Stage 1: Train GCAE ==========
    print("\n" + "="*60)
    print("Stage 1: Training GCAE Tokenizer")
    print("="*60)

    optimizer_s1 = optim.AdamW(
        model.gcae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Setup scheduler for Stage 1
    if args.scheduler == 'cosine':
        T_0 = max(1, args.stage1_epochs // 3)  # Ensure T_0 >= 1
        scheduler_s1 = CosineAnnealingWarmRestarts(
            optimizer_s1,
            T_0=T_0,
            T_mult=2,
            eta_min=args.min_lr
        )
    elif args.scheduler == 'plateau':
        scheduler_s1 = ReduceLROnPlateau(
            optimizer_s1,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=args.min_lr
        )
    else:
        scheduler_s1 = None

    for epoch in range(1, args.stage1_epochs + 1):
        loss = train_stage1(
            model, train_loader, optimizer_s1, scheduler_s1,
            device, epoch, args.grad_clip, augmentation
        )
        history['stage1_losses'].append(loss)
        history['stage1_lrs'].append(optimizer_s1.param_groups[0]['lr'])

        if args.scheduler == 'plateau' and scheduler_s1 is not None:
            scheduler_s1.step(loss)

        print(f"Epoch {epoch}/{args.stage1_epochs}: Loss = {loss:.6f}, LR = {optimizer_s1.param_groups[0]['lr']:.2e}")

    # Save Stage 1 checkpoint
    torch.save({
        'epoch': args.stage1_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_s1.state_dict(),
        'losses': history['stage1_losses'],
        'lrs': history['stage1_lrs']
    }, output_dir / 'gcae_checkpoint.pt')
    print(f"Stage 1 checkpoint saved")

    # ========== Stage 2: Train Transformer ==========
    print("\n" + "="*60)
    print("Stage 2: Training Transformer")
    print("="*60)

    # Only optimize transformer parameters
    transformer_params = list(model.transformer.parameters()) + \
                        list(model.pos_encoder.parameters())

    optimizer_s2 = optim.AdamW(
        transformer_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Setup scheduler for Stage 2
    if args.scheduler == 'cosine':
        T_0 = max(1, args.stage2_epochs // 3)  # Ensure T_0 >= 1
        scheduler_s2 = CosineAnnealingWarmRestarts(
            optimizer_s2,
            T_0=T_0,
            T_mult=2,
            eta_min=args.min_lr
        )
    elif args.scheduler == 'plateau':
        scheduler_s2 = ReduceLROnPlateau(
            optimizer_s2,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=args.min_lr
        )
    else:
        scheduler_s2 = None

    # Early stopping
    early_stopper = EarlyStopping(patience=args.patience, mode='max') if args.early_stopping else None

    best_auc = 0.0
    best_epoch = 0

    for epoch in range(1, args.stage2_epochs + 1):
        loss = train_stage2(
            model, train_loader, optimizer_s2, scheduler_s2,
            device, epoch, args.grad_clip, augmentation
        )
        history['stage2_losses'].append(loss)
        history['stage2_lrs'].append(optimizer_s2.param_groups[0]['lr'])

        # Evaluate
        if epoch % args.eval_interval == 0:
            metrics = evaluate(model, test_loader, device)
            auc = metrics['auc_roc']
            history['stage2_aucs'].append(auc)

            print(f"Epoch {epoch}/{args.stage2_epochs}: Loss = {loss:.6f}, "
                  f"AUC-ROC = {auc:.4f}, LR = {optimizer_s2.param_groups[0]['lr']:.2e}")

            # Scheduler step (plateau mode)
            if args.scheduler == 'plateau' and scheduler_s2 is not None:
                scheduler_s2.step(auc)

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_s2.state_dict(),
                    'metrics': metrics,
                    'best_auc': best_auc,
                    'history': history
                }, output_dir / 'best_model.pt')
                print(f"  *** New best model saved (AUC: {best_auc:.4f}) ***")

            # Early stopping check
            if early_stopper is not None:
                if early_stopper(auc):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best AUC was {best_auc:.4f} at epoch {best_epoch}")
                    break
        else:
            history['stage2_aucs'].append(None)
            print(f"Epoch {epoch}/{args.stage2_epochs}: Loss = {loss:.6f}, "
                  f"LR = {optimizer_s2.param_groups[0]['lr']:.2e}")

        # Cosine scheduler step
        if args.scheduler == 'cosine' and scheduler_s2 is not None:
            scheduler_s2.step()

    # Final checkpoint
    final_metrics = evaluate(model, test_loader, device)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_s2.state_dict(),
        'metrics': final_metrics,
        'history': history,
        'losses': history['stage2_losses']
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best AUC-ROC: {best_auc:.4f} at epoch {best_epoch}")
    print(f"Final AUC-ROC: {final_metrics['auc_roc']:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
