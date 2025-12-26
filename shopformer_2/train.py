#!/usr/bin/env python3
"""
Shopformer_2 Training Script.

Two-stage training pipeline:
- Stage 1: Train GCAE tokenizer (pose reconstruction)
- Stage 2: Train Transformer with frozen GCAE (token reconstruction)

Optimized for Apple Silicon with MPS backend.

Usage:
    python train.py --config configs/paper_config.yaml
    python train.py --config configs/paper_config.yaml --stage 2 --checkpoint checkpoints/stage1_best.pt
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.shopformer import Shopformer, build_shopformer
from data.poselift_dataset import PoseLiftDataset, PoseLiftDataModule
from utils.config import load_config, save_config
from utils.device import get_device, setup_mps_environment, clear_mps_cache
from utils.metrics import compute_metrics, compute_auc_roc


def get_optimizer(params, config: Dict, stage_cfg: Dict):
    """Create optimizer based on config (Adam or AdamW)."""
    optimizer_type = config.get('training', {}).get('optimizer', 'adamw').lower()
    lr = stage_cfg['learning_rate']
    weight_decay = stage_cfg.get('weight_decay', 0)

    if optimizer_type == 'adam':
        # Paper uses Adam without weight decay
        return torch.optim.Adam(params, lr=lr)
    else:
        # AdamW with weight decay
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, config: Dict, num_batches: int, stage: int):
    """Create learning rate scheduler."""
    scheduler_cfg = config.get('training', {}).get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', 'cosine_warmup').lower()

    # Paper uses constant LR (no scheduler)
    if scheduler_type == 'none' or scheduler_type == 'constant':
        return None

    if stage == 1:
        epochs = config['training']['stage1']['epochs']
    else:
        epochs = config['training']['stage2']['epochs']

    warmup_epochs = scheduler_cfg.get('warmup_epochs', 5)
    min_lr = scheduler_cfg.get('min_lr', 1e-6)

    # Account for gradient accumulation - scheduler steps once per optimizer step
    grad_accum = config['training'].get('gradient_accumulation', 1)
    steps_per_epoch = num_batches // grad_accum

    if scheduler_type == 'cosine_warmup':
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(min_lr / optimizer.defaults['lr'],
                      0.5 * (1 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == 'step':
        # Step LR: halve learning rate every N epochs
        step_size = scheduler_cfg.get('step_size', 10)
        gamma = scheduler_cfg.get('gamma', 0.5)
        # StepLR steps per epoch, so use epoch-based stepping
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size * steps_per_epoch,  # Convert epochs to steps
            gamma=gamma
        )

    elif scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=scheduler_cfg.get('factor', 0.5),
            patience=scheduler_cfg.get('patience', 5),
            min_lr=min_lr
        )

    return None


def train_stage1(
    model: Shopformer,
    train_loader: DataLoader,
    config: Dict,
    device: torch.device,
    checkpoint_dir: Path,
    writer: Optional[SummaryWriter] = None
) -> Shopformer:
    """
    Stage 1: Train GCAE tokenizer.

    Trains the GCAE to reconstruct input poses.
    """
    print("\n" + "=" * 60)
    print("Stage 1: Training GCAE Tokenizer")
    print("=" * 60)

    stage_cfg = config['training']['stage1']
    epochs = stage_cfg['epochs']
    grad_accum = config['training']['gradient_accumulation']
    grad_clip = config['training']['grad_clip']
    log_interval = config.get('logging', {}).get('log_interval', 10)

    # Use optimizer from config (Adam for paper alignment, AdamW otherwise)
    optimizer = get_optimizer(model.gcae.parameters(), config, stage_cfg)

    scheduler = get_scheduler(optimizer, config, len(train_loader), stage=1)

    best_loss = float('inf')
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, (poses, _) in enumerate(train_loader):
            poses = poses.to(device)

            # Forward through GCAE
            reconstructed, tokens = model.gcae(poses)
            loss = F.mse_loss(reconstructed, poses)

            # Gradient accumulation
            loss_scaled = loss / grad_accum
            loss_scaled.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.gcae.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None and not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()

                global_step += 1

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.6f} - LR: {current_lr:.2e}")

        avg_loss = epoch_loss / num_batches

        if writer:
            writer.add_scalar('Stage1/Loss', avg_loss, epoch)
            writer.add_scalar('Stage1/LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"\nEpoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'gcae_state_dict': model.gcae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / 'stage1_best.pt')
            print(f"  Saved best checkpoint (loss: {avg_loss:.6f})")

        # Periodic checkpoints
        save_freq = config.get('checkpoint', {}).get('save_frequency', 10)
        if (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'gcae_state_dict': model.gcae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / f'stage1_epoch{epoch + 1}.pt')

        # Clear MPS cache periodically
        if device.type == 'mps' and (epoch + 1) % 5 == 0:
            clear_mps_cache()

    # Save final checkpoint
    checkpoint = {
        'epoch': epochs - 1,
        'gcae_state_dict': model.gcae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config
    }
    torch.save(checkpoint, checkpoint_dir / 'stage1_final.pt')

    print(f"\nStage 1 complete. Best loss: {best_loss:.6f}")
    return model


def evaluate(
    model: Shopformer,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict]:
    """
    Evaluate model on test set.

    Returns:
        Tuple of (AUC-ROC, full metrics dict)
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for poses, labels in test_loader:
            poses = poses.to(device)
            scores = model.compute_anomaly_score(poses)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_labels, all_scores)
    return metrics['auc_roc'], metrics


def train_stage2(
    model: Shopformer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict,
    device: torch.device,
    checkpoint_dir: Path,
    writer: Optional[SummaryWriter] = None
) -> Tuple[Shopformer, float]:
    """
    Stage 2: Train Transformer with frozen GCAE.

    Trains the transformer to reconstruct GCAE tokens.
    """
    print("\n" + "=" * 60)
    print("Stage 2: Training Transformer (GCAE frozen)")
    print("=" * 60)

    # Freeze GCAE
    model.freeze_gcae()
    print(f"GCAE frozen. Trainable params: {model.get_num_parameters()['transformer']:,}")

    stage_cfg = config['training']['stage2']
    epochs = stage_cfg['epochs']
    grad_accum = config['training']['gradient_accumulation']
    grad_clip = config['training']['grad_clip']
    log_interval = config.get('logging', {}).get('log_interval', 10)

    early_stopping_cfg = config['training'].get('early_stopping', {})
    early_stopping_enabled = early_stopping_cfg.get('enabled', True)
    patience = early_stopping_cfg.get('patience', 20)
    min_delta = early_stopping_cfg.get('min_delta', 0.001)

    # Use optimizer from config (Adam for paper alignment, AdamW otherwise)
    optimizer = get_optimizer(model.transformer.parameters(), config, stage_cfg)

    scheduler = get_scheduler(optimizer, config, len(train_loader), stage=2)

    best_auc = 0.0
    patience_counter = 0
    global_step = 0

    for epoch in range(epochs):
        # Training
        model.transformer.train()
        model.gcae.eval()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, (poses, _) in enumerate(train_loader):
            poses = poses.to(device)

            # Get tokens from frozen GCAE
            with torch.no_grad():
                _, tokens = model.gcae(poses)
                tokens = tokens.detach()

            # Reconstruct with transformer
            reconstructed_tokens = model.transformer(tokens)
            loss = F.mse_loss(reconstructed_tokens, tokens)

            # Gradient accumulation
            loss_scaled = loss / grad_accum
            loss_scaled.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.transformer.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None and not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()

                global_step += 1

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.6f} - LR: {current_lr:.2e}")

        avg_loss = epoch_loss / num_batches

        # Evaluation
        auc, metrics = evaluate(model, test_loader, device)

        if writer:
            writer.add_scalar('Stage2/Loss', avg_loss, epoch)
            writer.add_scalar('Stage2/AUC_ROC', auc, epoch)
            writer.add_scalar('Stage2/AUC_PR', metrics['auc_pr'], epoch)
            writer.add_scalar('Stage2/LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  AUC-ROC: {auc:.4f} | AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

        # Update scheduler if reduce_on_plateau
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(auc)

        # Save best checkpoint
        if auc > best_auc + min_delta:
            best_auc = auc
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gcae_state_dict': model.gcae.state_dict(),
                'transformer_state_dict': model.transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc_roc': auc,
                'metrics': metrics,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / 'stage2_best.pt')
            print(f"  New best! Saved checkpoint (AUC: {auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        # Early stopping
        if early_stopping_enabled and patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

        # Periodic checkpoints
        save_freq = config.get('checkpoint', {}).get('save_frequency', 10)
        if (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gcae_state_dict': model.gcae.state_dict(),
                'transformer_state_dict': model.transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc_roc': auc,
                'metrics': metrics,
                'config': config
            }
            torch.save(checkpoint, checkpoint_dir / f'stage2_epoch{epoch + 1}.pt')

        # Clear MPS cache periodically
        if device.type == 'mps' and (epoch + 1) % 5 == 0:
            clear_mps_cache()

    # Save final checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'gcae_state_dict': model.gcae.state_dict(),
        'transformer_state_dict': model.transformer.state_dict(),
        'auc_roc': auc,
        'metrics': metrics,
        'config': config
    }
    torch.save(checkpoint, checkpoint_dir / 'stage2_final.pt')

    print(f"\nStage 2 complete. Best AUC-ROC: {best_auc:.4f}")
    return model, best_auc


def main():
    parser = argparse.ArgumentParser(description='Train Shopformer_2')
    parser.add_argument('--config', type=str, default='configs/paper_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=None,
                        help='Train only specific stage (default: both)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    args = parser.parse_args()

    # Setup MPS environment
    setup_mps_environment()

    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    config = load_config(str(config_path))

    # Get device
    device = get_device(config.get('training', {}).get('device', 'auto'))

    # Setup output directory
    if args.output_dir:
        checkpoint_dir = Path(args.output_dir)
    else:
        checkpoint_dir = script_dir / config.get('checkpoint', {}).get('save_dir', 'checkpoints')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = checkpoint_dir / timestamp

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    save_config(config, str(checkpoint_dir / 'config.yaml'))

    # Setup tensorboard
    use_tensorboard = config.get('logging', {}).get('use_tensorboard', True)
    writer = None
    if use_tensorboard:
        tensorboard_dir = checkpoint_dir / 'runs'
        writer = SummaryWriter(tensorboard_dir)

    # Print configuration
    print("\n" + "=" * 60)
    print("Shopformer_2 Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Sequence length: {config['model']['seq_len']}")
    print(f"Transformer: {config['model']['transformer']['num_heads']} heads, "
          f"{config['model']['transformer']['num_layers']} layers")
    print(f"Batch size: {config['training']['batch_size']} "
          f"(effective: {config['training']['batch_size'] * config['training']['gradient_accumulation']})")

    # Load data
    print("\nLoading data...")
    data_module = PoseLiftDataModule(config, num_workers=0)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    stats = data_module.get_stats()
    print(f"Train samples: {stats['train_samples']}")
    print(f"Test samples: {stats['test_samples']} "
          f"(Normal: {stats['test_normal']}, Anomaly: {stats['test_anomaly']})")

    # Create model
    print("\nCreating model...")
    model = build_shopformer(config).to(device)

    param_counts = model.get_num_parameters(trainable_only=False)
    print(f"Model parameters:")
    print(f"  GCAE: {param_counts['gcae']:,}")
    print(f"  Transformer: {param_counts['transformer']:,}")
    print(f"  Total: {param_counts['total']:,}")

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'gcae_state_dict' in checkpoint:
            model.gcae.load_state_dict(checkpoint['gcae_state_dict'])
            if 'transformer_state_dict' in checkpoint:
                model.transformer.load_state_dict(checkpoint['transformer_state_dict'])

    # Training
    if args.stage is None or args.stage == 1:
        model = train_stage1(
            model, train_loader, config, device, checkpoint_dir, writer
        )

    if args.stage is None or args.stage == 2:
        # If starting stage 2 without stage 1, load best GCAE
        if args.stage == 2 and args.checkpoint is None:
            stage1_checkpoint = checkpoint_dir / 'stage1_best.pt'
            if stage1_checkpoint.exists():
                print(f"\nLoading Stage 1 checkpoint: {stage1_checkpoint}")
                checkpoint = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
                model.gcae.load_state_dict(checkpoint['gcae_state_dict'])

        model, best_auc = train_stage2(
            model, train_loader, test_loader, config, device, checkpoint_dir, writer
        )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Load best checkpoint
    best_checkpoint = checkpoint_dir / 'stage2_best.pt'
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.freeze_gcae()
    auc, metrics = evaluate(model, test_loader, device)

    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")

    if writer:
        writer.add_hparams(
            {
                'transformer_heads': config['model']['transformer']['num_heads'],
                'transformer_layers': config['model']['transformer']['num_layers'],
                'ffn_dim': config['model']['transformer']['dim_feedforward'],
                'batch_size': config['training']['batch_size'],
                'stage2_lr': config['training']['stage2']['learning_rate'],
            },
            {
                'hparam/auc_roc': metrics['auc_roc'],
                'hparam/auc_pr': metrics['auc_pr'],
                'hparam/f1': metrics['f1'],
            }
        )
        writer.close()

    print(f"\nTraining complete! Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
