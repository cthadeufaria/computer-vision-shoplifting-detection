#!/usr/bin/env python3
"""
Shopformer_2 Evaluation Script.

Evaluates a trained model on the test set and generates detailed metrics.

Usage:
    python evaluate.py --checkpoint checkpoints/stage2_best.pt
    python evaluate.py --checkpoint checkpoints/stage2_best.pt --output results/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.shopformer import Shopformer, build_shopformer
from data.poselift_dataset import PoseLiftDataset, PoseLiftDataModule
from utils.config import load_config
from utils.device import get_device, setup_mps_environment
from utils.metrics import (
    compute_metrics,
    compute_auc_roc,
    compute_auc_pr,
    find_optimal_threshold,
    compute_video_level_metrics,
    print_metrics
)


def evaluate_frame_level(
    model: Shopformer,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate model at frame/sequence level.

    Returns:
        Tuple of (metrics dict, scores array, labels array)
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
    return metrics, all_scores, all_labels


def evaluate_video_level(
    model: Shopformer,
    test_dataset: PoseLiftDataset,
    device: torch.device,
    batch_size: int = 32,
    aggregation: str = 'max'
) -> Dict:
    """
    Evaluate model at video level.

    Aggregates frame scores to video scores.

    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Compute device
        batch_size: Batch size for inference
        aggregation: Score aggregation method ('max', 'mean', 'percentile_95')

    Returns:
        Video-level metrics
    """
    model.eval()

    # Collect scores per video
    video_scores: Dict[str, List[float]] = defaultdict(list)
    video_labels: Dict[str, int] = {}

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    with torch.no_grad():
        for batch_idx, (poses, labels) in enumerate(loader):
            poses = poses.to(device)
            scores = model.compute_anomaly_score(poses).cpu().numpy()

            # Get video info for each sample in batch
            start_idx = batch_idx * batch_size
            for i, score in enumerate(scores):
                sample_idx = start_idx + i
                if sample_idx >= len(test_dataset):
                    break

                info = test_dataset.get_video_info(sample_idx)
                video_id = info['video_id']

                video_scores[video_id].append(float(score))
                video_labels[video_id] = info['label']

    return compute_video_level_metrics(video_scores, video_labels, aggregation)


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str = 'ROC Curve'
):
    """Plot and save ROC curve."""
    auc, fpr, tpr = compute_auc_roc(labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_precision_recall_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str = 'Precision-Recall Curve'
):
    """Plot and save Precision-Recall curve."""
    auc_pr, precision, recall = compute_auc_pr(labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'AUC-PR = {auc_pr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_score_distribution(
    labels: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    threshold: float = None
):
    """Plot score distribution for normal vs anomaly."""
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    plt.figure(figsize=(10, 6))

    plt.hist(normal_scores, bins=50, alpha=0.6, label=f'Normal (n={len(normal_scores)})',
             density=True, color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label=f'Anomaly (n={len(anomaly_scores)})',
             density=True, color='red')

    if threshold is not None:
        plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')

    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Shopformer_2')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config (default: from checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--video-level', action='store_true',
                        help='Also compute video-level metrics')
    parser.add_argument('--plot', action='store_true',
                        help='Generate evaluation plots')
    args = parser.parse_args()

    # Setup
    setup_mps_environment()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Load config
    if args.config:
        config = load_config(args.config)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("No config found. Please provide --config argument.")

    # Get device
    device = get_device(config.get('training', {}).get('device', 'auto'))

    # Create model
    print("Creating model...")
    model = build_shopformer(config).to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'gcae_state_dict' in checkpoint:
        model.gcae.load_state_dict(checkpoint['gcae_state_dict'])
        if 'transformer_state_dict' in checkpoint:
            model.transformer.load_state_dict(checkpoint['transformer_state_dict'])

    model.freeze_gcae()
    model.eval()

    # Load data
    print("Loading test data...")
    data_module = PoseLiftDataModule(config, num_workers=0)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    print(f"Test samples: {len(data_module.test_dataset)}")

    # Frame-level evaluation
    print("\n" + "=" * 60)
    print("Frame-Level Evaluation")
    print("=" * 60)

    metrics, scores, labels = evaluate_frame_level(model, test_loader, device)
    print_metrics(metrics, prefix="  ")

    # Video-level evaluation
    video_metrics = None
    if args.video_level:
        print("\n" + "=" * 60)
        print("Video-Level Evaluation")
        print("=" * 60)

        for agg in ['max', 'mean', 'percentile_95']:
            print(f"\nAggregation: {agg}")
            video_metrics = evaluate_video_level(
                model, data_module.test_dataset, device, aggregation=agg
            )
            print_metrics(video_metrics, prefix="  ")

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        checkpoint_path = Path(args.checkpoint)
        output_dir = checkpoint_path.parent / 'evaluation'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    results = {
        'checkpoint': str(args.checkpoint),
        'frame_level': {k: float(v) for k, v in metrics.items()},
    }
    if video_metrics:
        results['video_level'] = {k: float(v) for k, v in video_metrics.items()}

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'metrics.json'}")

    # Generate plots
    if args.plot:
        print("\nGenerating plots...")

        plot_roc_curve(
            labels, scores,
            output_dir / 'roc_curve.png',
            title=f'ROC Curve (AUC = {metrics["auc_roc"]:.4f})'
        )
        print(f"  Saved: {output_dir / 'roc_curve.png'}")

        plot_precision_recall_curve(
            labels, scores,
            output_dir / 'pr_curve.png',
            title=f'Precision-Recall Curve (AUC = {metrics["auc_pr"]:.4f})'
        )
        print(f"  Saved: {output_dir / 'pr_curve.png'}")

        plot_score_distribution(
            labels, scores,
            output_dir / 'score_distribution.png',
            threshold=metrics['threshold']
        )
        print(f"  Saved: {output_dir / 'score_distribution.png'}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"F1:      {metrics['f1']:.4f}")

    if 'auc_roc' in checkpoint:
        original_auc = checkpoint['auc_roc']
        print(f"\nCheckpoint AUC: {original_auc:.4f}")
        diff = metrics['auc_roc'] - original_auc
        print(f"Difference: {diff:+.4f}")


if __name__ == '__main__':
    main()
