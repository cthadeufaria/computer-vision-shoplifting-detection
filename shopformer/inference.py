"""
Shopformer Inference Script

Load a trained model and run inference on pose sequences.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from models import Shopformer
from data.poselift_dataset import PoseLiftDataset, SyntheticPoseLiftDataset
from utils.metrics import compute_metrics, compute_auc_roc


def load_model(checkpoint_path: str, config_path: str = None, device: str = 'auto'):
    """Load trained Shopformer model."""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Load config
    if config_path is None:
        config_path = Path(checkpoint_path).parent / 'config.json'

    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'seq_len': 12,
            'num_keypoints': 17,
            'num_tokens': 2,
            'hidden_channels': 64,
            'latent_channels': 8,
            'transformer_heads': 2,
            'transformer_layers': 2,
            'dropout': 0.1
        }

    # Create model
    model = Shopformer(
        in_channels=2,
        hidden_channels=config.get('hidden_channels', 64),
        latent_channels=config.get('latent_channels', 8),
        num_keypoints=config.get('num_keypoints', 17),
        seq_len=config.get('seq_len', 12),
        num_tokens=config.get('num_tokens', 2),
        transformer_heads=config.get('transformer_heads', 2),
        transformer_layers=config.get('transformer_layers', 2),
        dropout=config.get('dropout', 0.1)
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device


def predict_poses(model, poses: np.ndarray, device) -> dict:
    """
    Run inference on pose sequences.

    Args:
        model: Trained Shopformer
        poses: Pose sequences of shape (N, 2, seq_len, num_keypoints)
        device: Torch device

    Returns:
        Dictionary with scores and predictions
    """
    if isinstance(poses, np.ndarray):
        poses = torch.FloatTensor(poses)

    poses = poses.to(device)

    with torch.no_grad():
        output = model(poses)

    scores = output['normality_score'].cpu().numpy()

    return {
        'scores': scores,
        'mean_score': float(np.mean(scores)),
        'max_score': float(np.max(scores)),
        'min_score': float(np.min(scores))
    }


def main():
    parser = argparse.ArgumentParser(description="Shopformer Inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to PoseLift test data')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic test data')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')

    args = parser.parse_args()

    print("Loading model...")
    model, device = load_model(args.checkpoint, args.config, args.device)
    print(f"Model loaded on {device}")

    # Load test data
    if args.use_synthetic:
        print("Using synthetic test data...")
        dataset = SyntheticPoseLiftDataset(
            num_samples=200,
            seq_len=12,
            anomaly_ratio=0.3
        )
    elif args.data_dir:
        print(f"Loading test data from {args.data_dir}...")
        dataset = PoseLiftDataset(
            data_dir=args.data_dir,
            split='test',
            seq_len=12
        )
    else:
        print("Error: Provide --data_dir or --use_synthetic")
        return

    print(f"Test samples: {len(dataset)}")

    # Run inference
    all_scores = []
    all_labels = []

    print("Running inference...")
    for i in range(len(dataset)):
        poses, label = dataset[i]
        poses = poses.unsqueeze(0)  # Add batch dimension

        result = predict_poses(model, poses, device)
        all_scores.append(result['scores'][0])
        all_labels.append(label.item())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_scores, threshold=args.threshold)
    auc, fpr, tpr = compute_auc_roc(all_labels, all_scores)

    print("\n=== Results ===")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")

    if args.output:
        results = {
            'metrics': metrics,
            'scores': all_scores.tolist(),
            'labels': all_labels.tolist()
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
