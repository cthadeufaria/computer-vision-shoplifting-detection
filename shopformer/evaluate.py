"""
Shopformer Evaluation Script

Evaluates trained model and saves comprehensive results.
Usage:
    python evaluate.py --checkpoint ./checkpoints/best_model.pt --data_dir ./data/PoseLift
    python evaluate.py --checkpoint ./checkpoints/best_model.pt --use_synthetic
"""

import argparse
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from models import Shopformer
from data.poselift_dataset import PoseLiftDataset, SyntheticPoseLiftDataset
from utils.metrics import compute_metrics, compute_auc_roc


def convert_to_serializable(obj):
    """Convert numpy types to JSON serializable Python types."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, device, config, checkpoint


def evaluate_model(model, dataset, device):
    """Run evaluation on dataset and return scores and labels."""
    all_scores = []
    all_labels = []

    print(f"Evaluating on {len(dataset)} samples...")

    with torch.no_grad():
        for i in range(len(dataset)):
            poses, label = dataset[i]
            poses = poses.unsqueeze(0).to(device)

            output = model(poses)
            score = output['normality_score'].cpu().numpy()[0]

            all_scores.append(score)
            all_labels.append(label.item())

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset)}")

    return np.array(all_scores), np.array(all_labels)


def load_training_history(checkpoint_dir: str):
    """Load training history from all checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    history = {}

    # Load GCAE checkpoint
    gcae_path = checkpoint_dir / 'gcae_checkpoint.pt'
    if gcae_path.exists():
        gcae = torch.load(gcae_path, map_location='cpu', weights_only=False)
        history['stage1_gcae'] = {
            'epochs_completed': gcae.get('epoch', 'N/A'),
            'final_loss': gcae['losses'][-1] if 'losses' in gcae else None,
            'loss_history': gcae.get('losses', [])
        }

    # Load best model checkpoint
    best_path = checkpoint_dir / 'best_model.pt'
    if best_path.exists():
        best = torch.load(best_path, map_location='cpu', weights_only=False)
        history['best_model'] = {
            'epoch': best.get('epoch', 'N/A'),
            'best_auc': best.get('best_auc', None),
            'metrics': best.get('metrics', {})
        }

    # Load final model checkpoint
    final_path = checkpoint_dir / 'final_model.pt'
    if final_path.exists():
        final = torch.load(final_path, map_location='cpu', weights_only=False)
        history['stage2_transformer'] = {
            'epochs_completed': final.get('epoch', 'N/A'),
            'loss_history': final.get('losses', [])
        }

    return history


def main():
    parser = argparse.ArgumentParser(description="Shopformer Evaluation")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to PoseLift test data')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic test data')
    parser.add_argument('--output', type=str, default='training_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--include_scores', action='store_true',
                        help='Include per-sample scores in output')

    args = parser.parse_args()

    print("=" * 60)
    print("Shopformer Model Evaluation")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, device, config, checkpoint = load_model(args.checkpoint, args.config, args.device)
    print(f"Model loaded on {device}")

    # Load test data
    if args.use_synthetic:
        print("\nUsing synthetic test data...")
        dataset = SyntheticPoseLiftDataset(
            num_samples=500,
            seq_len=config.get('seq_len', 12),
            anomaly_ratio=0.3
        )
    elif args.data_dir:
        print(f"\nLoading test data from {args.data_dir}...")
        dataset = PoseLiftDataset(
            data_dir=args.data_dir,
            split='test',
            seq_len=config.get('seq_len', 12)
        )
    else:
        print("Error: Provide --data_dir or --use_synthetic")
        return

    print(f"Test samples: {len(dataset)}")

    # Evaluate
    print("\nRunning evaluation...")
    scores, labels = evaluate_model(model, dataset, device)

    # Compute metrics
    metrics = compute_metrics(labels, scores)
    auc, fpr, tpr = compute_auc_roc(labels, scores)

    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  AUC-ROC:    {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)")
    print(f"  AUC-PR:     {metrics['auc_pr']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  Threshold:  {metrics['threshold']:.4f}")
    print("=" * 60)

    # Load training history
    checkpoint_dir = Path(args.checkpoint).parent
    training_history = load_training_history(checkpoint_dir)

    # Compile results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Shopformer',
        'description': 'Transformer-based shoplifting detection using pose sequences',
        'checkpoint_used': str(args.checkpoint),
        'test_data': 'synthetic' if args.use_synthetic else str(args.data_dir),
        'num_test_samples': len(dataset),

        'training_config': config,
        'training_history': training_history,

        'test_metrics': {
            'auc_roc': metrics['auc_roc'],
            'auc_pr': metrics['auc_pr'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'optimal_threshold': metrics['threshold']
        },

        'score_statistics': {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores))
        }
    }

    if args.include_scores:
        results['per_sample'] = {
            'scores': scores.tolist(),
            'labels': labels.tolist()
        }

    # Convert and save
    results = convert_to_serializable(results)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
