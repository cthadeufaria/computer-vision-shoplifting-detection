"""
Diagnostic utilities for analyzing Shopformer training.

Provides tools to:
- Analyze GCAE token discriminability (normal vs anomaly separation)
- Validate token quality before Stage 2 training
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader


def analyze_token_discriminability(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze GCAE token discriminability between normal and anomaly samples.

    This diagnostic should be run after Stage 1 (GCAE training) to verify
    that the tokenizer produces tokens with good separation between classes.

    Metrics computed:
    - token_variance: Average variance across token dimensions (information content)
    - inter_class_distance: Mean distance between normal and anomaly tokens
    - intra_class_variance_normal: Variance within normal class
    - intra_class_variance_anomaly: Variance within anomaly class
    - discriminability_ratio: inter_class_distance / sqrt(intra_class_variance)

    Args:
        model: Shopformer model with trained GCAE
        train_loader: DataLoader for training data (normal only)
        test_loader: DataLoader for test data (normal + anomaly)
        device: Torch device
        verbose: Print results

    Returns:
        Dictionary of diagnostic metrics
    """
    model.eval()

    # Collect tokens from training data (all normal)
    train_tokens = []
    with torch.no_grad():
        for poses, _ in train_loader:
            poses = poses.to(device)
            _, tokens = model.gcae(poses)
            train_tokens.append(tokens.cpu().numpy())

    train_tokens = np.concatenate(train_tokens, axis=0)

    # Collect tokens from test data (separated by label)
    test_normal_tokens = []
    test_anomaly_tokens = []

    with torch.no_grad():
        for poses, labels in test_loader:
            poses = poses.to(device)
            _, tokens = model.gcae(poses)
            tokens_np = tokens.cpu().numpy()
            labels_np = labels.numpy()

            # Separate by label
            normal_mask = labels_np == 0
            anomaly_mask = labels_np == 1

            if normal_mask.any():
                test_normal_tokens.append(tokens_np[normal_mask])
            if anomaly_mask.any():
                test_anomaly_tokens.append(tokens_np[anomaly_mask])

    test_normal_tokens = np.concatenate(test_normal_tokens, axis=0) if test_normal_tokens else np.array([])
    test_anomaly_tokens = np.concatenate(test_anomaly_tokens, axis=0) if test_anomaly_tokens else np.array([])

    # Flatten tokens for analysis: (N, num_tokens, dim) -> (N, num_tokens * dim)
    train_flat = train_tokens.reshape(train_tokens.shape[0], -1)
    normal_flat = test_normal_tokens.reshape(test_normal_tokens.shape[0], -1) if len(test_normal_tokens) > 0 else np.array([])
    anomaly_flat = test_anomaly_tokens.reshape(test_anomaly_tokens.shape[0], -1) if len(test_anomaly_tokens) > 0 else np.array([])

    # Compute metrics
    results = {}

    # 1. Token variance (information content)
    results['token_variance'] = float(np.mean(np.var(train_flat, axis=0)))
    results['token_variance_std'] = float(np.std(np.var(train_flat, axis=0)))

    # 2. Intra-class variance
    results['intra_class_variance_train'] = float(np.mean(np.var(train_flat, axis=0)))

    if len(normal_flat) > 0:
        results['intra_class_variance_normal'] = float(np.mean(np.var(normal_flat, axis=0)))
    else:
        results['intra_class_variance_normal'] = 0.0

    if len(anomaly_flat) > 0:
        results['intra_class_variance_anomaly'] = float(np.mean(np.var(anomaly_flat, axis=0)))
    else:
        results['intra_class_variance_anomaly'] = 0.0

    # 3. Inter-class distance (between normal and anomaly centroids)
    if len(normal_flat) > 0 and len(anomaly_flat) > 0:
        normal_centroid = np.mean(normal_flat, axis=0)
        anomaly_centroid = np.mean(anomaly_flat, axis=0)

        inter_class_dist = np.linalg.norm(normal_centroid - anomaly_centroid)
        results['inter_class_distance'] = float(inter_class_dist)

        # 4. Discriminability ratio (Fisher-like criterion)
        pooled_variance = (results['intra_class_variance_normal'] + results['intra_class_variance_anomaly']) / 2
        if pooled_variance > 1e-8:
            results['discriminability_ratio'] = float(inter_class_dist / np.sqrt(pooled_variance))
        else:
            results['discriminability_ratio'] = float('inf')
    else:
        results['inter_class_distance'] = 0.0
        results['discriminability_ratio'] = 0.0

    # 5. Token norm statistics
    results['token_norm_mean'] = float(np.mean(np.linalg.norm(train_flat, axis=1)))
    results['token_norm_std'] = float(np.std(np.linalg.norm(train_flat, axis=1)))

    # 6. Sample counts
    results['n_train'] = len(train_flat)
    results['n_test_normal'] = len(normal_flat)
    results['n_test_anomaly'] = len(anomaly_flat)

    if verbose:
        print("\n" + "=" * 60)
        print("GCAE Token Discriminability Analysis")
        print("=" * 60)
        print(f"\nSample counts:")
        print(f"  Train (normal):     {results['n_train']}")
        print(f"  Test normal:        {results['n_test_normal']}")
        print(f"  Test anomaly:       {results['n_test_anomaly']}")
        print(f"\nToken statistics:")
        print(f"  Token variance:     {results['token_variance']:.6f} (+/- {results['token_variance_std']:.6f})")
        print(f"  Token norm (mean):  {results['token_norm_mean']:.4f} (+/- {results['token_norm_std']:.4f})")
        print(f"\nClass separation:")
        print(f"  Intra-class var (normal):  {results['intra_class_variance_normal']:.6f}")
        print(f"  Intra-class var (anomaly): {results['intra_class_variance_anomaly']:.6f}")
        print(f"  Inter-class distance:      {results['inter_class_distance']:.6f}")
        print(f"  Discriminability ratio:    {results['discriminability_ratio']:.4f}")
        print("\nInterpretation:")
        if results['discriminability_ratio'] < 0.5:
            print("  WARNING: Low discriminability - GCAE may not be learning useful features")
            print("  Consider: longer training, different architecture, or data augmentation")
        elif results['discriminability_ratio'] < 1.0:
            print("  MODERATE: Some class separation, but transformer may struggle")
        else:
            print("  GOOD: Clear class separation in token space")
        print("=" * 60)

    return results


def analyze_reconstruction_error_distribution(
    model,
    test_loader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Analyze the distribution of reconstruction errors for normal vs anomaly samples.

    This helps diagnose whether the transformer is learning useful representations
    for anomaly detection.

    Args:
        model: Shopformer model (both GCAE and transformer trained)
        test_loader: DataLoader for test data
        device: Torch device
        verbose: Print results

    Returns:
        Dictionary of error distribution metrics
    """
    model.eval()

    normal_errors = []
    anomaly_errors = []

    with torch.no_grad():
        for poses, labels in test_loader:
            poses = poses.to(device)
            scores = model.compute_anomaly_score(poses)
            scores_np = scores.cpu().numpy()
            labels_np = labels.numpy()

            normal_mask = labels_np == 0
            anomaly_mask = labels_np == 1

            if normal_mask.any():
                normal_errors.extend(scores_np[normal_mask])
            if anomaly_mask.any():
                anomaly_errors.extend(scores_np[anomaly_mask])

    normal_errors = np.array(normal_errors)
    anomaly_errors = np.array(anomaly_errors)

    results = {}

    # Error statistics
    results['normal_error_mean'] = float(np.mean(normal_errors)) if len(normal_errors) > 0 else 0.0
    results['normal_error_std'] = float(np.std(normal_errors)) if len(normal_errors) > 0 else 0.0
    results['anomaly_error_mean'] = float(np.mean(anomaly_errors)) if len(anomaly_errors) > 0 else 0.0
    results['anomaly_error_std'] = float(np.std(anomaly_errors)) if len(anomaly_errors) > 0 else 0.0

    # Separation metrics
    if len(normal_errors) > 0 and len(anomaly_errors) > 0:
        # D-prime (signal detection theory)
        pooled_std = np.sqrt((results['normal_error_std']**2 + results['anomaly_error_std']**2) / 2)
        if pooled_std > 1e-8:
            results['d_prime'] = float((results['anomaly_error_mean'] - results['normal_error_mean']) / pooled_std)
        else:
            results['d_prime'] = 0.0

        # Overlap percentage (approximate)
        threshold = (results['normal_error_mean'] + results['anomaly_error_mean']) / 2
        normal_above = np.mean(normal_errors > threshold)
        anomaly_below = np.mean(anomaly_errors < threshold)
        results['overlap_pct'] = float((normal_above + anomaly_below) / 2 * 100)
    else:
        results['d_prime'] = 0.0
        results['overlap_pct'] = 100.0

    if verbose:
        print("\n" + "=" * 60)
        print("Reconstruction Error Distribution Analysis")
        print("=" * 60)
        print(f"\nNormal samples (n={len(normal_errors)}):")
        print(f"  Mean error: {results['normal_error_mean']:.6f}")
        print(f"  Std error:  {results['normal_error_std']:.6f}")
        print(f"\nAnomaly samples (n={len(anomaly_errors)}):")
        print(f"  Mean error: {results['anomaly_error_mean']:.6f}")
        print(f"  Std error:  {results['anomaly_error_std']:.6f}")
        print(f"\nSeparation metrics:")
        print(f"  D-prime:    {results['d_prime']:.4f}")
        print(f"  Overlap %:  {results['overlap_pct']:.1f}%")
        print("\nInterpretation:")
        if results['d_prime'] < 0.5:
            print("  WARNING: Poor separation - model cannot distinguish classes")
        elif results['d_prime'] < 1.0:
            print("  MODERATE: Some separation, but significant overlap")
        elif results['d_prime'] < 2.0:
            print("  GOOD: Clear separation between classes")
        else:
            print("  EXCELLENT: Strong separation")
        print("=" * 60)

    return results


def run_full_diagnostics(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Run all diagnostic analyses.

    Args:
        model: Shopformer model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        device: Torch device

    Returns:
        Dictionary containing all diagnostic results
    """
    results = {}

    print("\n" + "#" * 60)
    print("# SHOPFORMER DIAGNOSTIC REPORT")
    print("#" * 60)

    results['token_discriminability'] = analyze_token_discriminability(
        model, train_loader, test_loader, device, verbose=True
    )

    results['error_distribution'] = analyze_reconstruction_error_distribution(
        model, test_loader, device, verbose=True
    )

    print("\n" + "#" * 60)
    print("# END DIAGNOSTIC REPORT")
    print("#" * 60 + "\n")

    return results
