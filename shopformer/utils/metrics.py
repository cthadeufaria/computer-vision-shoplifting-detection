"""
Metrics for evaluating Shopformer_2 performance.

Includes frame-level and video-level evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)


def compute_auc_roc(
    labels: np.ndarray,
    scores: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Area Under ROC Curve.

    Args:
        labels: Ground truth labels (0 or 1)
        scores: Anomaly scores (higher = more anomalous)

    Returns:
        Tuple of (AUC score, FPR array, TPR array)
    """
    try:
        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        return auc, fpr, tpr
    except ValueError:
        # Handle case with only one class
        return 0.5, np.array([0, 1]), np.array([0, 1])


def compute_auc_pr(
    labels: np.ndarray,
    scores: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Area Under Precision-Recall Curve.

    Args:
        labels: Ground truth labels (0 or 1)
        scores: Anomaly scores (higher = more anomalous)

    Returns:
        Tuple of (AUC-PR score, precision array, recall array)
    """
    try:
        auc_pr = average_precision_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        return auc_pr, precision, recall
    except ValueError:
        return 0.0, np.array([0, 1]), np.array([1, 0])


def find_optimal_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    method: str = 'youden'
) -> float:
    """
    Find optimal classification threshold.

    Args:
        labels: Ground truth labels
        scores: Anomaly scores
        method: 'youden' (J-statistic) or 'f1' (maximize F1)

    Returns:
        Optimal threshold value
    """
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]
    elif method == 'f1':
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        # Avoid division by zero
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0
        )
        optimal_idx = np.argmax(f1_scores[:-1])  # Last element is always 0
        return thresholds[optimal_idx]
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        labels: Ground truth labels
        scores: Anomaly scores
        threshold: Classification threshold (if None, uses optimal from ROC)

    Returns:
        Dictionary of metrics
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    # AUC metrics
    auc_roc, _, _ = compute_auc_roc(labels, scores)
    auc_pr, _, _ = compute_auc_pr(labels, scores)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(labels, scores)

    # Binary predictions
    predictions = (scores >= threshold).astype(int)

    # Classification metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }


def compute_video_level_metrics(
    video_scores: Dict[str, List[float]],
    video_labels: Dict[str, int],
    aggregation: str = 'max'
) -> Dict[str, float]:
    """
    Compute video-level metrics by aggregating frame scores.

    Args:
        video_scores: Dict mapping video_id to list of frame scores
        video_labels: Dict mapping video_id to video label (0 or 1)
        aggregation: 'max', 'mean', or 'percentile_95'

    Returns:
        Dictionary of video-level metrics
    """
    video_agg_scores = []
    video_agg_labels = []

    for video_id, scores in video_scores.items():
        if video_id not in video_labels:
            continue

        scores = np.array(scores)

        if aggregation == 'max':
            agg_score = np.max(scores)
        elif aggregation == 'mean':
            agg_score = np.mean(scores)
        elif aggregation == 'percentile_95':
            agg_score = np.percentile(scores, 95)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        video_agg_scores.append(agg_score)
        video_agg_labels.append(video_labels[video_id])

    return compute_metrics(
        np.array(video_agg_labels),
        np.array(video_agg_scores)
    )


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for each line
    """
    print(f"{prefix}AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"{prefix}AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}F1 Score:  {metrics['f1']:.4f}")
    print(f"{prefix}Threshold: {metrics['threshold']:.4f}")
