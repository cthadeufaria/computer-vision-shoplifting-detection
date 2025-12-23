"""
Metrics for evaluating Shopformer performance.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
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
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc, fpr, tpr


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = None
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
    auc_roc = roc_auc_score(labels, scores)
    auc_pr = average_precision_score(labels, scores)

    if threshold is None:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        threshold = thresholds[optimal_idx]

    predictions = (scores >= threshold).astype(int)

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
