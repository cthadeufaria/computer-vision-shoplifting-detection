"""
Shopformer: Transformer-Based Framework for Detecting Shoplifting via Human Pose

This package implements the Shopformer architecture for pose-based shoplifting
detection as described in:

    Rashvand et al., "Shopformer: Transformer-Based Framework for Detecting
    Shoplifting via Human Pose", CVPR 2025

Architecture:
    - GCAE Tokenizer: Graph Convolutional Autoencoder based on ST-GCN
    - Transformer: Encoder-decoder for token reconstruction
    - Anomaly Detection: Reconstruction error as normality score

Usage:
    from shopformer import Shopformer

    model = Shopformer(
        num_keypoints=17,
        seq_len=12,
        num_tokens=2
    )

    # Get anomaly scores
    scores = model.get_anomaly_scores(pose_sequences)
"""

from .models import Shopformer, GCAE, ShopformerTransformer

__version__ = '0.1.0'
__all__ = ['Shopformer', 'GCAE', 'ShopformerTransformer']
