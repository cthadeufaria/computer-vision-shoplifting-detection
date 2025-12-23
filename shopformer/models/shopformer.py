"""
Shopformer: Complete Model

This module implements the full Shopformer architecture that combines:
1. GCAE Tokenizer: Encodes pose sequences into compact token representations
2. Transformer Encoder-Decoder: Reconstructs tokens for anomaly detection

The model uses a two-stage training approach:
- Stage 1: Train GCAE to learn pose embeddings
- Stage 2: Freeze GCAE encoder, train transformer for reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .gcae import GCAE, GCAEEncoder
from .transformer import ShopformerTransformer, PositionalEncoding


class Shopformer(nn.Module):
    """
    Shopformer: Transformer-Based Framework for Detecting Shoplifting via Human Pose.

    Architecture:
        - GCAE Tokenizer: Converts pose sequences to token embeddings
        - Transformer: Reconstructs tokens; high error indicates anomaly

    Reference:
        Rashvand et al., "Shopformer: Transformer-Based Framework for
        Detecting Shoplifting via Human Pose", CVPR 2025
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        latent_channels: int = 8,
        num_keypoints: int = 17,
        seq_len: int = 12,
        num_tokens: int = 2,
        gcae_layers: int = 4,
        transformer_heads: int = 2,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 64,
        dropout: float = 0.1,
        layout: str = 'coco',
        freeze_tokenizer: bool = False
    ):
        """
        Initialize Shopformer.

        Args:
            in_channels: Number of input channels (2 for x,y coordinates)
            hidden_channels: Hidden dimension in GCAE
            latent_channels: Latent dimension in GCAE tokens
            num_keypoints: Number of pose keypoints (17 for COCO, 18 for OpenPose)
            seq_len: Length of input pose sequence (frames)
            num_tokens: Number of tokens to generate
            gcae_layers: Number of layers in GCAE encoder/decoder
            transformer_heads: Number of attention heads
            transformer_layers: Number of transformer encoder/decoder layers
            transformer_ff_dim: Feed-forward dimension in transformer
            dropout: Dropout rate
            layout: Skeleton layout ('coco' or 'openpose')
            freeze_tokenizer: Whether to freeze GCAE encoder (Stage 2 training)
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        self.latent_channels = latent_channels

        # Token embedding dimension: latent_channels * num_keypoints
        self.embedding_dim = latent_channels * num_keypoints

        # GCAE Tokenizer
        self.gcae = GCAE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            num_keypoints=num_keypoints,
            seq_len=seq_len,
            num_tokens=num_tokens,
            num_layers=gcae_layers,
            dropout=dropout,
            layout=layout
        )

        # Transformer
        self.transformer = ShopformerTransformer(
            d_model=self.embedding_dim,
            nhead=transformer_heads,
            num_encoder_layers=transformer_layers,
            num_decoder_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout
        )

        # Positional encoding for tokens (used for computing normality score)
        self.pos_encoder = PositionalEncoding(
            self.embedding_dim,
            max_len=100,
            dropout=0.0  # No dropout for position encoding in scoring
        )

        self.freeze_tokenizer = freeze_tokenizer
        if freeze_tokenizer:
            self._freeze_gcae_encoder()

    def _freeze_gcae_encoder(self):
        """Freeze GCAE encoder parameters for Stage 2 training."""
        for param in self.gcae.encoder.parameters():
            param.requires_grad = False

    def unfreeze_tokenizer(self):
        """Unfreeze GCAE encoder parameters."""
        for param in self.gcae.encoder.parameters():
            param.requires_grad = True
        self.freeze_tokenizer = False

    def tokenize(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Convert pose sequences to tokens using GCAE encoder.

        Args:
            poses: Input pose sequences of shape (batch, channels, time, keypoints)
                   or (batch, time, keypoints, channels)

        Returns:
            Token embeddings of shape (batch, num_tokens, embedding_dim)
        """
        return self.gcae.encode(poses)

    def reconstruct_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct tokens using transformer.

        Args:
            tokens: Token embeddings (batch, num_tokens, embedding_dim)

        Returns:
            Reconstructed tokens (batch, num_tokens, embedding_dim)
        """
        return self.transformer(tokens)

    def compute_normality_score(
        self,
        tokens: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normality score based on reconstruction error.

        Lower score = more normal behavior
        Higher score = more anomalous (potential shoplifting)

        Args:
            tokens: Original tokens (with positional encoding)
            reconstructed: Reconstructed tokens

        Returns:
            Normality scores per sample (batch,)
        """
        # Add positional encoding to original tokens for comparison
        tokens_pe = self.pos_encoder.pe[:, :tokens.size(1), :].expand(
            tokens.size(0), -1, -1
        )
        tokens_with_pe = tokens + tokens_pe

        # Compute MSE reconstruction error
        error = F.mse_loss(reconstructed, tokens_with_pe, reduction='none')
        score = error.mean(dim=[1, 2])  # Average over tokens and dimensions

        return score

    def forward(
        self,
        poses: torch.Tensor,
        return_tokens: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            poses: Input pose sequences of shape (batch, channels, time, keypoints)
            return_tokens: Whether to return intermediate tokens

        Returns:
            Dictionary containing:
                - 'normality_score': Anomaly scores (higher = more anomalous)
                - 'reconstructed_tokens': Reconstructed token sequence
                - 'tokens': Original tokens (if return_tokens=True)
                - 'gcae_reconstructed': GCAE reconstruction (for Stage 1)
        """
        # Tokenize poses using GCAE encoder
        tokens = self.tokenize(poses)

        # Reconstruct tokens using transformer
        reconstructed_tokens = self.reconstruct_tokens(tokens)

        # Compute normality score
        normality_score = self.compute_normality_score(tokens, reconstructed_tokens)

        # GCAE reconstruction (for Stage 1 training)
        gcae_reconstructed = self.gcae.decode(tokens)

        output = {
            'normality_score': normality_score,
            'reconstructed_tokens': reconstructed_tokens,
            'gcae_reconstructed': gcae_reconstructed
        }

        if return_tokens:
            output['tokens'] = tokens

        return output

    def predict(self, poses: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict whether poses represent shoplifting.

        Args:
            poses: Input pose sequences
            threshold: Score threshold for classification

        Returns:
            Binary predictions (1 = shoplifting, 0 = normal)
        """
        with torch.no_grad():
            output = self.forward(poses)
            scores = output['normality_score']
            predictions = (scores > threshold).long()
        return predictions

    def get_anomaly_scores(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Get anomaly scores for pose sequences.

        Args:
            poses: Input pose sequences

        Returns:
            Anomaly scores (higher = more likely shoplifting)
        """
        with torch.no_grad():
            output = self.forward(poses)
        return output['normality_score']

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Shopformer':
        """
        Create Shopformer from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Shopformer instance
        """
        return cls(
            in_channels=config.get('in_channels', 2),
            hidden_channels=config.get('hidden_channels', 64),
            latent_channels=config.get('latent_channels', 8),
            num_keypoints=config.get('num_keypoints', 17),
            seq_len=config.get('seq_len', 12),
            num_tokens=config.get('num_tokens', 2),
            gcae_layers=config.get('gcae_layers', 4),
            transformer_heads=config.get('transformer_heads', 2),
            transformer_layers=config.get('transformer_layers', 2),
            transformer_ff_dim=config.get('transformer_ff_dim', 64),
            dropout=config.get('dropout', 0.1),
            layout=config.get('layout', 'coco'),
            freeze_tokenizer=config.get('freeze_tokenizer', False)
        )


class ShopformerStage1(nn.Module):
    """
    Shopformer Stage 1: GCAE Tokenizer Training.

    Trains the Graph Convolutional Autoencoder to learn meaningful
    pose embeddings through reconstruction.
    """

    def __init__(self, shopformer: Shopformer):
        super().__init__()
        self.gcae = shopformer.gcae

    def forward(self, poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for GCAE training.

        Args:
            poses: Input pose sequences

        Returns:
            Tuple of (reconstructed_poses, tokens)
        """
        reconstructed, tokens = self.gcae(poses)
        return reconstructed, tokens

    def compute_loss(
        self,
        poses: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GCAE reconstruction loss.

        Args:
            poses: Original poses
            reconstructed: Reconstructed poses

        Returns:
            MSE loss
        """
        return F.mse_loss(reconstructed, poses)


class ShopformerStage2(nn.Module):
    """
    Shopformer Stage 2: Transformer Training.

    Freezes GCAE encoder and trains transformer to reconstruct
    token sequences.
    """

    def __init__(self, shopformer: Shopformer):
        super().__init__()
        self.shopformer = shopformer

        # Freeze GCAE encoder
        for param in self.shopformer.gcae.encoder.parameters():
            param.requires_grad = False

    def forward(self, poses: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for transformer training.

        Args:
            poses: Input pose sequences

        Returns:
            Dictionary with tokens, reconstructed tokens, and scores
        """
        # Get tokens (frozen encoder)
        with torch.no_grad():
            tokens = self.shopformer.tokenize(poses)

        # Reconstruct with transformer
        reconstructed_tokens = self.shopformer.reconstruct_tokens(tokens)

        # Compute score
        normality_score = self.shopformer.compute_normality_score(
            tokens, reconstructed_tokens
        )

        return {
            'tokens': tokens,
            'reconstructed_tokens': reconstructed_tokens,
            'normality_score': normality_score
        }

    def compute_loss(
        self,
        tokens: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute transformer reconstruction loss.

        Args:
            tokens: Original tokens
            reconstructed: Reconstructed tokens

        Returns:
            MSE loss
        """
        # Add positional encoding to tokens for loss computation
        tokens_pe = self.shopformer.pos_encoder.pe[:, :tokens.size(1), :].expand(
            tokens.size(0), -1, -1
        )
        tokens_with_pe = tokens + tokens_pe

        return F.mse_loss(reconstructed, tokens_with_pe)
