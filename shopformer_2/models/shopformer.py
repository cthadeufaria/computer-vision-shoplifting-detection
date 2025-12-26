"""
Shopformer: Main Model.

Two-stage anomaly detection model combining:
1. GCAE (Graph Convolutional Autoencoder) for pose tokenization
2. Transformer encoder-decoder for token reconstruction

Anomaly score is computed from reconstruction error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

from .gcae import GCAE, GCAEEncoder
from .transformer import ShopformerTransformer, build_transformer


class Shopformer(nn.Module):
    """
    Shopformer anomaly detection model.

    Architecture:
        Input Poses -> GCAE Encoder -> Tokens -> Transformer -> Reconstructed Tokens
                            |                                           |
                            v                                           v
                    GCAE Decoder -> Reconstructed Poses          Anomaly Score

    Training:
        Stage 1: Train GCAE (pose reconstruction)
        Stage 2: Train Transformer with frozen GCAE (token reconstruction)

    Inference:
        Anomaly score = MSE(tokens, reconstructed_tokens)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Shopformer model.

        Args:
            config: Configuration dictionary with 'model' section
        """
        super().__init__()
        self.config = config
        model_cfg = config['model']

        # GCAE Tokenizer
        self.gcae = GCAE(
            in_channels=model_cfg['in_channels'],
            hidden_channels=model_cfg['gcae']['hidden_channels'],
            latent_channels=model_cfg['gcae']['latent_channels'],
            num_keypoints=model_cfg['num_keypoints'],
            seq_len=model_cfg['seq_len'],
            num_tokens=model_cfg['num_tokens'],
            num_layers=model_cfg['gcae'].get('num_layers', 4),
            dropout=model_cfg['gcae'].get('dropout', 0.1)
        )

        # Transformer for token reconstruction
        self.transformer = build_transformer(config)

        # Track GCAE frozen state
        self._gcae_frozen = False

        # Store dimensions for reference
        self.num_keypoints = model_cfg['num_keypoints']
        self.seq_len = model_cfg['seq_len']
        self.num_tokens = model_cfg['num_tokens']
        self.latent_channels = model_cfg['gcae']['latent_channels']

    def freeze_gcae(self):
        """
        Freeze GCAE parameters for Stage 2 training.

        Call this before Stage 2 to train only the transformer.
        """
        for param in self.gcae.parameters():
            param.requires_grad = False
        self.gcae.eval()
        self._gcae_frozen = True

    def unfreeze_gcae(self):
        """
        Unfreeze GCAE parameters.

        Call this if you want to fine-tune the entire model.
        """
        for param in self.gcae.parameters():
            param.requires_grad = True
        self._gcae_frozen = False

    def train(self, mode: bool = True):
        """
        Override train to keep GCAE in eval mode when frozen.
        """
        super().train(mode)
        if self._gcae_frozen and mode:
            self.gcae.eval()
        return self

    def forward(
        self,
        poses: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the full model.

        Args:
            poses: Input pose sequences, shape (batch, in_channels, seq_len, num_keypoints)
            return_all: If True, return all intermediate outputs

        Returns:
            If return_all=False:
                reconstructed_tokens: Shape (batch, num_tokens, latent_channels * num_keypoints)
            If return_all=True:
                Tuple of (reconstructed_poses, tokens, reconstructed_tokens)
        """
        # Encode poses to tokens via GCAE
        if self._gcae_frozen:
            with torch.no_grad():
                reconstructed_poses, tokens = self.gcae(poses)
                tokens = tokens.detach()
        else:
            reconstructed_poses, tokens = self.gcae(poses)

        # Reconstruct tokens with transformer
        reconstructed_tokens = self.transformer(tokens)

        if return_all:
            return reconstructed_poses, tokens, reconstructed_tokens

        return reconstructed_tokens

    def encode(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Encode poses to tokens using GCAE.

        Args:
            poses: Input poses, shape (batch, in_channels, seq_len, num_keypoints)

        Returns:
            tokens: Shape (batch, num_tokens, latent_channels * num_keypoints)
        """
        if self._gcae_frozen:
            with torch.no_grad():
                _, tokens = self.gcae(poses)
                return tokens.detach()
        else:
            _, tokens = self.gcae(poses)
            return tokens

    def compute_anomaly_score(
        self,
        poses: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute anomaly score for input poses.

        Higher scores indicate more anomalous (potential shoplifting) behavior.

        Args:
            poses: Input poses, shape (batch, in_channels, seq_len, num_keypoints)
            reduction: 'mean' for per-sample score, 'none' for per-token scores

        Returns:
            Anomaly scores, shape (batch,) if reduction='mean', else (batch, num_tokens)
        """
        self.eval()
        with torch.no_grad():
            # Get tokens and reconstructed tokens
            _, tokens = self.gcae(poses)
            reconstructed_tokens = self.transformer(tokens)

            # Compute MSE reconstruction error
            if reduction == 'mean':
                # Mean over tokens and features -> one score per sample
                scores = torch.mean((tokens - reconstructed_tokens) ** 2, dim=(1, 2))
            elif reduction == 'none':
                # Mean over features only -> one score per token
                scores = torch.mean((tokens - reconstructed_tokens) ** 2, dim=2)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

        return scores

    def compute_gcae_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute GCAE reconstruction loss (Stage 1).

        Args:
            poses: Input poses, shape (batch, in_channels, seq_len, num_keypoints)

        Returns:
            MSE loss for pose reconstruction
        """
        reconstructed_poses, _ = self.gcae(poses)
        return F.mse_loss(reconstructed_poses, poses)

    def compute_transformer_loss(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Compute Transformer reconstruction loss (Stage 2).

        Args:
            poses: Input poses, shape (batch, in_channels, seq_len, num_keypoints)

        Returns:
            MSE loss for token reconstruction
        """
        # Get tokens (frozen GCAE)
        if self._gcae_frozen:
            with torch.no_grad():
                _, tokens = self.gcae(poses)
                tokens = tokens.detach()
        else:
            _, tokens = self.gcae(poses)

        # Reconstruct with transformer
        reconstructed_tokens = self.transformer(tokens)

        return F.mse_loss(reconstructed_tokens, tokens)

    def get_num_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """
        Get parameter counts for model components.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Dictionary with parameter counts
        """
        def count_params(module, trainable_only):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        return {
            'gcae': count_params(self.gcae, trainable_only),
            'transformer': count_params(self.transformer, trainable_only),
            'total': count_params(self, trainable_only)
        }

    def load_gcae_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """
        Load GCAE weights from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce state dict matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'gcae_state_dict' in checkpoint:
            self.gcae.load_state_dict(checkpoint['gcae_state_dict'], strict=strict)
        elif 'model_state_dict' in checkpoint:
            # Try to extract GCAE weights from full model checkpoint
            gcae_state = {
                k.replace('gcae.', ''): v
                for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('gcae.')
            }
            self.gcae.load_state_dict(gcae_state, strict=strict)
        else:
            self.gcae.load_state_dict(checkpoint, strict=strict)

    def load_transformer_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """
        Load Transformer weights from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce state dict matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'transformer_state_dict' in checkpoint:
            self.transformer.load_state_dict(
                checkpoint['transformer_state_dict'], strict=strict
            )
        elif 'model_state_dict' in checkpoint:
            # Try to extract transformer weights from full model checkpoint
            transformer_state = {
                k.replace('transformer.', ''): v
                for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('transformer.')
            }
            self.transformer.load_state_dict(transformer_state, strict=strict)
        else:
            self.transformer.load_state_dict(checkpoint, strict=strict)


def build_shopformer(config: Dict[str, Any]) -> Shopformer:
    """
    Build Shopformer model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured Shopformer instance
    """
    return Shopformer(config)
