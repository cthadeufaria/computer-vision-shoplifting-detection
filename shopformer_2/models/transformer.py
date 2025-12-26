"""
Shopformer Transformer Module.

Paper-aligned implementation with:
- 12 attention heads
- 4 encoder + 4 decoder layers
- 64 FFN dimension (paper specification)
- 144 d_model (divisible by 12 heads)
- 144 embedding size = 8 channels * 18 keypoints (paper optimal)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.

    Uses the standard positional encoding from "Attention Is All You Need".
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 100
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ShopformerTransformer(nn.Module):
    """
    Transformer encoder-decoder for Shopformer.

    Paper specs:
    - 12 attention heads
    - 4 encoder layers + 4 decoder layers
    - 512 feed-forward dimension
    - 144 model dimension (144 / 12 = 12 per head)

    The transformer takes GCAE tokens and reconstructs them.
    Anomaly detection is based on reconstruction error.
    """

    def __init__(
        self,
        input_dim: int = 144,           # GCAE output: 8 * 18 = 144 (paper optimal)
        d_model: int = 144,             # 144 / 12 heads = 12 per head
        nhead: int = 12,                # Paper: 12 attention heads
        num_encoder_layers: int = 4,    # Paper: 4 encoder layers
        num_decoder_layers: int = 4,    # Paper: 4 decoder layers
        dim_feedforward: int = 64,      # Paper: 64 FFN dimension
        dropout: float = 0.1,
        max_seq_len: int = 100,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.needs_projection = (input_dim != d_model)

        # Projection layers only needed if dimensions don't match
        # With 18 keypoints: input_dim = 8 * 18 = 144 = d_model (no projection needed)
        if self.needs_projection:
            self.input_projection = nn.Linear(input_dim, d_model)
            self.output_projection = nn.Linear(d_model, input_dim)
        else:
            self.input_projection = nn.Identity()
            self.output_projection = nn.Identity()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for token reconstruction.

        Args:
            tokens: Input tokens from GCAE, shape (batch, num_tokens, input_dim)
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            src_key_padding_mask: Source padding mask
            tgt_key_padding_mask: Target padding mask

        Returns:
            Reconstructed tokens, shape (batch, num_tokens, input_dim)
        """
        # Project to d_model dimension
        x = self.input_projection(tokens)  # (batch, num_tokens, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encode
        memory = self.encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Decode (autoencoder style - reconstruct from encoded representation)
        output = self.decoder(
            x,  # Use same input as target for reconstruction
            memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # Project back to input dimension
        output = self.output_projection(output)  # (batch, num_tokens, input_dim)

        return output

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens to latent representation.

        Args:
            tokens: Input tokens, shape (batch, num_tokens, input_dim)

        Returns:
            Encoded representation, shape (batch, num_tokens, d_model)
        """
        x = self.input_projection(tokens)
        x = self.pos_encoder(x)
        return self.encoder(x)

    def decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent representation.

        Args:
            memory: Encoded representation, shape (batch, num_tokens, d_model)
            tokens: Target tokens for decoding, shape (batch, num_tokens, input_dim)

        Returns:
            Decoded tokens, shape (batch, num_tokens, input_dim)
        """
        x = self.input_projection(tokens)
        x = self.pos_encoder(x)
        output = self.decoder(x, memory)
        return self.output_projection(output)


class TransformerConfig:
    """Configuration class for ShopformerTransformer."""

    # Paper-aligned defaults
    INPUT_DIM = 144          # 8 * 18 = 144 (paper optimal embedding size)
    D_MODEL = 144            # Divisible by 12 heads (144/12=12)
    NHEAD = 12               # Paper specification
    NUM_ENCODER_LAYERS = 4   # Paper specification
    NUM_DECODER_LAYERS = 4   # Paper specification
    DIM_FEEDFORWARD = 64     # Paper specification (64, not 512)
    DROPOUT = 0.1
    MAX_SEQ_LEN = 100
    ACTIVATION = 'gelu'

    @classmethod
    def from_config(cls, config: dict) -> dict:
        """
        Create transformer kwargs from config dictionary.

        Args:
            config: Configuration dictionary with 'model.transformer' section

        Returns:
            Dictionary of kwargs for ShopformerTransformer
        """
        transformer_cfg = config.get('model', {}).get('transformer', {})

        return {
            'input_dim': transformer_cfg.get('input_dim', cls.INPUT_DIM),
            'd_model': transformer_cfg.get('d_model', cls.D_MODEL),
            'nhead': transformer_cfg.get('num_heads', cls.NHEAD),
            'num_encoder_layers': transformer_cfg.get('num_layers', cls.NUM_ENCODER_LAYERS),
            'num_decoder_layers': transformer_cfg.get('num_layers', cls.NUM_DECODER_LAYERS),
            'dim_feedforward': transformer_cfg.get('dim_feedforward', cls.DIM_FEEDFORWARD),
            'dropout': transformer_cfg.get('dropout', cls.DROPOUT),
        }


def build_transformer(config: dict) -> ShopformerTransformer:
    """
    Build transformer from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured ShopformerTransformer instance
    """
    kwargs = TransformerConfig.from_config(config)
    return ShopformerTransformer(**kwargs)
