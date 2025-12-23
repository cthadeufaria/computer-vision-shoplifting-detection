"""
Transformer Module for Shopformer

This module implements the transformer encoder-decoder architecture used in
Shopformer for reconstructing token sequences and computing normality scores.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sinusoidal functions.

    Adds positional information to token embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: Source sequence (batch, seq_len, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Padding mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tgt: Target sequence (batch, seq_len, d_model)
            memory: Encoder output (batch, seq_len, d_model)
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class ShopformerTransformer(nn.Module):
    """
    Transformer Encoder-Decoder for Shopformer.

    Takes token embeddings from GCAE and reconstructs them through
    an encoder-decoder architecture. Reconstruction error is used
    as anomaly score.
    """

    def __init__(
        self,
        d_model: int = 144,
        nhead: int = 2,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Initialize transformer.

        Args:
            d_model: Dimension of token embeddings (typically latent_channels * num_keypoints)
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feed-forward network
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode token sequence.

        Args:
            src: Source tokens (batch, seq_len, d_model)

        Returns:
            Encoded representation (batch, seq_len, d_model)
        """
        # Add positional encoding
        src = self.pos_encoder(src)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            src = layer(src)

        return src

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode from encoder output.

        Args:
            tgt: Target tokens (batch, seq_len, d_model)
            memory: Encoder output (batch, seq_len, d_model)

        Returns:
            Decoded output (batch, seq_len, d_model)
        """
        # Add positional encoding to target
        tgt = self.pos_encoder(tgt)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)

        return tgt

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: reconstruct token sequence.

        Args:
            tokens: Input tokens (batch, num_tokens, d_model)

        Returns:
            Reconstructed tokens (batch, num_tokens, d_model)
        """
        # Encode
        memory = self.encode(tokens)

        # Use encoder output as initial decoder input (autoencoder style)
        # Create shifted target by prepending a learned start token
        batch_size = tokens.size(0)
        start_token = torch.zeros(batch_size, 1, self.d_model, device=tokens.device)
        tgt = torch.cat([start_token, tokens[:, :-1, :]], dim=1)

        # Decode
        output = self.decode(tgt, memory)

        # Project to output space
        output = self.output_proj(output)

        return output

    def compute_reconstruction_error(
        self,
        tokens: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction error (normality score).

        Args:
            tokens: Original tokens
            reconstructed: Reconstructed tokens

        Returns:
            Reconstruction error per sample
        """
        # MSE per sample
        error = F.mse_loss(reconstructed, tokens, reduction='none')
        error = error.mean(dim=[1, 2])  # Average over tokens and dimensions
        return error
