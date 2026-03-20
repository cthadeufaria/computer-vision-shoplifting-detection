"""
Graph Convolutional Autoencoder (GCAE) Tokenizer Module

This module implements the GCAE tokenizer used in Shopformer_2, which consists of
stacked modified ST-GCN layers that map pose input sequences into lower-dimensional
token sequences.

The architecture combines:
- Spatial Graph Convolutional Networks (GCNs) for spatial relationships between keypoints
- Temporal Convolutional Networks (TCNs) for motion patterns over time

Adapted for MPS (Apple Silicon) compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def get_skeleton_adjacency(num_keypoints: int = 17, layout: str = 'coco') -> np.ndarray:
    """
    Generate adjacency matrix for human skeleton graph.

    Args:
        num_keypoints: Number of keypoints (17 for COCO, 18 for COCO+neck)
        layout: Skeleton layout type ('coco', 'coco_with_neck')

    Returns:
        Adjacency matrix of shape (num_keypoints, num_keypoints)
    """
    if layout == 'coco' and num_keypoints == 17:
        # COCO 17 keypoint connections
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (0, 5), (0, 6),  # Shoulders to nose
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12),  # Torso
            (11, 12),  # Hip connection
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]
    elif num_keypoints == 18 or layout == 'coco_with_neck':
        # COCO 17 + synthetic neck (index 17)
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        # 17: neck (synthetic - average of shoulders)
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (0, 17),  # Nose to neck
            (17, 5), (17, 6),  # Neck to shoulders
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12),  # Torso (shoulders to hips)
            (11, 12),  # Hip connection
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]
    else:
        raise ValueError(f"Unknown layout: {layout} with {num_keypoints} keypoints")

    # Create adjacency matrix
    adj = np.zeros((num_keypoints, num_keypoints))
    for i, j in edges:
        if i < num_keypoints and j < num_keypoints:
            adj[i, j] = 1
            adj[j, i] = 1

    # Add self-loops
    adj = adj + np.eye(num_keypoints)

    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Normalize adjacency matrix using symmetric normalization.

    D^(-0.5) * A * D^(-0.5)

    Args:
        adj: Adjacency matrix

    Returns:
        Normalized adjacency matrix
    """
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


class GraphConvolution(nn.Module):
    """
    Spatial Graph Convolution layer.

    Applies graph convolution to capture spatial relationships between keypoints.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj: torch.Tensor,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Register adjacency matrix as buffer (not a parameter)
        self.register_buffer('adj', adj)

        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time, num_keypoints)

        Returns:
            Output tensor of shape (batch, out_channels, time, num_keypoints)
        """
        batch, channels, time, num_kp = x.shape

        # Reshape for graph convolution: (batch * time, num_kp, channels)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch * time, num_kp, channels)

        # Graph convolution: A * X * W
        x = torch.matmul(self.adj, x)
        x = torch.matmul(x, self.weight)

        if self.bias is not None:
            x = x + self.bias

        # Reshape back: (batch, out_channels, time, num_kp)
        x = x.view(batch, time, num_kp, self.out_channels)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class TemporalConvolution(nn.Module):
    """
    Temporal Convolution layer.

    Applies 1D convolution along the temporal dimension to capture motion patterns.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        padding: int = 4,
        dilation: int = 1
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time, num_keypoints)

        Returns:
            Output tensor of shape (batch, out_channels, time, num_keypoints)
        """
        return self.bn(self.conv(x))


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Block.

    Combines spatial graph convolution and temporal convolution in a single block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        # Spatial graph convolution
        self.gcn = GraphConvolution(in_channels, out_channels, adj)

        # Temporal convolution
        self.tcn = TemporalConvolution(
            out_channels,
            out_channels,
            kernel_size=9,
            stride=stride,
            padding=4
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time, num_keypoints)

        Returns:
            Output tensor of shape (batch, out_channels, time/stride, num_keypoints)
        """
        res = self.residual(x)
        x = self.gcn(x)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.dropout(x)
        x = x + res
        x = self.relu(x)
        return x


class GCAEEncoder(nn.Module):
    """
    GCAE Encoder: Maps pose sequences to token embeddings.

    Uses stacked ST-GCN blocks to progressively reduce temporal dimension
    and increase channel dimension.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        out_channels: int = 8,
        num_keypoints: int = 17,
        seq_len: int = 24,
        num_tokens: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
        layout: str = 'coco'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_keypoints = num_keypoints
        self.seq_len = seq_len
        self.num_tokens = num_tokens

        # Get skeleton adjacency matrix
        adj = get_skeleton_adjacency(num_keypoints, layout)
        adj = normalize_adjacency(adj)
        adj = torch.FloatTensor(adj)

        # Input batch normalization
        self.bn_input = nn.BatchNorm1d(in_channels * num_keypoints)

        # Encoder layers with progressively increasing channels and decreasing time
        layers = []
        channels = [in_channels, hidden_channels, hidden_channels, hidden_channels, out_channels]
        strides = self._compute_strides(seq_len, num_tokens, num_layers)

        for i in range(num_layers):
            layers.append(
                STGCNBlock(
                    channels[i],
                    channels[i + 1],
                    adj,
                    stride=strides[i],
                    residual=True,
                    dropout=dropout
                )
            )

        self.layers = nn.ModuleList(layers)

        # Adaptive pooling to ensure exact num_tokens output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((num_tokens, num_keypoints))

    def _compute_strides(self, seq_len: int, num_tokens: int, num_layers: int) -> list:
        """
        Compute strides to reduce seq_len to exactly num_tokens.

        For seq_len=24, num_tokens=2, num_layers=4:
        24 -> 12 -> 6 -> 2 -> 2 with strides [2, 2, 3, 1]
        """
        strides = [1] * num_layers

        # Calculate total reduction needed
        total_reduction = seq_len // num_tokens  # 24/2 = 12

        # Find prime factorization of total_reduction
        remaining = total_reduction
        factors = []

        for p in [2, 3, 4, 5, 6]:
            while remaining % p == 0 and remaining > 1:
                factors.append(p)
                remaining = remaining // p

        if remaining > 1:
            factors.append(remaining)

        # Distribute factors across layers (prefer smaller factors first)
        factors.sort()
        for i, factor in enumerate(factors):
            if i < num_layers:
                strides[i] = factor

        # Sort strides descending to apply larger reductions later
        # This helps with the convolution receptive fields
        strides.sort(reverse=True)

        # Verify the reduction
        final_len = seq_len
        for s in strides:
            final_len = final_len // s

        self._final_len = final_len
        self._needs_pooling = (final_len != num_tokens)

        return strides

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time, num_keypoints)
               or (batch, time, num_keypoints, channels)

        Returns:
            Token embeddings of shape (batch, num_tokens, embedding_dim)
            where embedding_dim = out_channels * num_keypoints
        """
        # Handle different input formats
        if x.dim() == 4 and x.shape[-1] == self.in_channels:
            # (batch, time, num_keypoints, channels) -> (batch, channels, time, num_keypoints)
            x = x.permute(0, 3, 1, 2).contiguous()

        batch, channels, time, num_kp = x.shape

        # Batch normalization on input
        x = x.permute(0, 1, 3, 2).contiguous()  # (batch, channels, num_kp, time)
        x = x.view(batch, channels * num_kp, time)
        x = self.bn_input(x)
        x = x.view(batch, channels, num_kp, time)
        x = x.permute(0, 1, 3, 2).contiguous()  # (batch, channels, time, num_kp)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x)

        # Ensure exact num_tokens via adaptive pooling if needed
        if hasattr(self, '_needs_pooling') and self._needs_pooling:
            # MPS doesn't support adaptive pooling with non-divisible sizes
            # Fall back to CPU for this operation
            if x.device.type == 'mps':
                original_device = x.device
                x = x.cpu()
                x = self.adaptive_pool(x)
                x = x.to(original_device)
            else:
                x = self.adaptive_pool(x)

        # Reshape to token format: (batch, num_tokens, embedding_dim)
        batch, channels, time, num_kp = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, time, channels, num_kp)
        x = x.view(batch, time, channels * num_kp)  # (batch, num_tokens, embedding_dim)

        return x


class GCAEDecoder(nn.Module):
    """
    GCAE Decoder: Reconstructs pose sequences from token embeddings.

    Uses transposed convolutions to upsample temporal dimension.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_keypoints: int = 17,
        seq_len: int = 24,
        num_tokens: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
        layout: str = 'coco'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_keypoints = num_keypoints
        self.seq_len = seq_len
        self.num_tokens = num_tokens

        # Decoder layers
        self.initial_proj = nn.Linear(
            in_channels * num_keypoints,
            hidden_channels * num_keypoints
        )

        # Upsample temporal dimension
        upsample_factors = self._compute_upsample_factors(num_tokens, seq_len, num_layers)

        layers = []
        channels = [hidden_channels] * (num_layers - 1) + [out_channels]

        for i in range(num_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels
            out_ch = channels[i]

            if upsample_factors[i] > 1:
                layers.append(
                    nn.ConvTranspose2d(
                        in_ch, out_ch,
                        kernel_size=(upsample_factors[i], 1),
                        stride=(upsample_factors[i], 1)
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1)
                )

            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def _compute_upsample_factors(self, num_tokens: int, seq_len: int, num_layers: int) -> list:
        """Compute upsample factors to expand num_tokens to seq_len."""
        factors = [1] * num_layers
        current_len = num_tokens
        layer_idx = 0

        while current_len < seq_len and layer_idx < num_layers:
            if current_len * 2 <= seq_len:
                factors[layer_idx] = 2
                current_len = current_len * 2
            layer_idx += 1

        return factors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Token embeddings of shape (batch, num_tokens, embedding_dim)

        Returns:
            Reconstructed poses of shape (batch, channels, seq_len, num_keypoints)
        """
        batch, num_tokens, embed_dim = x.shape

        # Project to hidden dimension
        x = self.initial_proj(x)  # (batch, num_tokens, hidden_channels * num_keypoints)

        # Reshape for conv layers
        hidden_ch = x.shape[-1] // self.num_keypoints
        x = x.view(batch, num_tokens, hidden_ch, self.num_keypoints)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, hidden_channels, num_tokens, num_keypoints)

        # Pass through decoder layers
        x = self.layers(x)

        # Ensure output has correct temporal dimension
        if x.shape[2] != self.seq_len:
            x = F.interpolate(
                x,
                size=(self.seq_len, self.num_keypoints),
                mode='bilinear',
                align_corners=False
            )

        return x


class GCAE(nn.Module):
    """
    Graph Convolutional Autoencoder.

    Full autoencoder that encodes pose sequences into tokens and decodes them back.
    Used as the tokenizer module in Shopformer_2.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        latent_channels: int = 8,
        num_keypoints: int = 17,
        seq_len: int = 24,
        num_tokens: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
        layout: str = 'coco'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        self.embedding_dim = latent_channels * num_keypoints

        self.encoder = GCAEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=latent_channels,
            num_keypoints=num_keypoints,
            seq_len=seq_len,
            num_tokens=num_tokens,
            num_layers=num_layers,
            dropout=dropout,
            layout=layout
        )

        self.decoder = GCAEDecoder(
            in_channels=latent_channels,
            hidden_channels=hidden_channels,
            out_channels=in_channels,
            num_keypoints=num_keypoints,
            seq_len=seq_len,
            num_tokens=num_tokens,
            num_layers=num_layers,
            dropout=dropout,
            layout=layout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pose sequence to tokens."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode tokens to pose sequence."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input pose sequence of shape (batch, channels, time, num_keypoints)

        Returns:
            Tuple of (reconstructed, tokens)
        """
        tokens = self.encode(x)
        reconstructed = self.decode(tokens)
        return reconstructed, tokens

    def get_embedding_dim(self) -> int:
        """Get the token embedding dimension."""
        return self.embedding_dim
