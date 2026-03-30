"""Patch Embedder: Channel-independent 1D convolution.

Converts raw EEG [B, C, T] into patch embeddings [B, C, N, E]
using depthwise 1D convolution applied independently per channel.
"""

import torch
import torch.nn as nn


class PatchEmbedder(nn.Module):
    """Depthwise 1D convolution patch embedder.

    Args:
        patch_size: number of time samples per patch
        embed_dim: output embedding dimension (E)
    """

    def __init__(self, patch_size: int = 40, embed_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Single-channel conv applied to each channel independently
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] raw EEG signal
        Returns:
            [B, C, N, E] patch embeddings
              N = T // patch_size (number of patches per channel)
        """
        B, C, T = x.shape
        # Reshape to apply same conv to each channel: [B*C, 1, T]
        x = x.reshape(B * C, 1, T)
        # Conv1d: [B*C, 1, T] → [B*C, E, N]
        x = self.proj(x)
        # Reshape back: [B, C, N, E]
        N = x.shape[-1]
        x = x.reshape(B, C, self.embed_dim, N)
        x = x.permute(0, 1, 3, 2)  # [B, C, N, E]
        return x
