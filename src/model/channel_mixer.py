"""Dynamic Channel Mixer (LUNA/Laya style).

Uses learned queries and cross-attention to compress variable-channel
EEG into a fixed-size latent representation, with 3D electrode
coordinate encoding via Fourier features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierPositionEncoding(nn.Module):
    """Encode 3D electrode coordinates using random Fourier features.

    Args:
        coord_dim: input coordinate dimension (3 for xyz)
        embed_dim: output embedding dimension
    """

    def __init__(self, coord_dim: int = 3, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        # Fixed random projection matrix
        self.register_buffer(
            "B_mat", torch.randn(coord_dim, embed_dim // 2) * 2 * math.pi
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [C, 3] or [B, C, 3] electrode coordinates
        Returns:
            [C, embed_dim] or [B, C, embed_dim] Fourier features
        """
        proj = coords @ self.B_mat  # [..., embed_dim // 2]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DynamicChannelMixer(nn.Module):
    """Cross-attention channel mixer with learned queries.

    Projects variable-channel patch embeddings into fixed-size
    representation using spatial-aware cross-attention.

    Args:
        patch_embed_dim: input patch embedding dimension (E)
        num_queries: number of learned queries (Nq)
        mixer_dim: output dimension (D)
        fourier_dim: Fourier feature dimension for coordinates
        num_heads: number of attention heads
    """

    def __init__(
        self,
        patch_embed_dim: int = 64,
        num_queries: int = 8,
        mixer_dim: int = 256,
        fourier_dim: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.mixer_dim = mixer_dim

        # Coordinate encoding
        self.coord_encoder = FourierPositionEncoding(coord_dim=3, embed_dim=fourier_dim)

        # Project patch_embed + coord_features to attention dim
        self.channel_proj = nn.Linear(patch_embed_dim + fourier_dim, mixer_dim)

        # Learned query vectors
        self.queries = nn.Parameter(torch.randn(num_queries, mixer_dim) * 0.02)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=mixer_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output projection: flatten Nq queries into D
        self.output_proj = nn.Linear(num_queries * mixer_dim, mixer_dim)
        self.norm = nn.LayerNorm(mixer_dim)

    def forward(
        self,
        patches: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: [B, C, N, E] patch embeddings
            coords: [B, C, 3] or [C, 3] electrode coordinates
            padding_mask: [B, C] True = padded (invalid) channel.
                          Passed as key_padding_mask to cross-attention
                          so padded channels get zero attention weight.
        Returns:
            sequence: [B, N, D] fixed-size latent sequence
            attn_weights: [B, Nq, C] query-channel affinity (for specialization loss)
        """
        B, C, N, E = patches.shape

        # Encode electrode coordinates
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(B, -1, -1)  # [B, C, 3]
        coord_features = self.coord_encoder(coords)  # [B, C, fourier_dim]

        all_seq = []
        all_attn = []

        for t in range(N):
            # Get patches at time step t: [B, C, E]
            patch_t = patches[:, :, t, :]

            # Concatenate spatial features: [B, C, E + fourier_dim]
            kv = torch.cat([patch_t, coord_features], dim=-1)
            kv = self.channel_proj(kv)  # [B, C, mixer_dim]

            # Expand queries for batch: [B, Nq, mixer_dim]
            q = self.queries.unsqueeze(0).expand(B, -1, -1)

            # Cross-attention: queries attend over channels
            # key_padding_mask: True positions are ignored in attention
            out, attn = self.cross_attn(
                q, kv, kv, key_padding_mask=padding_mask
            )  # out: [B, Nq, mixer_dim]
            all_attn.append(attn)

            # Flatten queries into single vector: [B, Nq * mixer_dim] → [B, D]
            out = out.reshape(B, -1)
            out = self.output_proj(out)
            out = self.norm(out)
            all_seq.append(out)

        # Stack time steps: [B, N, D]
        sequence = torch.stack(all_seq, dim=1)
        # Average attention weights across time: [B, Nq, C]
        attn_weights = torch.stack(all_attn, dim=1).mean(dim=1)

        return sequence, attn_weights

    def query_specialization_loss(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Penalize redundant queries by minimizing off-diagonal of AA^T.

        Args:
            attn_weights: [B, Nq, C] query-channel affinity
        Returns:
            scalar loss
        """
        # A @ A^T: [B, Nq, Nq]
        sim = torch.bmm(attn_weights, attn_weights.transpose(1, 2))
        # Off-diagonal penalty
        eye = torch.eye(self.num_queries, device=sim.device).unsqueeze(0)
        off_diag = sim * (1 - eye)
        return off_diag.pow(2).mean()
