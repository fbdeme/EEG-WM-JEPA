"""EEG-JEPA: JEPA model adapted for EEG.

Replaces LeWM's ViT encoder with:
  - PatchEmbedder (channel-independent 1D Conv)
  - DynamicChannelMixer (3D coords + cross-attention)

Replaces action-conditioned prediction with unconditional
masked prediction (EEG pretraining has no actions).

Loss: pred_loss + λ_sigreg * SIGReg + λ_query * query_specialization
"""

import torch
import torch.nn as nn
from einops import rearrange

from src.model.patch_embedder import PatchEmbedder
from src.model.channel_mixer import DynamicChannelMixer
from src.model.lewm_modules import Transformer, Block, MLP, SIGReg
from src.model.masking import generate_batch_masks


class EEGPredictor(nn.Module):
    """Predicts masked patch embeddings from context.

    Unlike LeWM's ARPredictor (action-conditioned, causal),
    this is a bidirectional predictor that fills in masked positions.
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        hidden_dim: int,
        depth: int = 3,
        heads: int = 4,
        dim_head: int = 64,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.transformer = Transformer(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=Block,
        )

    def forward(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [B, N, D] encoder output (all positions)
            mask: [B, N] boolean (True = masked/target)
        Returns:
            [B, N, D] predictions
        """
        B, N, D = context.shape
        x = context.clone()

        # Replace masked positions with learnable mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        x = torch.where(mask_expanded, self.mask_token.expand(B, N, D), x)

        # Add positional embedding
        x = x + self.pos_embedding[:, :N]

        # Predict
        x = self.transformer(x)
        return x


class EEGJEPA(nn.Module):
    """Full EEG-JEPA model.

    Architecture:
        EEG [B, C, T] → PatchEmbedder → ChannelMixer → Encoder → Predictor
        Loss = MSE(pred, target) + λ₁·SIGReg + λ₂·QuerySpec
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        train_cfg = cfg["training"]

        num_patches = cfg["preprocessing"]["window_samples"] // model_cfg["patch_size"]
        embed_dim = model_cfg["mixer_dim"]

        # EEG-specific frontend
        self.patch_embedder = PatchEmbedder(
            patch_size=model_cfg["patch_size"],
            embed_dim=model_cfg["patch_embed_dim"],
        )
        self.channel_mixer = DynamicChannelMixer(
            patch_embed_dim=model_cfg["patch_embed_dim"],
            num_queries=model_cfg["num_queries"],
            mixer_dim=embed_dim,
            fourier_dim=model_cfg["fourier_features"],
            num_heads=model_cfg["num_heads_mixer"],
        )

        # Encoder (from LeWM architecture)
        self.encoder = Transformer(
            input_dim=embed_dim,
            hidden_dim=model_cfg["encoder_dim"],
            output_dim=model_cfg["encoder_dim"],
            depth=model_cfg["encoder_depth"],
            heads=model_cfg["encoder_heads"],
            dim_head=model_cfg["encoder_dim"] // model_cfg["encoder_heads"],
            mlp_dim=model_cfg["encoder_dim"] * 4,
            dropout=model_cfg["dropout"],
            block_class=Block,
        )

        # Projector (LeWM style: MLP with BatchNorm1d)
        self.projector = MLP(
            input_dim=model_cfg["encoder_dim"],
            hidden_dim=model_cfg["projector_hidden"],
            output_dim=model_cfg["projector_out"],
            norm_fn=nn.BatchNorm1d,
        )

        # Predictor (LeWM style: fixed dim_head, independent of encoder_dim)
        predictor_dim_head = model_cfg.get("predictor_dim_head", 64)
        predictor_mlp_dim = model_cfg.get(
            "predictor_mlp_dim", model_cfg["encoder_dim"] * 4
        )
        self.predictor = EEGPredictor(
            num_patches=num_patches,
            embed_dim=model_cfg["encoder_dim"],
            hidden_dim=model_cfg["encoder_dim"],
            depth=model_cfg["predictor_depth"],
            heads=model_cfg["predictor_heads"],
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            dropout=model_cfg["dropout"],
        )

        # Predictor projector (LeWM style)
        self.pred_projector = MLP(
            input_dim=model_cfg["encoder_dim"],
            hidden_dim=model_cfg["projector_hidden"],
            output_dim=model_cfg["projector_out"],
            norm_fn=nn.BatchNorm1d,
        )

        # SIGReg (from LeWM, as-is)
        self.sigreg = SIGReg(
            knots=17,
            num_proj=train_cfg["sigreg_num_projections"],
        )

        # Masking config
        self.mask_ratio = train_cfg["mask_ratio"]
        patch_duration = model_cfg["patch_size"] / cfg["preprocessing"]["target_srate"]
        self.block_min = max(1, int(train_cfg["mask_block_min_sec"] / patch_duration))
        self.block_max = max(
            self.block_min, int(train_cfg["mask_block_max_sec"] / patch_duration)
        )
        self.num_patches = num_patches

        # Loss weights
        self.lambda_sigreg = train_cfg["lambda_sigreg"]
        self.lambda_query = train_cfg["lambda_query"]

    def encode(
        self,
        eeg: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode EEG to latent sequence.

        Args:
            eeg: [B, C, T]
            coords: [B, C, 3] or [C, 3]
            padding_mask: [B, C] True = padded channel (optional)
        Returns:
            encoded: [B, N, D_enc]
            attn_weights: [B, Nq, C]
        """
        patches = self.patch_embedder(eeg)        # [B, C, N, E]
        mixed, attn = self.channel_mixer(
            patches, coords, padding_mask=padding_mask
        )  # [B, N, D]
        encoded = self.encoder(mixed)              # [B, N, D_enc]
        return encoded, attn

    def forward(self, batch: dict) -> dict:
        """Full forward pass with loss computation.

        Args:
            batch: dict with 'eeg' [B, C, T], 'coords' [B, C, 3] or [C, 3],
                   and optionally 'padding_mask' [B, C]
        Returns:
            dict with 'loss', 'pred_loss', 'sigreg_loss', 'query_loss'
        """
        eeg = batch["eeg"]
        coords = batch["coords"]
        padding_mask = batch.get("padding_mask")
        B = eeg.shape[0]

        # 1. Encode full sequence
        encoded, attn_weights = self.encode(
            eeg, coords, padding_mask=padding_mask
        )  # [B, N, D_enc]

        # 2. Generate masks
        mask = generate_batch_masks(
            B, self.num_patches,
            self.mask_ratio, self.block_min, self.block_max,
        ).to(eeg.device)  # [B, N]

        # 3. Project full encoding to target space
        #    Following Laya: stop-gradient on target branch
        #    Projector learns via SIGReg applied to projected space
        projected = self.projector(
            rearrange(encoded, "b n d -> (b n) d")
        )
        projected = rearrange(projected, "(b n) d -> b n d", b=B)
        target = projected.detach()  # stop-gradient on target for pred loss

        # 4. Predict masked positions
        pred = self.predictor(encoded, mask)  # [B, N, D_enc]
        pred = self.pred_projector(
            rearrange(pred, "b n d -> (b n) d")
        )
        pred = rearrange(pred, "(b n) d -> b n d", b=B)

        # 5. Compute losses (LeWM style)
        # Prediction loss: MSE only on masked positions
        mask_expanded = mask.unsqueeze(-1).float()
        num_masked = mask_expanded.sum().clamp(min=1.0)
        pred_loss = ((pred - target).pow(2) * mask_expanded).sum() / num_masked

        # SIGReg: on projected embeddings (keeps projector in gradient flow)
        sigreg_loss = self.sigreg(projected.transpose(0, 1))

        # Query specialization loss
        query_loss = self.channel_mixer.query_specialization_loss(attn_weights)

        # Total loss
        loss = (
            pred_loss
            + self.lambda_sigreg * sigreg_loss
            + self.lambda_query * query_loss
        )

        # Compute diagnostics (detached, no grad overhead)
        with torch.no_grad():
            # SIGReg internals: per-dimension variance and mean correlation
            flat = projected.detach().reshape(-1, projected.shape[-1])  # [B*N, D]
            emb_var = flat.var(dim=0)  # [D]
            emb_mean_var = emb_var.mean()
            emb_min_var = emb_var.min()
            # Correlation: off-diagonal of correlation matrix
            centered = flat - flat.mean(dim=0, keepdim=True)
            std = centered.std(dim=0, keepdim=True).clamp(min=1e-8)
            normed = centered / std
            corr = (normed.T @ normed) / normed.shape[0]  # [D, D]
            off_diag = corr - torch.eye(corr.shape[0], device=corr.device)
            mean_abs_corr = off_diag.abs().mean()

            # Actual mask ratio
            actual_mask_ratio = mask.float().mean()

        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "query_loss": query_loss,
            # Diagnostics
            "emb_mean_var": emb_mean_var,
            "emb_min_var": emb_min_var,
            "mean_abs_corr": mean_abs_corr,
            "actual_mask_ratio": actual_mask_ratio,
        }
