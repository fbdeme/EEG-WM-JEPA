"""Downstream evaluation heads for pretrained EEG-JEPA encoder.

Supports:
  - Linear probing (encoder frozen)
  - Fine-tuning (encoder trainable)
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Classification head on top of pretrained EEG-JEPA encoder.

    Takes the encoder's output sequence [B, N, D], pools it,
    and classifies through a linear layer.

    Args:
        encoder_dim: dimension of encoder output (D_enc)
        num_classes: number of target classes
    """

    def __init__(self, encoder_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.head = nn.Linear(encoder_dim, num_classes)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: [B, N, D] encoder output
        Returns:
            [B, num_classes] logits
        """
        # Average pooling over temporal patches
        pooled = encoded.mean(dim=1)  # [B, D]
        return self.head(pooled)


class DownstreamModel(nn.Module):
    """Wraps pretrained EEG-JEPA with a classification head.

    Args:
        eeg_jepa: pretrained EEGJEPA model
        num_classes: number of target classes
        freeze_encoder: if True, freeze all pretrained weights (linear probing)
    """

    def __init__(self, eeg_jepa, num_classes: int = 2, freeze_encoder: bool = True):
        super().__init__()
        self.eeg_jepa = eeg_jepa
        self.head = ClassificationHead(
            encoder_dim=eeg_jepa.encoder.output_proj.out_features
            if hasattr(eeg_jepa.encoder.output_proj, "out_features")
            else eeg_jepa.encoder.norm.normalized_shape[0],
            num_classes=num_classes,
        )

        if freeze_encoder:
            for param in self.eeg_jepa.parameters():
                param.requires_grad = False

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with 'eeg' and 'coords'
        Returns:
            [B, num_classes] logits
        """
        with torch.no_grad() if not self.training or not any(
            p.requires_grad for p in self.eeg_jepa.parameters()
        ) else torch.enable_grad():
            encoded, _ = self.eeg_jepa.encode(batch["eeg"], batch["coords"])

        return self.head(encoded)
