"""Forward pass verification test.

Tests the entire pipeline with dummy data:
  Stage 1: EEG standardization
  Stage 2: EEG-JEPA forward pass + loss computation
  Stage 3: Downstream classification head
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import yaml

from src.preprocessing.standardize import standardize_eeg
from src.preprocessing.dataset import make_dummy_dataset, EEGDataset
from src.model.eeg_jepa import EEGJEPA
from src.evaluation.downstream import DownstreamModel


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def test_stage1_preprocessing():
    """Test EEG standardization pipeline."""
    print("=" * 60)
    print("Stage 1: EEG Standardization")
    print("=" * 60)

    # Simulate raw EEG: 19 channels, 10 seconds at 500Hz
    raw = np.random.randn(19, 5000).astype(np.float32)
    print(f"  Input:  {raw.shape} (19ch, 10s @ 500Hz)")

    windows = standardize_eeg(raw, orig_srate=500, target_srate=200, window_sec=2.0)
    print(f"  Output: {windows.shape} (windows, channels, samples)")
    print(f"  → {windows.shape[0]} windows of 2s @ 200Hz")

    assert windows.shape[1] == 19
    assert windows.shape[2] == 400  # 2s * 200Hz
    print("  ✓ Stage 1 passed\n")
    return windows


def test_stage2_pretraining(cfg):
    """Test EEG-JEPA forward pass with loss computation."""
    print("=" * 60)
    print("Stage 2: EEG-JEPA Pretraining (forward pass)")
    print("=" * 60)

    model = EEGJEPA(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Create dummy batch
    dataset = make_dummy_dataset(num_samples=8, num_channels=19, window_samples=400)
    batch = {
        "eeg": dataset.windows[:8],
        "coords": dataset.coords,
    }
    print(f"  Input batch: eeg={batch['eeg'].shape}, coords={batch['coords'].shape}")

    # Forward pass
    output = model(batch)
    print(f"  Loss:        {output['loss'].item():.4f}")
    print(f"  Pred loss:   {output['pred_loss'].item():.4f}")
    print(f"  SIGReg loss: {output['sigreg_loss'].item():.4f}")
    print(f"  Query loss:  {output['query_loss'].item():.4f}")

    # Test backward
    output["loss"].backward()
    grad_ok = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    print(f"  Backward pass: {'✓' if grad_ok else '✗'}")
    print("  ✓ Stage 2 passed\n")
    return model


def test_stage3_downstream(model, cfg):
    """Test downstream classification head."""
    print("=" * 60)
    print("Stage 3: Downstream Classification")
    print("=" * 60)

    num_classes = cfg["downstream"]["num_classes"]

    # Linear probing (frozen encoder)
    downstream = DownstreamModel(model, num_classes=num_classes, freeze_encoder=True)
    frozen_params = sum(p.numel() for p in downstream.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in downstream.parameters())
    print(f"  Linear probing: {frozen_params:,} trainable / {total_params:,} total")

    dataset = make_dummy_dataset(num_samples=8, num_channels=19, window_samples=400)
    batch = {
        "eeg": dataset.windows[:8],
        "coords": dataset.coords,
    }

    downstream.eval()
    with torch.no_grad():
        logits = downstream(batch)
    print(f"  Output logits: {logits.shape}")
    assert logits.shape == (8, num_classes)

    # Compute classification loss
    labels = dataset.labels[:8]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    print(f"  CE Loss: {loss.item():.4f}")
    print("  ✓ Stage 3 passed\n")


def test_variable_channels(cfg):
    """Test that the model handles different channel counts."""
    print("=" * 60)
    print("Variable Channel Test")
    print("=" * 60)

    model = EEGJEPA(cfg)
    model.eval()

    for num_ch in [8, 19, 32, 64]:
        dataset = make_dummy_dataset(
            num_samples=4, num_channels=num_ch, window_samples=400
        )
        batch = {"eeg": dataset.windows[:4], "coords": dataset.coords}
        with torch.no_grad():
            output = model(batch)
        print(f"  {num_ch:3d} channels → loss={output['loss'].item():.4f} ✓")

    print("  ✓ Variable channel test passed\n")


if __name__ == "__main__":
    cfg = load_config()
    test_stage1_preprocessing()
    model = test_stage2_pretraining(cfg)
    test_stage3_downstream(model, cfg)
    test_variable_channels(cfg)
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
