"""Spatiotemporal Block Masking for JEPA pretraining.

Generates contiguous temporal block masks (60%, 500ms~1s blocks).
"""

import torch


def generate_block_mask(
    num_patches: int,
    mask_ratio: float = 0.6,
    block_min: int = 2,
    block_max: int = 5,
) -> torch.Tensor:
    """Generate a 1D block mask for temporal patches.

    Args:
        num_patches: total number of temporal patches (N)
        mask_ratio: target fraction to mask
        block_min: min block size in patches
        block_max: max block size in patches
    Returns:
        [N] boolean tensor (True = masked/target)
    """
    target_masked = int(num_patches * mask_ratio)
    mask = torch.zeros(num_patches, dtype=torch.bool)
    masked_count = 0

    while masked_count < target_masked:
        block_size = torch.randint(block_min, block_max + 1, (1,)).item()
        start = torch.randint(0, num_patches, (1,)).item()
        end = min(start + block_size, num_patches)
        new_masked = (~mask[start:end]).sum().item()
        mask[start:end] = True
        masked_count += new_masked

    return mask


def generate_batch_masks(
    batch_size: int,
    num_patches: int,
    mask_ratio: float = 0.6,
    block_min: int = 2,
    block_max: int = 5,
) -> torch.Tensor:
    """Generate independent masks for each sample in batch.

    Returns:
        [B, N] boolean tensor
    """
    return torch.stack([
        generate_block_mask(num_patches, mask_ratio, block_min, block_max)
        for _ in range(batch_size)
    ])
