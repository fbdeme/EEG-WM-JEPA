"""EEG Dataset with electrode coordinate support.

Wraps standardized EEG windows into PyTorch Dataset,
including 3D electrode coordinates for the Channel Mixer.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# Standard 10-20 system 19-channel coordinates (x, y, z on unit sphere)
# Used as fallback when dataset doesn't provide coordinates
STANDARD_1020_COORDS = {
    "Fp1": (-0.31, 0.95, 0.03), "Fp2": (0.31, 0.95, 0.03),
    "F7": (-0.81, 0.59, 0.03), "F3": (-0.55, 0.67, 0.47),
    "Fz": (0.00, 0.72, 0.69), "F4": (0.55, 0.67, 0.47),
    "F8": (0.81, 0.59, 0.03),
    "T3": (-1.00, 0.00, 0.03), "C3": (-0.71, 0.00, 0.71),
    "Cz": (0.00, 0.00, 1.00), "C4": (0.71, 0.00, 0.71),
    "T4": (1.00, 0.00, 0.03),
    "T5": (-0.81, -0.59, 0.03), "P3": (-0.55, -0.67, 0.47),
    "Pz": (0.00, -0.72, 0.69), "P4": (0.55, -0.67, 0.47),
    "T6": (0.81, -0.59, 0.03),
    "O1": (-0.31, -0.95, 0.03), "O2": (0.31, -0.95, 0.03),
}


class EEGDataset(Dataset):
    """Dataset for standardized EEG windows.

    Args:
        windows: [num_windows, C, T] numpy array of standardized EEG
        coords: [C, 3] numpy array of electrode 3D coordinates
        labels: optional [num_windows] array of labels for downstream tasks
    """

    def __init__(
        self,
        windows: np.ndarray,
        coords: np.ndarray,
        labels: np.ndarray | None = None,
    ):
        self.windows = torch.from_numpy(windows).float()
        self.coords = torch.from_numpy(coords).float()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "eeg": self.windows[idx],       # [C, T]
            "coords": self.coords,           # [C, 3] (shared across samples)
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def make_dummy_dataset(
    num_samples: int = 100,
    num_channels: int = 19,
    window_samples: int = 400,
    num_classes: int = 2,
) -> EEGDataset:
    """Create a dummy dataset for testing.

    Args:
        num_samples: number of windows
        num_channels: number of EEG channels
        window_samples: samples per window (e.g. 400 = 2s at 200Hz)
        num_classes: number of classes for labels
    Returns:
        EEGDataset with random data
    """
    windows = np.random.randn(num_samples, num_channels, window_samples).astype(
        np.float32
    )
    # Use first num_channels entries from standard 10-20 as coords
    coord_list = list(STANDARD_1020_COORDS.values())[:num_channels]
    if num_channels > len(coord_list):
        # Pad with random coords for extra channels
        extra = num_channels - len(coord_list)
        coord_list.extend(
            [tuple(np.random.randn(3)) for _ in range(extra)]
        )
    coords = np.array(coord_list, dtype=np.float32)
    labels = np.random.randint(0, num_classes, size=num_samples)
    return EEGDataset(windows, coords, labels)
