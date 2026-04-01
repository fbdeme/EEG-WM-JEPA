"""Streaming EEG Dataset for HuggingFace-hosted preprocessed data.

Loads parquet shards from a HuggingFace dataset repo via streaming,
decodes binary EEG/coords, and applies CAR on-the-fly.
Provides a collate_fn that handles variable channel counts via padding.
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset


class StreamingEEGDataset(IterableDataset):
    """Streams preprocessed EEG windows from HuggingFace.

    Each sample is decoded from parquet binary fields into tensors.
    CAR is already applied during offline preprocessing.

    Args:
        hf_repo: HuggingFace dataset repo ID
        split: dataset split name
        shuffle_buffer: number of samples to buffer for shuffling
        seed: random seed for shuffling
    """

    def __init__(
        self,
        hf_repo: str,
        split: str = "train",
        shuffle_buffer: int = 10000,
        seed: int = 42,
    ):
        self.hf_repo = hf_repo
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _decode_sample(self, sample: dict) -> dict:
        """Decode a single sample from parquet binary to tensors."""
        n_ch = sample["num_channels"]
        ws = sample["window_samples"]

        eeg = np.frombuffer(sample["eeg"], dtype=np.float32).reshape(n_ch, ws)
        coords = np.frombuffer(
            sample["coords"], dtype=np.float32
        ).reshape(n_ch, 3)

        return {
            "eeg": torch.from_numpy(eeg.copy()),        # [C, T]
            "coords": torch.from_numpy(coords.copy()),  # [C, 3]
            "num_channels": n_ch,
        }

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset(
            self.hf_repo,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(
            buffer_size=self.shuffle_buffer, seed=self.seed
        )

        for sample in ds:
            yield self._decode_sample(sample)


def eeg_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-channel EEG samples with padding.

    Pads all samples to the maximum channel count in the batch,
    producing a padding_mask for the Channel Mixer.

    Args:
        batch: list of dicts from StreamingEEGDataset

    Returns:
        dict with:
            eeg: [B, max_C, T] zero-padded
            coords: [B, max_C, 3] zero-padded
            padding_mask: [B, max_C] True = padded (invalid) channel
            num_channels: [B] original channel counts
    """
    max_c = max(s["num_channels"] for s in batch)
    T = batch[0]["eeg"].shape[1]
    B = len(batch)

    eeg = torch.zeros(B, max_c, T)
    coords = torch.zeros(B, max_c, 3)
    padding_mask = torch.ones(B, max_c, dtype=torch.bool)  # True = padded
    num_channels = torch.zeros(B, dtype=torch.long)

    for i, sample in enumerate(batch):
        c = sample["num_channels"]
        eeg[i, :c] = sample["eeg"]
        coords[i, :c] = sample["coords"]
        padding_mask[i, :c] = False  # real channels
        num_channels[i] = c

    return {
        "eeg": eeg,
        "coords": coords,
        "padding_mask": padding_mask,
        "num_channels": num_channels,
    }
