"""MDS-based EEG Dataset using MosaicML Streaming.

Streams preprocessed EEG data from HuggingFace via MDS format.
Handles local caching with automatic LRU eviction for disk-constrained servers.

Usage:
    dataset = MDSEEGDataset(
        remote="hf://fbdeme/reve-mds/train",
        cache_limit="50gb",
    )
    loader = DataLoader(dataset, batch_size=512, collate_fn=eeg_collate_fn)
"""

import numpy as np
import torch
from streaming import StreamingDataset


class MDSEEGDataset(StreamingDataset):
    """Streaming EEG dataset backed by MDS shards on HuggingFace.

    Inherits from StreamingDataset which handles:
    - Background prefetching of remote shards
    - Local disk caching with LRU eviction (cache_limit)
    - Deterministic shuffling across epochs
    - Multi-worker DataLoader compatibility

    Args:
        remote: Remote path (e.g. "hf://fbdeme/reve-mds/train")
        local: Local cache directory
        cache_limit: Max local cache size (e.g. "50gb"). LRU eviction when exceeded.
        shuffle: Whether to shuffle samples
        shuffle_seed: Seed for deterministic shuffling
        batch_size: Batch size (helps streaming optimize prefetch)
        predownload: Number of samples to prefetch ahead
    """

    def __init__(
        self,
        remote: str,
        local: str = "/tmp/eeg_mds_cache",
        cache_limit: str | None = "50gb",
        shuffle: bool = True,
        shuffle_seed: int = 42,
        batch_size: int = 512,
        predownload: int | None = None,
    ):
        if predownload is None:
            predownload = batch_size * 8  # prefetch 8 batches ahead

        super().__init__(
            remote=remote,
            local=local,
            cache_limit=cache_limit,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            batch_size=batch_size,
            predownload=predownload,
        )

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)

        n_ch = sample["num_channels"]
        ws = sample["window_samples"]

        eeg = np.frombuffer(sample["eeg"], dtype=np.float32).reshape(n_ch, ws)
        coords = np.frombuffer(sample["coords"], dtype=np.float32).reshape(n_ch, 3)

        return {
            "eeg": torch.from_numpy(eeg.copy()),
            "coords": torch.from_numpy(coords.copy()),
            "num_channels": n_ch,
        }
