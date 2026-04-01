"""REVE dataset loader utilities.

Handles loading raw memmap .npy files from the brain-bzh/reve-dataset
HuggingFace repository, including metadata CSVs and position files.
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

REVE_REPO = "brain-bzh/reve-dataset"


def get_available_indices() -> set[int]:
    """Get the set of recording indices that actually exist on HF.

    Some recordings in df_big.csv are missing from the repo
    (license restrictions prevent redistribution).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_tree(
        REVE_REPO, repo_type="dataset", path_in_repo="data"
    )
    return {
        int(f.path.split("_")[-1].replace(".npy", ""))
        for f in files
        if f.path.endswith(".npy")
    }


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download and parse all three REVE metadata CSVs.

    Returns:
        df_big: one row per recording (big_recording_index, duration, n_chans)
        df_corrected: one row per session (big_recording_index, duration, flag_remove, flag_reduce, n_chans_to_remove)
        df_stats: one row per recording (big_recording_index, n_sessions, n_chans)
    """
    df_big = pd.read_csv(
        hf_hub_download(REVE_REPO, "df_big.csv", repo_type="dataset")
    )
    df_corrected = pd.read_csv(
        hf_hub_download(REVE_REPO, "df_corrected.csv", repo_type="dataset")
    )
    df_stats = pd.read_csv(
        hf_hub_download(REVE_REPO, "df_stats.csv", repo_type="dataset")
    )
    return df_big, df_corrected, df_stats


def load_recording_eeg(
    big_idx: int, duration: int, n_chans: int
) -> np.ndarray:
    """Download and load a single EEG recording as memmap.

    The REVE data files are raw float32 without numpy headers.

    Args:
        big_idx: big_recording_index from df_big
        duration: total time samples
        n_chans: number of channels

    Returns:
        [duration, n_chans] float32 array
    """
    path = hf_hub_download(
        REVE_REPO,
        f"data/recording_-_eeg_-_{big_idx}.npy",
        repo_type="dataset",
    )
    return np.memmap(path, dtype="float32", mode="r", shape=(duration, n_chans))


def load_recording_positions(big_idx: int) -> np.ndarray:
    """Download and load electrode positions for a recording.

    Position files are standard .npy format.

    Args:
        big_idx: big_recording_index

    Returns:
        [n_chans, 3] float32 array of 3D coordinates
    """
    path = hf_hub_download(
        REVE_REPO,
        f"positions/recording_-_positions_-_{big_idx}.npy",
        repo_type="dataset",
    )
    return np.load(path)


def load_recording_stats(
    big_idx: int, n_sessions: int, n_chans: int
) -> np.ndarray:
    """Download and load per-session normalization stats.

    Stats files are raw float32 memmap with shape (n_sessions, 2, n_chans).
    Index 0 = mean, index 1 = std.

    Args:
        big_idx: big_recording_index
        n_sessions: number of sessions
        n_chans: number of channels

    Returns:
        [n_sessions, 2, n_chans] float32 array
    """
    path = hf_hub_download(
        REVE_REPO,
        f"stats/recording_-_stats_-_{big_idx}.npy",
        repo_type="dataset",
    )
    return np.memmap(
        path, dtype="float32", mode="r", shape=(n_sessions, 2, n_chans)
    )


def get_session_boundaries(
    df_corrected: pd.DataFrame, big_idx: int
) -> list[dict]:
    """Get session boundaries within a recording.

    Sessions are concatenated along the time axis in the .npy file.
    Each session's duration tells us where it starts and ends.

    Args:
        df_corrected: session-level metadata
        big_idx: big_recording_index

    Returns:
        List of dicts with 'start', 'end', 'duration', 'flag_remove',
        'channels_to_remove' (list of int indices)
    """
    sessions_df = df_corrected[
        df_corrected["big_recording_index"] == big_idx
    ].reset_index(drop=True)

    sessions = []
    offset = 0
    for _, row in sessions_df.iterrows():
        dur = int(row["duration"])
        # Parse flag_reduce: can be 'False' or a list like '[16]' or '[3, 5]'
        flag_reduce = row["flag_reduce"]
        if isinstance(flag_reduce, str) and flag_reduce != "False":
            channels_to_remove = ast.literal_eval(flag_reduce)
            if isinstance(channels_to_remove, int):
                channels_to_remove = [channels_to_remove]
        else:
            channels_to_remove = []

        sessions.append({
            "start": offset,
            "end": offset + dur,
            "duration": dur,
            "flag_remove": bool(row["flag_remove"]),
            "channels_to_remove": channels_to_remove,
        })
        offset += dur

    return sessions
