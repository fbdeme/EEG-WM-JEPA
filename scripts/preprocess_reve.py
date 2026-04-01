"""Offline preprocessing: REVE raw memmap → HuggingFace parquet.

Pipeline per recording:
  1. Download .npy + positions from brain-bzh/reve-dataset
  2. Split into sessions (skip flagged ones)
  3. Remove flagged channels per session
  4. Apply CAR (Common Average Reference)
  5. Apply bandpass filter (0.5–75Hz)
  6. Segment into 2-second windows (400 samples at 200Hz)
  7. Yield windows as dicts → push to HF dataset repo as parquet shards

Usage:
  uv run python scripts/preprocess_reve.py \
    --hf-repo your-username/reve-preprocessed \
    --start-idx 0 --end-idx 10  # process recordings 0..9

Disk usage is minimal: one recording at a time, deleted after upload.
"""

import argparse
import gc
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.reve_loader import (
    get_available_indices,
    load_metadata,
    load_recording_eeg,
    load_recording_positions,
    get_session_boundaries,
)
from src.preprocessing.standardize import (
    common_average_reference,
    bandpass_filter,
    segment_windows,
)

import shutil

SRATE = 200  # REVE data is already at 200Hz
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(SRATE * WINDOW_SEC)  # 400
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 75.0
MIN_CHANNELS = 7  # skip recordings with fewer usable channels
CLIP_VALUE = 15.0  # clip extreme values (same as REVE)
SHARD_SIZE = 5000  # windows per parquet shard


def _clear_hf_cache(big_idx: int):
    """Remove cached .npy and position files for a recording."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    for repo_dir in cache_dir.iterdir():
        if not repo_dir.is_dir():
            continue
        blobs = repo_dir / "blobs"
        snapshots = repo_dir / "snapshots"
        # Delete snapshot symlinks and blobs matching this recording
        if snapshots.exists():
            for snap in snapshots.iterdir():
                for f in snap.rglob(f"*{big_idx}*"):
                    # Resolve symlink to blob, delete both
                    if f.is_symlink():
                        blob = f.resolve()
                        blob.unlink(missing_ok=True)
                    f.unlink(missing_ok=True)


def process_session(
    eeg_data: np.ndarray,
    positions: np.ndarray,
    session: dict,
) -> list[dict]:
    """Process a single session into standardized windows.

    Args:
        eeg_data: [duration, n_chans] full recording memmap
        positions: [n_chans, 3] electrode positions
        session: dict with start, end, channels_to_remove, flag_remove

    Returns:
        List of window dicts ready for parquet
    """
    if session["flag_remove"]:
        return []

    # Extract session slice: [session_dur, n_chans]
    segment = np.array(
        eeg_data[session["start"]:session["end"]], dtype=np.float32
    )
    pos = positions.copy()

    # Remove flagged channels
    channels_to_remove = session["channels_to_remove"]
    if channels_to_remove:
        keep_mask = np.ones(segment.shape[1], dtype=bool)
        for ch_idx in channels_to_remove:
            if ch_idx < len(keep_mask):
                keep_mask[ch_idx] = False
        segment = segment[:, keep_mask]
        pos = pos[keep_mask]

    n_chans = segment.shape[1]
    if n_chans < MIN_CHANNELS:
        return []

    # Transpose to [C, T] for our preprocessing functions
    signal = segment.T  # [C, T]

    # CAR (before any padding, as designed)
    signal = common_average_reference(signal)

    # Bandpass filter
    if signal.shape[1] >= 20:  # need minimum samples for filter
        try:
            signal = bandpass_filter(
                signal, SRATE, BANDPASS_LOW, BANDPASS_HIGH
            )
        except ValueError:
            return []

    # Clip extreme values
    signal = np.clip(signal, -CLIP_VALUE, CLIP_VALUE)

    # Segment into 2-second windows: [num_windows, C, 400]
    try:
        windows = segment_windows(signal, SRATE, WINDOW_SEC)
    except ValueError:
        return []

    # Build output dicts
    results = []
    coords_flat = pos.astype(np.float32).tobytes()
    for i in range(windows.shape[0]):
        results.append({
            "eeg": windows[i].astype(np.float32).tobytes(),
            "coords": coords_flat,
            "num_channels": n_chans,
            "window_samples": WINDOW_SAMPLES,
        })

    return results


def process_recording(
    big_idx: int,
    duration: int,
    n_chans: int,
    df_corrected,
) -> list[dict]:
    """Process an entire recording into windows.

    Args:
        big_idx: recording index
        duration: total samples
        n_chans: channel count
        df_corrected: session metadata

    Returns:
        List of window dicts
    """
    try:
        eeg_data = load_recording_eeg(big_idx, duration, n_chans)
        positions = load_recording_positions(big_idx)
    except Exception as e:
        print(f"  [SKIP] Recording {big_idx}: download failed - {e}")
        return []

    # Verify positions match channels
    if positions.shape[0] != n_chans:
        print(
            f"  [SKIP] Recording {big_idx}: position mismatch "
            f"({positions.shape[0]} vs {n_chans})"
        )
        return []

    sessions = get_session_boundaries(df_corrected, big_idx)
    all_windows = []
    for sess in sessions:
        windows = process_session(eeg_data, positions, sess)
        all_windows.extend(windows)

    return all_windows


def make_parquet_table(windows: list[dict]) -> pa.Table:
    """Convert window dicts to a PyArrow table."""
    return pa.table({
        "eeg": pa.array(
            [w["eeg"] for w in windows], type=pa.binary()
        ),
        "coords": pa.array(
            [w["coords"] for w in windows], type=pa.binary()
        ),
        "num_channels": pa.array(
            [w["num_channels"] for w in windows], type=pa.int32()
        ),
        "window_samples": pa.array(
            [w["window_samples"] for w in windows], type=pa.int32()
        ),
    })


def upload_shard(
    table: pa.Table,
    shard_idx: int,
    hf_repo: str,
    api: HfApi,
    tmp_dir: Path,
):
    """Write a parquet shard locally and upload to HF."""
    filename = f"train/shard_{shard_idx:06d}.parquet"
    local_path = tmp_dir / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, local_path)

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=f"data/{filename}",
        repo_id=hf_repo,
        repo_type="dataset",
    )
    local_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess REVE dataset for EEG-JEPA"
    )
    parser.add_argument(
        "--hf-repo",
        required=True,
        help="HuggingFace dataset repo (e.g. username/reve-preprocessed)",
    )
    parser.add_argument(
        "--start-idx", type=int, default=0,
        help="Start index in sorted recording list",
    )
    parser.add_argument(
        "--end-idx", type=int, default=None,
        help="End index (exclusive) in sorted recording list",
    )
    parser.add_argument(
        "--shard-size", type=int, default=SHARD_SIZE,
        help="Windows per parquet shard",
    )
    parser.add_argument(
        "--tmp-dir", type=str, default="/tmp/reve_preprocess",
        help="Temporary directory for parquet files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process but don't upload (save shards locally)",
    )
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    # Create repo if needed
    if not args.dry_run:
        api.create_repo(
            args.hf_repo, repo_type="dataset", private=True, exist_ok=True
        )

    # Load metadata
    print("Loading REVE metadata...")
    df_big, df_corrected, df_stats = load_metadata()

    # Filter to only recordings that actually exist on HF
    print("Checking available files on HF...")
    available = get_available_indices()
    df_big = df_big[
        df_big["big_recording_index"].isin(available)
    ].sort_values("big_recording_index").reset_index(drop=True)
    print(f"Available recordings: {len(df_big)} / {len(available)}")

    end_idx = args.end_idx or len(df_big)
    recordings = df_big.iloc[args.start_idx:end_idx]

    print(
        f"Processing recordings {args.start_idx}..{end_idx} "
        f"({len(recordings)} recordings)"
    )

    # Process recordings, accumulate windows, flush as shards
    buffer = []
    shard_idx = args.start_idx * 100  # offset shard numbering to avoid collisions
    total_windows = 0

    for _, row in tqdm(
        recordings.iterrows(), total=len(recordings), desc="Recordings"
    ):
        big_idx = int(row["big_recording_index"])
        duration = int(row["duration"])
        n_chans = int(row["n_chans"])

        file_size_gb = (duration * n_chans * 4) / (1024**3)
        print(
            f"\n  Recording {big_idx}: {n_chans}ch, "
            f"{duration/SRATE/3600:.1f}h, {file_size_gb:.1f}GB"
        )

        windows = process_recording(big_idx, duration, n_chans, df_corrected)
        print(f"  → {len(windows)} windows")
        buffer.extend(windows)
        total_windows += len(windows)

        # Flush when buffer is large enough
        while len(buffer) >= args.shard_size:
            shard_data = buffer[:args.shard_size]
            buffer = buffer[args.shard_size:]
            table = make_parquet_table(shard_data)

            if args.dry_run:
                local_path = tmp_dir / f"shard_{shard_idx:06d}.parquet"
                pq.write_table(table, local_path)
                print(f"  [DRY] Saved {local_path} ({len(shard_data)} windows)")
            else:
                upload_shard(table, shard_idx, args.hf_repo, api, tmp_dir)
                print(f"  [UP] Shard {shard_idx} ({len(shard_data)} windows)")

            shard_idx += 1
            del table, shard_data
            gc.collect()

        # Clear HF cache for this recording to save disk
        _clear_hf_cache(big_idx)
        gc.collect()

    # Flush remaining buffer
    if buffer:
        table = make_parquet_table(buffer)
        if args.dry_run:
            local_path = tmp_dir / f"shard_{shard_idx:06d}.parquet"
            pq.write_table(table, local_path)
            print(f"  [DRY] Saved {local_path} ({len(buffer)} windows)")
        else:
            upload_shard(table, shard_idx, args.hf_repo, api, tmp_dir)
            print(f"  [UP] Shard {shard_idx} ({len(buffer)} windows)")
        shard_idx += 1

    print(f"\nDone! Total windows: {total_windows}, Shards: {shard_idx}")


if __name__ == "__main__":
    main()
