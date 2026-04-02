"""Parallel offline preprocessing: REVE raw memmap → HuggingFace parquet.

Uses multiprocessing to process multiple recordings simultaneously.
Each worker: download → preprocess → save shards locally.
Main process: periodically batch-uploads accumulated shards to HF.

Usage:
  uv run python scripts/preprocess_parallel.py \
    --hf-repo your-username/reve-preprocessed \
    --start-idx 55 --end-idx 322 --workers 4
"""

import argparse
import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

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

SRATE = 200
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(SRATE * WINDOW_SEC)
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 75.0
MIN_CHANNELS = 7
CLIP_VALUE = 15.0
SHARD_SIZE = 5000


def process_session(eeg_data, positions, session):
    """Process a single session into standardized windows."""
    if session["flag_remove"]:
        return []

    segment = np.array(
        eeg_data[session["start"]:session["end"]], dtype=np.float32
    )
    pos = positions.copy()

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

    signal = segment.T

    signal = common_average_reference(signal)

    if signal.shape[1] >= 20:
        try:
            signal = bandpass_filter(signal, SRATE, BANDPASS_LOW, BANDPASS_HIGH)
        except ValueError:
            return []

    signal = np.clip(signal, -CLIP_VALUE, CLIP_VALUE)

    try:
        windows = segment_windows(signal, SRATE, WINDOW_SEC)
    except ValueError:
        return []

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


def process_single_recording(args_tuple):
    """Worker function: process one recording and save shards locally.

    Returns (big_idx, num_windows, num_shards, error_msg)
    """
    big_idx, duration, n_chans, df_corrected_subset, out_dir, shard_offset = args_tuple

    try:
        eeg_data = load_recording_eeg(big_idx, duration, n_chans)
        positions = load_recording_positions(big_idx)
    except Exception as e:
        return (big_idx, 0, 0, f"download failed: {e}")

    if positions.shape[0] != n_chans:
        return (big_idx, 0, 0, f"position mismatch ({positions.shape[0]} vs {n_chans})")

    sessions = get_session_boundaries(df_corrected_subset, big_idx)

    all_windows = []
    for sess in sessions:
        windows = process_session(eeg_data, positions, sess)
        all_windows.extend(windows)

    if not all_windows:
        return (big_idx, 0, 0, None)

    # Write shards locally
    shard_idx = shard_offset
    num_shards = 0
    buffer = all_windows

    while len(buffer) >= SHARD_SIZE:
        shard_data = buffer[:SHARD_SIZE]
        buffer = buffer[SHARD_SIZE:]
        table = pa.table({
            "eeg": pa.array([w["eeg"] for w in shard_data], type=pa.binary()),
            "coords": pa.array([w["coords"] for w in shard_data], type=pa.binary()),
            "num_channels": pa.array([w["num_channels"] for w in shard_data], type=pa.int32()),
            "window_samples": pa.array([w["window_samples"] for w in shard_data], type=pa.int32()),
        })
        local_path = out_dir / f"train/shard_{shard_idx:06d}.parquet"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, local_path)
        shard_idx += 1
        num_shards += 1
        del table, shard_data

    # Remaining
    if buffer:
        table = pa.table({
            "eeg": pa.array([w["eeg"] for w in buffer], type=pa.binary()),
            "coords": pa.array([w["coords"] for w in buffer], type=pa.binary()),
            "num_channels": pa.array([w["num_channels"] for w in buffer], type=pa.int32()),
            "window_samples": pa.array([w["window_samples"] for w in buffer], type=pa.int32()),
        })
        local_path = out_dir / f"train/shard_{shard_idx:06d}.parquet"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, local_path)
        num_shards += 1
        del table

    # Clear HF cache for this recording
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        for repo_dir in cache_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            snapshots = repo_dir / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    for f in snap.rglob(f"*{big_idx}*"):
                        if f.is_symlink():
                            blob = f.resolve()
                            blob.unlink(missing_ok=True)
                        f.unlink(missing_ok=True)

    del all_windows, buffer
    gc.collect()

    return (big_idx, len(all_windows) if not all_windows else sum(1 for _ in []), 0, None)


def process_single_recording_v2(args_tuple):
    """Simplified: returns (big_idx, num_windows, num_shards, error)."""
    big_idx, duration, n_chans, df_corrected, out_dir, shard_offset = args_tuple

    try:
        eeg_data = load_recording_eeg(big_idx, duration, n_chans)
        positions = load_recording_positions(big_idx)
    except Exception as e:
        return (big_idx, 0, 0, f"download failed: {e}")

    if positions.shape[0] != n_chans:
        return (big_idx, 0, 0, f"position mismatch")

    sessions = get_session_boundaries(df_corrected, big_idx)

    total_windows = 0
    shard_idx = shard_offset
    num_shards = 0
    buffer = []

    for sess in sessions:
        windows = process_session(eeg_data, positions, sess)
        buffer.extend(windows)

        # Flush incrementally (don't accumulate entire recording)
        while len(buffer) >= SHARD_SIZE:
            shard_data = buffer[:SHARD_SIZE]
            buffer = buffer[SHARD_SIZE:]
            table = pa.table({
                "eeg": pa.array([w["eeg"] for w in shard_data], type=pa.binary()),
                "coords": pa.array([w["coords"] for w in shard_data], type=pa.binary()),
                "num_channels": pa.array([w["num_channels"] for w in shard_data], type=pa.int32()),
                "window_samples": pa.array([w["window_samples"] for w in shard_data], type=pa.int32()),
            })
            local_path = out_dir / f"train/shard_{shard_idx:06d}.parquet"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, local_path)
            total_windows += len(shard_data)
            shard_idx += 1
            num_shards += 1
            del table, shard_data

    # Remaining
    if buffer:
        table = pa.table({
            "eeg": pa.array([w["eeg"] for w in buffer], type=pa.binary()),
            "coords": pa.array([w["coords"] for w in buffer], type=pa.binary()),
            "num_channels": pa.array([w["num_channels"] for w in buffer], type=pa.int32()),
            "window_samples": pa.array([w["window_samples"] for w in buffer], type=pa.int32()),
        })
        local_path = out_dir / f"train/shard_{shard_idx:06d}.parquet"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, local_path)
        total_windows += len(buffer)
        num_shards += 1
        del table

    # Clear HF cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        for repo_dir in cache_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            snapshots = repo_dir / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    for f in snap.rglob(f"*{big_idx}*"):
                        if f.is_symlink():
                            blob = f.resolve()
                            blob.unlink(missing_ok=True)
                        f.unlink(missing_ok=True)

    gc.collect()
    return (big_idx, total_windows, num_shards, None)


def batch_upload(tmp_dir: Path, hf_repo: str, max_retries: int = 3):
    """Upload all local parquet files as a single commit."""
    from huggingface_hub import HfApi
    api = HfApi()

    parquet_files = sorted(tmp_dir.rglob("*.parquet"))
    if not parquet_files:
        return 0

    n_files = len(parquet_files)
    total_size = sum(f.stat().st_size for f in parquet_files)

    for attempt in range(max_retries):
        try:
            api.upload_folder(
                folder_path=str(tmp_dir),
                path_in_repo="data",
                repo_id=hf_repo,
                repo_type="dataset",
                allow_patterns="**/*.parquet",
            )
            for f in parquet_files:
                f.unlink()
            return n_files
        except Exception as e:
            print(f"  [RETRY {attempt+1}/{max_retries}] Upload: {e}")
            time.sleep(10 * (attempt + 1))

    print(f"  [FAIL] Upload failed after {max_retries} retries")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Parallel REVE preprocessing"
    )
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--upload-every", type=int, default=20,
                        help="Upload after this many shards accumulate")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/reve_preprocess")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi
    api = HfApi()

    if not args.dry_run:
        api.create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)

    print("Loading REVE metadata...")
    df_big, df_corrected, _ = load_metadata()

    print("Checking available files on HF...")
    available = get_available_indices()
    df_big = df_big[
        df_big["big_recording_index"].isin(available)
    ].sort_values("big_recording_index").reset_index(drop=True)

    end_idx = args.end_idx or len(df_big)
    recordings = df_big.iloc[args.start_idx:end_idx]

    print(f"Processing {len(recordings)} recordings with {args.workers} workers")
    print(f"Upload every {args.upload_every} shards")
    print()

    # Prepare tasks with unique shard offsets
    tasks = []
    shard_offset = args.start_idx * 100
    for i, (_, row) in enumerate(recordings.iterrows()):
        big_idx = int(row["big_recording_index"])
        duration = int(row["duration"])
        n_chans = int(row["n_chans"])
        # Each recording gets a unique shard range (max 200 shards per recording)
        rec_offset = shard_offset + i * 200
        tasks.append((big_idx, duration, n_chans, df_corrected, tmp_dir, rec_offset))

    total_windows = 0
    total_shards = 0
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_recording_v2, task): task[0]
            for task in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Recordings"):
            big_idx = futures[future]
            try:
                rec_idx, n_windows, n_shards, error = future.result()
                if error:
                    print(f"  [SKIP] Recording {rec_idx}: {error}")
                    failed += 1
                else:
                    total_windows += n_windows
                    total_shards += n_shards
                    completed += 1
                    if n_windows > 0:
                        print(f"  [OK] Recording {rec_idx}: {n_windows} windows, {n_shards} shards")
            except Exception as e:
                print(f"  [ERROR] Recording {big_idx}: {e}")
                failed += 1

            # Periodic batch upload
            local_shards = list(tmp_dir.rglob("*.parquet"))
            if not args.dry_run and len(local_shards) >= args.upload_every:
                size_gb = sum(f.stat().st_size for f in local_shards) / 1e9
                print(f"\n  [UPLOAD] {len(local_shards)} shards ({size_gb:.1f}GB)...")
                uploaded = batch_upload(tmp_dir, args.hf_repo)
                print(f"  [UPLOAD] {uploaded} shards uploaded")

    # Final upload
    local_shards = list(tmp_dir.rglob("*.parquet"))
    if not args.dry_run and local_shards:
        size_gb = sum(f.stat().st_size for f in local_shards) / 1e9
        print(f"\n  [UPLOAD] Final: {len(local_shards)} shards ({size_gb:.1f}GB)...")
        uploaded = batch_upload(tmp_dir, args.hf_repo)
        print(f"  [UPLOAD] {uploaded} shards uploaded")

    print(f"\nDone! Completed: {completed}, Failed: {failed}")
    print(f"Total windows: {total_windows}, Total shards: {total_shards}")


if __name__ == "__main__":
    main()
