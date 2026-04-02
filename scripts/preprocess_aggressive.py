"""Aggressive parallel preprocessing: process → upload → delete immediately.

Designed for disk-constrained servers. Each recording is uploaded as soon as
it finishes processing, then local files are deleted immediately.

Upload runs in a background thread so processing never blocks on network I/O.

Usage:
  uv run python scripts/preprocess_aggressive.py \
    --hf-repo fbdeme/reve-preprocessed \
    --start-idx 81 --end-idx 322 --workers 8
"""

import argparse
import gc
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

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
    """Worker: process one recording, save shards locally, clean HF cache."""
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

    # Aggressively clear HF cache for this recording
    del eeg_data, positions
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        for repo_dir in cache_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            blobs = repo_dir / "blobs"
            if blobs.exists():
                for blob in blobs.iterdir():
                    try:
                        blob.unlink(missing_ok=True)
                    except Exception:
                        pass
            snapshots = repo_dir / "snapshots"
            if snapshots.exists():
                for snap in snapshots.iterdir():
                    for f in snap.rglob(f"*{big_idx}*"):
                        if f.is_symlink():
                            try:
                                blob = f.resolve()
                                blob.unlink(missing_ok=True)
                            except Exception:
                                pass
                        f.unlink(missing_ok=True)

    gc.collect()
    return (big_idx, total_windows, num_shards, None)


def upload_worker(upload_queue: Queue, hf_repo: str, tmp_dir: Path):
    """Background thread: upload shards as they become available."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()

    while True:
        item = upload_queue.get()
        if item is None:  # poison pill
            upload_queue.task_done()
            break

        files_to_upload = item
        if not files_to_upload:
            upload_queue.task_done()
            continue

        # Filter to files that still exist
        files_to_upload = [f for f in files_to_upload if f.exists()]
        if not files_to_upload:
            upload_queue.task_done()
            continue

        size_gb = sum(f.stat().st_size for f in files_to_upload) / 1e9
        print(f"  [UPLOAD] {len(files_to_upload)} shards ({size_gb:.1f}GB)...", flush=True)

        for attempt in range(3):
            try:
                ops = [
                    CommitOperationAdd(
                        path_in_repo=f"data/train/{f.name}",
                        path_or_fileobj=str(f),
                    )
                    for f in files_to_upload
                ]
                api.create_commit(
                    repo_id=hf_repo,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"Add {len(files_to_upload)} shards ({size_gb:.1f}GB)",
                )
                # Delete after successful upload
                for f in files_to_upload:
                    f.unlink(missing_ok=True)
                print(f"  [UPLOAD OK] {len(files_to_upload)} shards uploaded & deleted", flush=True)
                break
            except Exception as e:
                print(f"  [UPLOAD RETRY {attempt+1}/3] {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            print(f"  [UPLOAD FAIL] {len(files_to_upload)} shards kept locally", flush=True)

        upload_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Aggressive REVE preprocessing")
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--upload-every", type=int, default=5,
                        help="Upload after this many shards accumulate (default: 5 = ~5GB)")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/reve_preprocess")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    # Start fresh
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi
    api = HfApi()

    if not args.dry_run:
        api.create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)

    print("Loading REVE metadata...", flush=True)
    df_big, df_corrected, _ = load_metadata()

    print("Checking available files on HF...", flush=True)
    available = get_available_indices()
    df_big = df_big[
        df_big["big_recording_index"].isin(available)
    ].sort_values("big_recording_index").reset_index(drop=True)

    end_idx = args.end_idx or len(df_big)
    recordings = df_big.iloc[args.start_idx:end_idx]

    print(f"Processing {len(recordings)} recordings (idx {args.start_idx}→{end_idx})", flush=True)
    print(f"Workers: {args.workers}, Upload every: {args.upload_every} shards", flush=True)
    print(flush=True)

    # Start background upload thread
    upload_queue = Queue(maxsize=4)  # backpressure if uploads fall behind
    upload_thread = threading.Thread(
        target=upload_worker,
        args=(upload_queue, args.hf_repo, tmp_dir),
        daemon=True,
    )
    if not args.dry_run:
        upload_thread.start()

    # Prepare tasks
    tasks = []
    shard_offset = args.start_idx * 100
    for i, (_, row) in enumerate(recordings.iterrows()):
        big_idx = int(row["big_recording_index"])
        duration = int(row["duration"])
        n_chans = int(row["n_chans"])
        rec_offset = shard_offset + i * 200
        tasks.append((big_idx, duration, n_chans, df_corrected, tmp_dir, rec_offset))

    total_windows = 0
    total_shards = 0
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_recording, task): task[0]
            for task in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Recordings"):
            big_idx = futures[future]
            try:
                rec_idx, n_windows, n_shards, error = future.result()
                if error:
                    print(f"  [SKIP] Recording {rec_idx}: {error}", flush=True)
                    failed += 1
                else:
                    total_windows += n_windows
                    total_shards += n_shards
                    completed += 1
                    if n_windows > 0:
                        print(f"  [OK] Rec {rec_idx}: {n_windows} win, {n_shards} shards", flush=True)
            except Exception as e:
                print(f"  [ERROR] Recording {big_idx}: {e}", flush=True)
                failed += 1

            # Check for accumulated shards and queue upload
            local_shards = sorted(tmp_dir.rglob("*.parquet"))
            if not args.dry_run and len(local_shards) >= args.upload_every:
                upload_queue.put(list(local_shards))

    # Final upload of any remaining shards
    local_shards = sorted(tmp_dir.rglob("*.parquet"))
    if not args.dry_run and local_shards:
        upload_queue.put(list(local_shards))

    # Signal upload thread to stop and wait
    if not args.dry_run:
        upload_queue.put(None)
        upload_queue.join()

    print(f"\nDone! Completed: {completed}, Failed: {failed}", flush=True)
    print(f"Total windows: {total_windows}, Total shards: {total_shards}", flush=True)


if __name__ == "__main__":
    main()
