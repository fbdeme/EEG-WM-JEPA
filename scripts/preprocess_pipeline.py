"""Pipeline-based preprocessing with dedicated worker stages.

Stage 1 (Download, 2 threads): Download raw EEG → pass to preprocessing queue
Stage 2 (Preprocess, 4 processes): Process EEG → write shards locally
Stage 3 (Upload, 1 thread): Upload shards to HF → delete local files
Cache cleanup happens immediately after download is consumed.

Usage:
  uv run python scripts/preprocess_pipeline.py \
    --hf-repo fbdeme/reve-preprocessed \
    --start-idx 81 --end-idx 322
"""

import argparse
import gc
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

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


# ─── Stage 1: Download ───────────────────────────────────────────────
def download_recording(big_idx, duration, n_chans):
    """Download a single recording and return raw numpy arrays."""
    eeg_data = load_recording_eeg(big_idx, duration, n_chans)
    positions = load_recording_positions(big_idx)
    # Convert to numpy immediately so we can clear HF cache
    eeg_np = np.array(eeg_data, dtype=np.float32)
    pos_np = np.array(positions, dtype=np.float32)
    del eeg_data, positions
    return eeg_np, pos_np


def clear_hf_cache():
    """Aggressively clear all HF cache blobs."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return
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


def download_worker(task_queue: Queue, preprocess_queue: Queue, stats: dict):
    """Thread: download recordings and feed to preprocess queue."""
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.task_done()
            break

        big_idx, duration, n_chans, df_corrected, shard_offset = item
        try:
            eeg_np, pos_np = download_recording(big_idx, duration, n_chans)
            # Clear cache immediately after loading into memory
            clear_hf_cache()
            preprocess_queue.put((big_idx, eeg_np, pos_np, df_corrected, shard_offset))
            stats["downloaded"] += 1
            print(f"  [DL] Rec {big_idx} downloaded ({eeg_np.shape}), cache cleared", flush=True)
        except Exception as e:
            print(f"  [DL FAIL] Rec {big_idx}: {e}", flush=True)
            stats["failed"] += 1
        finally:
            task_queue.task_done()


# ─── Stage 2: Preprocess ─────────────────────────────────────────────
def process_session(eeg_data, positions, session):
    """Process a single session into standardized windows."""
    if session["flag_remove"]:
        return []

    segment = eeg_data[session["start"]:session["end"]]
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


def preprocess_worker(preprocess_queue: Queue, out_dir: Path, upload_queue: Queue, stats: dict):
    """Thread: preprocess recordings and write shards."""
    while True:
        try:
            item = preprocess_queue.get(timeout=5)
        except Empty:
            if stats.get("downloads_done"):
                break
            continue

        if item is None:
            preprocess_queue.task_done()
            break

        big_idx, eeg_np, pos_np, df_corrected, shard_offset = item

        try:
            if pos_np.shape[0] != eeg_np.shape[1]:
                print(f"  [SKIP] Rec {big_idx}: position mismatch", flush=True)
                stats["failed"] += 1
                preprocess_queue.task_done()
                continue

            sessions = get_session_boundaries(df_corrected, big_idx)

            total_windows = 0
            shard_idx = shard_offset
            num_shards = 0
            buffer = []

            for sess in sessions:
                windows = process_session(eeg_np, pos_np, sess)
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
                    # Queue for upload immediately
                    upload_queue.put(local_path)
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
                upload_queue.put(local_path)
                del table

            stats["completed"] += 1
            stats["total_windows"] += total_windows
            stats["total_shards"] += num_shards
            done = stats["completed"] + stats["failed"]
            print(f"  [OK] Rec {big_idx}: {total_windows} win, {num_shards} shards ({done}/{stats['total_recordings']})", flush=True)

        except Exception as e:
            print(f"  [PROC ERROR] Rec {big_idx}: {e}", flush=True)
            stats["failed"] += 1

        del eeg_np, pos_np
        gc.collect()
        preprocess_queue.task_done()


# ─── Stage 3: Upload ─────────────────────────────────────────────────
def upload_worker(upload_queue: Queue, hf_repo: str, stats: dict, upload_batch_size: int = 5):
    """Thread: batch-upload shards to HF and delete locally."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()

    pending = []

    def flush_upload(files):
        files = [f for f in files if f.exists()]
        if not files:
            return
        size_gb = sum(f.stat().st_size for f in files) / 1e9
        for attempt in range(3):
            try:
                ops = [
                    CommitOperationAdd(
                        path_in_repo=f"data/train/{f.name}",
                        path_or_fileobj=str(f),
                    )
                    for f in files
                ]
                api.create_commit(
                    repo_id=hf_repo,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"Add {len(files)} shards ({size_gb:.1f}GB)",
                )
                for f in files:
                    f.unlink(missing_ok=True)
                stats["uploaded"] += len(files)
                print(f"  [UP] {len(files)} shards ({size_gb:.1f}GB) uploaded, total: {stats['uploaded']}", flush=True)
                return
            except Exception as e:
                print(f"  [UP RETRY {attempt+1}/3] {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        print(f"  [UP FAIL] {len(files)} shards kept locally", flush=True)

    while True:
        try:
            item = upload_queue.get(timeout=10)
        except Empty:
            # Flush any pending
            if pending:
                flush_upload(pending)
                pending = []
            if stats.get("processing_done"):
                break
            continue

        if item is None:
            if pending:
                flush_upload(pending)
            upload_queue.task_done()
            break

        pending.append(item)
        upload_queue.task_done()

        if len(pending) >= upload_batch_size:
            flush_upload(pending)
            pending = []


def main():
    parser = argparse.ArgumentParser(description="Pipeline REVE preprocessing")
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--dl-workers", type=int, default=2,
                        help="Download threads (keep low to limit disk usage)")
    parser.add_argument("--proc-workers", type=int, default=4,
                        help="Preprocessing threads")
    parser.add_argument("--upload-batch", type=int, default=5,
                        help="Upload shards in batches of N")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/reve_preprocess")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
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
    total = len(recordings)

    print(f"Processing {total} recordings (idx {args.start_idx}→{end_idx})", flush=True)
    print(f"DL workers: {args.dl_workers}, Proc workers: {args.proc_workers}", flush=True)
    print(f"Upload batch size: {args.upload_batch}", flush=True)
    print(flush=True)

    # Shared stats (thread-safe enough for counters)
    stats = {
        "downloaded": 0,
        "completed": 0,
        "failed": 0,
        "uploaded": 0,
        "total_windows": 0,
        "total_shards": 0,
        "total_recordings": total,
        "downloads_done": False,
        "processing_done": False,
    }

    # Queues (bounded to limit memory/disk usage)
    task_queue = Queue(maxsize=args.dl_workers + 2)
    preprocess_queue = Queue(maxsize=args.proc_workers + 2)
    upload_queue = Queue(maxsize=20)

    # Start upload worker (1 thread)
    upload_thread = threading.Thread(
        target=upload_worker,
        args=(upload_queue, args.hf_repo, stats, args.upload_batch),
        daemon=True,
    )
    if not args.dry_run:
        upload_thread.start()

    # Start preprocess workers (N threads)
    proc_threads = []
    for _ in range(args.proc_workers):
        t = threading.Thread(
            target=preprocess_worker,
            args=(preprocess_queue, tmp_dir, upload_queue, stats),
            daemon=True,
        )
        t.start()
        proc_threads.append(t)

    # Start download workers (M threads)
    dl_threads = []
    for _ in range(args.dl_workers):
        t = threading.Thread(
            target=download_worker,
            args=(task_queue, preprocess_queue, stats),
            daemon=True,
        )
        t.start()
        dl_threads.append(t)

    # Feed tasks to download queue
    shard_offset = args.start_idx * 100
    for i, (_, row) in enumerate(recordings.iterrows()):
        big_idx = int(row["big_recording_index"])
        duration = int(row["duration"])
        n_chans = int(row["n_chans"])
        rec_offset = shard_offset + i * 200
        task_queue.put((big_idx, duration, n_chans, df_corrected, rec_offset))

    # Signal download workers to stop
    for _ in range(args.dl_workers):
        task_queue.put(None)

    # Wait for downloads to finish
    task_queue.join()
    stats["downloads_done"] = True
    print(f"\n[STAGE 1 DONE] All downloads finished", flush=True)

    # Wait for preprocessing to finish
    preprocess_queue.join()
    stats["processing_done"] = True
    print(f"[STAGE 2 DONE] All preprocessing finished", flush=True)

    # Signal upload to finish
    if not args.dry_run:
        upload_queue.put(None)
        upload_thread.join(timeout=300)

    print(f"\n{'='*50}", flush=True)
    print(f"DONE! Completed: {stats['completed']}, Failed: {stats['failed']}", flush=True)
    print(f"Total windows: {stats['total_windows']}, Total shards: {stats['total_shards']}", flush=True)
    print(f"Uploaded: {stats['uploaded']} shards", flush=True)


if __name__ == "__main__":
    main()
