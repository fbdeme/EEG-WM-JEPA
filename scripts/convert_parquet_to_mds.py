"""Convert HF parquet dataset to MDS format for MosaicML Streaming.

Reads from existing HF parquet repo, writes MDS shards to a new HF repo
or local directory.

Usage:
  # Convert and upload to HF
  uv run python scripts/convert_parquet_to_mds.py \
    --src fbdeme/reve-preprocessed \
    --dst hf://fbdeme/reve-mds/train

  # Convert to local directory
  uv run python scripts/convert_parquet_to_mds.py \
    --src fbdeme/reve-preprocessed \
    --dst /data/reve-mds/train
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from streaming import MDSWriter
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert parquet to MDS")
    parser.add_argument("--src", required=True, help="Source HF dataset repo")
    parser.add_argument("--dst", required=True, help="Destination (hf://repo/path or local path)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-shard", type=int, default=10000,
                        help="Samples per MDS shard")
    args = parser.parse_args()

    columns = {
        "eeg": "bytes",
        "coords": "bytes",
        "num_channels": "int",
        "window_samples": "int",
    }

    print(f"Loading source: {args.src}", flush=True)
    ds = load_dataset(args.src, split=args.split, streaming=True)

    print(f"Writing MDS to: {args.dst}", flush=True)
    with MDSWriter(
        out=args.dst,
        columns=columns,
        size_limit=256 * 1024 * 1024,  # 256MB per shard
    ) as writer:
        for i, sample in enumerate(tqdm(ds, desc="Converting")):
            writer.write({
                "eeg": sample["eeg"],
                "coords": sample["coords"],
                "num_channels": sample["num_channels"],
                "window_samples": sample["window_samples"],
            })

    print(f"Done! Converted {i + 1} samples.", flush=True)


if __name__ == "__main__":
    main()
