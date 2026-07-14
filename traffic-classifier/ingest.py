"""
ingest.py  –  Script 1

Loads all training CSV files from dataset/train, builds the 125-step
aggregated timeseries for each sample, computes statistical features,
fits a global z-score normaliser, and stores every sample (vector +
aggregated timeseries + metadata) in a Zilliz (Milvus) collection.

Environment variables (place in a .env file or export before running):
    ZILLIZ_URI      – e.g. https://<cluster>.zillizcloud.com:19530
    ZILLIZ_TOKEN    – API token (user:password or cloud token)

Usage:
    python ingest.py
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from utils.stats_utils import (
    extract_timeseries,
    aggregate_timeseries,
    compute_stats,
    stats_to_raw_vector,
    normalise_vector,
    FEATURE_KEYS,
    VECTOR_DIM,
    AGG_STEPS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

ZILLIZ_URI   = os.getenv("ZILLIZ_URI",   "http://localhost:19530")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN", "")

DATASET_ROOT = Path("dataset/train")
PLATFORMS    = ["Netflix", "Stan", "Youtube"]
COLLECTION_NAME = "video_traffic"

# ---------------------------------------------------------------------------
# Zilliz / Milvus helpers
# ---------------------------------------------------------------------------

def connect_zilliz():
    if ZILLIZ_TOKEN:
        connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    else:
        connections.connect(alias="default", uri=ZILLIZ_URI)
    print(f"[INFO] Connected to Zilliz/Milvus at {ZILLIZ_URI}")


def create_collection() -> Collection:
    """
    Create (or re-use) the Milvus collection.

    Schema
    ------
    id            : auto-increment primary key
    vector        : FLOAT_VECTOR of dim VECTOR_DIM  (normalised stats)
    platform      : VARCHAR  – e.g. "Netflix"
    video         : VARCHAR  – e.g. "vid1"
    filename      : VARCHAR  – e.g. "Netflix_vid1_1.csv"
    agg_timeseries: VARCHAR  – JSON-encoded list of 125 floats
    """
    if utility.has_collection(COLLECTION_NAME):
        print(f"[INFO] Collection '{COLLECTION_NAME}' already exists – dropping it for a fresh ingest.")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id",             dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name="vector",         dtype=DataType.FLOAT_VECTOR,  dim=VECTOR_DIM),
        FieldSchema(name="platform",       dtype=DataType.VARCHAR,       max_length=64),
        FieldSchema(name="video",          dtype=DataType.VARCHAR,       max_length=64),
        FieldSchema(name="filename",       dtype=DataType.VARCHAR,       max_length=256),
        FieldSchema(name="agg_timeseries", dtype=DataType.VARCHAR,       max_length=8192),
    ]
    schema = CollectionSchema(fields=fields, description="Video traffic timeseries features")
    col = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"[INFO] Collection '{COLLECTION_NAME}' created.")
    return col


def build_index(col: Collection):
    index_params = {
        "metric_type": "L2",
        "index_type":  "IVF_FLAT",
        "params":      {"nlist": 128},
    }
    col.create_index(field_name="vector", index_params=index_params)
    print("[INFO] Index created on 'vector' field.")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def iter_csv_files(root: Path):
    """
    Yield (platform, video, filepath) for every CSV under root.
    root/
      <platform>/
        <video>/
          <filename>.csv
    """
    for platform in PLATFORMS:
        platform_dir = root / platform
        if not platform_dir.exists():
            print(f"[WARN] Platform directory not found: {platform_dir}")
            continue
        for video_dir in sorted(platform_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video = video_dir.name
            for csv_file in sorted(video_dir.glob("*.csv")):
                yield platform, video, csv_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 1. Collect all samples & raw feature vectors
    # ------------------------------------------------------------------
    records = []   # list of dicts: platform, video, filename, raw_vec, agg_ts

    print("[INFO] Loading training data …")
    for platform, video, filepath in tqdm(list(iter_csv_files(DATASET_ROOT))):
        try:
            raw_ts  = extract_timeseries(str(filepath))
            agg_ts  = aggregate_timeseries(raw_ts)
            st      = compute_stats(agg_ts)
            raw_vec = stats_to_raw_vector(st)
        except Exception as exc:
            print(f"[WARN] Skipping {filepath}: {exc}")
            continue

        records.append({
            "platform": platform,
            "video":    video,
            "filename": filepath.name,
            "raw_vec":  raw_vec,
            "agg_ts":   agg_ts,
        })

    if not records:
        print("[ERROR] No records loaded. Exiting.")
        return

    print(f"[INFO] Loaded {len(records)} samples.")

    # ------------------------------------------------------------------
    # 2. Fit global z-score normaliser on the training set
    # ------------------------------------------------------------------
    all_raw = np.stack([r["raw_vec"] for r in records])   # (N, VECTOR_DIM)
    global_mean = all_raw.mean(axis=0)
    global_std  = all_raw.std(axis=0, ddof=1)
    # Avoid division by zero
    global_std  = np.where(global_std == 0, 1.0, global_std)

    # Save normaliser so the agent can use the same parameters
    norm_path = Path("normaliser.npz")
    np.savez(norm_path, mean=global_mean, std=global_std)
    print(f"[INFO] Normaliser saved to {norm_path}")

    # ------------------------------------------------------------------
    # 3. Connect to Zilliz and prepare collection
    # ------------------------------------------------------------------
    connect_zilliz()
    col = create_collection()

    # ------------------------------------------------------------------
    # 4. Insert data in batches
    # ------------------------------------------------------------------
    BATCH_SIZE = 500
    batch_vectors   = []
    batch_platforms = []
    batch_videos    = []
    batch_filenames = []
    batch_agg_ts    = []

    def flush_batch():
        if not batch_vectors:
            return
        col.insert([
            batch_vectors,
            batch_platforms,
            batch_videos,
            batch_filenames,
            batch_agg_ts,
        ])
        batch_vectors.clear()
        batch_platforms.clear()
        batch_videos.clear()
        batch_filenames.clear()
        batch_agg_ts.clear()

    print("[INFO] Inserting records into Zilliz …")
    for r in tqdm(records):
        norm_vec = normalise_vector(r["raw_vec"], global_mean, global_std)
        batch_vectors.append(norm_vec.tolist())
        batch_platforms.append(r["platform"])
        batch_videos.append(r["video"])
        batch_filenames.append(r["filename"])
        batch_agg_ts.append(json.dumps(r["agg_ts"].tolist()))

        if len(batch_vectors) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    col.flush()
    print(f"[INFO] Inserted {col.num_entities} entities.")

    # ------------------------------------------------------------------
    # 5. Build index and load collection for searching
    # ------------------------------------------------------------------
    build_index(col)
    col.load()
    print("[INFO] Ingest complete. Collection is ready for search.")


if __name__ == "__main__":
    main()
