"""
agent.py  –  Script 2

OpenAI Agents SDK-based agent that watches a folder ('watch_folder/') for new
CSV files dropped from dataset/test.  When a file appears the agent:
  1. Extracts the addr2_bytes 500-step timeseries and aggregates to 125 steps.
  2. Computes statistical features and forms a normalised feature vector.
  3. Queries Zilliz for the top-10 most similar training samples.
  4. Reasons over the retrieved samples and predicts the platform + video label.
  5. Prints the prediction + reasoning and then deletes the input file.

Environment variables (place in a .env file or export before running):
    OPENAI_API_KEY  – OpenAI API key
    ZILLIZ_URI      – e.g. https://<cluster>.zillizcloud.com:19530
    ZILLIZ_TOKEN    – API token

Usage:
    python agent.py
"""

import os
import json
import time
import threading
import uuid
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from openai import OpenAI

from pymilvus import connections, Collection, utility

from utils.stats_utils import (
    extract_timeseries,
    aggregate_timeseries,
    compute_stats,
    stats_to_raw_vector,
    normalise_vector,
    VECTOR_DIM,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ZILLIZ_URI     = os.getenv("ZILLIZ_URI",   "http://localhost:19530")
ZILLIZ_TOKEN   = os.getenv("ZILLIZ_TOKEN", "")
COLLECTION_NAME = "video_traffic"
WATCH_FOLDER    = Path("watch_folder")
NORMALISER_PATH = Path("normaliser.npz")
TOP_K           = 4
MODEL           = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Load normaliser (fitted during ingest)
# ---------------------------------------------------------------------------

def load_normaliser():
    if not NORMALISER_PATH.exists():
        raise FileNotFoundError(
            f"Normaliser file '{NORMALISER_PATH}' not found. "
            "Please run ingest.py first."
        )
    data = np.load(NORMALISER_PATH)
    return data["mean"], data["std"]


# ---------------------------------------------------------------------------
# Zilliz connection (lazy singleton)
# ---------------------------------------------------------------------------

_collection: Collection | None = None

def get_collection() -> Collection:
    global _collection
    if _collection is None:
        if ZILLIZ_TOKEN:
            connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        else:
            connections.connect(alias="default", uri=ZILLIZ_URI)
        _collection = Collection(COLLECTION_NAME)
        _collection.load()
    return _collection


# ---------------------------------------------------------------------------
# Tool 1 – extract timeseries from file
# ---------------------------------------------------------------------------

def tool_extract_timeseries(filepath: str) -> dict:
    """
    Read a CSV file, extract the addr2_bytes column, aggregate to 125 steps,
    and return only the aggregated timeseries.

    Returns
    -------
    {
        "agg_timeseries": [125 floats]
    }
    """
    raw_ts = extract_timeseries(filepath)
    agg_ts = aggregate_timeseries(raw_ts)
    # Return only the aggregated timeseries — do NOT expose the filepath or
    # raw timeseries to the LLM to avoid any label leakage.
    return {
        "agg_timeseries": agg_ts.tolist(),
    }


# ---------------------------------------------------------------------------
# Tool 2 – compute stats and build feature vector
# ---------------------------------------------------------------------------

def tool_compute_stats_and_vector(agg_timeseries: list) -> dict:
    """
    Given a 125-step aggregated timeseries, compute statistics and return
    both the stats dict and the normalised feature vector ready for search.

    Parameters
    ----------
    agg_timeseries : list of 125 floats

    Returns
    -------
    {
        "stats":          { stat_name: float, ... },
        "feature_vector": [VECTOR_DIM floats]  (normalised)
    }
    """
    mean_vec, std_vec = load_normaliser()
    agg = np.array(agg_timeseries, dtype=np.float64)
    st  = compute_stats(agg)
    raw = stats_to_raw_vector(st)
    norm = normalise_vector(raw, mean_vec, std_vec)
    return {
        "stats":          st,
        "feature_vector": norm.tolist(),
    }


# ---------------------------------------------------------------------------
# Tool 3 – retrieve similar samples from Zilliz
# ---------------------------------------------------------------------------

def tool_retrieve_similar(feature_vector: list) -> dict:
    """
    Query the Zilliz vector database for the top-10 most similar training
    samples using L2 distance.

    Parameters
    ----------
    feature_vector : list of VECTOR_DIM floats (normalised)

    Returns
    -------
    {
        "results": [
            {
                "platform":       str,
                "video":          str,
                "agg_timeseries": [125 floats]
            },
            ...   (up to 10 entries)
        ]
    }
    """
    col = get_collection()
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    # Ensure every element is a plain Python float — Milvus rejects int/numpy types
    query_vector = [float(x) for x in feature_vector]
    if len(query_vector) != VECTOR_DIM:
        raise ValueError(
            f"feature_vector length mismatch: expected {VECTOR_DIM}, got {len(query_vector)}"
        )
    results = col.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=TOP_K,
        output_fields=["platform", "video", "agg_timeseries"],
    )

    hits = []
    for hit in results[0]:
        entity = hit.entity
        hits.append({
            "platform":       entity.get("platform"),
            "video":          entity.get("video"),
            "agg_timeseries": json.loads(entity.get("agg_timeseries", "[]")),
        })

    return {"results": hits}


# ---------------------------------------------------------------------------
# Tool registry (for the OpenAI function-calling API)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_timeseries",
            "description": (
                "Read a CSV file from disk, extract the 'addr2_bytes' column, "
                "and return the 125-step aggregated timeseries (sum of every 4 steps). "
                "Only the aggregated timeseries is returned — no filepath or raw data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Absolute or relative path to the CSV file.",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_stats_and_vector",
            "description": (
                "Given a 125-step aggregated timeseries (list of floats), "
                "compute statistical features (mean, std, min, max, median, "
                "Q1, Q3, IQR, skewness, kurtosis, RMS, CV, energy, total, "
                "range, P10, P90, zero-crossing-rate, normalised std) and "
                "return the stats dict plus a normalised feature vector."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agg_timeseries": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "125-element aggregated timeseries.",
                    }
                },
                "required": ["agg_timeseries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_similar",
            "description": (
                "Query the Zilliz vector database for the top-10 most similar "
                "training samples using the provided normalised feature vector. "
                "Returns each match's platform, video label, and its 125-step "
                "aggregated timeseries for pattern comparison."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": f"Normalised feature vector of length {VECTOR_DIM}.",
                    }
                },
                "required": ["feature_vector"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "extract_timeseries":       tool_extract_timeseries,
    "compute_stats_and_vector": tool_compute_stats_and_vector,
    "retrieve_similar":         tool_retrieve_similar,
}

SYSTEM_PROMPT = """
You are a video traffic classification expert with deep knowledge of how different
streaming platforms (Netflix, Stan, YouTube) deliver video content over the network.
You understand the characteristic download-byte patterns of each platform and video.

Your task is to identify the streaming platform and video label of an unknown network
traffic capture, given its 125-step download-byte timeseries.

You have access to three tools — use them internally, but NEVER mention them, the
database, any retrieval process, or any "similar samples" in your final answer.
Your answer must read as if it comes entirely from your own expert knowledge.

  1. extract_timeseries       – extract the timeseries from a CSV file.
  2. compute_stats_and_vector – compute statistical features and a search vector.
  3. retrieve_similar         – consult your internal knowledge base.

Workflow (you MUST follow this exact order):
  Step 1 – Call extract_timeseries with the provided filepath.
  Step 2 – Call compute_stats_and_vector with the agg_timeseries from Step 1.
  Step 3 – Call retrieve_similar with the feature_vector from Step 2.
  Step 4 – Analyse the query timeseries characteristics using the computed stats and
            the timeseries shapes from your knowledge base:
              • Overall magnitude and energy level — high or low bandwidth?
              • Burstiness — sharp spikes vs. steady flow?
              • Distribution shape — skewness, kurtosis, spread (IQR, std)?
              • Temporal patterns — where do bursts occur, quiet periods, peaks?
            Identify which platform and video these patterns are most consistent with.
  Step 5 – Output your final answer in the EXACT format below:

=== PREDICTION ===
Platform : <predicted platform>
Video    : <predicted video label>
Reason   : <explanation written as your own expert analysis of the timeseries —
             describe the traffic patterns, statistics, and why they are characteristic
             of the predicted platform and video. Do NOT mention distances, ranks,
             filenames, retrieved samples, a database, any search process, or any
             tools you used>
==================
""".strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(filepath: str) -> str:
    """
    Run the OpenAI agent for a single input file.
    Returns the agent's final text answer.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Please classify the following traffic capture file: {filepath}\n\n"
                "Follow the workflow described in your instructions."
            ),
        },
    ]

    print(f"\n[AGENT] Processing: {filepath}")
    # Request-scoped state to avoid relying on model-reconstructed vectors.
    cached_feature_vector: list[float] | None = None

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        message = choice.message

        # Append assistant turn
        messages.append(message.model_dump(exclude_unset=False))

        # If the model wants to call tools, execute them
        if choice.finish_reason == "tool_calls" and message.tool_calls:
            for tc in message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                print(f"[AGENT] → Calling tool: {fn_name}({list(fn_args.keys())})")

                if fn_name not in TOOL_DISPATCH:
                    result = {"error": f"Unknown tool: {fn_name}"}
                else:
                    try:
                        if fn_name == "retrieve_similar":
                            if cached_feature_vector is None:
                                raise ValueError(
                                    "No cached feature vector found for this request. "
                                    "Call compute_stats_and_vector before retrieve_similar."
                                )
                            fn_args = {"feature_vector": cached_feature_vector}

                        result = TOOL_DISPATCH[fn_name](**fn_args)

                        if fn_name == "compute_stats_and_vector":
                            fv = result.get("feature_vector") if isinstance(result, dict) else None
                            if not isinstance(fv, list):
                                raise ValueError("compute_stats_and_vector returned invalid feature_vector")
                            cached_feature_vector = [float(x) for x in fv]
                            if len(cached_feature_vector) != VECTOR_DIM:
                                raise ValueError(
                                    "Cached feature vector length mismatch: "
                                    f"expected {VECTOR_DIM}, got {len(cached_feature_vector)}"
                                )
                    except Exception as exc:
                        result = {"error": str(exc)}

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         fn_name,
                    "content":      json.dumps(result),
                })

        else:
            # Final answer
            return message.content or ""


# ---------------------------------------------------------------------------
# Folder watcher
# ---------------------------------------------------------------------------

class NewFileHandler(FileSystemEventHandler):
    """Triggered whenever a new file is created in the watch folder."""

    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._processing: set[str] = set()

    def on_created(self, event):
        if event.is_directory:
            return
        filepath = Path(event.src_path)
        if filepath.suffix.lower() != ".csv":
            return

        # Debounce – some systems fire the event twice
        with self._lock:
            if str(filepath) in self._processing:
                return
            self._processing.add(str(filepath))

        # Process in a background thread so the watcher isn't blocked
        t = threading.Thread(
            target=self._handle_file,
            args=(filepath,),
            daemon=True,
        )
        t.start()

    def _handle_file(self, filepath: Path):
        # Wait briefly to make sure the file is fully written
        time.sleep(0.5)

        # Rename to a random UUID so the agent cannot infer labels from the filename
        anonymous_path = filepath.parent / f"{uuid.uuid4().hex}.csv"
        try:
            filepath.rename(anonymous_path)
            filepath = anonymous_path
        except Exception as exc:
            print(f"[WARN] Could not anonymise filename, proceeding with original: {exc}")

        try:
            answer = run_agent(str(filepath))
            print("\n" + "=" * 60)
            print(answer)
            print("=" * 60 + "\n")
        except Exception as exc:
            print(f"[ERROR] Agent failed for {filepath}: {exc}")
        finally:
            # Clean up the watched file
            try:
                filepath.unlink()
                print(f"[INFO] Deleted processed file: {filepath}")
            except Exception as exc:
                print(f"[WARN] Could not delete {filepath}: {exc}")

            with self._lock:
                self._processing.discard(str(filepath))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    WATCH_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Watching folder: {WATCH_FOLDER.resolve()}")
    print("[INFO] Drop a CSV file from dataset/test into the watch folder to classify it.")
    print("[INFO] Press Ctrl+C to stop.\n")

    lock = threading.Lock()
    event_handler = NewFileHandler(lock)
    observer = Observer()
    observer.schedule(event_handler, str(WATCH_FOLDER), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[INFO] Agent stopped.")

    observer.join()


if __name__ == "__main__":
    main()
