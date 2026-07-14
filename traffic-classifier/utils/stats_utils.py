"""
utils/stats_utils.py

Utility functions for:
  - Extracting and aggregating the addr2_bytes timeseries from a CSV file
  - Computing statistical features from the aggregated timeseries
  - Normalising those features to form a fixed-length feature vector
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_STEPS = 500          # expected number of rows in each CSV (excluding header)
AGG_STEPS = 125          # 500 / 4
AGG_WINDOW = 4           # how many raw steps to sum together
VECTOR_DIM = 19          # number of features in the normalised vector


# ---------------------------------------------------------------------------
# Timeseries extraction
# ---------------------------------------------------------------------------

def extract_timeseries(filepath: str) -> np.ndarray:
    """
    Read a CSV file, extract the 'addr2_bytes' column, strip whitespace from
    column names, and return the raw 500-step timeseries as a numpy array.

    Raises ValueError if the column is missing or the length is wrong.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    if "addr2_bytes" not in df.columns:
        raise ValueError(
            f"Column 'addr2_bytes' not found in {filepath}. "
            f"Available columns: {list(df.columns)}"
        )

    series = df["addr2_bytes"].values.astype(np.float64)

    if len(series) != RAW_STEPS:
        raise ValueError(
            f"Expected {RAW_STEPS} rows in {filepath}, got {len(series)}."
        )

    return series


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_timeseries(series: np.ndarray) -> np.ndarray:
    """
    Aggregate a 500-step timeseries to 125 steps by summing every 4 consecutive
    values.
    """
    if len(series) != RAW_STEPS:
        raise ValueError(
            f"aggregate_timeseries expects a {RAW_STEPS}-element array, "
            f"got {len(series)}."
        )
    return series.reshape(AGG_STEPS, AGG_WINDOW).sum(axis=1)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(agg_series: np.ndarray) -> dict:
    """
    Compute a rich set of statistics from the 125-step aggregated timeseries.

    Returns a dict with the following keys:
        mean, std, min, max, median,
        q1, q3, iqr,
        skewness, kurtosis,
        rms,
        cv,                  # coefficient of variation
        energy,              # sum of squares
        total,               # sum of all values
        range_val,           # max - min
        p10, p90,            # 10th and 90th percentiles
        zero_crossing_rate   # fraction of consecutive pairs that cross zero (mean-centred)
    """
    s = agg_series.astype(np.float64)

    mean    = float(np.mean(s))
    std     = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
    minimum = float(np.min(s))
    maximum = float(np.max(s))
    median  = float(np.median(s))
    q1      = float(np.percentile(s, 25))
    q3      = float(np.percentile(s, 75))
    iqr     = q3 - q1
    skewness  = float(scipy_stats.skew(s))
    kurtosis  = float(scipy_stats.kurtosis(s))          # excess kurtosis
    rms       = float(np.sqrt(np.mean(s ** 2)))
    cv        = (std / mean) if mean != 0.0 else 0.0
    energy    = float(np.sum(s ** 2))
    total     = float(np.sum(s))
    range_val = maximum - minimum
    p10       = float(np.percentile(s, 10))
    p90       = float(np.percentile(s, 90))

    # Zero-crossing rate on mean-centred signal
    centred = s - mean
    zc = float(
        np.sum(np.diff(np.sign(centred)) != 0) / (len(s) - 1)
    ) if len(s) > 1 else 0.0

    return {
        "mean":               mean,
        "std":                std,
        "min":                minimum,
        "max":                maximum,
        "median":             median,
        "q1":                 q1,
        "q3":                 q3,
        "iqr":                iqr,
        "skewness":           skewness,
        "kurtosis":           kurtosis,
        "rms":                rms,
        "cv":                 cv,
        "energy":             energy,
        "total":              total,
        "range":              range_val,
        "p10":                p10,
        "p90":                p90,
        "zero_crossing_rate": zc,
        "agg_std_norm":       std / (maximum + 1e-9),  # spread relative to peak
    }


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# Feature order – must stay consistent between ingest and query time
FEATURE_KEYS = [
    "mean", "std", "min", "max", "median",
    "q1", "q3", "iqr",
    "skewness", "kurtosis",
    "rms", "cv", "energy", "total", "range",
    "p10", "p90",
    "zero_crossing_rate", "agg_std_norm",
]

assert len(FEATURE_KEYS) == VECTOR_DIM, (
    f"FEATURE_KEYS has {len(FEATURE_KEYS)} entries but VECTOR_DIM={VECTOR_DIM}"
)


def stats_to_raw_vector(stats: dict) -> np.ndarray:
    """Convert a stats dict to a raw (un-normalised) numpy vector."""
    return np.array([stats[k] for k in FEATURE_KEYS], dtype=np.float64)


def normalise_vector(
    raw_vec: np.ndarray,
    mean_vec: np.ndarray,
    std_vec: np.ndarray,
) -> np.ndarray:
    """Z-score normalise a raw feature vector."""
    denom = np.where(std_vec == 0, 1.0, std_vec)
    return (raw_vec - mean_vec) / denom


def stats_to_normalised_vector(
    stats: dict,
    mean_vec: np.ndarray,
    std_vec: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper: dict → raw → normalised vector."""
    raw = stats_to_raw_vector(stats)
    return normalise_vector(raw, mean_vec, std_vec)
