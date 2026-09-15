"""
src/data_loading.py
====================
Data layer of the five-layer pipeline (Section IV-A/B, Table II).

Four public sources make up the data layer:
    - CICIDS2017   (network IDS,  ~2.8M flows, 78 features)
    - UNSW-NB15    (network IDS,  ~2.5M rows,  49 features)
    - SWaT / WADI  (ICS testbeds, 51 / 123 features, physical process data)
    - CVE / NVD    (vulnerability context: severity + exploit-maturity
                     features attached to assets a model has already
                     flagged -- not traffic data itself)

None of CICIDS2017, UNSW-NB15, SWaT, or WADI can be auto-downloaded inside
a notebook: the network sets are large Kaggle mirrors and the two ICS
testbeds require a signed access request to iTrust (SUTD). See
`data/README.md` for exact acquisition steps.

To keep the pipeline runnable end-to-end without any of that -- e.g. for a
first Colab smoke test, or for grading a repo that a reviewer will not run
with restricted-access data -- every loader below falls back to a
structurally-matched *synthetic* dataset (same column count, same rough
class imbalance, same dtypes) when the real file is not present. Synthetic
data is clearly flagged in the returned `meta` dict and should never be
used to report real results.
"""

import glob
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import DATASETS, RAW_DATA_DIR, RANDOM_SEED
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Synthetic fallbacks
# ---------------------------------------------------------------------------
def _make_synthetic_tabular(
    n_rows: int,
    n_features: int,
    attack_ratio: float,
    name: str,
    time_ordered: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build a synthetic classification table that mimics a real IDS/ICS export:
    a mix of continuous and count-like features, a handful of features that
    are actually informative for the label (so models/explainers have
    something real to find), human-readable column names, and a label
    column. If `time_ordered` is True, rows are left in a fixed order so a
    time-based split behaves as it would on SWaT/WADI.
    """
    rng = np.random.default_rng(seed)
    n_informative = max(4, n_features // 10)

    informative = rng.normal(0, 1, size=(n_rows, n_informative))
    noise = rng.exponential(1.0, size=(n_rows, n_features - n_informative))
    X = np.hstack([informative, noise])

    weights = rng.normal(0, 1, size=n_informative)
    logits = informative @ weights
    logits += rng.normal(0, 0.5, size=n_rows)  # label noise
    threshold = np.quantile(logits, 1 - attack_ratio)
    y = (logits >= threshold).astype(int)

    informative_names = [f"{name}_signal_feat_{i+1}" for i in range(n_informative)]
    noise_names = [f"{name}_raw_feat_{i+1}" for i in range(n_features - n_informative)]
    columns = informative_names + noise_names

    df = pd.DataFrame(X, columns=columns)
    # a couple of categorical-looking columns, common in real IDS exports
    df[f"{name}_protocol"] = rng.choice(["TCP", "UDP", "ICMP"], size=n_rows)
    df[f"{name}_service"] = rng.choice(["http", "dns", "modbus", "other"], size=n_rows)
    df["label"] = y

    if time_ordered:
        df.insert(0, "timestamp", pd.date_range("2024-01-01", periods=n_rows, freq="min"))
    return df


def _synthetic_for(dataset_name: str, n_rows: int = 20000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    shape = {
        "cicids2017": dict(n_features=78, attack_ratio=0.20, time_ordered=False),
        "unsw_nb15": dict(n_features=49, attack_ratio=0.30, time_ordered=False),
        "swat": dict(n_features=51, attack_ratio=0.12, time_ordered=True),
        "wadi": dict(n_features=123, attack_ratio=0.06, time_ordered=True),
    }[dataset_name]
    return _make_synthetic_tabular(n_rows=n_rows, name=dataset_name, seed=seed, **shape)


# ---------------------------------------------------------------------------
# Real-file loaders
# ---------------------------------------------------------------------------
def _find_raw_files(dataset_name: str) -> list:
    pattern_dir = os.path.join(RAW_DATA_DIR, dataset_name)
    if not os.path.isdir(pattern_dir):
        return []
    return sorted(glob.glob(os.path.join(pattern_dir, "*.csv")))


def _harmonize_label(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Map each dataset's native label scheme onto a single binary `label`
    column (0 = benign/normal, 1 = attack/malicious). Multi-class attack-type
    columns are collapsed to binary here; keep the original column (renamed
    `attack_type`) for anyone who wants the finer-grained target later.
    """
    label_aliases = ["Label", "label", "attack_cat", "Attack", "Normal/Attack"]
    label_col = next((c for c in label_aliases if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            f"Could not find a label column for '{dataset_name}'. "
            f"Expected one of {label_aliases}. Found: {list(df.columns)[:10]}..."
        )

    raw = df[label_col].astype(str).str.strip().str.upper()
    benign_tokens = {"BENIGN", "NORMAL", "0"}
    df["attack_type"] = df[label_col]
    df["label"] = (~raw.isin(benign_tokens)).astype(int)
    if label_col != "label":
        df = df.drop(columns=[label_col])
    return df


def load_dataset(
    dataset_name: str,
    n_synthetic_rows: int = 20000,
    force_synthetic: bool = False,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load one of the four data-layer sources.

    Returns
    -------
    df   : DataFrame including a binary `label` column (and, for SWaT/WADI,
           a `timestamp` column used by the time-based split).
    meta : dict with at least {'dataset': str, 'synthetic': bool, 'role': str}
    """
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Options: {list(DATASETS)}")

    info = DATASETS[dataset_name]
    files = [] if force_synthetic else _find_raw_files(dataset_name)

    if not files:
        logger.warning(
            "No raw files found for '%s' under %s/%s -- using a synthetic "
            "stand-in. See data/README.md to plug in the real dataset.",
            dataset_name, RAW_DATA_DIR, dataset_name,
        )
        df = _synthetic_for(dataset_name, n_rows=n_synthetic_rows, seed=seed)
        meta = {"dataset": dataset_name, "synthetic": True, "role": info["role"], "n_files": 0}
        return df, meta

    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = _harmonize_label(df, dataset_name)

    meta = {
        "dataset": dataset_name,
        "synthetic": False,
        "role": info["role"],
        "n_files": len(files),
        "source_files": files,
    }
    return df, meta


# ---------------------------------------------------------------------------
# CVE / NVD enrichment (Section IV-B)
# ---------------------------------------------------------------------------
def load_cve_enrichment(force_synthetic: bool = False, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    CVE/NVD provides severity and exploit-maturity features attached to
    assets a model has already flagged -- it is never used as traffic data
    directly. Expects a CSV with at least [cve_id, cvss_score,
    exploit_maturity, affected_service]; falls back to a small synthetic
    lookup table matching that schema.
    """
    path = os.path.join(RAW_DATA_DIR, "cve_nvd", "nvd_cve_enrichment.csv")
    if not force_synthetic and os.path.exists(path):
        return pd.read_csv(path)

    logger.warning(
        "No CVE/NVD enrichment file found at %s -- using a synthetic "
        "vulnerability-context table. See data/README.md for the NVD feed URL.",
        path,
    )
    rng = np.random.default_rng(seed)
    n = 200
    return pd.DataFrame(
        {
            "cve_id": [f"CVE-2025-{1000 + i}" for i in range(n)],
            "cvss_score": rng.uniform(1.0, 10.0, size=n).round(1),
            "exploit_maturity": rng.choice(
                ["unproven", "proof_of_concept", "functional", "high"], size=n
            ),
            "affected_service": rng.choice(
                ["http", "modbus", "dns", "smb", "ssh", "other"], size=n
            ),
        }
    )


def enrich_with_cve(
    df: pd.DataFrame,
    service_col: str,
    cve_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Attach severity/exploit-maturity context to each row by joining on the
    service it involves, following the enrichment approach used in [15]:
    CVE/NVD features augment already-flagged assets rather than serving as
    primary traffic features.
    """
    if cve_table is None:
        cve_table = load_cve_enrichment()
    if service_col not in df.columns:
        logger.warning("Service column '%s' not found; skipping CVE enrichment.", service_col)
        return df

    agg = (
        cve_table.groupby("affected_service")
        .agg(cvss_score_mean=("cvss_score", "mean"), cvss_score_max=("cvss_score", "max"))
        .reset_index()
        .rename(columns={"affected_service": service_col})
    )
    return df.merge(agg, on=service_col, how="left")
