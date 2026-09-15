"""
src/preprocessing.py
=====================
Preprocessing & feature engineering layer, and the data-split layer
(Section IV-B/C of the paper).

- Removes duplicate and malformed records.
- Encodes categorical features (one-hot; keeps human-readable column names,
  since an explanation attached to an unlabeled column does not help an
  analyst act on it).
- Splits 70/15/15: stratified by class for CICIDS2017/UNSW-NB15, by time
  window for SWaT/WADI (a random split would leak adjacent time steps
  across partitions).
- Resamples the *training* split only: SMOTE for the interpretable track,
  nothing here for the high-performance track (those get class weighting
  inside the model, not the data -- see src/models.py), since oversampling
  is incompatible with bagging and boosting.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, SPLIT_RATIOS
from src.utils import get_logger, human_readable

logger = get_logger(__name__)


def clean(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Drop duplicates and rows that are malformed after coercion to numeric."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_candidates = [c for c in numeric_candidates if c != label_col]
    df[numeric_candidates] = df[numeric_candidates].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_candidates, how="all").reset_index(drop=True)
    df[numeric_candidates] = df[numeric_candidates].fillna(df[numeric_candidates].median())

    logger.info("clean(): %d -> %d rows (%d removed)", before, len(df), before - len(df))
    return df


def encode_features(
    df: pd.DataFrame, label_col: str = "label", drop_cols=("timestamp", "attack_type")
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    One-hot encode categorical columns and separate features from the label.
    Column names stay human-readable end to end (Section IV-C).
    """
    drop_cols = [c for c in drop_cols if c in df.columns]
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col] + drop_cols)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, prefix=cat_cols)

    # get_dummies() emits bool columns; cast everything to float64 so every
    # downstream consumer (sklearn, SHAP's C extension, LIME) sees one dtype.
    X = X.astype("float64")

    X.columns = human_readable(X.columns)
    return X, y, X.columns.tolist()


def split_dataset(
    df: pd.DataFrame,
    strategy: str,
    label_col: str = "label",
    ratios: Dict[str, float] = None,
    seed: int = RANDOM_SEED,
) -> Dict[str, pd.DataFrame]:
    """
    70/15/15 split. `strategy` is 'stratified' (CICIDS2017, UNSW-NB15) or
    'time' (SWaT, WADI -- split by row order / timestamp, never shuffled).
    """
    ratios = ratios or SPLIT_RATIOS
    train_frac, val_frac, test_frac = ratios["train"], ratios["val"], ratios["test"]
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    if strategy == "stratified":
        train_df, temp_df = train_test_split(
            df, train_size=train_frac, stratify=df[label_col], random_state=seed
        )
        relative_val = val_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(
            temp_df, train_size=relative_val, stratify=temp_df[label_col], random_state=seed
        )
    elif strategy == "time":
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        n = len(df)
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)
        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown split strategy '{strategy}'")

    logger.info(
        "split_dataset(%s): train=%d val=%d test=%d", strategy, len(train_df), len(val_df), len(test_df)
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, seed: int = RANDOM_SEED):
    """
    Resample the training split for the interpretable track. Only ever
    called on the training partition -- val/test stay untouched so metrics
    reflect the real class balance.
    """
    minority_count = y_train.value_counts().min()
    k_neighbors = max(1, min(5, minority_count - 1))
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "apply_smote(): %d -> %d rows (class balance now %s)",
        len(X_train), len(X_res), dict(y_res.value_counts()),
    )
    return X_res, y_res


def prepare_splits(
    df: pd.DataFrame, split_strategy: str, label_col: str = "label", seed: int = RANDOM_SEED
) -> Dict[str, object]:
    """
    Run the full preprocessing + split layer and return everything downstream
    layers need: encoded X/y for train/val/test, the SMOTE-resampled training
    set for the interpretable track, and the feature-name list.
    """
    df = clean(df, label_col=label_col)
    parts = split_dataset(df, strategy=split_strategy, label_col=label_col, seed=seed)

    # Fit the one-hot encoding on the union of columns seen across splits so
    # train/val/test always align, then encode each split.
    encoded = {}
    feature_names = None
    for name, part_df in parts.items():
        X, y, cols = encode_features(part_df, label_col=label_col)
        encoded[name] = (X, y)
        feature_names = cols if feature_names is None else feature_names

    # Align val/test columns to the training columns (missing -> 0, extras dropped)
    X_train, y_train = encoded["train"]
    aligned = {"train": (X_train, y_train)}
    for name in ("val", "test"):
        X, y = encoded[name]
        X = X.reindex(columns=X_train.columns, fill_value=0)
        aligned[name] = (X, y)

    X_train_smote, y_train_smote = apply_smote(aligned["train"][0], aligned["train"][1], seed=seed)

    return {
        "X_train": aligned["train"][0], "y_train": aligned["train"][1],
        "X_train_smote": X_train_smote, "y_train_smote": y_train_smote,
        "X_val": aligned["val"][0], "y_val": aligned["val"][1],
        "X_test": aligned["test"][0], "y_test": aligned["test"][1],
        "feature_names": list(X_train.columns),
    }
