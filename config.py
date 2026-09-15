"""
config.py
=========
Single source of truth for paths, random seed, dataset registry, and the
hyperparameter grids used by the two modeling tracks. Every other module
imports from here so that a setting only has to change in one place.

Mirrors the methodology in Section IV of the paper:
    "Explainable Artificial Intelligence for Cyber Risk Assessment in
    High-Risk Critical Infrastructure Systems"
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Data layer registry (Table II)
# ---------------------------------------------------------------------------
# `label_col` is the binary target after harmonisation (0 = benign/normal,
# 1 = malicious/attack). `split` selects the splitting strategy used in
# Section IV-B/C: "stratified" for the two network-IDS sets, "time" for the
# two ICS testbeds (split by time window so adjacent time steps never leak
# across partitions).
DATASETS = {
    "cicids2017": {
        "role": "Network IDS",
        "expected_files": ["cicids2017.csv"],
        "label_col": "label",
        "split": "stratified",
        "kaggle": "cicdataset/cicids2017",
    },
    "unsw_nb15": {
        "role": "Network IDS",
        "expected_files": ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"],
        "label_col": "label",
        "split": "stratified",
        "kaggle": "mrwellsdavid/unsw-nb15",
    },
    "swat": {
        "role": "ICS testbed",
        "expected_files": ["SWaT_Dataset_Normal_v1.csv", "SWaT_Dataset_Attack_v0.csv"],
        "label_col": "label",
        "split": "time",
        "source": "iTrust, Centre for Research in Cyber Security (request-based access)",
    },
    "wadi": {
        "role": "ICS testbed",
        "expected_files": ["WADI_14days.csv", "WADI_attackdata.csv"],
        "label_col": "label",
        "split": "time",
        "source": "iTrust, Centre for Research in Cyber Security (request-based access)",
    },
    "cve_nvd": {
        "role": "Vulnerability context (enrichment only, not traffic data)",
        "expected_files": ["nvd_cve_enrichment.csv"],
        "label_col": None,
        "split": None,
        "source": "NVD JSON feeds (https://nvd.nist.gov/vuln/data-feeds) or NVD REST API",
    },
}

# 70 / 15 / 15 split (Section IV-B/F)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------
# Preprocessing (Section IV-C)
# ---------------------------------------------------------------------------
# Interpretable-track models are resampled with SMOTE; high-performance
# tree-ensemble models instead get class weighting, since oversampling is
# incompatible with bagging/boosting.
SMOTE_TRACK = ["logistic_regression", "decision_tree", "ebm"]
CLASS_WEIGHT_TRACK = ["random_forest", "xgboost", "lightgbm"]

# ---------------------------------------------------------------------------
# Models and hyperparameter grids (Table III)
# ---------------------------------------------------------------------------
INTERPRETABLE_MODELS = ["logistic_regression", "decision_tree", "ebm"]
HIGH_PERFORMANCE_MODELS = ["random_forest", "xgboost", "lightgbm"]

PARAM_GRIDS = {
    "logistic_regression": {"C": [0.01, 0.1, 1, 10]},
    "decision_tree": {"max_depth": [3, 5, 7, 10]},
    "ebm": {"interactions": [0, 10]},
    "random_forest": {"n_estimators": [100, 300, 500]},
    "xgboost": {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 0.3],
    },
    "lightgbm": {"num_leaves": [31, 63, 127]},
}

CV_FOLDS = 5
CV_SCORING = "f1"  # optimizing F1, not accuracy, per Section IV-D

# ---------------------------------------------------------------------------
# Explainability layer
# ---------------------------------------------------------------------------
TOP_K_FEATURES = 10          # top-k features used for fidelity / stability
SHAP_BACKGROUND_SIZE = 100   # background sample for SHAP explainers
LIME_N_FLAGGED_INSTANCES = 25  # "same flagged instances" sampled for LIME
N_PERTURBATIONS = 10         # repeats per instance in the fidelity test
N_NEIGHBORS_STABILITY = 1    # nearest neighbours used for the stability test

# ---------------------------------------------------------------------------
# Quick mode (smoke tests / first Colab run)
# ---------------------------------------------------------------------------
# When True, grids/CV/sample sizes shrink drastically so the whole five-layer
# pipeline can be exercised end-to-end in well under a minute on synthetic
# data, purely to confirm the wiring is correct before a full run.
QUICK_MODE = False


def quick_param_grids():
    """Return a single-point grid per model, for fast smoke testing."""
    return {
        "logistic_regression": {"C": [1]},
        "decision_tree": {"max_depth": [5]},
        "ebm": {"interactions": [0]},
        "random_forest": {"n_estimators": [100]},
        "xgboost": {"max_depth": [4], "learning_rate": [0.1]},
        "lightgbm": {"num_leaves": [31]},
    }
