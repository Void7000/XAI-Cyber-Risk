"""
src/utils.py
============
Small shared helpers used across every layer of the pipeline.
"""

import json
import logging
import os
import random
import sys

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix every RNG the pipeline touches so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a configured stdout logger, safe to call repeatedly."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def human_readable(feature_names):
    """
    Preprocessing keeps every feature human-readable (Section IV-C): this is
    a defensive pass that just strips internal separators some raw datasets
    ship with (e.g. 'Flow Byts/s' -> 'Flow Byts/s' unchanged, ' Fwd Pkt Len '
    -> 'Fwd Pkt Len'), so a SHAP/LIME plot column never shows an opaque
    encoded name.
    """
    return [str(f).strip().replace("_", " ") for f in feature_names]
