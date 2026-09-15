"""
src/evaluation.py
==================
Evaluation layer (Section IV-E).

Predictive performance: Precision, Recall, F1, PR-AUC, ROC-AUC.

Explanation quality, computed for every model-explainer combination:
    - fidelity  : perturb the features an explanation ranks as important and
                  measure whether the model's output changes accordingly.
    - stability : Jaccard similarity of the top-k feature sets between
                  neighboring instances (Eq. 1 in the paper).
    - ground-truth agreement : rank correlation / top-k overlap between a
                  SHAP or LIME ranking and the interpretable track's
                  intrinsic ranking, operationalizing "SHAP and LIME outputs
                  are measured against this ground truth rather than
                  accepted on their own" (Section IV-D).
    - usability : a *structured peer review*, not something a script can
                  compute alone. This module builds the review packet and
                  aggregates completed reviews; humans do the scoring.
"""

from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import N_NEIGHBORS_STABILITY, N_PERTURBATIONS, RANDOM_SEED, TOP_K_FEATURES
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Predictive performance
# ---------------------------------------------------------------------------
def predictive_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
    metrics["roc_auc"] = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan")
    return {k: float(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Fidelity: perturb the top-k features an explanation flags, measure Δ output
# ---------------------------------------------------------------------------
def fidelity_score(
    model,
    X_explain: pd.DataFrame,
    topk_per_row: List[Set[str]],
    feature_names: List[str],
    n_perturbations: int = N_PERTURBATIONS,
    seed: int = RANDOM_SEED,
    include_random_control: bool = True,
) -> Dict[str, float]:
    """
    For each explained instance, replace its top-k "important" feature
    values with random draws from that feature's empirical distribution and
    measure the mean absolute change in predicted probability of the
    positive class. A faithful explanation's top-k features should move the
    prediction more than a random k-feature set does, so a random-feature
    control is reported alongside for context.
    """
    rng = np.random.default_rng(seed)
    baseline_proba = model.predict_proba(X_explain)[:, 1]
    topk_deltas, random_deltas = [], []

    for pos, (idx, topk) in enumerate(zip(X_explain.index, topk_per_row)):
        instance = X_explain.loc[idx]
        topk = [f for f in topk if f in feature_names]
        if not topk:
            continue
        random_k = list(rng.choice(feature_names, size=len(topk), replace=False))

        for cols, bucket in ((topk, topk_deltas), (random_k, random_deltas)):
            diffs = []
            for _ in range(n_perturbations):
                perturbed = instance.copy()
                for c in cols:
                    perturbed[c] = rng.choice(X_explain[c].values)
                new_proba = model.predict_proba(perturbed.values.reshape(1, -1))[:, 1][0]
                diffs.append(abs(new_proba - baseline_proba[pos]))
            bucket.append(float(np.mean(diffs)))

    result = {"fidelity_topk": float(np.mean(topk_deltas)) if topk_deltas else 0.0}
    if include_random_control:
        result["fidelity_random_control"] = float(np.mean(random_deltas)) if random_deltas else 0.0
        result["fidelity_ratio"] = (
            result["fidelity_topk"] / result["fidelity_random_control"]
            if result["fidelity_random_control"] > 0
            else float("nan")
        )
    return result


# ---------------------------------------------------------------------------
# Stability: Eq. 1, J(A,B) = |A ∩ B| / |A ∪ B|, for neighboring instances
# ---------------------------------------------------------------------------
def stability_score(
    X_explain: pd.DataFrame,
    topk_per_row: List[Set[str]],
    n_neighbors: int = N_NEIGHBORS_STABILITY,
) -> float:
    """Average Jaccard similarity of the top-k feature set between each
    explained instance and its nearest neighbor(s) in scaled feature space."""
    if len(X_explain) < 2:
        return float("nan")

    X_scaled = StandardScaler().fit_transform(X_explain.values)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(X_explain))).fit(X_scaled)
    _, indices = nn.kneighbors(X_scaled)

    jaccards = []
    for i in range(len(X_explain)):
        neighbor_ids = indices[i][1:]  # drop self at position 0
        for j in neighbor_ids:
            A, B = topk_per_row[i], topk_per_row[j]
            union = A | B
            jaccards.append(len(A & B) / len(union) if union else 0.0)
    return float(np.mean(jaccards)) if jaccards else 0.0


# ---------------------------------------------------------------------------
# Agreement with the interpretable track's intrinsic ground truth
# ---------------------------------------------------------------------------
def compare_to_intrinsic(
    explainer_importance: pd.Series, intrinsic_importance: pd.Series, k: int = TOP_K_FEATURES
) -> Dict[str, float]:
    """Spearman rank correlation and top-k Jaccard overlap between a SHAP/LIME
    global ranking and the interpretable track's intrinsic ranking."""
    shared = explainer_importance.index.intersection(intrinsic_importance.index)
    if len(shared) < 2:
        return {"spearman_vs_intrinsic": float("nan"), "topk_jaccard_vs_intrinsic": float("nan")}

    rho, _ = spearmanr(explainer_importance[shared], intrinsic_importance[shared])
    topk_a = set(explainer_importance.head(k).index)
    topk_b = set(intrinsic_importance.head(k).index)
    union = topk_a | topk_b
    jaccard = len(topk_a & topk_b) / len(union) if union else 0.0
    return {"spearman_vs_intrinsic": float(rho), "topk_jaccard_vs_intrinsic": float(jaccard)}


# ---------------------------------------------------------------------------
# Usability: structured peer review (human-scored, not automatable)
# ---------------------------------------------------------------------------
def generate_review_packet(
    explanations: Dict[str, List[list]], output_path: str
) -> pd.DataFrame:
    """
    Build a blank review packet: one row per (model, explainer, instance)
    with the top factors an explanation surfaced, and empty 1-5 columns for
    a reviewer to rate clarity, completeness, actionability, and trust.
    `explanations` keys are "model_name/explainer_name" strings mapping to a
    list of per-instance (feature, weight) lists.
    """
    rows = []
    for key, per_instance in explanations.items():
        model_name, explainer_name = key.split("/", 1)
        for i, exp in enumerate(per_instance):
            top_factors = "; ".join(f"{f} ({w:+.3f})" for f, w in exp[:5]) if exp else ""
            rows.append(
                {
                    "model": model_name,
                    "explainer": explainer_name,
                    "instance_id": i,
                    "top_factors": top_factors,
                    "clarity_1to5": "",
                    "completeness_1to5": "",
                    "actionability_1to5": "",
                    "trust_1to5": "",
                    "reviewer_comments": "",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Usability review packet written to %s (%d rows) -- fill in and re-load.", output_path, len(df))
    return df


def aggregate_usability_scores(review_csv_paths: List[str]) -> pd.DataFrame:
    """
    Combine one completed review packet per reviewer into a per
    (model, explainer) usability score: mean of the four 1-5 criteria, plus
    the cross-reviewer standard deviation as a simple agreement signal.
    """
    frames = []
    for i, path in enumerate(review_csv_paths):
        df = pd.read_csv(path)
        df["reviewer"] = f"reviewer_{i + 1}"
        frames.append(df)
    all_reviews = pd.concat(frames, ignore_index=True)

    score_cols = ["clarity_1to5", "completeness_1to5", "actionability_1to5", "trust_1to5"]
    all_reviews[score_cols] = all_reviews[score_cols].apply(pd.to_numeric, errors="coerce")
    all_reviews["usability_score"] = all_reviews[score_cols].mean(axis=1)

    summary = (
        all_reviews.groupby(["model", "explainer"])["usability_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["model", "explainer", "usability_mean", "usability_std", "n_ratings"]
    return summary


# ---------------------------------------------------------------------------
# Comparison table (Section V: "a comparative table ... for all
# model-explainer combinations")
# ---------------------------------------------------------------------------
def build_comparison_table(
    predictive: Dict[str, Dict[str, float]],
    explanation_quality: Dict[str, Dict[str, float]],
    usability: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    predictive          : {model_name: {precision, recall, f1, pr_auc, roc_auc}}
    explanation_quality  : {"model_name/explainer_name": {fidelity_topk, ...,
                             stability, spearman_vs_intrinsic, ...}}
    usability            : optional output of aggregate_usability_scores()
    """
    rows = []
    for key, eq_metrics in explanation_quality.items():
        model_name, explainer_name = key.split("/", 1)
        row = {"model": model_name, "explainer": explainer_name}
        row.update(predictive.get(model_name, {}))
        row.update(eq_metrics)
        rows.append(row)
    table = pd.DataFrame(rows)

    if usability is not None and not usability.empty:
        table = table.merge(usability, on=["model", "explainer"], how="left")

    sort_cols = [c for c in ("f1", "model") if c in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    return table
