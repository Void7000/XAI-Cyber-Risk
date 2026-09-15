"""
src/explainability.py
======================
Explainability layer (Section IV-D): three kinds of explanation.

1. Intrinsic  -- coefficients (Logistic Regression), impurity-based feature
   importance / decision path (Decision Tree), and shape functions (EBM)
   from the interpretable track. These serve as the ground truth that
   SHAP/LIME outputs are measured against (Section IV-D, closing the
   baseline gap from Section III).

2. SHAP (TreeSHAP) -- exact global and local attributions for the three
   high-performance models.

3. LIME -- a local surrogate model explaining the same flagged instances
   SHAP explains, for the high-performance models.

All three return "top-k feature sets" in a common shape so evaluation.py can
compute fidelity, stability, and ground-truth agreement identically
regardless of which explainer produced them.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import lime.lime_tabular
import numpy as np
import pandas as pd
import shap

from config import LIME_N_FLAGGED_INSTANCES, RANDOM_SEED, SHAP_BACKGROUND_SIZE, TOP_K_FEATURES
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Intrinsic explanations (interpretable track -> ground truth)
# ---------------------------------------------------------------------------
def intrinsic_global_importance(model, model_name: str, feature_names: List[str]) -> pd.Series:
    """Return a descending Series of feature -> importance, the reference
    ranking that SHAP/LIME rankings are compared against."""
    if model_name == "logistic_regression":
        scores = np.abs(model.coef_[0])

    elif model_name == "decision_tree":
        scores = model.feature_importances_

    elif model_name == "ebm":
        global_exp = model.explain_global()
        data = global_exp.data()
        # interpret pairs 'names' with 'scores'; interaction terms (named
        # "A x B") are dropped here since they don't map onto a single
        # feature for top-k comparison against SHAP/LIME.
        pairs = [
            (n, s) for n, s in zip(data["names"], data["scores"]) if " x " not in str(n)
        ]
        names = [p[0] for p in pairs]
        scores = np.array([p[1] for p in pairs])
        return pd.Series(scores, index=names).sort_values(ascending=False)

    else:
        raise ValueError(f"No intrinsic explanation defined for '{model_name}'")

    return pd.Series(scores, index=feature_names).sort_values(ascending=False)


def intrinsic_local_explanation(model, model_name: str, instance: pd.Series, feature_names: List[str]):
    """
    A single-instance intrinsic explanation:
      - Logistic Regression: coefficient * feature value (signed contribution)
      - Decision Tree: the sequence of (feature, threshold, direction) nodes
        the instance actually traverses -- the decision path
      - EBM: per-feature shape-function value at this instance's value
    """
    if model_name == "logistic_regression":
        contrib = model.coef_[0] * instance.values
        return pd.Series(contrib, index=feature_names).sort_values(key=np.abs, ascending=False)

    if model_name == "decision_tree":
        tree = model.tree_
        node_indicator = model.decision_path(instance.values.reshape(1, -1))
        node_ids = node_indicator.indices
        path = []
        for node_id in node_ids:
            if tree.feature[node_id] < 0:  # leaf
                continue
            feat = feature_names[tree.feature[node_id]]
            thresh = tree.threshold[node_id]
            direction = "<=" if instance[feat] <= thresh else ">"
            path.append(f"{feat} {direction} {thresh:.3f}")
        return path

    if model_name == "ebm":
        local_exp = model.explain_local(instance.values.reshape(1, -1))
        data = local_exp.data(0)
        pairs = [(n, s) for n, s in zip(data["names"], data["scores"]) if " x " not in str(n)]
        return pd.Series(dict(pairs)).sort_values(key=np.abs, ascending=False)

    raise ValueError(f"No intrinsic local explanation defined for '{model_name}'")


# ---------------------------------------------------------------------------
# Shared: pick the "flagged instances" both SHAP and LIME will explain
# ---------------------------------------------------------------------------
def get_flagged_instances(
    model, X: pd.DataFrame, n: int = LIME_N_FLAGGED_INSTANCES, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Sample instances the model actually predicts positive (malicious) --
    SHAP and LIME then explain the same set, as in Section IV-D."""
    preds = model.predict(X)
    flagged_idx = np.where(preds == 1)[0]
    if len(flagged_idx) == 0:
        logger.warning("No positive predictions to explain; falling back to a random sample.")
        flagged_idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(flagged_idx, size=min(n, len(flagged_idx)), replace=False)
    return X.iloc[chosen]


# ---------------------------------------------------------------------------
# 2. SHAP (TreeSHAP)
# ---------------------------------------------------------------------------
def positive_class_shap_values(raw, n_features: int) -> np.ndarray:
    """Normalize SHAP's several return shapes to a single (n, n_features)
    array of positive-class attributions, across shap/sklearn/xgboost/lightgbm
    versions (list-of-per-class-arrays, or a single 3-D (n, features, classes)
    array). Used everywhere a SHAP explanation is computed so every caller
    handles every model family identically."""
    if isinstance(raw, list):
        return raw[1] if len(raw) > 1 else raw[0]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        return raw[:, :, 1] if raw.shape[2] > 1 else raw[:, :, 0]
    return raw


def compute_shap_topk(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    feature_names: List[str],
    k: int = TOP_K_FEATURES,
) -> Tuple[pd.DataFrame, List[Set[str]], pd.Series]:
    """
    TreeSHAP global + local attributions for a high-performance model.

    Returns
    -------
    shap_df       : (n_explain, n_features) signed SHAP values
    topk_per_row  : list of top-k feature name sets, one per explained instance
    global_import : mean(|SHAP|) per feature, descending
    """
    bg = X_background.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(X_background)), random_state=RANDOM_SEED
    )
    explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
    raw = explainer.shap_values(X_explain)
    values = positive_class_shap_values(raw, len(feature_names))

    shap_df = pd.DataFrame(values, columns=feature_names, index=X_explain.index)
    topk_per_row = [
        set(shap_df.loc[i].abs().sort_values(ascending=False).head(k).index) for i in shap_df.index
    ]
    global_importance = shap_df.abs().mean(axis=0).sort_values(ascending=False)
    return shap_df, topk_per_row, global_importance


# ---------------------------------------------------------------------------
# 3. LIME
# ---------------------------------------------------------------------------
def _lime_feature_name(description: str, feature_names: List[str]) -> str:
    """LIME's discretized rule text (e.g. '3.00 < duration <= 5.00') embeds
    the true feature name; recover it by longest-match against known names."""
    for name in sorted(feature_names, key=len, reverse=True):
        if name in description:
            return name
    return description  # fallback: return the raw rule text


def compute_lime_topk(
    model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    feature_names: List[str],
    k: int = TOP_K_FEATURES,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Set[str]], List[list]]:
    """
    LIME local surrogate explanations for the same flagged instances SHAP
    explained. Returns per-instance top-k feature sets and the raw
    (feature, weight) lists for downstream reporting/usability review.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_background.values,
        feature_names=feature_names,
        class_names=["benign", "malicious"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )

    topk_per_row, raw_explanations = [], []
    for _, row in X_explain.iterrows():
        exp = explainer.explain_instance(
            row.values, model.predict_proba, num_features=k, labels=(1,)
        )
        pairs = exp.as_list(label=1)
        resolved = [(_lime_feature_name(desc, feature_names), w) for desc, w in pairs]
        raw_explanations.append(resolved)
        topk_per_row.append({name for name, _ in resolved})
    return topk_per_row, raw_explanations


def lime_global_importance(raw_explanations: List[list], feature_names: List[str]) -> pd.Series:
    """
    Collapse per-instance LIME (feature, weight) lists into a single global
    ranking -- mean(|weight|) per feature across explained instances -- so
    LIME can be compared to the intrinsic ground truth the same way SHAP is.
    """
    sums, counts = defaultdict(float), defaultdict(int)
    for row in raw_explanations:
        for name, weight in row:
            if name in feature_names:
                sums[name] += abs(weight)
                counts[name] += 1
    means = {name: sums[name] / counts[name] for name in sums}
    return pd.Series(means).sort_values(ascending=False)
