"""
src/models.py
=============
Modeling layer (Section IV-D, Table III): two parallel tracks trained on
identical data splits.

Interpretable track  -> Logistic Regression, depth-limited Decision Tree,
                         Explainable Boosting Machine (EBM). Resampled with
                         SMOTE (see src/preprocessing.apply_smote).
High-performance track -> Random Forest, XGBoost, LightGBM. Trained on the
                         original class distribution with class weighting
                         instead of oversampling.

Every model is tuned with stratified 5-fold CV optimizing F1 (not accuracy,
since accuracy is misleading under the class imbalance in both network
datasets).
"""

from typing import Dict, Tuple

from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import CV_FOLDS, CV_SCORING, PARAM_GRIDS, RANDOM_SEED, quick_param_grids
from src.utils import get_logger

logger = get_logger(__name__)


def _base_estimator(model_name: str, y_train_for_weight=None, seed: int = RANDOM_SEED):
    """Instantiate an untuned estimator for `model_name` with the right
    imbalance handling for its track already wired in."""
    if model_name == "logistic_regression":
        return LogisticRegression(penalty="l2", max_iter=2000, random_state=seed)

    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)

    if model_name == "ebm":
        return ExplainableBoostingClassifier(max_bins=256, random_state=seed)

    if model_name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1)

    if model_name == "xgboost":
        scale_pos_weight = 1.0
        if y_train_for_weight is not None:
            pos = max(int((y_train_for_weight == 1).sum()), 1)
            neg = max(int((y_train_for_weight == 0).sum()), 1)
            scale_pos_weight = neg / pos
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "lightgbm":
        return LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=-1, verbosity=-1)

    raise ValueError(f"Unknown model '{model_name}'")


def train_and_tune(
    model_name: str,
    X_train,
    y_train,
    quick_mode: bool = False,
    cv_folds: int = CV_FOLDS,
    scoring: str = CV_SCORING,
    seed: int = RANDOM_SEED,
) -> Tuple[object, Dict]:
    """
    Grid-search `model_name` over the grid in Table III (or a 1-point grid
    in quick_mode) using stratified k-fold CV optimizing F1. Returns the
    refit best estimator and a small dict of CV diagnostics.
    """
    grids = quick_param_grids() if quick_mode else PARAM_GRIDS
    param_grid = grids[model_name]
    folds = 2 if quick_mode else cv_folds

    estimator = _base_estimator(model_name, y_train_for_weight=y_train, seed=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    search = GridSearchCV(
        estimator, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True
    )
    logger.info("Tuning %s over %s (cv=%d, scoring=%s)...", model_name, param_grid, folds, scoring)
    search.fit(X_train, y_train)

    diagnostics = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
    }
    logger.info("%s best params: %s (CV F1=%.4f)", model_name, search.best_params_, search.best_score_)
    return search.best_estimator_, diagnostics


def train_interpretable_track(splits: dict, quick_mode: bool = False, seed: int = RANDOM_SEED):
    """Train LR, Decision Tree, EBM on the SMOTE-resampled training split."""
    X_train, y_train = splits["X_train_smote"], splits["y_train_smote"]
    models, diagnostics = {}, {}
    for name in ("logistic_regression", "decision_tree", "ebm"):
        model, diag = train_and_tune(name, X_train, y_train, quick_mode=quick_mode, seed=seed)
        models[name] = model
        diagnostics[name] = diag
    return models, diagnostics


def train_high_performance_track(splits: dict, quick_mode: bool = False, seed: int = RANDOM_SEED):
    """Train RF, XGBoost, LightGBM on the original (class-weighted) training split."""
    X_train, y_train = splits["X_train"], splits["y_train"]
    models, diagnostics = {}, {}
    for name in ("random_forest", "xgboost", "lightgbm"):
        model, diag = train_and_tune(name, X_train, y_train, quick_mode=quick_mode, seed=seed)
        models[name] = model
        diagnostics[name] = diag
    return models, diagnostics
