"""
src/pipeline.py
================
Runs the full five-layer pipeline end to end for one dataset:

    data -> preprocessing/split -> [interpretable | high-performance] models
    -> explainability -> evaluation -> comparison table -> saved artifacts
    for the presentation layer (dashboard/app.py).

Each layer is independently testable (see scripts/smoke_test.py); this
module just wires them together in the order Fig. 1 in the paper describes.
"""

import os

import joblib
import pandas as pd

from config import DATASETS, QUICK_MODE, RANDOM_SEED, RESULTS_DIR, TOP_K_FEATURES
from src.data_loading import load_dataset
from src.evaluation import (
    build_comparison_table,
    compare_to_intrinsic,
    fidelity_score,
    generate_review_packet,
    predictive_metrics,
    stability_score,
)
from src.explainability import (
    compute_lime_topk,
    compute_shap_topk,
    get_flagged_instances,
    intrinsic_global_importance,
    lime_global_importance,
)
from src.models import train_high_performance_track, train_interpretable_track
from src.preprocessing import prepare_splits
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def run_full_pipeline(
    dataset_name: str,
    quick_mode: bool = QUICK_MODE,
    force_synthetic: bool = False,
    seed: int = RANDOM_SEED,
    top_k: int = TOP_K_FEATURES,
    n_synthetic_rows: int = 20000,
) -> dict:
    set_seed(seed)
    out_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Layer 1: data ----------------------------------------------------
    df, meta = load_dataset(
        dataset_name, n_synthetic_rows=n_synthetic_rows, force_synthetic=force_synthetic, seed=seed
    )
    split_strategy = DATASETS[dataset_name]["split"]
    logger.info("Loaded %s: %d rows (synthetic=%s)", dataset_name, len(df), meta["synthetic"])

    # ---- Layers 2-3: preprocessing + split ---------------------------------
    splits = prepare_splits(df, split_strategy=split_strategy, seed=seed)
    feature_names = splits["feature_names"]

    # ---- Layer 4: parallel modeling tracks ---------------------------------
    interp_models, interp_diag = train_interpretable_track(splits, quick_mode=quick_mode, seed=seed)
    hp_models, hp_diag = train_high_performance_track(splits, quick_mode=quick_mode, seed=seed)
    all_models = {**interp_models, **hp_models}

    # ---- Predictive metrics, every model ----
    predictive = {}
    for name, model in all_models.items():
        y_pred = model.predict(splits["X_test"])
        y_proba = model.predict_proba(splits["X_test"])[:, 1]
        predictive[name] = predictive_metrics(splits["y_test"], y_pred, y_proba)

    # ---- Intrinsic ground truth from the interpretable track ----
    intrinsic_rankings = {
        name: intrinsic_global_importance(model, name, feature_names)
        for name, model in interp_models.items()
    }
    # Logistic Regression's ranking is the primary reference ground truth:
    # it is the most directly interpretable of the three intrinsic models.
    reference_ranking = intrinsic_rankings["logistic_regression"]

    # ---- Layer 5: explainability + evaluation, high-performance track ----
    explanation_quality, review_material = {}, {}

    for name, model in hp_models.items():
        flagged = get_flagged_instances(model, splits["X_test"], seed=seed)

        shap_df, shap_topk, shap_global = compute_shap_topk(
            model, splits["X_train"], flagged, feature_names, k=top_k
        )
        lime_topk, lime_raw = compute_lime_topk(
            model, splits["X_train"], flagged, feature_names, k=top_k, seed=seed
        )
        lime_global = lime_global_importance(lime_raw, feature_names)

        for explainer_name, topk_sets, global_ranking in (
            ("shap", shap_topk, shap_global),
            ("lime", lime_topk, lime_global),
        ):
            fid = fidelity_score(model, flagged, topk_sets, feature_names, seed=seed)
            stab = stability_score(flagged, topk_sets)
            gt = compare_to_intrinsic(global_ranking, reference_ranking, k=top_k)
            explanation_quality[f"{name}/{explainer_name}"] = {**fid, "stability": stab, **gt}

        review_material[f"{name}/shap"] = [
            [
                (f, float(shap_df.loc[i, f]))
                for f in shap_df.loc[i].abs().sort_values(ascending=False).head(top_k).index
            ]
            for i in shap_df.index
        ]
        review_material[f"{name}/lime"] = lime_raw

    # ---- Interpretable track: fidelity/stability of its own intrinsic ranking ----
    for name, model in interp_models.items():
        ranking = intrinsic_rankings[name]
        topk_set = set(ranking.head(top_k).index)
        sample = splits["X_test"].sample(n=min(25, len(splits["X_test"])), random_state=seed)
        topk_sets = [topk_set for _ in range(len(sample))]
        fid = fidelity_score(model, sample, topk_sets, feature_names, seed=seed)
        stab = stability_score(sample, topk_sets)
        explanation_quality[f"{name}/intrinsic"] = {**fid, "stability": stab}

    # ---- Usability: blank structured-review packet for a human reviewer ----
    review_path = os.path.join(out_dir, "usability_review_packet.csv")
    generate_review_packet(review_material, review_path)

    # ---- Comparison table (Section V) ----
    comparison_table = build_comparison_table(predictive, explanation_quality)
    comparison_table.to_csv(os.path.join(out_dir, "comparison_table.csv"), index=False)

    # ---- Presentation-layer bundle for dashboard/app.py ----
    best_model_name = comparison_table.iloc[0]["model"] if not comparison_table.empty else "random_forest"
    best_model = all_models[best_model_name]
    sample_idx = splits["X_test"].sample(n=min(200, len(splits["X_test"])), random_state=seed).index
    dashboard_bundle = {
        "dataset_name": dataset_name,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "feature_names": feature_names,
        "X_test_sample": splits["X_test"].loc[sample_idx],
        "y_test_sample": splits["y_test"].loc[sample_idx],
        "reference_ranking": reference_ranking,
    }
    joblib.dump(dashboard_bundle, os.path.join(out_dir, "dashboard_bundle.joblib"))

    logger.info("Pipeline complete for %s -- artifacts in %s", dataset_name, out_dir)

    return {
        "meta": meta,
        "splits": splits,
        "interpretable_models": interp_models,
        "high_performance_models": hp_models,
        "diagnostics": {**interp_diag, **hp_diag},
        "predictive": predictive,
        "explanation_quality": explanation_quality,
        "comparison_table": comparison_table,
        "out_dir": out_dir,
    }


def run_all_datasets(dataset_names=None, **kwargs) -> pd.DataFrame:
    """Run the pipeline over several datasets and stack their comparison
    tables (with a `dataset` column) for a cross-dataset view."""
    dataset_names = dataset_names or [d for d in DATASETS if DATASETS[d]["split"] is not None]
    tables = []
    for name in dataset_names:
        result = run_full_pipeline(name, **kwargs)
        table = result["comparison_table"].copy()
        table.insert(0, "dataset", name)
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "comparison_table_all_datasets.csv"), index=False)
    return combined
