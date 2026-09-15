"""
dashboard/app.py
=================
Presentation layer (Section IV-A, bottom of Fig. 1): an analyst-facing
dashboard showing a flagged event's risk score, its top contributing
factors, and a global model-behavior overview.

Run after at least one `scripts/run_pipeline.py` call has populated
`results/<dataset>/`:

    streamlit run dashboard/app.py

The dashboard reads only the artifacts the pipeline already saved
(`dashboard_bundle.joblib`, `comparison_table.csv`) -- it does not retrain
anything, so switching datasets in the sidebar is instant.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import shap
import streamlit as st

from config import RESULTS_DIR, TOP_K_FEATURES
from src.explainability import intrinsic_local_explanation, positive_class_shap_values

INTERPRETABLE = {"logistic_regression", "decision_tree", "ebm"}

st.set_page_config(page_title="XAI Cyber-Risk Dashboard", layout="wide")


@st.cache_data
def available_datasets():
    root = Path(RESULTS_DIR)
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "dashboard_bundle.joblib").exists()
    )


@st.cache_resource
def load_bundle(dataset_name: str):
    return joblib.load(Path(RESULTS_DIR) / dataset_name / "dashboard_bundle.joblib")


@st.cache_data
def load_comparison_table(dataset_name: str):
    path = Path(RESULTS_DIR) / dataset_name / "comparison_table.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def local_shap_for_instance(model, X_background: pd.DataFrame, instance: pd.Series, feature_names):
    bg = X_background.sample(n=min(100, len(X_background)), random_state=42)
    explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
    raw = explainer.shap_values(instance.to_frame().T)
    values = positive_class_shap_values(raw, len(feature_names))[0]
    return pd.Series(values, index=feature_names).sort_values(key=abs, ascending=False)


def main():
    st.title("Analyst Dashboard -- XAI Cyber-Risk Assessment")
    st.caption(
        "Presentation layer of the five-layer pipeline: risk score, top contributing "
        "factors, and a global model-behavior overview for a flagged event."
    )

    datasets = available_datasets()
    if not datasets:
        st.warning(
            "No results found yet. Run the pipeline first, e.g.:\n\n"
            "`python scripts/run_pipeline.py --dataset cicids2017`\n\n"
            "or the fast synthetic check: `python scripts/smoke_test.py`"
        )
        return

    with st.sidebar:
        dataset_name = st.selectbox("Dataset", datasets)
        bundle = load_bundle(dataset_name)
        st.markdown(f"**Deployed model:** `{bundle['best_model_name']}`")
        st.markdown(
            "Selected as the top row of the comparison table "
            "(highest F1 among all model-explainer combinations)."
        )

    model = bundle["best_model"]
    model_name = bundle["best_model_name"]
    feature_names = bundle["feature_names"]
    X_sample = bundle["X_test_sample"]
    y_sample = bundle["y_test_sample"]

    tab_event, tab_global, tab_compare = st.tabs(
        ["Flagged event", "Global model behavior", "Model-explainer comparison"]
    )

    # ------------------------------------------------------------------ #
    # Tab 1: a single flagged event -- risk score + top contributing factors
    # ------------------------------------------------------------------ #
    with tab_event:
        proba = model.predict_proba(X_sample)[:, 1]
        options = list(X_sample.index)
        labels = {
            idx: f"Instance {idx}  |  risk={p:.2f}  |  true label={int(y_sample.loc[idx])}"
            for idx, p in zip(options, proba)
        }
        chosen = st.selectbox("Choose an instance from the held-out test sample", options, format_func=lambda i: labels[i])

        instance = X_sample.loc[chosen]
        risk_score = model.predict_proba(instance.to_frame().T)[0, 1]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Risk score (P[malicious])", f"{risk_score:.1%}")
            st.metric("True label", "Malicious" if y_sample.loc[chosen] == 1 else "Benign")

        with col2:
            st.subheader("Top contributing factors")
            if model_name in INTERPRETABLE:
                explanation = intrinsic_local_explanation(model, model_name, instance, feature_names)
                if isinstance(explanation, list):  # decision-tree path
                    st.write("Decision path this instance followed:")
                    for step in explanation:
                        st.markdown(f"- {step}")
                else:
                    st.bar_chart(explanation.head(TOP_K_FEATURES))
            else:
                shap_series = local_shap_for_instance(
                    model, bundle["X_test_sample"], instance, feature_names
                )
                st.bar_chart(shap_series.head(TOP_K_FEATURES))
                st.caption("SHAP local attribution (TreeSHAP), signed contribution to the risk score.")

    # ------------------------------------------------------------------ #
    # Tab 2: global model-behavior overview
    # ------------------------------------------------------------------ #
    with tab_global:
        st.subheader(f"Global feature importance -- {model_name}")
        ranking = bundle["reference_ranking"]
        st.caption(
            "Reference ranking from the interpretable track's Logistic Regression "
            "coefficients -- the intrinsic ground truth SHAP/LIME are evaluated against."
        )
        st.bar_chart(ranking.head(TOP_K_FEATURES))

    # ------------------------------------------------------------------ #
    # Tab 3: full comparison table (Section V)
    # ------------------------------------------------------------------ #
    with tab_compare:
        table = load_comparison_table(dataset_name)
        if table.empty:
            st.info("No comparison table found for this dataset.")
        else:
            st.dataframe(table, width="stretch")
            st.caption(
                "Precision/Recall/F1/PR-AUC/ROC-AUC are predictive metrics; fidelity, "
                "stability, and the vs-intrinsic columns are explanation-quality metrics "
                "(Section IV-E). Usability columns populate once "
                "`usability_review_packet.csv` has been scored and aggregated."
            )


if __name__ == "__main__":
    main()
