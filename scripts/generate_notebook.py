"""
One-off generator for notebooks/XAI_Cyber_Risk_Colab.ipynb. Run this if you
want to regenerate the notebook after editing its cell content below; the
.ipynb file itself is the deliverable, this script is not part of the
pipeline.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# --------------------------------------------------------------------------
md(r"""# XAI for Cyber Risk Assessment -- Colab walkthrough

Runs the five-layer pipeline from *"Explainable Artificial Intelligence for
Cyber Risk Assessment in High-Risk Critical Infrastructure Systems"*
end to end: data -> preprocessing/split -> two model tracks -> SHAP/LIME/
intrinsic explanations -> evaluation -> comparison table.

**First time here?** Just Run All. Every loader falls back to a
structurally-matched synthetic dataset when real data isn't found under
`data/raw/`, so this notebook works with nothing downloaded. See
`data/README.md` in the repo for how to plug in the real datasets
(CICIDS2017, UNSW-NB15 are Kaggle mirrors; SWaT/WADI need an iTrust access
request).""")

# --------------------------------------------------------------------------
md("## 0. Setup")

code(r"""# If you're opening this notebook directly in Colab (not via a cloned repo),
# uncomment the clone + %cd below. If you already have the repo (e.g. you
# uploaded it, or Colab opened it from GitHub with the repo alongside it),
# skip straight to the pip install.

# !git clone https://github.com/<your-username>/xai-cyber-risk.git
# %cd xai-cyber-risk

!pip install -q -r requirements.txt""")

code(r"""import sys
sys.path.insert(0, ".")

from config import DATASETS, TOP_K_FEATURES
from src.pipeline import run_full_pipeline, run_all_datasets
from src.utils import set_seed

set_seed(42)
list(DATASETS.keys())""")

# --------------------------------------------------------------------------
md(r"""## 1. Fast wiring check

Synthetic data, 1-point hyperparameter grids, 2-fold CV -- confirms every
layer runs before spending time on the full grids or real data. Takes about
a minute (EBM is the slow step).""")

code(r"""quick_result = run_full_pipeline(
    "cicids2017", quick_mode=True, force_synthetic=True, seed=42
)
quick_result["comparison_table"]""")

# --------------------------------------------------------------------------
md(r"""## 2. A closer look at each layer

The cells below re-run the same steps `run_full_pipeline()` just did, one
layer at a time, so you can inspect intermediate output -- useful for a
report or a presentation appendix.""")

code(r"""from src.data_loading import load_dataset

DATASET_NAME = "cicids2017"   # change to unsw_nb15 / swat / wadi
df, meta = load_dataset(DATASET_NAME, force_synthetic=True)  # drop force_synthetic once data/raw/ is populated
print(meta)
df.head()""")

code(r"""from src.preprocessing import prepare_splits

split_strategy = DATASETS[DATASET_NAME]["split"]
splits = prepare_splits(df, split_strategy=split_strategy, seed=42)
feature_names = splits["feature_names"]
{k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in splits.items() if k != "feature_names"}""")

code(r"""from src.models import train_interpretable_track, train_high_performance_track

# quick_mode=True below for notebook speed; set False for the full Table III grid
interp_models, interp_diag = train_interpretable_track(splits, quick_mode=True, seed=42)
hp_models, hp_diag = train_high_performance_track(splits, quick_mode=True, seed=42)
{**interp_diag, **hp_diag}""")

code(r"""from src.evaluation import predictive_metrics
import pandas as pd

predictive = {}
for name, model in {**interp_models, **hp_models}.items():
    y_pred = model.predict(splits["X_test"])
    y_proba = model.predict_proba(splits["X_test"])[:, 1]
    predictive[name] = predictive_metrics(splits["y_test"], y_pred, y_proba)

pd.DataFrame(predictive).T""")

code(r"""from src.explainability import (
    intrinsic_global_importance, get_flagged_instances,
    compute_shap_topk, compute_lime_topk, lime_global_importance,
)

# Intrinsic ground truth from the interpretable track
reference_ranking = intrinsic_global_importance(
    interp_models["logistic_regression"], "logistic_regression", feature_names
)

# SHAP + LIME for one high-performance model, on the same flagged instances
hp_name = "random_forest"
flagged = get_flagged_instances(hp_models[hp_name], splits["X_test"], seed=42)
shap_df, shap_topk, shap_global = compute_shap_topk(
    hp_models[hp_name], splits["X_train"], flagged, feature_names, k=TOP_K_FEATURES
)
lime_topk, lime_raw = compute_lime_topk(
    hp_models[hp_name], splits["X_train"], flagged, feature_names, k=TOP_K_FEATURES, seed=42
)

reference_ranking.head(10)""")

code(r"""from src.evaluation import fidelity_score, stability_score, compare_to_intrinsic

fid = fidelity_score(hp_models[hp_name], flagged, shap_topk, feature_names, seed=42)
stab = stability_score(flagged, shap_topk)
gt = compare_to_intrinsic(shap_global, reference_ranking, k=TOP_K_FEATURES)
print("SHAP explanation quality for", hp_name)
{**fid, "stability": stab, **gt}""")

code(r"""import matplotlib.pyplot as plt

shap_global.head(TOP_K_FEATURES).sort_values().plot(
    kind="barh", figsize=(6, 4), title=f"Global |SHAP| importance -- {hp_name}"
)
plt.xlabel("mean(|SHAP value|)")
plt.tight_layout()
plt.show()""")

# --------------------------------------------------------------------------
md(r"""## 3. Full pipeline run

Runs every layer for real, with the full Table III hyperparameter grids and
5-fold CV. This is much slower than the quick check above -- expect several
minutes per dataset on synthetic data, and considerably longer on the real
network datasets. Set `force_synthetic=False` once `data/raw/` is
populated (see `data/README.md`).""")

code(r"""FULL_RUN = False  # flip to True to actually run the full grids

if FULL_RUN:
    result = run_full_pipeline(DATASET_NAME, quick_mode=False, force_synthetic=True, seed=42)
    display(result["comparison_table"])
else:
    print("Skipped -- flip FULL_RUN to True when you're ready to spend the time.")""")

# --------------------------------------------------------------------------
md(r"""## 4. Cross-dataset comparison

Runs the whole pipeline over every dataset that has a split strategy
defined (CICIDS2017, UNSW-NB15, SWaT, WADI) and stacks their comparison
tables. Slow for the same reason as above -- start with `quick_mode=True`.""")

code(r"""all_tables = run_all_datasets(quick_mode=True, force_synthetic=True, seed=42)
all_tables""")

code(r"""import matplotlib.pyplot as plt

pivot = all_tables.pivot_table(index="model", columns="dataset", values="f1", aggfunc="mean")
pivot.plot(kind="bar", figsize=(8, 4), title="F1 by model and dataset (quick mode, synthetic)")
plt.ylabel("F1")
plt.tight_layout()
plt.show()""")

# --------------------------------------------------------------------------
md(r"""## 5. Usability review

`run_full_pipeline()` already wrote a blank review packet to
`results/<dataset>/usability_review_packet.csv` -- one row per
(model, explainer, instance) with the top factors an explanation surfaced
and empty 1-5 columns for a human reviewer to score. Download it, score it
yourself or with a couple of peers (one CSV per reviewer), then aggregate:""")

code(r"""from src.evaluation import aggregate_usability_scores

# after downloading, scoring, and re-uploading one completed CSV per reviewer:
# usability = aggregate_usability_scores([
#     "results/cicids2017/usability_review_reviewer1.csv",
#     "results/cicids2017/usability_review_reviewer2.csv",
# ])
# usability""")

# --------------------------------------------------------------------------
md(r"""## 6. Presentation layer (optional, best-effort in Colab)

`dashboard/app.py` is a Streamlit app meant to run locally
(`streamlit run dashboard/app.py`) once `results/<dataset>/` exists. Colab
doesn't expose local ports directly, so the common workaround is a tunnel
(`localtunnel` or `ngrok`). This is optional and environment-dependent --
if it doesn't work in your Colab runtime, run the dashboard locally instead.""")

code(r"""# Optional -- uncomment to try. Click the printed localtunnel URL once it appears.
# !npm install -q -g localtunnel
# get_ipython().system_raw("streamlit run dashboard/app.py --server.port 8501 &")
# !npx localtunnel --port 8501""")

nb["cells"] = cells
nbf.write(nb, "notebooks/XAI_Cyber_Risk_Colab.ipynb")
print("Notebook written.")
