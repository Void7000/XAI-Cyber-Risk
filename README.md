# XAI for Cyber Risk Assessment in High-Risk Critical Infrastructure

Code implementation of the five-layer methodology from *"Explainable
Artificial Intelligence for Cyber Risk Assessment in High-Risk Critical
Infrastructure Systems"* (Mahfuz, Ahamed, Singha -- Information Security
Management). The paper is a literature review plus a proposed methodology;
this repo is the pipeline that methodology describes, built to run on
Google Colab.

Two model tracks are trained on identical splits of the same data,
explained with SHAP, LIME, and each interpretable model's own intrinsic
explanation, then scored on both predictive performance and explanation
quality (fidelity, stability, and agreement with an intrinsic ground truth):

```
Data layer (CICIDS2017, UNSW-NB15, SWaT, WADI, CVE/NVD)
        |
Preprocessing & feature engineering (cleaning, encoding, SMOTE / class weighting)
        |
Data split (stratified 70/15/15, or time-based for SWaT/WADI)
        |
   /-------------------------\
Interpretable track       High-performance track
(LR, Decision Tree, EBM)  (Random Forest, XGBoost, LightGBM)
   \-------------------------/
        |
Explainability layer (intrinsic | TreeSHAP | LIME)
        |
Evaluation layer (Precision/Recall/F1/PR-AUC/ROC-AUC, fidelity, stability, usability)
        |
Presentation layer (Streamlit analyst dashboard)
```

## Quickstart on Google Colab

```python
!git clone https://github.com/<your-username>/xai-cyber-risk.git
%cd xai-cyber-risk
!pip install -q -r requirements.txt

# ~1 minute wiring check on synthetic data -- confirms everything imports
# and runs before you spend time on real data or the full grids
!python scripts/smoke_test.py
```

Then open `notebooks/XAI_Cyber_Risk_Colab.ipynb` and run it top to bottom,
or drive the pipeline directly:

```python
import sys; sys.path.insert(0, ".")
from src.pipeline import run_full_pipeline

result = run_full_pipeline("cicids2017", quick_mode=True, force_synthetic=True)
result["comparison_table"]
```

Set `force_synthetic=False` (the default) once you've placed real data under
`data/raw/` -- see [`data/README.md`](data/README.md) for exactly where
each of the four sources goes and how to get it, including the iTrust
request process for SWaT/WADI. Until then, every run automatically falls
back to a structurally-matched synthetic dataset instead of failing, so the
repo is always runnable, including for anyone grading it without access to
the gated datasets.

## Local setup

```bash
git clone https://github.com/<your-username>/xai-cyber-risk.git
cd xai-cyber-risk
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

python scripts/smoke_test.py                             # ~1 min, synthetic, wiring check
python scripts/run_pipeline.py --dataset cicids2017       # full run, real or synthetic data
python scripts/run_pipeline.py --dataset all              # every dataset with a split strategy
streamlit run dashboard/app.py                             # presentation layer
```

`scripts/run_pipeline.py` flags:

| Flag | Effect |
|---|---|
| `--dataset {cicids2017,unsw_nb15,swat,wadi,all}` | which data-layer source to run |
| `--quick` | 1-point hyperparameter grids, 2-fold CV -- fast wiring runs |
| `--force-synthetic` | ignore any real files under `data/raw/` |
| `--n-rows N` | synthetic row count (default 20000); lower it for a faster ICS run, since EBM tuning scales with feature count and WADI has 123 |
| `--seed N` | random seed (default 42) |

## Repository layout

```
config.py                     paths, seed, dataset registry, Table III hyperparameter grids
src/
  data_loading.py             data layer: real loaders + synthetic fallbacks + CVE enrichment
  preprocessing.py            cleaning, encoding, SMOTE, stratified/time split
  models.py                   both tracks, grid-search tuning (Table III)
  explainability.py           intrinsic / TreeSHAP / LIME, normalized to top-k feature sets
  evaluation.py                predictive metrics, fidelity, stability, ground-truth agreement,
                                usability review packet + aggregation
  pipeline.py                  wires the five layers together, saves results/ artifacts
scripts/
  run_pipeline.py              CLI entry point
  smoke_test.py                fast synthetic wiring check
dashboard/
  app.py                       Streamlit presentation layer
notebooks/
  XAI_Cyber_Risk_Colab.ipynb   Colab-ready walkthrough of the whole pipeline
data/
  README.md                    where to get each of the four sources, and where to put them
  raw/<dataset>/                you place real CSVs here (git-ignored)
results/<dataset>/              per-dataset outputs: comparison_table.csv, dashboard_bundle.joblib,
                                  usability_review_packet.csv (git-ignored, regenerate by re-running)
```

## What each evaluation metric means

- **Predictive**: Precision, Recall, F1, PR-AUC, ROC-AUC on the held-out
  test split. Models are tuned on F1, not accuracy, because both network
  datasets are class-imbalanced.
- **Fidelity**: for a sample of flagged instances, the top-k features an
  explanation names as important are replaced with random draws from that
  feature's own distribution, and the mean absolute change in predicted
  risk score is measured. A random-feature control is reported alongside so
  you can see whether the "important" features actually move the
  prediction more than an arbitrary same-size set does (`fidelity_ratio`).
- **Stability** (Eq. 1 in the paper): for each explained instance, its
  nearest neighbor in feature space is found, and the Jaccard similarity of
  their top-k feature sets is computed. Averaged across the sample.
- **Ground-truth agreement**: Spearman rank correlation and top-k Jaccard
  overlap between a SHAP/LIME global ranking and the interpretable track's
  intrinsic ranking (Logistic Regression coefficients by default) --
  operationalizing the paper's point that SHAP/LIME outputs should be
  measured against an intrinsic baseline rather than accepted on their own.
- **Usability**: not automatable. `evaluation.generate_review_packet()`
  writes a CSV with each explanation's top factors and blank 1-5 columns
  for a human reviewer to score (clarity, completeness, actionability,
  trust); `evaluation.aggregate_usability_scores()` combines one completed
  CSV per reviewer into a mean + inter-rater standard deviation per
  model-explainer pair. Score `results/<dataset>/usability_review_packet.csv`
  yourself (or recruit a couple of peers) and re-load it to fill in that
  column of the comparison table.

## Notes and honest limitations

- **Binary classification.** The pipeline treats every source as
  benign-vs-malicious. `_harmonize_label()` in `src/data_loading.py`
  collapses multi-class attack-type columns to binary but keeps the
  original as `attack_type` if you want to extend this to multi-class.
- **`xgboost<3.0` is pinned deliberately.** SHAP's `TreeExplainer` cannot
  yet parse the `base_score` format XGBoost >= 3.0 writes
  ([shap#4288](https://github.com/shap/shap/issues/4288)); this repo pins
  around it rather than working around it at explanation time.
- **EBM is slow at width.** `ExplainableBoostingClassifier` tuning time
  scales with feature count; on WADI's 123 features it is the slowest step
  in the pipeline by a wide margin. Use `--quick` and a smaller `--n-rows`
  for iteration, and expect the full Table III grid on real data to take
  meaningfully longer than on the network datasets.
- **Synthetic data is for wiring checks only.** It is generated to have a
  handful of genuinely informative features so models and explainers have
  something real to find, but it is not a substitute for CICIDS2017,
  UNSW-NB15, SWaT, or WADI -- never report synthetic-data numbers as
  results.
