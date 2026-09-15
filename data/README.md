# Data layer -- acquiring the four sources

The pipeline never ships real traffic data (CICIDS2017/UNSW-NB15 are multi-GB
Kaggle mirrors; SWaT/WADI require a signed access request). Every loader in
`src/data_loading.py` falls back to a structurally-matched **synthetic**
dataset when it can't find real files, so `scripts/smoke_test.py` and
`scripts/run_pipeline.py --force-synthetic` always work with nothing
downloaded. This file is how you swap in the real thing.

Drop CSVs under `data/raw/<dataset_name>/` exactly as named below --
`load_dataset()` auto-detects them and stops using synthetic data.

## CICIDS2017 (network IDS, Table II: ~2.8M flows, 78 features)

- Kaggle mirror: `cicdataset/cicids2017` (or the original at
  https://www.unb.ca/cic/datasets/ids-2017.html)
- Place every daily CSV (`Monday-WorkingHours.pcap_ISCX.csv`, etc.) directly
  under `data/raw/cicids2017/`. The loader concatenates all `*.csv` files it
  finds there and harmonizes the `Label` column to binary.

## UNSW-NB15 (network IDS, ~2.5M rows, 49 features)

- Kaggle mirror: `mrwellsdavid/unsw-nb15` (or
  https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- Place `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` under
  `data/raw/unsw_nb15/`. The loader concatenates both and harmonizes
  `label`/`attack_cat`.

## SWaT and WADI (ICS testbeds, 51 / 123 features)

These are **not** downloadable via a script. Both are distributed by iTrust,
Centre for Research in Cyber Security at the Singapore University of
Technology and Design, and require submitting a dataset-request form (an
academic email and a short project description are typically enough; access
is usually granted within a few days):

- Request portal: https://itrust.sutd.edu.sg/itrust-labs_datasets/
- SWaT: place `SWaT_Dataset_Normal_v1.csv` and `SWaT_Dataset_Attack_v0.csv`
  under `data/raw/swat/`.
- WADI: place `WADI_14days.csv` and `WADI_attackdata.csv` under
  `data/raw/wadi/`.

Both testbeds are physical process data recorded continuously, so
`config.py` marks their split strategy as `"time"`: rows are ordered by
`timestamp` and split by position, never shuffled, so adjacent time steps
never leak across train/val/test.

## CVE / NVD (vulnerability-context enrichment, not traffic data)

- NVD JSON data feeds: https://nvd.nist.gov/vuln/data-feeds
- NVD REST API (no bulk download needed for a small enrichment table):
  https://nvd.nist.gov/developers/vulnerabilities
- Build a CSV with columns `cve_id, cvss_score, exploit_maturity,
  affected_service` and place it at
  `data/raw/cve_nvd/nvd_cve_enrichment.csv`. `enrich_with_cve()` in
  `src/data_loading.py` joins this onto flagged rows by service, following
  the enrichment approach in [15] (severity/exploit-maturity features
  attached to an asset a model already flagged, not used as primary traffic
  features).

## Notes

- `data/raw/` is git-ignored (see `.gitignore`) -- real datasets are large
  and several are access-gated, so they should never be committed. Only the
  `.gitkeep` placeholders are tracked, which is what keeps the expected
  folder layout visible in the repo.
- Multi-class `attack_cat`/`Attack` columns are collapsed to a binary
  `label` (0 = benign/normal, 1 = attack/malicious) by
  `_harmonize_label()`; the original column is kept as `attack_type` if you
  want to build a multi-class variant later.
