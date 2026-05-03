## Earthquake Declustering Thesis Workspace

This repository is the main thesis workspace for earthquake catalog declustering using machine learning and statistical methods.

The structure is now organized by workflow stage and by learning style, with separate locations for `supervised` and `unsupervised` work.

## Repository Structure

```text
.
├── data/
│   ├── supervised/
│   │   ├── raw/
│   │   └── processed/
│   └── unsupervised/
│       ├── raw/
│       └── processed/
├── notebooks/
│   ├── supervised/
│   └── unsupervised/
├── experiments/
│   ├── supervised/
│   └── unsupervised/
├── results/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   └── catalogs/
├── scripts/
│   └── gmt/
├── docs/
│   ├── methodology/
│   └── papers/
└── declustering-earthquake-catalog/
    ├── scripts/
    └── src/declustering/
```

## Current Contents 

- `data/unsupervised/` contains the current processed SOM + DBSCAN and GMT-related datasets.
- `notebooks/unsupervised/` contains exploratory, fractal-dimension, and SOM + DBSCAN notebooks.
- `experiments/unsupervised/` contains DBSCAN, HDBSCAN, GMM, autoencoder, and KDM experiments.
- `results/figures/unsupervised/`, `results/reports/unsupervised/`, and `results/catalogs/unsupervised/` contain generated outputs grouped by method.
- `declustering-earthquake-catalog/` remains the reusable code layer for turning the notebook work into scripts and modules.

## Working Convention

Use the following rule of thumb when adding new material:

1. Put source catalogs in `data/<mode>/raw/`.
2. Put cleaned or engineered datasets in `data/<mode>/processed/`.
3. Keep interactive analysis in `notebooks/<mode>/`.
4. Keep comparison studies and one-off trials in `experiments/<mode>/`.
5. Save generated figures, tables, catalogs, and reports in `results/.../<mode>/`.
6. Promote stable logic into `declustering-earthquake-catalog/src/declustering/`.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

- The `supervised/` folders are prepared for future thesis material even though the current repository content is mostly unsupervised.
- Generated GMT figures now live under `results/figures/unsupervised/gmt/`, while the GMT shell script lives under `scripts/gmt/`.
