# Declustering Earthquake Catalog Methodology Note

This note tracks the intended migration path from notebook-heavy thesis work to the reusable package in `declustering-earthquake-catalog/`.

## Source Inputs

- primary notebook source: `notebooks/unsupervised/som_dbscan/05_SOM_DBSCAN.ipynb`
- supporting experiment source: `experiments/unsupervised/KDM/`
- processed inputs: `data/unsupervised/processed/`

## Package Goal

Move stable logic into:

- `declustering-earthquake-catalog/src/declustering/preprocess.py`
- `declustering-earthquake-catalog/src/declustering/som.py`
- `declustering-earthquake-catalog/src/declustering/dbscan.py`
- `declustering-earthquake-catalog/src/declustering/som_dbscan.py`
- `declustering-earthquake-catalog/src/declustering/utils.py`

## Expected Outputs

Reusable scripts should eventually read from `data/<mode>/...` and write generated artifacts to:

- `results/figures/<mode>/`
- `results/tables/<mode>/`
- `results/reports/<mode>/`
- `results/catalogs/<mode>/`
