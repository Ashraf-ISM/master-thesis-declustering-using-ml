# Repository Roadmap

This document captures the intended direction for converting the thesis workspace into a reproducible research repository.

## Immediate Goals

1. Define one canonical declustering workflow.
2. Promote stable notebook logic into versioned Python modules.
3. Separate raw data, processed data, exploratory work, and final outputs.
4. Document how each figure and table used in the thesis is generated.

## Suggested Canonical Pipeline

```text
data/<mode>/raw
  -> preprocessing
  -> feature engineering
  -> declustering model
  -> evaluation / validation
  -> results/figures, results/tables, results/reports, results/catalogs
```

## Recommended Refactor Order

### 1. Data Layer

- add a dataset manifest describing source, coverage, and preprocessing
- keep large generated outputs out of core source directories
- avoid mixing raw catalogs with analysis results

### 2. Method Layer

Refactor the strongest code paths first:

- `experiments/unsupervised/KDM/KDM_NewZealand_Implementation.py`
- `notebooks/unsupervised/som_dbscan/05_SOM_DBSCAN.ipynb`

Target modules inside `declustering-earthquake-catalog/src/declustering/`:

- `preprocess.py`
- `som.py`
- `dbscan.py`
- `som_dbscan.py`
- `utils.py`

### 3. Reproducibility Layer

Create runnable scripts for:

- preprocessing a catalog
- running the chosen declustering method
- producing validation metrics
- exporting final plots and tables

### 4. Thesis Output Layer

Final outputs should converge into:

- `results/figures/<mode>/`
- `results/tables/<mode>/`
- `results/reports/<mode>/`
- `results/catalogs/<mode>/`

Each final figure, table, report, or catalog should have a short provenance note.

## Cleanup Notes

- `experiments/unsupervised/` contains the current exploratory unsupervised work
- `scripts/gmt/` contains the GMT plotting shell script
- `notebooks/unsupervised/fractal_dimension/` contains the retained fractal-dimension notebooks
