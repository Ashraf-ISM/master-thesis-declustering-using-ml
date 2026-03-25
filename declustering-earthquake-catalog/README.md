# Declustering Earthquake Catalog

This subproject is the intended home for the final reusable codebase behind the thesis.

## Purpose

The notebook work in the main repository is exploratory. This folder should gradually become the reproducible implementation that:

- loads earthquake catalogs
- preprocesses spatial, temporal, and magnitude features
- runs declustering methods
- exports labelled catalogs and diagnostics

## Current State

The package structure is present, and this folder now focuses on reusable code rather than storing notebook outputs or generated data.

Primary contents:

- `src/declustering/`
- `scripts/`

The most useful source material for filling these modules currently lives in:

- `../notebooks/unsupervised/som_dbscan/`
- `../experiments/unsupervised/KDM/`
- `../docs/methodology/`

## Near-Term Goal

Refactor the thesis notebooks into importable functions and runnable scripts so this folder becomes the canonical implementation layer for the dissertation.
