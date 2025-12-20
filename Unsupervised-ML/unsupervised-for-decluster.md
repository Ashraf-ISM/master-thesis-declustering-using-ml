# 🌍 Unsupervised Machine Learning for Earthquake Catalog Declustering  
### Focus: Self-Organizing Maps (SOM) & DBSCAN

## 📌 Project Overview

Earthquake catalogs contain both **independent seismic events (mainshocks)** and **dependent events (aftershocks, foreshocks, swarms)**.  
Traditional declustering techniques (e.g., Gardner–Knopoff, Reasenberg) rely on **empirical space–time windows**, which may not generalize well across tectonic regions.

This project explores **unsupervised machine learning approaches** for earthquake catalog declustering, with a primary focus on:

- **Self-Organizing Maps (SOM)**
- **Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**

The goal is to **identify seismic clusters automatically**, without predefined physical thresholds, and separate **clustered events** from **background seismicity**.

---

## 🎯 Objectives

- Apply **unsupervised ML techniques** to earthquake catalogs
- Identify seismic clusters based on **spatio-temporal-magnitude features**
- Compare clustering behavior of **SOM vs DBSCAN**
- Extract a **declustered catalog** suitable for:
  - Seismic hazard analysis
  - b-value estimation
  - Seismicity rate modeling

---

## 📂 Dataset Description

The earthquake catalog typically contains the following parameters:

| Parameter | Description |
|---------|-------------|
| Latitude | Event latitude (degrees) |
| Longitude | Event longitude (degrees) |
| Depth | Hypocentral depth (km) |
| Magnitude | Event magnitude (Mw / ML / Mb) |
| Time | Origin time (UTC) |

📌 **Note:** The catalog is preprocessed to remove missing or inconsistent entries before ML application.

---

## ⚙️ Feature Engineering

The following features are derived and used for clustering:

- **Spatial Features**
  - Latitude
  - Longitude
  - Depth

- **Temporal Features**
  - Inter-event time (Δt)
  - Event time index

- **Magnitude Features**
  - Magnitude
  - Magnitude difference with previous event (ΔM)

All features are **normalized/scaled** prior to ML training to avoid dominance of any single parameter.

---

## 🧠 Methods Used

### 1️⃣ Self-Organizing Maps (SOM)

**SOM** is a neural-network-based unsupervised learning algorithm that projects high-dimensional data onto a **low-dimensional (2D) grid** while preserving topological relationships.

**Why SOM for Declustering?**
- Captures **nonlinear relationships**
- Provides **visual interpretability**
- Effective in identifying seismic patterns and regimes

**Workflow:**
1. Train SOM using seismic features
2. Analyze neuron activation patterns
3. Group neurons representing clustered vs background events
4. Label events accordingly

---

### 2️⃣ DBSCAN (Density-Based Spatial Clustering)

**DBSCAN** identifies clusters based on **event density** in feature space and labels sparse points as noise.

**Why DBSCAN for Declustering?**
- No need to specify number of clusters
- Naturally separates **noise (background seismicity)**
- Well-suited for aftershock sequences

**Key Parameters:**
- `eps`: Neighborhood radius
- `min_samples`: Minimum points to form a cluster

**Workflow:**
1. Apply DBSCAN on scaled seismic features
2. Identify dense seismic clusters
3. Treat noise points as background events

---

## 📊 Output & Results

The project produces:

- Cluster labels for each earthquake
- Identification of:
  - Main clusters (aftershock sequences)
  - Background seismicity
- A **declustered earthquake catalog**
- Visualizations:
  - Spatial cluster maps
  - Time-series cluster evolution
  - SOM component planes

---

## 🧪 Evaluation Strategy

Since declustering is **unsupervised**, evaluation is based on:

- Visual inspection of spatial–temporal clusters
- Comparison with known seismic sequences
- Reduction in short-term clustering
- Stability of b-value after declustering

---

## 🛠️ Technologies & Libraries

- **Python**
- `numpy`
- `pandas`
- `scikit-learn`
- `minisom`
- `matplotlib`
- `seaborn`

---

## 📁 Project Structure

```text
├── data/
│   └── earthquake_catalog.csv
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── som_declustering.ipynb
│   └── dbscan_declustering.ipynb
├── results/
│   ├── clustered_catalog.csv
│   └── declustered_catalog.csv
├── figures/
│   ├── som_maps.png
│   └── dbscan_clusters.png
├── README.md
└── requirements.txt
