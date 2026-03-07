# KDM DECLUSTERING — SIMPLE GUIDE FOR NEW ZEALAND EARTHQUAKE DATA
## What the Paper Does & How to Apply It to Your 380,000 Events

---

## WHAT IS THE PAPER TRYING TO DO?

Earthquake catalogues mix two types of events:
- **Background (non-crisis)** → independent earthquakes, random in time/space
- **Crisis events** → aftershocks & swarms triggered by other quakes

"Declustering" = separating these two classes.

The paper's method (called **KDM**) uses a neural network called a
**Self-Organising Map (SOM)** to do this automatically — no manual
thresholds, no assumptions about distributions.

---

## THE METHOD IN 5 PLAIN STEPS

```
[Your Catalogue]
      ↓
STEP 1: Compute 22 features per event
      ↓
STEP 2: Train a SOM (4×4 neural grid)
      ↓
STEP 3: Find clusters using Agglomerative Clustering
      ↓
STEP 4: Classify clusters as crisis/non-crisis (probabilistically)
      ↓
STEP 5: Every event gets a probability score
```

---

## YOUR DATA IS ALREADY HALF-WAY THERE!

Your NZ dataset already has the pre-computed features the paper needs:

| Paper Feature | Your Column | Meaning |
|---|---|---|
| T1...T10 | T1, T2, ..., T10 | Temporal distance to 10 nearest neighbours |
| R1...R10 | R1, R2, ..., R10 | Spatial distance to 10 nearest neighbours |
| Mn | Mn | Magnitude ratio (STA/LTA style) |
| bval | bval | Local b-value |

**You do NOT need to recompute these. Just load and use directly.**

---

## STEP-BY-STEP WHAT EACH SECTION OF CODE DOES

### SECTION 1 — Load & Clean
```python
df = load_nz_data('your_file.csv')
df_clean = clean_data(df)
```
- Loads your CSV
- Drops rows where T/R/Mn/bval are missing
- Drops events with `i+ = 0` (no neighbours yet)

---

### SECTION 2 — Build Feature Matrix
```python
X_scaled, feature_cols, scaler = build_feature_matrix(df_clean)
```
- Extracts the 22 columns [T1..T10, R1..R10, Mn, bval]
- Scales everything to [0, 1] using MinMaxScaler
- **Why scale?** SOM uses Euclidean distances, so all features must be comparable

---

### SECTION 3 — Train the SOM
```python
som = train_som_batch(X_scaled, grid_size=4, n_iterations=100000, n_samples=7000)
```
- Creates a **4×4 grid** of 16 "neurons"
- Each neuron learns to represent a group of similar events
- Paper found 4×4 is optimal for the 2-class (crisis/non-crisis) problem
- For 380k events: use `n_samples=140000` if you have >16GB RAM

**How SOM works (simple):**
1. Start with 16 random neurons
2. For each earthquake, find the closest neuron (Best Matching Unit = BMU)
3. Move that neuron + its neighbours slightly toward the earthquake
4. Repeat 100,000 times → neurons settle into meaningful clusters

---

### SECTION 4 — Check Training Quality
```python
qe, te = compute_som_errors(som, X_scaled[:10000])
```
- **QE (Quantisation Error)** → how well neurons represent data. Lower = better
- **TE (Topological Error)** → how well the 2D grid preserves structure. Lower = better
- Both should be stable / not changing much = training complete

---

### SECTION 5 — Assign Events to Nodes
```python
bmu_list = assign_bmu(som, X_scaled)
```
- Every event gets assigned to one of the 16 neurons (its "home node")
- Events on the same node are similar in feature space

---

### SECTION 6 — Agglomerative Clustering
```python
node_labels_2d, node_labels, w_flat = cluster_som_nodes(som, n_clusters=16)
```
- Groups the 16 neurons into meaningful clusters
- Uses Ward linkage (minimises within-cluster variance)
- Clusters emerge as regions of similar seismic behaviour

---

### SECTION 7 — Probabilistic Classification (THE CORE)
```python
classification_results = probabilistic_classification(centroids)
```

For each SOM cluster k, compute two scores:

**Ak (Crisis score)** — large if:
- Events are CLOSE in space/time to neighbours (small R, small T)
- High magnitude ratio (Mn is large)
- b-value close to 1

**Bk (Non-crisis score)** — large if:
- Events are FAR from neighbours (large R, large T)
- Low magnitude ratio (Mn is small)
- b-value deviates from 1

Then convert to probabilities using **softmax**:
```
P(crisis)     = e^Ak / (e^Ak + e^Bk)
P(non-crisis) = e^Bk / (e^Ak + e^Bk)
```

**Confidence** = how decisive the classification is:
```
Confidence = |0.5 - max(P_crisis, P_non_crisis)| / 0.5
```
(0 = completely uncertain, 1 = completely certain)

---

### SECTION 8 — Assign Labels Back
```python
df_labelled = assign_event_labels(df_clean, event_cluster, classification_results)
```
Adds 5 new columns to your dataframe:
- `cluster_id` — which SOM cluster this event belongs to
- `P_crisis` — probability of being a crisis event (0–1)
- `P_non_crisis` — probability of being background (0–1)
- `label` — 'crisis' or 'non_crisis'
- `confidence` — how certain the classification is

---

## HOW TO RUN (3 LINES)

```python
# Install once:
# pip install minisom scikit-learn pandas numpy scipy matplotlib

from KDM_NewZealand_Implementation import run_kdm_pipeline

results, som_model, class_results = run_kdm_pipeline(
    filepath     = 'your_nz_earthquakes.csv',
    grid_size    = 4,
    n_iterations = 100000,
    n_samples    = 7000
)
```

---

## OUTPUT FILES PRODUCED

| File | What it shows |
|---|---|
| `NZ_declustered.csv` | Your full catalogue with labels + probabilities |
| `NZ_SOM_map.png` | 2D SOM grid coloured by cluster & class |
| `NZ_probability_maps.png` | P(crisis), P(non-crisis), confidence maps |
| `NZ_spatial_classification.png` | NZ map coloured by class |
| `NZ_cumulative_curves.png` | Time curves (crisis = staircase, background = linear) |
| `NZ_magnitude_distribution.png` | Magnitude histogram by class |
| `NZ_feature_importance.png` | Which features drove the classification |

---

## TUNING TIPS FOR YOUR 380,000 EVENT DATASET

| Issue | Solution |
|---|---|
| Training too slow | Reduce `n_iterations=50000` or `n_samples=3000` |
| Memory error | Add `X_scaled = X_scaled[:200000]` to subsample |
| Too many crisis events | Raise probability threshold: filter `P_crisis > 0.7` |
| Too few clusters | Change `n_clusters` from 16 to 8 or 4 |
| Results unstable | Increase `n_samples` to 140000 |

**Expected runtime on a laptop:**
- 380k events, 100k iterations: ~15–25 minutes
- Using `n_samples=7000`: ~5–10 minutes

---

## EXPECTED RESULTS FOR NZ

Based on the paper's results on comparable datasets (Taiwan, 20 years, M≥3):
- Crisis events: roughly **70–85%** of catalogue
- Non-crisis: roughly **15–30%**
- NZ has many subduction zone events → expect high crisis fraction

The **cumulative curve validation** is key:
- Non-crisis curve should be roughly linear (constant background rate)
- Crisis curve should show clear steps at major NZ earthquake sequences

---

## COLUMN REFERENCE FROM YOUR DATASET

| Your column | Used for |
|---|---|
| `latitude`, `longitude` | Spatial plots only (not SOM features directly) |
| `time` | Temporal axis in cumulative plots |
| `magnitude` | Magnitude distribution plots |
| `T1–T10` | **Core SOM features** (temporal distances) |
| `R1–R10` | **Core SOM features** (spatial distances) |
| `Mn` | **Core SOM feature** (magnitude ratio) |
| `bval` | **Core SOM feature** (local b-value) |
| `n_child`, `n_parent` | Informational (not used in SOM) |
| `i+`, `N+` | Neighbour counts (used to filter clean events) |
