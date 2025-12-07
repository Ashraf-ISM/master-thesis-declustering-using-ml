# 🌍 Machine Learning–Based Declustering of Earthquake Catalogs

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EA4335?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Research-yellow?style=for-the-badge)

**Adaptive, threshold-free framework for separating mainshocks from aftershocks using physics-informed machine learning**

[📖 Documentation](#documentation) • [🚀 Quick Start](#quick-start) • [📊 Results](#results) • [🤝 Contributing](#contributing)

</div>

---

## 🎯 Overview

Earthquake catalogs contain both **independent background events** (mainshocks) and **dependent triggered events** (foreshocks/aftershocks). Accurately separating these populations—known as **declustering**—is fundamental for:

- 🗺️ Seismic hazard assessment and building codes
- 📈 Statistical seismology and b-value estimation
- ⚡ Earthquake forecasting and early warning systems
- 🏗️ Infrastructure planning and risk mitigation

### ⚠️ The Problem with Traditional Methods

Classical declustering techniques (Reasenberg, Gardner-Knopoff, NND) rely on **fixed empirical thresholds**:

```
❌ Fixed time-distance windows
❌ Regional parameter tuning required
❌ Poor performance in complex tectonic zones
❌ Misclassification of overlapping sequences
```

### 💡 Our Solution

**Adaptive ML framework** trained on physics-based synthetic catalogs:

```
✅ Threshold-free classification
✅ Data-driven feature learning
✅ Regionally transferable
✅ Physically interpretable results
```

---

## 🔬 Methodology

### 1️⃣ **ETAS Synthetic Catalog Generation**

We use the **Epidemic-Type Aftershock Sequence (ETAS)** model to generate labeled training data:

```math
λ(t, x, y | ℋₜ) = μu(x,y) + Σᵢ:ₜᵢ<ₜ ξ(t - tᵢ, x - xᵢ, y - yᵢ; mᵢ)
```

**Key ETAS Parameters (New Zealand):**

| Parameter | Symbol | Value | Physical Meaning |
|-----------|--------|-------|------------------|
| Background Rate | μ | 0.4766 | Spontaneous earthquake rate |
| Productivity | K | 4.9184 | Aftershock generation capacity |
| Mag Productivity | α | 1.2334 | Magnitude-productivity scaling |
| Temporal Decay | p | 1.0051 | Aftershock decay rate |
| Spatial Scale | D | 0.0022 | Typical aftershock distance |
| Spatial Decay | q | 1.6122 | Distance decay exponent |
| Mag-Spatial Scale | γ | 0.4476 | Magnitude-rupture scaling |

**Output:** Synthetic catalog with ground-truth labels
- `Label 0`: Background events (mainshocks)
- `Label 1`: Triggered events (aftershocks)

---

### 2️⃣ **Nearest-Neighbor Distance (NND) Analysis**

NND measures space-time-energy proximity between earthquakes:

```python
# Core NND Formula
η_ij = T_ij × R_ij

where:
    T_ij = t_ij × 10^(-b·mᵢ/2)     # Rescaled Time
    R_ij = r_ij^df × 10^(-b·mᵢ/2)  # Rescaled Distance
    
    t_ij: inter-event time
    r_ij: spatial distance
    mᵢ: magnitude of event i
    df: fractal dimension (~1.57 for NZ)
    b: Gutenberg-Richter b-value (~1.0)
```

**Why NND Works:**

<div align="center">

| η Value | Interpretation |
|---------|---------------|
| η → 0 | **Close clustering** → Likely aftershock |
| η → ∞ | **Distant events** → Likely independent |

</div>

The bimodal distribution of η naturally separates the two populations!

---

### 3️⃣ **Feature Engineering**

Five physics-informed features extracted for each earthquake pair:

| Feature | Formula | Physical Interpretation |
|---------|---------|------------------------|
| **Rescaled Time** | `T = t × 10^(-b·m/2)` | Temporal triggering likelihood |
| **Rescaled Distance** | `R = r^df × 10^(-b·m/2)` | Spatial coupling strength |
| **Magnitude Difference** | `Δm = mⱼ - mᵢ` | Energy hierarchy (independence indicator) |
| **NND Metric** | `η = T × R` | Composite space-time distance |
| **Parent Index** | `i` | Most probable parent event |

**Feature Importance (XGBoost):**

```
███████████████████████████████████ η (NND Metric)        35.2%
█████████████████████████████ R (Rescaled Distance) 28.7%
███████████████████ T (Rescaled Time)        19.4%
████████████ Δm (Magnitude Diff)      11.8%
█████ Parent Index                4.9%
```

---

### 4️⃣ **Model Training & Selection**

Four supervised ML models tested on ETAS synthetic catalogs:

<div align="center">

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| 🏆 **XGBoost** | **97.44%** | **97.66%** | **98.74%** | **98.20%** |
| Gradient Boosting | 97.11% | 97.06% | 98.89% | 97.97% |
| Random Forest | 96.72% | 96.22% | 95.15% | 97.91% |
| SVM | 94.36% | 94.48% | 94.36% | 94.40% |

</div>

**Confusion Matrix (XGBoost):**

```
                 Predicted
              Background  Triggered
Actual  
Background     94.4%       5.6%      ✅ High specificity
Triggered      1.3%       98.7%      ✅ Excellent sensitivity
```

**Winner: XGBoost** 
- Best balance of precision/recall
- Robust to class imbalance
- Fast inference on large catalogs

---

## 🗺️ Case Study: New Zealand Seismicity

### Dataset Overview

<div align="center">

| **Property** | **Value** |
|--------------|-----------|
| 📅 Time Period | 1980 – 2024 (44 years) |
| 📍 Region | Pacific-Australian Plate Boundary |
| 🌐 Total Events | 396,267 earthquakes |
| 📏 Magnitude Range | Mw ≥ 2.2 (completeness threshold) |
| 🧮 Fractal Dimension | df ≈ 1.568 |
| 🎯 Tectonic Features | Alpine Fault, Hikurangi Subduction Zone |

</div>

### Tectonic Context

```
🏔️ Alpine Fault (South Island)
   └─ Strike-slip boundary
   └─ ~30mm/yr plate motion
   └─ M7+ earthquake recurrence ~300 years

🌊 Hikurangi Subduction Zone (North Island)  
   └─ Pacific plate subducting beneath Australian plate
   └─ Slow slip events and megathrust potential
   └─ Dense seismicity and tsunami hazard
```

---

## 📊 Results

### Classification Performance

**XGBoost Declustering Results:**

<div align="center">

| Event Type | Count | Percentage |
|------------|-------|------------|
| 🟢 **Background** (Mainshocks) | 230,758 | **58.23%** |
| 🔴 **Triggered** (Aftershocks) | 165,509 | **41.75%** |

</div>

### Spatial Distribution

**Key Observations:**

1. **Background Events (Independent):**
   - 🏔️ Concentrated along Alpine Fault (South Island)
   - 🌊 Distributed across Hikurangi Subduction Zone (North Island)
   - ✅ Consistent with tectonic plate boundaries

2. **Triggered Events (Aftershocks):**
   - 💥 Dense clusters near **Canterbury region** (2010-2011 sequence)
   - ⚡ Major clustering around **Kaikōura Mw 7.8** (2016)
   - 📍 Concentrated aftershock zones validate physical model

### Temporal Evolution

```
Major Earthquake Sequences Identified:

2010-2011 Canterbury Sequence
├─ Darfield Mw 7.1 (Sep 2010)
├─ Christchurch Mw 6.3 (Feb 2011)
└─ ~10,000 aftershocks detected

2013 Cook Strait/Seddon
├─ Lake Grassmere Mw 6.5 (Jul 2013)
└─ ~2,500 aftershocks

2016 Kaikōura Earthquake
├─ Mainshock Mw 7.8 (Nov 2016)
└─ ~15,000+ aftershocks (ongoing)
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/earthquake-declustering.git
cd earthquake-declustering

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Generate ETAS Synthetic Catalog

```python
from etas_model import ETASSimulator

# Initialize with New Zealand parameters
simulator = ETASSimulator(
    mu=0.4766, k=4.9184, alpha=1.2334,
    p=1.0051, d=0.0022, q=1.6122, gamma=0.4476
)

# Generate synthetic catalog
synthetic_catalog = simulator.simulate(
    duration=365*10,  # 10 years
    magnitude_threshold=2.2
)
```

#### 2. Extract NND Features

```python
from nnd_analysis import NNDFeatureExtractor

# Initialize feature extractor
extractor = NNDFeatureExtractor(b_value=1.0, fractal_dim=1.568)

# Compute features
features = extractor.extract_features(synthetic_catalog)
# Returns: [T, R, Δm, η, parent_index]
```

#### 3. Train XGBoost Model

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Prepare training data
X = features[['T', 'R', 'dm', 'eta', 'parent_idx']]
y = synthetic_catalog['label']  # 0: background, 1: triggered

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic'
)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

#### 4. Apply to Real Catalog

```python
# Load New Zealand catalog
nz_catalog = pd.read_csv('new_zealand_catalog.csv')

# Extract features
real_features = extractor.extract_features(nz_catalog)

# Predict
predictions = model.predict(real_features)
nz_catalog['event_type'] = predictions  # 0: background, 1: triggered

# Export declustered catalog
background_events = nz_catalog[nz_catalog['event_type'] == 0]
background_events.to_csv('nz_declustered_background.csv', index=False)
```

---

## 📈 Model Comparison with Traditional Methods

### Quantitative Comparison

<div align="center">

| Method | Background % | Triggered % | Adaptability | Computation |
|--------|--------------|-------------|--------------|-------------|
| **XGBoost (Ours)** | **58.23%** | **41.75%** | ✅ High | Fast |
| Gradient Boosting | 55.36% | 44.64% | ✅ High | Fast |
| Random Forest | 68.80% | 31.20% | ✅ Medium | Medium |
| SVM | 72.69% | 27.01% | ⚠️ Medium | Slow |
| Gardner-Knopoff | ~65% | ~35% | ❌ Fixed | Fast |
| Reasenberg | ~60% | ~40% | ❌ Fixed | Fast |

</div>

### Advantages Over Classical Methods

| Aspect | Traditional | ML-Based (Ours) |
|--------|-------------|-----------------|
| **Threshold Selection** | Manual calibration | Learned from data |
| **Regional Transfer** | Re-tune parameters | Direct application |
| **Complex Patterns** | Misses overlapping | Captures nuances |
| **Physical Interpretation** | Window-based | Feature importance |
| **Uncertainty** | Binary decision | Probability estimates |

---

## 🔍 Feature Importance Analysis

**What Makes an Earthquake an Aftershock?**

Our XGBoost model reveals the key discriminating factors:

```python
Feature Importance Ranking:

1. η (NND Metric) ████████████████████████████████ 35.2%
   → Composite space-time proximity is the strongest signal
   
2. R (Rescaled Distance) ████████████████████████ 28.7%
   → Spatial clustering heavily influences classification
   
3. T (Rescaled Time) ███████████████ 19.4%
   → Recent events more likely to be triggered
   
4. Δm (Magnitude Difference) █████████ 11.8%
   → Smaller events after larger ones = aftershocks
   
5. Parent Index ███ 4.9%
   → Linking to specific parent event adds context
```

**Key Insight:** The composite NND metric (η = T × R) is the most powerful single predictor, validating the physical basis of the approach.

---

## 🌐 Regional Transferability

### Tested Regions (Ongoing)

| Region | Status | Dataset Size | Tectonic Setting |
|--------|--------|--------------|------------------|
| 🇳🇿 **New Zealand** | ✅ Complete | 396,267 | Subduction + Strike-slip |
| 🇺🇸 **Southern California** | 🔄 In Progress | ~500,000 | Strike-slip (San Andreas) |
| 🇯🇵 **Japan** | ⏳ Planned | ~1,000,000+ | Subduction (Pacific Ring) |
| 🇮🇹 **Italy** | ⏳ Planned | ~200,000 | Extensional tectonics |

**Hypothesis:** NND-based features should transfer well across tectonic settings since they encode fundamental earthquake physics.

---

## 🛠️ Advanced Configuration

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    XGBClassifier(objective='binary:logistic'),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Custom ETAS Parameters

```python
# Estimate parameters from your catalog
from etas_model import ETASParameterEstimator

estimator = ETASParameterEstimator()
params = estimator.fit(your_catalog)

print(f"Estimated parameters:")
print(f"μ = {params['mu']:.4f}")
print(f"K = {params['k']:.4f}")
print(f"α = {params['alpha']:.4f}")
# ... etc
```

---

## 📚 Documentation

### Key References

1. **ETAS Model:**
   - Ogata, Y. (1988). *Statistical models for earthquake occurrences and residual analysis for point processes*. JASA.
   
2. **NND Analysis:**
   - Zaliapin, I., & Ben-Zion, Y. (2013). *Earthquake clusters in southern California*. JGR: Solid Earth.

3. **Machine Learning Application:**
   - Aden-Antoniów, F., Frank, W. B., & Seydoux, L. (2022). *An adaptable random forest model for the declustering of earthquake catalogs*. JGR: Solid Earth.

4. **This Work:**
   - Ashraf, M., & Jana, N. (2025). *Machine learning approach to earthquake declustering: Application to New Zealand earthquake catalogue*. [In Preparation]

### Mathematical Foundations

<details>
<summary><b>Click to expand: ETAS Model Derivation</b></summary>

The conditional intensity function describes the instantaneous earthquake rate:

```math
λ(t, x, y | ℋₜ) = μ(x, y) + Σᵢ:ₜᵢ<ₜ k₀ e^α(mᵢ - m₀) · g(t - tᵢ) · f(x - xᵢ, y - yᵢ; mᵢ)
```

Where:
- **Background term:** μ(x, y) = spatially varying Poisson process
- **Triggering kernel:** Product of temporal and spatial components
- **Temporal:** g(t) = (p-1)/c · (1 + t/c)^(-p) (Omori-Utsu law)
- **Spatial:** f(r; m) = (q-1)/(π d² e^γm) · (1 + r²/(d² e^γm))^(-q)

</details>

<details>
<summary><b>Click to expand: NND Rescaling Derivation</b></summary>

The rescaling accounts for magnitude-dependent triggering:

```math
T_ij = t_ij · 10^(-b·mᵢ/2)
R_ij = r_ij^df · 10^(-b·mᵢ/2)
```

This normalization ensures:
- Larger earthquakes (higher m) → larger search windows
- Self-consistent scaling with rupture dimensions
- Fractal dimension df accounts for spatial distribution

</details>

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas for Contribution

- 🌍 **Regional Testing:** Apply to new earthquake catalogs
- 🧮 **Feature Engineering:** Propose new physics-based features
- 🤖 **Model Development:** Test deep learning architectures
- 📊 **Visualization:** Improve result presentation
- 📝 **Documentation:** Enhance tutorials and examples

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/earthquake-declustering.git
cd earthquake-declustering

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

### Contribution Guidelines

1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open Pull Request

---

## 📊 Performance Benchmarks

### Computational Efficiency

<div align="center">

| Operation | Time (396K events) | Memory |
|-----------|-------------------|--------|
| NND Feature Extraction | ~45 seconds | ~2.5 GB |
| XGBoost Training | ~12 seconds | ~1.8 GB |
| Inference | ~8 seconds | ~1.2 GB |
| **Total Pipeline** | **~65 seconds** | **~2.5 GB** |

*Tested on: Intel i7-10700K, 32GB RAM*

</div>

### Scalability

```python
# Performance scales linearly with catalog size
Catalog Size    Processing Time
-----------    ----------------
10K events     →  2 seconds
100K events    → 15 seconds
396K events    → 65 seconds
1M events      → ~180 seconds (estimated)
```

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{ashraf2025earthquake,
  title={Machine Learning Approach to Earthquake Declustering: 
         Application to New Zealand Earthquake Catalogue},
  author={Ashraf, Md},
  school={Indian Institute of Technology (ISM) Dhanbad},
  year={2025},
  supervisor={Jana, Niptika},
  department={Applied Geophysics}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dr. Niptika Jana** – Research supervision and guidance
- **GNS Science** – New Zealand earthquake catalog (GeoNet)
- **USGS** – Seismic data and catalog tools
- **XGBoost Development Team** – Excellent ML library

---

## 📧 Contact

**Md Ashraf**  
M.Sc. (Tech.) Applied Geophysics  
Indian Institute of Technology (ISM) Dhanbad

📧 Email: [23mc0049@iitism.ac.in](mailto:23mc0049@iitism.ac.in)  
🔗 LinkedIn: [linkedin.com/in/md-ashraf](https://linkedin.com/in/md-ashraf)  
🐙 GitHub: [@yourusername](https://github.com/yourusername)

---

<div align="center">

### 🌟 Star this repository if you find it useful!

**Made with ❤️ for safer earthquake monitoring**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=earthquake-declustering)
![GitHub stars](https://img.shields.io/github/stars/yourusername/earthquake-declustering?style=social)

[⬆ Back to Top](#-machine-learningbased-declustering-of-earthquake-catalogs)

</div>
