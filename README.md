# Machine Learning–Based Declustering of Earthquake Catalogs

This repository contains the research work and implementation for my Master’s thesis:  
**"Machine Learning–Based Declustering of Earthquake Catalogs: A Case Study on New Zealand Seismicity (1980–2024)"**  
at **IIT (ISM) Dhanbad**, under the supervision of **Dr. Niptika Jana**.

---

## 📖 About the Project

The project aims to improve earthquake catalog declustering using machine learning techniques. Traditional methods (e.g., Gardner–Knopoff, Reasenberg, ZMAP) often rely on fixed spatio-temporal windows, which may not generalize well across tectonic regions.  
In this work, I explore **nearest-neighbor distance (NND)** and machine learning–based approaches for robust separation of **background seismicity** from **aftershock sequences**, focusing on **New Zealand earthquake data (1980–2024)**.

Key Highlights:
- Dataset: **New Zealand National Seismic Catalog (1980–2024)**
- Tools: **Python, Scikit-learn, Pandas, Matplotlib, GMT, ZMAP**
- Methods:  
  - Classical declustering (Gardner–Knopoff, Reasenberg)  
  - Nearest Neighbor Distance (NND)  
  - Supervised & unsupervised ML approaches  
- Deliverables:  
  - Cleaned and declustered earthquake catalogs  
  - Visualizations of seismicity patterns  
  - Comparative evaluation of methods  

---

## 📂 Folder Structure

```bash
data/           -> raw and processed catalogs
notebooks/      -> Jupyter notebooks for analysis
src/            -> Python modules for data processing and models
results/        -> figures, logs, and tables
thesis/         -> LaTeX source files of thesis
---
# Installation

git clone https://github.com/ashraf-iit-ism/master-thesis-declustering.git
cd master-thesis-declustering

# Using Conda
conda env create -f environment.yml
conda activate thesis-env

# OR using pip
pip install -r requirements.txt
---
# workflow
# Step 1: Preprocess catalog
python src/data_processing/clean_catalog.py --input data/raw/nz_catalog.csv --output data/processed/clean.csv

# Step 2: Run declustering model
python src/models/decluster.py --method nnd --input data/processed/clean.csv --output data/processed/declustered.csv

# Step 3: Visualize results
jupyter notebook notebooks/04_visualization.ipynb
