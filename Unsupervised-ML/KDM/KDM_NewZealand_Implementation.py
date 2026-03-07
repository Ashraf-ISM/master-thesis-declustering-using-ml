"""
==============================================================================
KDM (Kohonen Map Declustering Method) — Complete Implementation
for New Zealand Earthquake Catalogue (~380,000 events)
==============================================================================

PAPER: "Unsupervised probabilistic machine learning applied to seismicity
        declustering" by Septier et al.

This script follows the paper's exact methodology, adapted for your NZ dataset.
Run sections one at a time if memory is a concern.

COLUMN MAPPING from your dataset:
    - latitude, longitude  → spatial coordinates
    - time (decimal years) → temporal coordinate
    - magnitude            → Mn (average magnitude feature)
    - bval                 → b-value feature
    - T1–T10               → temporal distances to 10 nearest neighbours
    - R1–R10               → spatial distances to 10 nearest neighbours

==============================================================================
INSTALL REQUIRED LIBRARIES (run once in terminal):
    pip install minisom scikit-learn pandas numpy scipy matplotlib seaborn
==============================================================================
"""

# ============================================================
# SECTION 0: IMPORTS
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom           # pip install minisom
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

print("All libraries loaded successfully!")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def output_file(filename):
    """Build a cross-platform path in the local outputs folder."""
    return str(OUTPUT_DIR / filename)


# ============================================================
# SECTION 1: LOAD AND CLEAN YOUR DATA
# ============================================================

def load_nz_data(filepath):
    """
    Load the New Zealand earthquake CSV.
    Adjust sep=',' or sep='\t' based on your file.
    """
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def clean_data(df):
    """
    Keep only rows that have all the pre-computed features
    (T1-T10, R1-R10, Mn, bval). Events with i+=0 have no neighbours
    yet and we drop them.
    """
    # Required feature columns — already in your dataset!
    t_cols = [f'T{i}' for i in range(1, 11)]   # T1 to T10
    r_cols = [f'R{i}' for i in range(1, 11)]   # R1 to R10
    other   = ['Mn', 'bval']

    required = t_cols + r_cols + other
    df_clean = df.dropna(subset=required).copy()

    # Drop rows where i+ = 0 (no neighbours computed yet)
    if 'i+' in df_clean.columns:
        df_clean = df_clean[df_clean['i+'] > 0].copy()

    print(f"Clean data shape: {df_clean.shape}")
    print(f"Removed {len(df) - len(df_clean)} rows with missing features")
    return df_clean.reset_index(drop=True)


# ============================================================
# SECTION 2: FEATURE MATRIX
# ============================================================

def build_feature_matrix(df):
    """
    Build the 22-dimensional feature vector per event:
      - T1..T10  : temporal distances to 10 nearest neighbours (already in df)
      - R1..R10  : spatial distances to 10 nearest neighbours  (already in df)
      - Mn       : normalised average magnitude ratio (STA/LTA style)
      - bval     : local b-value

    YOUR DATASET ALREADY HAS ALL 22 FEATURES — great!
    We just need to extract and normalise them.
    """
    t_cols = [f'T{i}' for i in range(1, 11)]
    r_cols = [f'R{i}' for i in range(1, 11)]
    feature_cols = t_cols + r_cols + ['Mn', 'bval']

    X = df[feature_cols].values.astype(np.float32)

    # Replace any remaining NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalise to [0, 1] — SOM is distance-based, so scaling is critical
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Features: {feature_cols}")
    return X_scaled, feature_cols, scaler


# ============================================================
# SECTION 3: SOM TRAINING
# ============================================================

def train_som(X_scaled, grid_size=4, n_iterations=100000, n_samples=140000,
              random_seed=42):
    """
    Train the Self-Organising Map.

    HYPERPARAMETERS (from paper Table 1):
        grid_size    = 4  →  4x4 = 16 nodes (best for 2-class problem)
        n_iterations = 100,000
        n_samples    = 7,000 (per iteration batch)

    For 380,000 events, the paper recommends up to 140,000 samples
    for large datasets (see Italy Cat4 in Table 1).
    Adjust n_samples=140000 if you have enough RAM.
    """
    n_features = X_scaled.shape[1]  # 22

    print(f"\nTraining SOM: {grid_size}x{grid_size} grid, "
          f"{n_features} features, {n_iterations} iterations...")

    som = MiniSom(
        x=grid_size,
        y=grid_size,
        input_len=n_features,
        sigma=1.5,           # neighbourhood radius (typical: ~grid_size/2)
        learning_rate=0.5,   # initial learning rate
        random_seed=random_seed
    )

    # Initialise weights using PCA (faster convergence)
    som.pca_weights_init(X_scaled)

    # Train using random samples (memory-efficient for 380k events)
    np.random.seed(random_seed)
    n_data = X_scaled.shape[0]

    for i in range(n_iterations):
        # Randomly pick one sample per iteration
        idx = np.random.randint(0, n_data)
        som.update(X_scaled[idx], som.winner(X_scaled[idx]),
                   i, n_iterations)

        if (i + 1) % 10000 == 0:
            print(f"  Iteration {i+1}/{n_iterations} done")

    print("SOM training complete!")
    return som


def train_som_batch(X_scaled, grid_size=4, n_iterations=100000,
                    n_samples=7000, random_seed=42):
    """
    Alternative: batch training (faster, uses MiniSom built-in).
    Use this if the loop above is too slow.
    """
    n_features = X_scaled.shape[1]
    som = MiniSom(grid_size, grid_size, n_features,
                  sigma=1.5, learning_rate=0.5, random_seed=random_seed)
    som.pca_weights_init(X_scaled)

    # Sample n_samples rows for training
    idx = np.random.choice(len(X_scaled), size=min(n_samples, len(X_scaled)),
                           replace=False)
    X_train = X_scaled[idx]

    print(f"Batch training SOM on {len(X_train)} samples...")
    som.train_random(X_train, n_iterations, verbose=True)
    print("SOM training complete!")
    return som


# ============================================================
# SECTION 4: COMPUTE SOM ERRORS (HYPERPARAMETER TUNING)
# ============================================================

def compute_som_errors(som, X_scaled):
    """
    Compute Quantisation Error (QE) and Topological Error (TE).
    Use these to confirm your hyperparameters are good.

    QE → smaller is better (map resolution)
    TE → closer to 0 is better (topology preserved)
    """
    qe = som.quantization_error(X_scaled)
    te = som.topographic_error(X_scaled)
    print(f"Quantisation Error (QE): {qe:.4f}")
    print(f"Topological Error  (TE): {te:.4f}")
    return qe, te


# ============================================================
# SECTION 5: ASSIGN EVENTS TO SOM NODES
# ============================================================

def assign_bmu(som, X_scaled):
    """
    For each event, find its Best Matching Unit (BMU) — the SOM node
    it maps to. Returns (row, col) for each event.
    """
    print("Assigning events to BMU nodes...")
    bmu_list = np.array([som.winner(x) for x in X_scaled])
    print(f"BMU assignment done. Shape: {bmu_list.shape}")
    return bmu_list   # shape: (n_events, 2)


# ============================================================
# SECTION 6: AGGLOMERATIVE CLUSTERING ON SOM NODES
# ============================================================

def cluster_som_nodes(som, n_clusters=None):
    """
    Run Agglomerative Clustering on the SOM weight vectors to
    identify meaningful clusters in the 2D grid.

    n_clusters: if None, the paper uses ~14-16 clusters for real data.
                Try n_clusters between 4 and 16.
    """
    weights = som.get_weights()  # shape: (grid_x, grid_y, n_features)
    grid_x, grid_y, n_feat = weights.shape
    w_flat = weights.reshape(grid_x * grid_y, n_feat)  # flatten to 2D

    if n_clusters is None:
        n_clusters = grid_x * grid_y  # one cluster per node initially

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    node_labels = clustering.fit_predict(w_flat)
    node_labels_2d = node_labels.reshape(grid_x, grid_y)

    print(f"Agglomerative clustering: {n_clusters} clusters found")
    return node_labels_2d, node_labels, w_flat


# ============================================================
# SECTION 7: PROBABILISTIC CLASSIFICATION
# ============================================================

def compute_cluster_centroids(X_scaled, bmu_list, node_labels_2d,
                               feature_cols):
    """
    For each SOM cluster k, compute:
      - mean T (average temporal distances over 10 neighbours)
      - mean R (average spatial distances over 10 neighbours)
      - mean Mn (magnitude ratio)
      - mean bval

    These become the centroid coordinates used in the probabilistic formula.
    """
    grid_x = node_labels_2d.shape[0]
    grid_y = node_labels_2d.shape[1]

    # Map each event to its cluster label
    event_cluster = np.array([
        node_labels_2d[bmu[0], bmu[1]] for bmu in bmu_list
    ])

    t_cols = [f'T{i}' for i in range(1, 11)]
    r_cols = [f'R{i}' for i in range(1, 11)]

    t_indices = [feature_cols.index(c) for c in t_cols]
    r_indices = [feature_cols.index(c) for c in r_cols]
    mn_index  = feature_cols.index('Mn')
    bv_index  = feature_cols.index('bval')

    unique_clusters = np.unique(event_cluster)
    centroids = {}

    for k in unique_clusters:
        mask = event_cluster == k
        cluster_data = X_scaled[mask]

        centroids[k] = {
            'T_mean': cluster_data[:, t_indices].mean(),
            'R_mean': cluster_data[:, r_indices].mean(),
            'Mn_mean': cluster_data[:, mn_index].mean(),
            'bval_mean': cluster_data[:, bv_index].mean(),
            'n_events': mask.sum()
        }

    print(f"Centroids computed for {len(unique_clusters)} clusters")
    return centroids, event_cluster


def probabilistic_classification(centroids):
    """
    Implement Equations 7-14 from the paper.

    CRISIS events have:
      → HIGH number of neighbours (small T and R distances)
      → HIGH magnitude ratio (Mn)
      → b-value close to 1 (from crisis perspective)

    NON-CRISIS events have:
      → LOW neighbours (large T and R distances)
      → LOW magnitude ratio
      → b-value close to 1 (background)

    Returns: dict of {cluster_k: {'P_crisis': float, 'P_non_crisis': float,
                                   'label': 'crisis'/'non_crisis',
                                   'confidence': float}}
    """
    # Extract feature arrays across all clusters
    all_T  = np.array([centroids[k]['T_mean']    for k in centroids])
    all_R  = np.array([centroids[k]['R_mean']    for k in centroids])
    all_Mn = np.array([centroids[k]['Mn_mean']   for k in centroids])
    all_bv = np.array([centroids[k]['bval_mean'] for k in centroids])

    T_max, T_min = all_T.max(), all_T.min()
    R_max, R_min = all_R.max(), all_R.min()
    Mn_max, Mn_min = all_Mn.max(), all_Mn.min()

    results = {}
    cluster_keys = list(centroids.keys())

    for i, k in enumerate(cluster_keys):
        Tk  = all_T[i]
        Rk  = all_R[i]
        Mnk = all_Mn[i]
        bvk = all_bv[i]

        # --- Equation 7: ECmax (deviation from maximum) ---
        # --- Equation 8: ECmin (deviation from minimum) ---
        def ECmax(val, vmax):
            return abs(vmax - val) / (vmax + 1e-10)

        def ECmin(val, vmin):
            return abs(vmin - val) / (abs(vmin) + 1e-10)

        # --- Equation 9: EC1 (deviation from b-value of 1) ---
        def EC1(val):
            return abs(1 - val) / 1.0

        # --- Equation 10: Ak (crisis score) ---
        # Crisis: large R (far in space), large T (far in time)... wait —
        # CORRECTED: Crisis events are CLOSE to neighbours (small R, small T)
        # and have HIGH Mn. So:
        #   Ak uses ECmax(R) + ECmax(T) + ECmin(Mn) + EC1(bval)
        #   (if cluster is FAR from max R/T → it's close → crisis-like)
        Ak = (ECmax(Rk, R_max) + ECmax(Tk, T_max) +
              ECmin(Mnk, Mn_min) + EC1(bvk))

        # --- Equation 11: Bk (non-crisis score) ---
        Bk = (ECmin(Rk, R_min) + ECmin(Tk, T_min) +
              ECmax(Mnk, Mn_max) - EC1(bvk))

        # --- Equations 13-14: Softmax to get probabilities ---
        exp_A = np.exp(Ak)
        exp_B = np.exp(Bk)
        P_crisis     = exp_A / (exp_A + exp_B)
        P_non_crisis = exp_B / (exp_A + exp_B)

        # --- Equation 15: Confidence ---
        confidence = abs(0.5 - max(P_crisis, P_non_crisis)) / 0.5

        label = 'crisis' if P_crisis > P_non_crisis else 'non_crisis'

        results[k] = {
            'P_crisis':     P_crisis,
            'P_non_crisis': P_non_crisis,
            'label':        label,
            'confidence':   confidence,
            'Ak': Ak,
            'Bk': Bk
        }

    return results


# ============================================================
# SECTION 8: ASSIGN LABELS BACK TO EVENTS
# ============================================================

def assign_event_labels(df, event_cluster, classification_results):
    """
    Map each event's cluster label → crisis / non_crisis probability.
    Adds columns to the dataframe:
        - cluster_id
        - P_crisis
        - P_non_crisis
        - label ('crisis' or 'non_crisis')
        - confidence
    """
    df = df.copy()
    df['cluster_id']   = event_cluster
    df['P_crisis']     = df['cluster_id'].map(
        lambda k: classification_results[k]['P_crisis'])
    df['P_non_crisis'] = df['cluster_id'].map(
        lambda k: classification_results[k]['P_non_crisis'])
    df['label']        = df['cluster_id'].map(
        lambda k: classification_results[k]['label'])
    df['confidence']   = df['cluster_id'].map(
        lambda k: classification_results[k]['confidence'])

    n_crisis     = (df['label'] == 'crisis').sum()
    n_non_crisis = (df['label'] == 'non_crisis').sum()
    print(f"\nClassification Results:")
    print(f"  Crisis events     : {n_crisis:,}  ({100*n_crisis/len(df):.1f}%)")
    print(f"  Non-crisis events : {n_non_crisis:,}  ({100*n_non_crisis/len(df):.1f}%)")

    return df


# ============================================================
# SECTION 9: VISUALISATIONS
# ============================================================

def plot_som_map(som, node_labels_2d, classification_results):
    """Plot the 2D SOM grid coloured by cluster and class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: SOM clusters
    ax = axes[0]
    ax.set_title('SOM 2D Map — Clusters', fontsize=13)
    colors_cluster = plt.cm.tab20(
        node_labels_2d / node_labels_2d.max())
    ax.imshow(node_labels_2d, cmap='tab20', origin='lower')
    ax.set_xlabel('SOM x-coordinate')
    ax.set_ylabel('SOM y-coordinate')
    plt.colorbar(ax.images[0], ax=ax, label='Cluster ID')

    # Right: Crisis vs Non-crisis
    ax2 = axes[1]
    ax2.set_title('Classification Map', fontsize=13)
    grid_x, grid_y = node_labels_2d.shape
    class_map = np.zeros((grid_x, grid_y))
    for i in range(grid_x):
        for j in range(grid_y):
            k = node_labels_2d[i, j]
            if k in classification_results:
                class_map[i, j] = classification_results[k]['P_crisis']

    im = ax2.imshow(class_map, cmap='RdYlGn_r', origin='lower',
                    vmin=0, vmax=1)
    ax2.set_xlabel('SOM x-coordinate')
    ax2.set_ylabel('SOM y-coordinate')
    plt.colorbar(im, ax=ax2, label='P(crisis)')

    plt.tight_layout()
    plt.savefig(output_file('NZ_SOM_map.png'), dpi=150)
    plt.show()
    print("SOM map saved.")


def plot_probability_maps(som, node_labels_2d, classification_results):
    """Plot non-crisis prob, crisis prob, and confidence."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    grid_x, grid_y = node_labels_2d.shape

    maps = {
        'P(non-crisis)': 'P_non_crisis',
        'P(crisis)':     'P_crisis',
        'Confidence':    'confidence'
    }
    cmaps = ['YlGn', 'YlOrRd', 'Blues']

    for ax, (title, key), cmap in zip(axes, maps.items(), cmaps):
        grid = np.zeros((grid_x, grid_y))
        for i in range(grid_x):
            for j in range(grid_y):
                k = node_labels_2d[i, j]
                if k in classification_results:
                    grid[i, j] = classification_results[k][key]
        im = ax.imshow(grid, cmap=cmap, origin='lower', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('SOM x')
        ax.set_ylabel('SOM y')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Probabilistic Classification — New Zealand Catalogue',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file('NZ_probability_maps.png'), dpi=150)
    plt.show()
    print("Probability maps saved.")


def plot_spatial_classification(df_labelled):
    """Map of NZ showing crisis vs background events."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    crisis     = df_labelled[df_labelled['label'] == 'crisis']
    non_crisis = df_labelled[df_labelled['label'] == 'non_crisis']

    ax = axes[0]
    ax.scatter(non_crisis['longitude'], non_crisis['latitude'],
               c='steelblue', s=1, alpha=0.3, label=f'Non-crisis ({len(non_crisis):,})')
    ax.scatter(crisis['longitude'], crisis['latitude'],
               c='crimson', s=1, alpha=0.3, label=f'Crisis ({len(crisis):,})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Distribution — Crisis vs Non-Crisis')
    ax.legend(markerscale=5)

    # Probability map
    ax2 = axes[1]
    sc = ax2.scatter(df_labelled['longitude'], df_labelled['latitude'],
                     c=df_labelled['P_crisis'], cmap='RdYlGn_r',
                     s=1, alpha=0.4, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax2, label='P(crisis)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Spatial Distribution — P(crisis)')

    plt.tight_layout()
    plt.savefig(output_file('NZ_spatial_classification.png'), dpi=150)
    plt.show()
    print("Spatial map saved.")


def plot_cumulative_curves(df_labelled):
    """
    Cumulative event curves over time — signature validation plot.
    Background curve should be roughly linear; crisis curve shows steps.
    """
    # Use decimal time column
    time_col = 'time' if 'time' in df_labelled.columns else 'Year'

    df_sorted = df_labelled.sort_values(time_col)
    crisis     = df_sorted[df_sorted['label'] == 'crisis']
    non_crisis = df_sorted[df_sorted['label'] == 'non_crisis']

    fig, ax = plt.subplots(figsize=(14, 5))

    total_len = len(df_sorted)
    ax.plot(df_sorted[time_col],
            np.arange(total_len) / total_len,
            'k--', linewidth=1, label='Full catalogue', alpha=0.6)

    c_len = len(crisis)
    ax.plot(crisis[time_col],
            np.arange(c_len) / total_len,
            'r:', linewidth=1.5, label=f'Crisis ({c_len:,})')

    nc_len = len(non_crisis)
    ax.plot(non_crisis[time_col],
            np.arange(nc_len) / total_len,
            'b-', linewidth=1.5, label=f'Non-crisis ({nc_len:,})')

    ax.set_xlabel('Time (decimal years)')
    ax.set_ylabel('Proportion of events')
    ax.set_title('Cumulative Event Curves — New Zealand KDM Declustering')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file('NZ_cumulative_curves.png'), dpi=150)
    plt.show()
    print("Cumulative curves saved.")


def plot_magnitude_distribution(df_labelled):
    """Magnitude distribution: crisis vs non-crisis (like Fig 13 in paper)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, 9, 0.5)

    crisis     = df_labelled[df_labelled['label'] == 'crisis']['magnitude']
    non_crisis = df_labelled[df_labelled['label'] == 'non_crisis']['magnitude']

    ax.hist(crisis, bins=bins, density=True, alpha=0.6,
            color='orange', label=f'Crisis ({len(crisis):,})')
    ax.hist(non_crisis, bins=bins, density=True, alpha=0.6,
            color='steelblue', label=f'Non-crisis ({len(non_crisis):,})')

    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Density (%)')
    ax.set_title('Magnitude Distribution — Crisis vs Non-Crisis (NZ)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file('NZ_magnitude_distribution.png'), dpi=150)
    plt.show()


# ============================================================
# SECTION 10: FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(X_scaled, df_labelled, feature_cols):
    """
    Three importance metrics from the paper (Section 3.3.5):

    1. SIGNIFICANCE  = variance of each feature across SOM nodes
    2. MEANINGFULNESS= how distinctive each feature is for one class
    3. CORRELATION   = Pearson correlation between feature and class label
    """
    label_binary = (df_labelled['label'] == 'crisis').astype(int).values

    # Subsample for speed (use up to 50k)
    n_use = min(50000, len(X_scaled))
    idx   = np.random.choice(len(X_scaled), n_use, replace=False)
    X_sub = X_scaled[idx]
    y_sub = label_binary[idx]

    significance   = X_sub.var(axis=0)

    # Meaningfulness per class
    crisis_mask     = y_sub == 1
    non_crisis_mask = y_sub == 0
    X_crisis     = X_sub[crisis_mask]
    X_non_crisis = X_sub[non_crisis_mask]

    meaningfulness_crisis     = []
    meaningfulness_non_crisis = []
    correlations              = []

    for f in range(X_sub.shape[1]):
        fmax = X_sub[:, f].max()
        fmin = X_sub[:, f].min()
        rng  = fmax - fmin + 1e-10

        # Meaningfulness: how concentrated values are in one class
        m_crisis = (fmax - X_crisis[:, f].mean()) / rng
        m_non    = (fmax - X_non_crisis[:, f].mean()) / rng
        meaningfulness_crisis.append(m_crisis)
        meaningfulness_non_crisis.append(m_non)

        # Correlation with crisis label
        correlations.append(abs(np.corrcoef(X_sub[:, f], y_sub)[0, 1]))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x_pos = range(len(feature_cols))

    axes[0, 0].bar(x_pos, correlations, color='purple', alpha=0.7)
    axes[0, 0].set_title('Correlation with Crisis Class')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(feature_cols, rotation=90, fontsize=7)
    axes[0, 0].set_ylabel('|Correlation|')

    axes[0, 1].bar(x_pos, significance, color='navy', alpha=0.7)
    axes[0, 1].set_title('Significance (Variance)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(feature_cols, rotation=90, fontsize=7)
    axes[0, 1].set_ylabel('Variance')

    axes[1, 0].bar(x_pos, meaningfulness_crisis, color='crimson', alpha=0.7)
    axes[1, 0].set_title('Meaningfulness — Crisis Class')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(feature_cols, rotation=90, fontsize=7)

    axes[1, 1].bar(x_pos, meaningfulness_non_crisis,
                   color='steelblue', alpha=0.7)
    axes[1, 1].set_title('Meaningfulness — Non-Crisis Class')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(feature_cols, rotation=90, fontsize=7)

    plt.suptitle('Feature Importance Analysis — NZ Catalogue', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file('NZ_feature_importance.png'), dpi=150)
    plt.show()
    print("Feature importance plots saved.")

    return significance, correlations


# ============================================================
# SECTION 11: SAVE RESULTS
# ============================================================

def save_results(df_labelled, filepath=None):
    """Save the labelled catalogue."""
    if filepath is None:
        filepath = output_file('NZ_declustered.csv')

    output_cols = ['event', 'DateTime', 'latitude', 'longitude', 'depth',
                   'magnitude', 'time', 'P_crisis', 'P_non_crisis',
                   'label', 'confidence', 'cluster_id']

    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in df_labelled.columns]
    df_out = df_labelled[output_cols]
    df_out.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    print(f"Total events saved: {len(df_out):,}")
    return df_out


# ============================================================
# SECTION 12: MAIN PIPELINE — RUN THIS
# ============================================================

def run_kdm_pipeline(filepath, grid_size=4, n_iterations=100000,
                     n_samples=7000, n_clusters=None):
    """
    Full KDM pipeline. Call this with your CSV file path.

    Example:
        results = run_kdm_pipeline('your_nz_data.csv')

    PARAMETERS:
        filepath     : path to your NZ earthquake CSV
        grid_size    : SOM grid dimension (4 → 4x4=16 nodes, paper default)
        n_iterations : SOM training iterations (100,000 for most datasets)
        n_samples    : training samples per run (7,000 or 140,000 for large data)
        n_clusters   : agglomerative clusters (None = auto = grid_size²)
    """
    print("=" * 60)
    print("  KDM DECLUSTERING — NEW ZEALAND EARTHQUAKE CATALOGUE")
    print("=" * 60)

    # Step 1: Load
    df = load_nz_data(filepath)

    # Step 2: Clean
    df_clean = clean_data(df)

    # Step 3: Feature matrix
    X_scaled, feature_cols, scaler = build_feature_matrix(df_clean)

    # Step 4: Train SOM (use batch version for speed)
    som = train_som_batch(X_scaled, grid_size=grid_size,
                          n_iterations=n_iterations, n_samples=n_samples)

    # Step 5: SOM errors
    compute_som_errors(som, X_scaled[:10000])  # use subset for speed

    # Step 6: Assign BMU
    bmu_list = assign_bmu(som, X_scaled)

    # Step 7: Agglomerative clustering on SOM nodes
    node_labels_2d, node_labels, w_flat = cluster_som_nodes(
        som, n_clusters=n_clusters if n_clusters else grid_size * grid_size)

    # Step 8: Compute centroids
    centroids, event_cluster = compute_cluster_centroids(
        X_scaled, bmu_list, node_labels_2d, feature_cols)

    # Step 9: Probabilistic classification
    classification_results = probabilistic_classification(centroids)

    # Step 10: Assign labels to events
    df_labelled = assign_event_labels(df_clean, event_cluster,
                                      classification_results)

    # Step 11: Plots
    print("\nGenerating visualisations...")
    plot_som_map(som, node_labels_2d, classification_results)
    plot_probability_maps(som, node_labels_2d, classification_results)
    plot_spatial_classification(df_labelled)
    plot_cumulative_curves(df_labelled)
    plot_magnitude_distribution(df_labelled)
    compute_feature_importance(X_scaled, df_labelled, feature_cols)

    # Step 12: Save
    df_out = save_results(df_labelled)

    print("\n" + "=" * 60)
    print("  KDM PIPELINE COMPLETE!")
    print("=" * 60)
    return df_labelled, som, classification_results


# ============================================================
# HOW TO RUN (at the bottom of your script or in Jupyter):
# ============================================================
if __name__ == "__main__":
    # ── CHANGE THIS PATH TO YOUR ACTUAL FILE ──
    YOUR_CSV = "som_feature_nz_real_catalog.csv"

    results, som_model, class_results = run_kdm_pipeline(
        filepath     = YOUR_CSV,
        grid_size    = 4,       # 4x4 grid (paper recommendation)
        n_iterations = 100000,  # 100k iterations
        n_samples    = 140000,    # increase to 140000 if RAM allows
        n_clusters   = None     # auto = 16 clusters for 4x4 grid
    )

    print("\nSample output:")
    print(results[['latitude', 'longitude', 'magnitude',
                   'P_crisis', 'P_non_crisis', 'label',
                   'confidence']].head(10))
