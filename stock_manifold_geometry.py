# stock_manifold_geometry.py
# Geometric analysis of stock feature manifold
# Estimates intrinsic dimensionality, curvature, noise structure
# Provides architecture recommendations for AE training

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from scipy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')

# ── Feature definitions ───────────────────────────────────────────────────────
EXCLUDE_COLS = {'ticker', 'open', 'high', 'low', 'close', 'volume',
                'Mkt_RF', 'SMB', 'HML', 'RF',
                'high_252d', 'low_252d', 'high_63d', 'low_63d',
                'sma_21', 'sma_63', 'sma_252', 'vol_ma21',
                'up_vol_ratio', 'kurt_63d'}

print("="*80)
print("STOCK MANIFOLD GEOMETRY ANALYSIS")
print("Pre-AE architecture recommendations")
print("="*80)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading data...")

manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
tickers = manifest['ticker'].tolist()

with open(DATA_DIR / 'stock_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

sample_df = pd.read_parquet(DATA_DIR / f'stock_clean_{tickers[0]}.parquet')
feature_cols = [c for c in sample_df.columns 
                if c not in EXCLUDE_COLS
                and sample_df[c].dtype in [np.float64, np.float32]]

print(f"Features: {len(feature_cols)}")

# Load subset for analysis (30 random stocks, 3000 obs per stock max)
np.random.seed(42)
sample_tickers = np.random.choice(tickers, size=min(30, len(tickers)), replace=False)

data_chunks = []
print(f"Loading {len(sample_tickers)} stocks...")

for ticker in sample_tickers:
    f = DATA_DIR / f'stock_clean_{ticker}.parquet'
    if not f.exists():
        continue
    
    df = pd.read_parquet(f)
    
    if len(df) > 3000:
        df = df.sample(n=3000, random_state=42)
    
    X = df[feature_cols].values.astype(np.float32)
    
    # Basic cleaning
    X = np.where(np.isinf(X), np.nan, X)
    col_meds = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_meds[j]
    X = np.nan_to_num(X, nan=0.0)
    
    data_chunks.append(X)

X_all = np.vstack(data_chunks)
print(f"Loaded: {X_all.shape[0]:,} observations")

# Scale
X_scaled = scaler.transform(X_all).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Intrinsic Dimensionality (MLE method)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TEST 1: INTRINSIC DIMENSIONALITY (MLE)")
print("="*80)

def estimate_intrinsic_dim_mle(X, k=20, n_samples=5000):
    """
    Maximum Likelihood Estimation of intrinsic dimensionality.
    Based on Levina & Bickel (2005).
    """
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_sample)
    distances, indices = nbrs.kneighbors(X_sample)
    
    # Remove self (distance 0)
    distances = distances[:, 1:]
    
    # MLE formula
    dims = []
    for i in range(len(X_sample)):
        r = distances[i]
        if r[-1] > 0 and r[0] > 0:
            # Avoid log(0)
            log_ratio = np.log(r[-1] / r[:-1])
            dim_est = (k - 1) / np.sum(log_ratio)
            if 0 < dim_est < 100:  # Sanity check
                dims.append(dim_est)
    
    return np.median(dims), np.std(dims)

intrinsic_dim, intrinsic_std = estimate_intrinsic_dim_mle(X_scaled, k=20)
print(f"MLE intrinsic dimension: {intrinsic_dim:.1f} ± {intrinsic_std:.1f}")
print(f"Ambient dimension:       {X_scaled.shape[1]}")
print(f"Compression ratio:       {X_scaled.shape[1] / intrinsic_dim:.1f}x")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: PCA Eigenvalue Spectrum (scree plot + noise model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TEST 2: PCA EIGENVALUE SPECTRUM")
print("="*80)

pca_full = PCA()
pca_full.fit(X_scaled)
eigenvalues = pca_full.explained_variance_

# Marchenko-Pastur noise floor
n, d = X_scaled.shape
gamma = d / n
lambda_plus = (1 + np.sqrt(gamma))**2
lambda_minus = (1 - np.sqrt(gamma))**2

print(f"\nMarchenko-Pastur noise bounds: [{lambda_minus:.3f}, {lambda_plus:.3f}]")

# Find where eigenvalues drop below noise floor
signal_dims = np.sum(eigenvalues > lambda_plus)
print(f"Signal dimensions (above noise): {signal_dims}")

# Find elbow via second derivative
log_eig = np.log(eigenvalues + 1e-10)
diff2 = np.diff(log_eig, 2)
elbow_idx = np.argmax(np.abs(diff2)) + 2
print(f"Elbow (2nd derivative):          {elbow_idx}")

# Cumulative variance
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
dim_80 = np.argmax(cumvar >= 0.80) + 1
dim_90 = np.argmax(cumvar >= 0.90) + 1
dim_95 = np.argmax(cumvar >= 0.95) + 1

print(f"\nDimensions for variance thresholds:")
print(f"  80%: {dim_80}")
print(f"  90%: {dim_90}")
print(f"  95%: {dim_95}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Local vs Global Structure (trustworthiness & continuity)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TEST 3: LOCAL VS GLOBAL STRUCTURE")
print("="*80)

def trustworthiness_continuity(X_high, X_low, k=20, n_samples=2000):
    """
    Trustworthiness: do neighbors in low-D stay neighbors in high-D?
    Continuity: do neighbors in high-D stay neighbors in low-D?
    """
    if len(X_high) > n_samples:
        idx = np.random.choice(len(X_high), n_samples, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]
    
    n = len(X_high)
    
    # Neighbors in high-D
    nbrs_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    _, idx_high = nbrs_high.kneighbors(X_high)
    
    # Neighbors in low-D
    nbrs_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)
    _, idx_low = nbrs_low.kneighbors(X_low)
    
    # Trustworthiness
    trust = 0
    for i in range(n):
        neighbors_low = set(idx_low[i, 1:])  # exclude self
        neighbors_high = set(idx_high[i, 1:])
        # Neighbors in low-D that are NOT in high-D
        intruders = neighbors_low - neighbors_high
        if intruders:
            # Rank penalty
            for j in intruders:
                r_high = np.where(idx_high[i] == j)[0]
                if len(r_high) > 0:
                    trust += (r_high[0] - k)
    
    trust = 1 - (2 / (n * k * (2*n - 3*k - 1))) * trust
    
    # Continuity
    cont = 0
    for i in range(n):
        neighbors_high = set(idx_high[i, 1:])
        neighbors_low = set(idx_low[i, 1:])
        # Neighbors in high-D that are NOT in low-D
        missing = neighbors_high - neighbors_low
        if missing:
            for j in missing:
                r_low = np.where(idx_low[i] == j)[0]
                if len(r_low) > 0:
                    cont += (r_low[0] - k)
    
    cont = 1 - (2 / (n * k * (2*n - 3*k - 1))) * cont
    
    return trust, cont

# Test at different dimensionalities
print("\nNeighbor preservation across dimensions:")
for d in [8, 12, 16, 20]:
    pca_d = PCA(n_components=d)
    X_low = pca_d.fit_transform(X_scaled)
    trust, cont = trustworthiness_continuity(X_scaled, X_low, k=15)
    print(f"  {d:2d}D: Trust={trust:.3f}, Continuity={cont:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Feature Redundancy (correlation + mutual information)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TEST 4: FEATURE REDUNDANCY")
print("="*80)

# Correlation matrix
corr_matrix = np.corrcoef(X_scaled.T)
np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation

# Find highly correlated pairs (|r| > 0.9)
high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        if abs(corr_matrix[i, j]) > 0.9:
            high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_matrix[i, j]))

print(f"\nHighly correlated feature pairs (|r| > 0.9): {len(high_corr_pairs)}")
for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:10]:
    print(f"  {f1:20s} ~ {f2:20s}  r={r:6.3f}")

# Average correlation per feature
avg_corr = np.abs(corr_matrix).mean(axis=1)
redundant_features = [feature_cols[i] for i in np.argsort(-avg_corr)[:5]]
print(f"\nMost redundant features (high avg |r|):")
for i, feat in enumerate(redundant_features):
    print(f"  {feat:25s} avg|r|={avg_corr[np.argsort(-avg_corr)[i]]:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Distribution Shape (Gaussianity test)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TEST 5: DISTRIBUTION SHAPE")
print("="*80)

# Per-feature kurtosis and skew
feature_kurt = []
feature_skew = []
heavy_tailed = []

for i, feat in enumerate(feature_cols):
    k = kurtosis(X_scaled[:, i])
    s = skew(X_scaled[:, i])
    feature_kurt.append(k)
    feature_skew.append(s)
    
    if abs(k) > 3:  # Heavy tails
        heavy_tailed.append((feat, k, s))

print(f"\nHeavy-tailed features (|kurtosis| > 3): {len(heavy_tailed)}")
for feat, k, s in sorted(heavy_tailed, key=lambda x: -abs(x[1]))[:5]:
    print(f"  {feat:25s} kurt={k:6.2f}, skew={s:6.2f}")

avg_kurt = np.mean(np.abs(feature_kurt))
print(f"\nMean |kurtosis|: {avg_kurt:.2f}")
if avg_kurt > 2:
    print("  → Recommend: Huber loss or robust AE (heavy tails)")
else:
    print("  → Recommend: MSE loss (near-Gaussian)")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Curvature Estimation (via MDS stress)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TEST 6: MANIFOLD CURVATURE")
print("="*80)

def estimate_curvature(X, n_samples=1000):
    """
    Rough curvature estimate via distortion in low-D embedding.
    Negative = hyperbolic, Positive = spherical, Zero = flat
    """
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    
    # Pairwise distances in high-D
    D_high = squareform(pdist(X_sample, metric='euclidean'))
    
    # Embed in lower dimension
    pca_low = PCA(n_components=10)
    X_low = pca_low.fit_transform(X_sample)
    D_low = squareform(pdist(X_low, metric='euclidean'))
    
    # Compare distance distributions
    # Hyperbolic: large distances compress more than small distances
    # Spherical: small distances expand more than large distances
    
    quantiles = [0.1, 0.5, 0.9]
    q_high = np.quantile(D_high[D_high > 0], quantiles)
    q_low = np.quantile(D_low[D_low > 0], quantiles)
    
    ratios = q_low / (q_high + 1e-10)
    
    # If large distances compress (ratio decreases), hyperbolic
    # If small distances expand (ratio increases), spherical
    curvature_est = ratios[2] - ratios[0]  # Large - small ratio
    
    return curvature_est, ratios

curv_est, ratios = estimate_curvature(X_scaled)
print(f"Curvature estimate: {curv_est:.3f}")
print(f"Distance ratio (10%/50%/90%): {ratios[0]:.3f} / {ratios[1]:.3f} / {ratios[2]:.3f}")

if curv_est < -0.1:
    print("  → Manifold appears HYPERBOLIC (tree-like)")
    print("  → Recommend: Hyperbolic encoder (Poincaré ball)")
elif curv_est > 0.1:
    print("  → Manifold appears SPHERICAL (clustered)")
    print("  → Recommend: Spherical constraints or VAE")
else:
    print("  → Manifold appears FLAT (Euclidean)")
    print("  → Recommend: Standard autoencoder")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SUMMARY & ARCHITECTURE RECOMMENDATIONS")
print("="*80)

# Synthesize results
recommended_dim = int(np.median([intrinsic_dim, signal_dims, elbow_idx, dim_90]))
print(f"\n1. LATENT DIMENSION")
print(f"   MLE intrinsic dim:    {intrinsic_dim:.1f}")
print(f"   Signal dims (noise):  {signal_dims}")
print(f"   Elbow:                {elbow_idx}")
print(f"   90% variance:         {dim_90}")
print(f"   → RECOMMENDED:        {recommended_dim}D bottleneck")

print(f"\n2. ARCHITECTURE TYPE")
if curv_est < -0.1:
    print(f"   Curvature: {curv_est:.3f} (hyperbolic)")
    print(f"   → RECOMMENDED: Hyperbolic encoder")
else:
    print(f"   Curvature: {curv_est:.3f} (flat)")
    print(f"   → RECOMMENDED: Standard feedforward AE")

print(f"\n3. LOSS FUNCTION")
print(f"   Mean |kurtosis|: {avg_kurt:.2f}")
if avg_kurt > 2:
    print(f"   → RECOMMENDED: Huber loss (δ=1.0)")
else:
    print(f"   → RECOMMENDED: MSE loss")

print(f"\n4. FEATURE PRUNING")
if len(high_corr_pairs) > 0:
    print(f"   High correlation pairs: {len(high_corr_pairs)}")
    print(f"   → CONSIDER: Drop one feature from each pair")
else:
    print(f"   No highly redundant features detected")

print(f"\n5. NEIGHBOR PRESERVATION")
trust_12, cont_12 = trustworthiness_continuity(X_scaled, 
                                               PCA(n_components=12).fit_transform(X_scaled))
print(f"   12D trustworthiness:  {trust_12:.3f}")
print(f"   12D continuity:       {cont_12:.3f}")
if trust_12 > 0.9 and cont_12 > 0.9:
    print(f"   → Excellent local structure preservation")
elif trust_12 > 0.8 or cont_12 > 0.8:
    print(f"   → Good structure preservation")
else:
    print(f"   → Poor structure — consider nonlinear methods")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

results = {
    'intrinsic_dim_mle': float(intrinsic_dim),
    'signal_dims': int(signal_dims),
    'elbow': int(elbow_idx),
    'dim_90pct': int(dim_90),
    'recommended_latent_dim': int(recommended_dim),
    'curvature_estimate': float(curv_est),
    'mean_kurtosis': float(avg_kurt),
    'high_corr_pairs': len(high_corr_pairs),
    'heavy_tailed_features': len(heavy_tailed),
    'trustworthiness_12d': float(trust_12) if 'trust_12' in locals() else None,
    'continuity_12d': float(cont_12) if 'cont_12' in locals() else None,
    'recommendations': {
        'latent_dim': int(recommended_dim),
        'architecture': 'hyperbolic' if curv_est < -0.1 else 'standard',
        'loss': 'huber' if avg_kurt > 2 else 'mse',
        'prune_features': len(high_corr_pairs) > 5,
    }
}

with open(DATA_DIR / 'manifold_geometry.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {DATA_DIR / 'manifold_geometry.json'}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Stock Manifold Geometry Analysis', fontsize=14, fontweight='bold')

# Plot 1: Scree plot with noise floor
ax = axes[0, 0]
ax.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', linewidth=2, markersize=4)
ax.axhline(lambda_plus, color='red', linestyle='--', alpha=0.7, label='Noise floor')
ax.axvline(elbow_idx, color='green', linestyle='--', alpha=0.7, label=f'Elbow (d={elbow_idx})')
ax.axvline(recommended_dim, color='blue', linestyle='--', alpha=0.7, 
           label=f'Recommended (d={recommended_dim})')
ax.set_xlabel('Component')
ax.set_ylabel('Eigenvalue')
ax.set_title('PCA Eigenvalue Spectrum')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Cumulative variance
ax = axes[0, 1]
ax.plot(range(1, len(cumvar)+1), cumvar, 'o-', linewidth=2, markersize=4)
ax.axhline(0.90, color='red', linestyle='--', alpha=0.7, label='90% variance')
ax.axvline(dim_90, color='green', linestyle='--', alpha=0.7, label=f'd={dim_90}')
ax.axvline(recommended_dim, color='blue', linestyle='--', alpha=0.7, 
           label=f'Recommended (d={recommended_dim})')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Variance Explained')
ax.set_title('Cumulative Variance')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Feature correlation heatmap (top 15 features)
ax = axes[1, 0]
top_features = np.argsort(-avg_corr)[:15]
corr_subset = corr_matrix[np.ix_(top_features, top_features)]
im = ax.imshow(np.abs(corr_subset), cmap='YlOrRd', vmin=0, vmax=1)
ax.set_xticks(range(len(top_features)))
ax.set_yticks(range(len(top_features)))
ax.set_xticklabels([feature_cols[i] for i in top_features], rotation=90, fontsize=7)
ax.set_yticklabels([feature_cols[i] for i in top_features], fontsize=7)
ax.set_title('Feature Correlation (Top 15 Most Redundant)')
plt.colorbar(im, ax=ax, label='|Correlation|')

# Plot 4: Distribution shape (kurtosis histogram)
ax = axes[1, 1]
ax.hist(feature_kurt, bins=20, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Gaussian (kurt=0)')
ax.axvline(3, color='orange', linestyle='--', alpha=0.7, label='Heavy tail threshold')
ax.axvline(-3, color='orange', linestyle='--', alpha=0.7)
ax.set_xlabel('Kurtosis')
ax.set_ylabel('Number of Features')
ax.set_title('Feature Distribution Shape')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / 'manifold_geometry.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Plot saved: {DATA_DIR / 'manifold_geometry.png'}")
print("\n" + "="*80)
print("GEOMETRY ANALYSIS COMPLETE")
print("="*80)
print(f"\nRecommended AE architecture:")
print(f"  Input:      {len(feature_cols)}D")
print(f"  Encoder:    {len(feature_cols)} → 64 → 32 → {recommended_dim}")
print(f"  Decoder:    {recommended_dim} → 32 → 64 → {len(feature_cols)}")
print(f"  Loss:       {'Huber (δ=1.0)' if avg_kurt > 2 else 'MSE'}")
print(f"  Type:       {'Hyperbolic' if curv_est < -0.1 else 'Standard'}")