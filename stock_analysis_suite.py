# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:48:40 2026

@author: Justin.Sanford
"""

# stock_analysis_suite.py
# Comprehensive exploratory analysis on stock features before AE training
# Replicates F1 telemetry analysis structure: PCA, clustering, linear probes, consistency

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')

# ── Feature definitions ───────────────────────────────────────────────────────
EXCLUDE_COLS = {'ticker', 'open', 'high', 'low', 'close', 'volume',
                'Mkt_RF', 'SMB', 'HML', 'RF',
                'high_252d', 'low_252d', 'high_63d', 'low_63d',
                'sma_21', 'sma_63', 'sma_252', 'vol_ma21',
                'up_vol_ratio', 'kurt_63d'}  # Exclude problematic features

print("="*80)
print("STOCK FEATURE ANALYSIS SUITE")
print("Comprehensive pre-AE validation and baseline establishment")
print("="*80)

# ── Step 1: Load clean data ───────────────────────────────────────────────────
print("\nStep 1: Loading clean stock data...")

manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
tickers  = manifest['ticker'].tolist()

# Load scaler
with open(DATA_DIR / 'stock_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Sample one stock to get feature list
sample_df = pd.read_parquet(DATA_DIR / f'stock_clean_{tickers[0]}.parquet')
feature_cols = [c for c in sample_df.columns 
                if c not in EXCLUDE_COLS
                and sample_df[c].dtype in [np.float64, np.float32]]

print(f"Features: {len(feature_cols)}")
print(f"Scaler dimensions: {len(scaler.scale_)}")

# Load subset of stocks for analysis (50 random stocks, max 5000 obs per stock)
np.random.seed(42)
sample_tickers = np.random.choice(tickers, size=min(50, len(tickers)), replace=False)

data_chunks = []
meta_chunks = []

print(f"Loading {len(sample_tickers)} stocks...")

for ticker in sample_tickers:
    f = DATA_DIR / f'stock_clean_{ticker}.parquet'
    if not f.exists():
        continue
    
    try:
        df = pd.read_parquet(f)
        
        # Subsample if too large
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
        
        X = df[feature_cols].values.astype(np.float32)
        
        # Basic cleaning
        X = np.where(np.isinf(X), np.nan, X)
        col_meds = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_meds[j]
        X = np.nan_to_num(X, nan=0.0)
        
        data_chunks.append(X)
        
        # Metadata
        meta = pd.DataFrame({
            'ticker': ticker,
            'date': df.index,
            'ret_1d': df['ret_1d'] if 'ret_1d' in df.columns else 0,
            'alpha_resid': df['alpha_resid'] if 'alpha_resid' in df.columns else 0,
        })
        meta_chunks.append(meta)
        
    except Exception as e:
        print(f"  Failed {ticker}: {e}")
        continue

X_all = np.vstack(data_chunks)
meta_all = pd.concat(meta_chunks, ignore_index=True)

print(f"Loaded: {X_all.shape[0]:,} observations from {len(sample_tickers)} stocks")
print(f"Feature matrix: {X_all.shape}")

# Scale
X_scaled = scaler.transform(X_all).astype(np.float32)

# ── Step 2: PCA variance explained ────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 2: PCA VARIANCE EXPLAINED")
print("="*80)

pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

print("\nComponents needed for variance thresholds:")
for thresh in [0.70, 0.80, 0.90, 0.95, 0.99]:
    n = np.argmax(cumvar >= thresh) + 1
    print(f"  {thresh*100:.0f}%: {n:3d} components (of {len(feature_cols)})")

# Work with 95% variance
N_COMPONENTS = int(np.argmax(cumvar >= 0.95)) + 1
print(f"\nUsing {N_COMPONENTS} components for downstream analysis")

pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

# Top features per component
print("\nTop features per component (first 10 PCs):")
loadings = pd.DataFrame(
    pca.components_,
    columns=feature_cols,
    index=[f'PC{i+1}' for i in range(N_COMPONENTS)]
)

for pc in list(loadings.index)[:10]:
    top = loadings.loc[pc].abs().sort_values(ascending=False).head(3)
    var = pca.explained_variance_ratio_[int(pc[2:])-1]
    print(f"  {pc} ({var:.3f}): {', '.join(top.index)}")

# ── Step 3: Elbow + HDBSCAN ──────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 3: CLUSTERING IN PCA SPACE")
print("="*80)

# K-means elbow
print("\nK-means elbow (on PCA space):")
inertias = []
ks = range(2, 21)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    inertias.append(km.inertia_)

# Find elbow via second derivative
diffs = np.diff(inertias)
diffs2 = np.diff(diffs)
elbow_k = list(ks)[np.argmin(diffs2) + 1]
print(f"  Elbow at k={elbow_k}")

# HDBSCAN sweep
print("\nHDBSCAN sweep on PCA space:")
for min_size in [50, 100, 200, 500, 1000]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    coverage   = (labels != -1).sum() / len(labels)
    
    if n_clusters > 1:
        try:
            labeled_mask = labels != -1
            sil = silhouette_score(X_pca[labeled_mask], labels[labeled_mask])
            db  = davies_bouldin_score(X_pca[labeled_mask], labels[labeled_mask])
            print(f"  min_size={min_size:4d} → {n_clusters:2d} clusters, "
                  f"noise={n_noise:5d} ({100*(1-coverage):4.1f}%), "
                  f"sil={sil:.3f}, db={db:.3f}")
        except:
            print(f"  min_size={min_size:4d} → {n_clusters:2d} clusters, "
                  f"noise={n_noise:5d} ({100*(1-coverage):4.1f}%)")

# Best HDBSCAN
best_size = 200
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=best_size,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom'
)
labels = clusterer.fit_predict(X_pca)
meta_all['cluster'] = labels
meta_all['strength'] = clusterer.probabilities_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()

print(f"\nSelected min_size={best_size}: {n_clusters} clusters, {n_noise} noise")

if n_clusters > 1 and (labels != -1).sum() > 10:
    labeled_mask = labels != -1
    sil = silhouette_score(X_pca[labeled_mask], labels[labeled_mask])
    db  = davies_bouldin_score(X_pca[labeled_mask], labels[labeled_mask])
    print(f"Silhouette:      {sil:.4f}")
    print(f"Davies-Bouldin:  {db:.4f}")

# Cluster composition
print("\nCluster composition:")
for label in sorted(meta_all['cluster'].unique())[:10]:  # Show first 10
    subset = meta_all[meta_all['cluster'] == label]
    name = 'NOISE' if label == -1 else f'Cluster {label}'
    
    mean_ret   = subset['ret_1d'].mean()
    std_ret    = subset['ret_1d'].std()
    mean_alpha = subset['alpha_resid'].mean()
    
    print(f"\n  {name} (n={len(subset):,}):")
    print(f"    Mean return:      {mean_ret:.4f}")
    print(f"    Std return:       {std_ret:.4f}")
    print(f"    Mean alpha resid: {mean_alpha:.4f}")
    print(f"    Top tickers:      {subset['ticker'].value_counts().head(3).to_dict()}")

# ── Step 4: Per-stock centroids ──────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 4: STOCK CENTROIDS IN PCA SPACE")
print("="*80)

X_pca_df = pd.DataFrame(X_pca[:, :8],
                         columns=[f'PC{i+1}' for i in range(8)])
X_pca_df['ticker']      = meta_all['ticker'].values
X_pca_df['ret_1d']      = meta_all['ret_1d'].values
X_pca_df['alpha_resid'] = meta_all['alpha_resid'].values

stock_centroids = X_pca_df.groupby('ticker')[[f'PC{i+1}' for i in range(8)]].mean()
stock_ret       = X_pca_df.groupby('ticker')['ret_1d'].mean()
stock_alpha     = X_pca_df.groupby('ticker')['alpha_resid'].mean()

stock_centroids['MeanReturn'] = stock_ret
stock_centroids['MeanAlpha']  = stock_alpha

print("\nStock centroids (sorted by mean return):")
print(stock_centroids.sort_values('MeanReturn').round(4).head(10).to_string())
print("\n...")
print(stock_centroids.sort_values('MeanReturn').round(4).tail(10).to_string())

# ── Step 5: Linear probes ─────────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 5: LINEAR PROBES ON PCA SPACE")
print("="*80)

# Filter out NaN values for each probe
valid_ret = ~np.isnan(meta_all['ret_1d'].values)
valid_alpha = ~np.isnan(meta_all['alpha_resid'].values)

print(f"\nValid observations:")
print(f"  ret_1d:      {valid_ret.sum():,} / {len(valid_ret):,}")
print(f"  alpha_resid: {valid_alpha.sum():,} / {len(valid_alpha):,}")

# Probe 1 — predict return from PCA
ridge = Ridge(alpha=1.0)
ret_scores = cross_val_score(ridge, X_pca[valid_ret], meta_all.loc[valid_ret, 'ret_1d'].values,
                              cv=5, scoring='r2')
print(f"\nPredict ret_1d from PCA (Ridge R²):")
print(f"  CV scores: {ret_scores.round(3)}")
print(f"  Mean R²:   {ret_scores.mean():.3f} ± {ret_scores.std():.3f}")

# Probe 2 — predict alpha residual from PCA
if valid_alpha.sum() > 1000:  # Need enough samples
    alpha_scores = cross_val_score(ridge, X_pca[valid_alpha], meta_all.loc[valid_alpha, 'alpha_resid'].values,
                                    cv=5, scoring='r2')
    print(f"\nPredict alpha_resid from PCA (Ridge R²):")
    print(f"  CV scores: {alpha_scores.round(3)}")
    print(f"  Mean R²:   {alpha_scores.mean():.3f} ± {alpha_scores.std():.3f}")
else:
    print(f"\nSkipping alpha_resid probe — insufficient valid samples")

# Probe 3 — predict ticker identity from PCA
le_ticker = LabelEncoder()
ticker_labels = le_ticker.fit_transform(meta_all['ticker'].values)
logreg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
ticker_scores = cross_val_score(logreg, X_pca, ticker_labels,
                                 cv=5, scoring='accuracy')
baseline_ticker = 1 / meta_all['ticker'].nunique()
print(f"\nPredict Ticker identity from PCA (LogReg accuracy):")
print(f"  CV scores: {ticker_scores.round(3)}")
print(f"  Mean acc:  {ticker_scores.mean():.3f} ± {ticker_scores.std():.3f}")
print(f"  Baseline:  {baseline_ticker:.3f}")
print(f"  Lift:      {ticker_scores.mean()/baseline_ticker:.1f}x")

# Probe 4 — predict year from PCA
meta_all['year'] = pd.to_datetime(meta_all['date']).dt.year
le_year = LabelEncoder()
year_labels = le_year.fit_transform(meta_all['year'].values)
year_scores = cross_val_score(logreg, X_pca, year_labels,
                               cv=5, scoring='accuracy')
baseline_year = 1 / meta_all['year'].nunique()
print(f"\nPredict Year from PCA (LogReg accuracy):")
print(f"  CV scores: {year_scores.round(3)}")
print(f"  Mean acc:  {year_scores.mean():.3f} ± {year_scores.std():.3f}")
print(f"  Baseline:  {baseline_year:.3f}")
print(f"  Lift:      {year_scores.mean()/baseline_year:.1f}x")

# ── Step 6: Stock consistency ─────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 6: STOCK CONSISTENCY IN PCA SPACE")
print("="*80)

# How much does each stock move around in PC space over time?
# Low variance = stable behavior, high variance = regime shifts
stock_consistency = []

for ticker in sample_tickers:
    mask = meta_all['ticker'] == ticker
    if mask.sum() < 100:  # Need at least 100 obs
        continue
    
    coords = X_pca[mask, :8]
    variance = coords.var(axis=0).mean()
    
    stock_consistency.append({
        'ticker': ticker,
        'StyleVariance': variance,
        'MeanReturn': meta_all.loc[mask, 'ret_1d'].mean(),
        'StdReturn': meta_all.loc[mask, 'ret_1d'].std(),
        'N': mask.sum()
    })

cons_df = pd.DataFrame(stock_consistency).sort_values('StyleVariance')

print("\nStock consistency (lower variance = more stable behavior):")
print(cons_df.round(4).head(10).to_string(index=False))
print("\n...")
print(cons_df.round(4).tail(10).to_string(index=False))

# Correlation: variance vs return/volatility
if len(cons_df) > 10:
    corr_ret = stats.pearsonr(cons_df['StyleVariance'], cons_df['MeanReturn'])
    corr_vol = stats.pearsonr(cons_df['StyleVariance'], cons_df['StdReturn'])
    print(f"\nCorrelations:")
    print(f"  StyleVariance ~ MeanReturn: r={corr_ret[0]:.3f}, p={corr_ret[1]:.4f}")
    print(f"  StyleVariance ~ StdReturn:  r={corr_vol[0]:.3f}, p={corr_vol[1]:.4f}")

# ── Step 7: Feature importance via PCA loadings ──────────────────────────────
print("\n" + "="*80)
print("STEP 7: FEATURE IMPORTANCE VIA PCA LOADINGS")
print("="*80)

# Which features contribute most to variance?
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.abs(pca.components_[:8]).sum(axis=0)  # Sum abs loadings across first 8 PCs
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 15 features by contribution to first 8 PCs:")
print(feature_importance.head(15).to_string(index=False))

# ── Step 8: Plots ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 8: GENERATING PLOTS")
print("="*80)

# Reduce to 2D for visualization
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Stock Feature Analysis — PCA Space\n'
             f'{len(sample_tickers)} stocks, {len(X_all):,} observations',
             fontsize=14, fontweight='bold')

# Plot A — colored by return
sc1 = axes[0, 0].scatter(X_pca2[:, 0], X_pca2[:, 1],
                         c=meta_all['ret_1d'].values,
                         cmap='RdYlGn', s=10, alpha=0.4,
                         vmin=-0.05, vmax=0.05)
plt.colorbar(sc1, ax=axes[0, 0], label='1-day return')
axes[0, 0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
axes[0, 0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
axes[0, 0].set_title('Colored by 1-Day Return')
axes[0, 0].grid(True, alpha=0.3)

# Plot B — colored by alpha residual
sc2 = axes[0, 1].scatter(X_pca2[:, 0], X_pca2[:, 1],
                         c=meta_all['alpha_resid'].values,
                         cmap='coolwarm', s=10, alpha=0.4,
                         vmin=-0.02, vmax=0.02)
plt.colorbar(sc2, ax=axes[0, 1], label='Alpha residual')
axes[0, 1].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
axes[0, 1].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
axes[0, 1].set_title('Colored by Alpha Residual (FF3)')
axes[0, 1].grid(True, alpha=0.3)

# Plot C — colored by cluster
if n_clusters > 1:
    sc3 = axes[1, 0].scatter(X_pca2[:, 0], X_pca2[:, 1],
                             c=labels, cmap='tab10', s=10, alpha=0.4)
    axes[1, 0].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1, 0].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1, 0].set_title(f'HDBSCAN Clusters (min_size={best_size})')
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'No clusters found', 
                    ha='center', va='center', transform=axes[1, 0].transAxes)

# Plot D — elbow curve
axes[1, 1].plot(list(ks), inertias, 'o-', linewidth=2, markersize=6)
axes[1, 1].axvline(elbow_k, color='red', linestyle='--', alpha=0.7, 
                   label=f'Elbow at k={elbow_k}')
axes[1, 1].set_xlabel('Number of clusters (k)')
axes[1, 1].set_ylabel('Inertia')
axes[1, 1].set_title('K-Means Elbow Curve')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / 'stock_pca_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: stock_pca_analysis.png")

# Stock consistency plot
if len(cons_df) > 5:
    fig, ax = plt.subplots(figsize=(12, 8))
    cons_sorted = cons_df.sort_values('StyleVariance').head(30)
    
    bars = ax.barh(cons_sorted['ticker'], cons_sorted['StyleVariance'],
                   alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Mean PC Variance Over Time (lower = more consistent)')
    ax.set_title('Stock Behavioral Consistency in PCA Space\n'
                 'Low variance = stable dynamics, high variance = regime shifts')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'stock_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: stock_consistency.png")

# ── Step 9: Save outputs ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("STEP 9: SAVING OUTPUTS")
print("="*80)

# Save PCA model
with open(DATA_DIR / 'stock_pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
print("  Saved: stock_pca.pkl")

# Save analysis results
results = {
    'timestamp': datetime.now().isoformat(),
    'n_stocks': len(sample_tickers),
    'n_observations': len(X_all),
    'n_features': len(feature_cols),
    'n_pca_components_95pct': int(N_COMPONENTS),
    'elbow_k': int(elbow_k),
    'hdbscan_n_clusters': int(n_clusters),
    'hdbscan_min_size': int(best_size),
    'probes': {
        'return_r2': float(ret_scores.mean()),
        'alpha_r2': float(alpha_scores.mean()),
        'ticker_accuracy': float(ticker_scores.mean()),
        'ticker_lift': float(ticker_scores.mean() / baseline_ticker),
        'year_accuracy': float(year_scores.mean()),
        'year_lift': float(year_scores.mean() / baseline_year),
    },
    'feature_importance_top10': feature_importance.head(10).to_dict('records'),
}

with open(DATA_DIR / 'stock_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Saved: stock_analysis_results.json")

# Save stock centroids and consistency
stock_centroids.to_csv(DATA_DIR / 'stock_centroids.csv')
print("  Saved: stock_centroids.csv")

if len(cons_df) > 0:
    cons_df.to_csv(DATA_DIR / 'stock_consistency.csv', index=False)
    print("  Saved: stock_consistency.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey findings:")
print(f"  • {N_COMPONENTS} components explain 95% variance (of {len(feature_cols)} features)")
print(f"  • Elbow at k={elbow_k} clusters")
print(f"  • Return prediction R²: {ret_scores.mean():.3f}")
print(f"  • Alpha prediction R²:  {alpha_scores.mean():.3f}")
print(f"  • Ticker ID lift:       {ticker_scores.mean()/baseline_ticker:.1f}x")
print(f"  • Year ID lift:         {year_scores.mean()/baseline_year:.1f}x")
print(f"\nReady for AE training — baseline established")