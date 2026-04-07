# -*- coding: utf-8 -*-
"""
stock_ae_extended.py
Extended manifold analysis — run after stock_ae_eval.py

Sections:
  1. Multi-relation constraint profiles
  2. Policy distance matrix + unsupervised clustering
  3. Sample efficiency for new listings (IPO analog)
  4. Temporal stability of constraint profiles across regimes
  5. Daily cross-sectional manifold dispersion as regime indicator
  6. Finance ↔ F1 axis analog mapping
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.special import rel_entr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
PLOT_DIR = DATA_DIR / 'plots' / 'extended'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Plots:  {PLOT_DIR}")

# ── Model (inferred from checkpoint) ─────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


def build_model_from_ckpt(ckpt):
    """Infer architecture from checkpoint weight shapes."""
    s    = ckpt['model_state']
    h1   = s['encoder.0.weight'].shape[0]
    h2   = s['encoder.4.weight'].shape[0]
    inp  = s['encoder.0.weight'].shape[1]
    lat  = s['encoder.8.weight'].shape[0]

    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(inp, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(h1, h2),  nn.LayerNorm(h2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(h2, lat),
            )
            self.decoder = nn.Sequential(
                nn.Linear(lat, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(h2, h1),  nn.LayerNorm(h1), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(h1, inp),
            )
        def encode(self, x): return self.encoder(x)
        def decode(self, z): return self.decoder(z)
        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    model = _AE().to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, inp, lat


def encode_dataset(model, loader):
    model.eval()
    latents = []
    with torch.no_grad():
        for X_batch in loader:
            _, z = model(X_batch.to(DEVICE))
            latents.append(z.cpu().numpy())
    return np.vstack(latents)


def load_stocks(tickers, feature_cols, date_filter=None):
    """
    Load clean parquet files.
    date_filter = ('before'|'after', 'YYYY-MM-DD') or None
    Returns X, y_ret, y_alpha, dates, ticker_labels
    """
    Xs, y_rets, y_alphas, all_dates, all_tickers = [], [], [], [], []
    for ticker in tickers:
        f = DATA_DIR / f'stock_clean_{ticker}.parquet'
        if not f.exists():
            continue
        try:
            df    = pd.read_parquet(f)
            dates = pd.to_datetime(df.index)

            if date_filter is not None:
                direction, cutoff = date_filter
                ts   = pd.Timestamp(cutoff)
                mask = (dates < ts) if direction == 'before' else (dates >= ts)
                df, dates = df[mask], dates[mask]

            if len(df) < 63:
                continue

            X = df[feature_cols].values.astype(np.float32)
            X = np.where(np.isinf(X), np.nan, X)
            col_meds = np.nanmedian(X, axis=0)
            for j in range(X.shape[1]):
                m = np.isnan(X[:, j])
                X[m, j] = col_meds[j]
            X = np.nan_to_num(X, nan=0.0)

            y_ret   = df['ret_1d'].values      if 'ret_1d'      in df.columns else np.zeros(len(df))
            y_alpha = df['alpha_resid'].values  if 'alpha_resid' in df.columns else np.zeros(len(df))

            valid = ~np.isnan(y_ret) & ~np.isnan(y_alpha)
            if valid.sum() < 63:
                continue

            Xs.append(X[valid])
            y_rets.append(y_ret[valid])
            y_alphas.append(y_alpha[valid])
            all_dates.append(dates[valid])
            all_tickers.extend([ticker] * valid.sum())
        except Exception as e:
            print(f"  Skip {ticker}: {e}")

    return (np.vstack(Xs), np.concatenate(y_rets), np.concatenate(y_alphas),
            np.concatenate(all_dates), np.array(all_tickers))


# ── Load models + data ────────────────────────────────────────────────────────
# Temporal model  → sections 1, 4, 5, 6  (time-based phenomena, regime analysis)
# Fold-1 model    → sections 2, 3         (stock-level characterization, new listings)

print("\nLoading temporal model (sections 1/4/5/6)...")
ckpt_temp         = torch.load(DATA_DIR / 'ae_temporal_best.pt', map_location=DEVICE, weights_only=False)
model_temp, input_dim, latent_dim = build_model_from_ckpt(ckpt_temp)
scaler_temp       = ckpt_temp['scaler']
feature_cols      = ckpt_temp['feature_cols']   # canonical feature list

print("Loading fold-1 model (sections 2/3)...")
ckpt_fold1        = torch.load(DATA_DIR / 'ae_fold1_best.pt', map_location=DEVICE, weights_only=False)
model_fold1, _, _ = build_model_from_ckpt(ckpt_fold1)
scaler_fold1      = ckpt_fold1['scaler']
# fold1 feature_cols should match — verify
assert ckpt_fold1['feature_cols'] == feature_cols, \
    "Feature mismatch between temporal and fold-1 checkpoints"

manifest    = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

# ── Load + encode with temporal model (pre-2023 only — what it was trained on) ─
print("\nLoading pre-2023 data (temporal model scope)...")
X_all, y_ret_all, y_alpha_all, dates_all, tickers_all = load_stocks(
    all_tickers, feature_cols, date_filter=('before', '2023-01-01'))
print(f"  {len(X_all):,} observations  |  {len(np.unique(tickers_all))} stocks")

X_all_s = scaler_temp.transform(X_all)
loader  = DataLoader(StockDataset(X_all_s), batch_size=8192, shuffle=False)
print("Encoding with temporal model...")
Z_temp = encode_dataset(model_temp, loader)
print(f"  Z_temp shape: {Z_temp.shape}")

dates_all = np.array(dates_all)

# ── Load + encode fold-1 val stocks only (what fold-1 never saw) ──────────────
print("\nLoading fold-1 val stocks (fold-1 model scope)...")
import json as _json
with open(DATA_DIR / 'ae_training_results.json') as _f:
    _tr = _json.load(_f)
fold1_val_tickers = _tr['cv_results'][0]['val_tickers']

X_fold, y_ret_fold, y_alpha_fold, dates_fold, tickers_fold = load_stocks(
    fold1_val_tickers, feature_cols)
print(f"  {len(X_fold):,} observations  |  {len(np.unique(tickers_fold))} stocks")

X_fold_s  = scaler_fold1.transform(X_fold)
loader_f1 = DataLoader(StockDataset(X_fold_s), batch_size=8192, shuffle=False)
print("Encoding with fold-1 model...")
Z_fold = encode_dataset(model_fold1, loader_f1)
print(f"  Z_fold shape: {Z_fold.shape}")

dates_fold   = np.array(dates_fold)

# Convenience aliases — Z_all / tickers_all point to temporal throughout
Z_all = Z_temp


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Multi-Relation Constraint Profiles
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 1: MULTI-RELATION CONSTRAINT PROFILES")
print("=" * 80)
print("Which market phenomena have stable axis-level constraint signatures?")

DIAG_SIGN_THRESH   = 0.15   # |sign_consistency - 0.5| threshold
DIAG_EFFECT_THRESH = 0.15   # effect size threshold
MIN_DIAG_AXES      = 3      # learnable if >= this many diagnostic axes

def constraint_profile(Z, dates, signal, n_quantiles=5, min_per_group=3):
    """
    Compute constraint profile for a signal.
    For each date, rank signal into quantiles cross-sectionally.
    Return per-axis sign consistency and effect size for Q_top vs Q_bottom.
    """
    unique_dates = np.unique(dates)
    q_top   = n_quantiles
    q_bot   = 1
    deltas  = []

    for d in unique_dates:
        mask_d = dates == d
        sig_d  = signal[mask_d]
        Z_d    = Z[mask_d]
        valid  = ~np.isnan(sig_d)
        if valid.sum() < n_quantiles * min_per_group:
            continue

        sig_v = sig_d[valid]
        Z_v   = Z_d[valid]
        ranks = pd.Series(sig_v).rank(pct=True)
        q_labels = pd.cut(ranks, bins=n_quantiles,
                          labels=range(1, n_quantiles + 1))

        top_mask = q_labels == q_top
        bot_mask = q_labels == q_bot
        if top_mask.sum() < min_per_group or bot_mask.sum() < min_per_group:
            continue

        delta = Z_v[top_mask.values].mean(axis=0) - Z_v[bot_mask.values].mean(axis=0)
        deltas.append(delta)

    if len(deltas) < 50:
        return None

    deltas = np.array(deltas)
    sign_cons   = (deltas > 0).mean(axis=0)
    effect_size = np.abs(deltas.mean(axis=0)) / (deltas.std(axis=0) + 1e-8)
    diagnostic  = (np.abs(sign_cons - 0.5) >= DIAG_SIGN_THRESH) & \
                  (effect_size >= DIAG_EFFECT_THRESH)

    return {
        'n_dates':         len(deltas),
        'sign_consistency': sign_cons,
        'effect_size':      effect_size,
        'diagnostic':       diagnostic,
        'n_diagnostic':     int(diagnostic.sum()),
        'learnable':        diagnostic.sum() >= MIN_DIAG_AXES,
    }


# Define relations to test
# Each is (name, signal_array, description)
# All signals are cross-sectionally ranked per date
relations = []

# Alpha quintile (already done in eval — replicate for comparison)
relations.append(('alpha_Q5vQ1',   y_alpha_all, 'FF3 alpha (top vs bottom quintile)'))

# Return quintile
relations.append(('ret_Q5vQ1',     y_ret_all,   'Daily return (top vs bottom quintile)'))

# Volatility quintile — low vol vs high vol (inverted: Q1=low vol = "stable")
if 'vol_21d' in feature_cols:
    idx = feature_cols.index('vol_21d')
    relations.append(('vol_low_Q1vQ5', -X_all[:, idx], 'Low vol vs high vol'))

# Momentum quintile
if 'ret_63d' in feature_cols:
    idx = feature_cols.index('ret_63d')
    relations.append(('momentum_Q5vQ1', X_all[:, idx], '63d momentum (top vs bottom)'))

# Price position quintile (52w high proximity)
if 'pos_252d' in feature_cols:
    idx = feature_cols.index('pos_252d')
    relations.append(('pos252_Q5vQ1', X_all[:, idx], '52-week position (top vs bottom)'))

# Drawdown quintile (least drawdown vs most)
if 'drawdown' in feature_cols:
    idx = feature_cols.index('drawdown')
    relations.append(('drawdown_Q1vQ5', -X_all[:, idx], 'Least drawdown vs most'))

# Momentum acceleration
if 'mom_accel' in feature_cols:
    idx = feature_cols.index('mom_accel')
    relations.append(('mom_accel_Q5vQ1', X_all[:, idx], 'Momentum acceleration'))

# Volume surge
if 'vol_ratio_v' in feature_cols:
    idx = feature_cols.index('vol_ratio_v')
    relations.append(('vol_surge_Q5vQ1', X_all[:, idx], 'Volume surge (top vs bottom)'))

print(f"\nTesting {len(relations)} relation types...")
print(f"\n  {'Relation':<22} {'Dates':>7} {'Diag axes':>10} {'Learnable':>10}  Profile summary")
print(f"  {'─'*75}")

profiles      = {}
profile_vecs  = {}   # for cross-relation similarity

for name, signal, desc in relations:
    result = constraint_profile(Z_all, dates_all, signal)
    if result is None:
        print(f"  {name:<22} insufficient data")
        continue

    profiles[name] = result
    profiles[name]['desc'] = desc

    sc   = result['sign_consistency']
    diag = result['diagnostic']
    diag_str = ', '.join(f'z{i}({"↑" if sc[i]>0.5 else "↓"})' for i in np.where(diag)[0])

    print(f"  {name:<22} {result['n_dates']:>7,} {result['n_diagnostic']:>10}/12 "
          f"{'✓ YES' if result['learnable'] else '  no':>10}  {diag_str[:50]}")

    # Weighted sign vector for cross-relation similarity
    profile_vecs[name] = (sc - 0.5) * result['effect_size']

# Cross-relation constraint similarity
print(f"\n  Cross-relation constraint similarity (cosine):")
names_learned = [n for n, r in profiles.items() if r['learnable']]

if len(names_learned) >= 2:
    sim_matrix = np.zeros((len(names_learned), len(names_learned)))
    for i, n1 in enumerate(names_learned):
        for j, n2 in enumerate(names_learned):
            v1, v2 = profile_vecs[n1], profile_vecs[n2]
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
            sim_matrix[i, j] = np.dot(v1, v2) / denom if denom > 0 else 0

    print(f"\n  {'':22}", end='')
    for n in names_learned:
        print(f"  {n[:10]:>12}", end='')
    print()
    for i, n1 in enumerate(names_learned):
        print(f"  {n1:<22}", end='')
        for j in range(len(names_learned)):
            print(f"  {sim_matrix[i,j]:>12.3f}", end='')
        print()

    mean_off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / \
                    (len(names_learned) * (len(names_learned) - 1))
    print(f"\n  Mean off-diagonal similarity: {mean_off_diag:.3f}")
    print(f"  (Atlas semantic relations: ~0.019 — near-orthogonal)")

    # Plot constraint profiles heatmap
    n_rel = len(relations)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n_rel * 0.7 + 2)))

    # Sign consistency heatmap
    sc_matrix = np.array([profiles[n]['sign_consistency']
                           for n in profiles if n in profiles])
    rel_names = [n for n in profiles]
    sns.heatmap(sc_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0.5, vmin=0, vmax=1,
                xticklabels=[f'z{i}' for i in range(latent_dim)],
                yticklabels=rel_names,
                ax=axes[0])
    axes[0].set_title('Sign Consistency per Axis\n(>0.65 or <0.35 = diagnostic)')

    # Cross-relation similarity
    if len(names_learned) >= 2:
        sns.heatmap(sim_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1,
                    xticklabels=[n[:12] for n in names_learned],
                    yticklabels=[n[:12] for n in names_learned],
                    ax=axes[1])
        axes[1].set_title('Cross-Relation Constraint Similarity\n(near-0 = orthogonal)')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'constraint_profiles_multi.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: constraint_profiles_multi.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Policy Distance Matrix + Unsupervised Clustering
# (fold-1 model — unseen stocks, generalizes across names not time)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 2: POLICY DISTANCE MATRIX  [fold-1 model, unseen val stocks]")
print("=" * 80)
print("Do stocks cluster by sector/factor without supervision?")
print("Analog: F1 teammate pairs at 2nd percentile of policy distances")

# Compute per-stock policy distributions (centroid + covariance)
print("\nComputing per-stock policy distributions...")
stock_policies = {}
for ticker in np.unique(tickers_fold):
    mask = tickers_fold == ticker
    if mask.sum() < 63:
        continue
    z_t = Z_fold[mask]
    stock_policies[ticker] = {
        'centroid':   z_t.mean(axis=0),
        'cov':        np.cov(z_t.T) + np.eye(latent_dim) * 1e-4,
        'stability':  float(z_t.var(axis=0).mean()),
        'n_obs':      int(mask.sum()),
    }

print(f"  Stocks with policy distributions: {len(stock_policies)}")

# Bhattacharyya distance between policy distributions
def bhattacharyya(mu1, cov1, mu2, cov2):
    sigma = (cov1 + cov2) / 2
    try:
        sigma_inv = np.linalg.inv(sigma)
        diff      = mu1 - mu2
        term1     = (1/8) * diff @ sigma_inv @ diff
        sign1, ld1 = np.linalg.slogdet(sigma)
        sign2, ld2 = np.linalg.slogdet(cov1)
        sign3, ld3 = np.linalg.slogdet(cov2)
        if sign1 <= 0 or sign2 <= 0 or sign3 <= 0:
            return np.linalg.norm(mu1 - mu2)  # fallback to centroid distance
        term2 = 0.5 * (ld1 - 0.5 * (ld2 + ld3))
        return float(term1 + term2)
    except np.linalg.LinAlgError:
        return np.linalg.norm(mu1 - mu2)

# Sample up to 500 stocks for distance matrix (full 2000x2000 is slow)
rng         = np.random.RandomState(42)
sample_n    = min(500, len(stock_policies))
sample_tickers = rng.choice(list(stock_policies.keys()), sample_n, replace=False)

print(f"  Computing {sample_n}×{sample_n} distance matrix...")
D = np.zeros((sample_n, sample_n))
for i in range(sample_n):
    for j in range(i + 1, sample_n):
        t1, t2 = sample_tickers[i], sample_tickers[j]
        p1, p2 = stock_policies[t1], stock_policies[t2]
        d = bhattacharyya(p1['centroid'], p1['cov'], p2['centroid'], p2['cov'])
        D[i, j] = D[j, i] = max(0, d)   # clip negatives from numerical issues

# Clip extreme values
D_clipped = np.clip(D, 0, np.percentile(D[D > 0], 99))

print(f"  Distance matrix: min={D[D>0].min():.3f}  "
      f"median={np.median(D[D>0]):.3f}  "
      f"max={D[D>0].max():.3f}")

# Most similar pairs
upper_idx = np.triu_indices(sample_n, k=1)
distances = D[upper_idx]
sorted_idx = np.argsort(distances)

print(f"\n  10 most similar pairs (lowest Bhattacharyya distance):")
for k in range(min(10, len(sorted_idx))):
    idx = sorted_idx[k]
    i, j = upper_idx[0][idx], upper_idx[1][idx]
    t1, t2 = sample_tickers[i], sample_tickers[j]
    print(f"    {t1:<8} ↔ {t2:<8}  d={distances[idx]:.3f}")

print(f"\n  10 most distinct pairs (highest Bhattacharyya distance):")
for k in range(1, min(11, len(sorted_idx))):
    idx = sorted_idx[-k]
    i, j = upper_idx[0][idx], upper_idx[1][idx]
    t1, t2 = sample_tickers[i], sample_tickers[j]
    print(f"    {t1:<8} ↔ {t2:<8}  d={distances[idx]:.3f}")

# Hierarchical clustering
print(f"\n  Hierarchical clustering on policy distances...")
D_sym = (D_clipped + D_clipped.T) / 2
np.fill_diagonal(D_sym, 0)

try:
    linkage_mat = linkage(squareform(D_sym), method='ward')
    n_clusters  = 8
    cluster_labels = fcluster(linkage_mat, t=n_clusters, criterion='maxclust')
    print(f"  Cluster sizes (k={n_clusters}):")
    for c in range(1, n_clusters + 1):
        members = sample_tickers[cluster_labels == c]
        print(f"    Cluster {c}: n={len(members)}  "
              f"examples: {', '.join(members[:5])}")

    # Plot dendrogram (subsample for readability)
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(linkage_mat, ax=ax, no_labels=True, truncate_mode='lastp', p=40,
               color_threshold=np.percentile(linkage_mat[:, 2], 70))
    ax.set_title(f'Policy Distance Dendrogram — {sample_n} Stocks\n'
                 f'(Ward linkage on Bhattacharyya distances)')
    ax.set_ylabel('Distance')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'policy_distance_dendrogram.png', dpi=150)
    plt.close()
    print(f"  Plot saved: policy_distance_dendrogram.png")
except Exception as e:
    print(f"  Clustering failed: {e}")

# Distance distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(distances, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
axes[0].axvline(np.percentile(distances, 2), color='red', linestyle='--',
                label='2nd percentile')
axes[0].set_xlabel('Bhattacharyya Distance')
axes[0].set_ylabel('Count')
axes[0].set_title('Policy Distance Distribution')
axes[0].legend()

# Stability vs distance from field centroid
stab_vals = np.array([stock_policies[t]['stability'] for t in sample_tickers])
field_centroid = np.mean([stock_policies[t]['centroid'] for t in sample_tickers], axis=0)
dist_from_field = np.array([np.linalg.norm(stock_policies[t]['centroid'] - field_centroid)
                             for t in sample_tickers])
axes[1].scatter(stab_vals, dist_from_field, alpha=0.3, s=10, color='orange')
r_val, p_val = pearsonr(stab_vals, dist_from_field)
axes[1].set_xlabel('Manifold Stability (latent variance)')
axes[1].set_ylabel('Distance from Field Centroid')
axes[1].set_title(f'Stability vs Field Distance\nr={r_val:.3f}, p={p_val:.4f}')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'policy_distance_analysis.png', dpi=150)
plt.close()
print(f"  Plot saved: policy_distance_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Sample Efficiency for New Listings
# (fold-1 model — unseen stocks, stock-level characterization)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 3: SAMPLE EFFICIENCY — NEW LISTING ANALOG  [fold-1 model]")
print("=" * 80)
print("How quickly can we characterize a stock from limited history?")
print("Practical: newly listed / spun-off stocks with 30-90 days of data")

# Split stocks by stability quartile using fold data
all_stable_tickers = list(stock_policies.keys())
stab_all   = np.array([stock_policies[t]['stability'] for t in all_stable_tickers])
q25, q75   = np.percentile(stab_all, 25), np.percentile(stab_all, 75)

stable_group   = [t for t, s in zip(all_stable_tickers, stab_all) if s <= q25]
unstable_group = [t for t, s in zip(all_stable_tickers, stab_all) if s >= q75]

# For each group: bootstrap centroid error at various n
sample_sizes   = [5, 10, 21, 42, 63, 126, 252]
rng2           = np.random.RandomState(42)

def bootstrap_centroid_error(tickers, n_obs, n_boot=30, max_stocks=50):
    errors = []
    for ticker in tickers[:max_stocks]:
        mask  = tickers_fold == ticker
        z_t   = Z_fold[mask]
        if len(z_t) < n_obs * 2:
            continue
        true_c = z_t.mean(axis=0)
        for _ in range(n_boot):
            idx = rng2.choice(len(z_t), n_obs, replace=False)
            errors.append(np.linalg.norm(z_t[idx].mean(axis=0) - true_c))
    return np.mean(errors) if errors else np.nan

print(f"\n  Stable group:   n={len(stable_group)} stocks")
print(f"  Unstable group: n={len(unstable_group)} stocks")
print(f"\n  {'n obs':<8} {'Stable':>10} {'Unstable':>12} {'Ratio':>8} {'Practical context'}")
print(f"  {'─'*60}")

stable_errs, unstable_errs = [], []
contexts = {5: '1 week', 10: '2 weeks', 21: '1 month', 42: '2 months',
            63: '3 months', 126: '6 months', 252: '1 year'}

for n in sample_sizes:
    se = bootstrap_centroid_error(stable_group,   n)
    ue = bootstrap_centroid_error(unstable_group, n)
    stable_errs.append(se)
    unstable_errs.append(ue)
    ratio = ue / se if se > 0 and not np.isnan(se) else np.nan
    ctx   = contexts.get(n, '')
    print(f"  {n:<8} {se:>10.4f} {ue:>12.4f} {ratio:>8.2f}x  {ctx}")

# Where does unstable stock reach stable's n=63 precision?
target_precision = stable_errs[sample_sizes.index(63)] if 63 in sample_sizes else None
if target_precision:
    crossover = None
    for n, ue in zip(sample_sizes, unstable_errs):
        if not np.isnan(ue) and ue <= target_precision:
            crossover = n
            break
    if crossover:
        print(f"\n  Unstable stock reaches stable's 3-month precision at n={crossover} obs "
              f"({contexts.get(crossover, '')})")
    else:
        print(f"\n  Unstable stocks never reach stable's 3-month precision within {max(sample_sizes)} obs")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(sample_sizes, stable_errs,   'o-',  label='Stable (Q1)',   color='steelblue', lw=2)
axes[0].plot(sample_sizes, unstable_errs, 's--', label='Unstable (Q4)', color='crimson',   lw=2)
if target_precision:
    axes[0].axhline(target_precision, color='steelblue', alpha=0.3, linestyle=':',
                    label='Stable @ 63-day precision')
axes[0].set_xlabel('Trading days of history')
axes[0].set_ylabel('Centroid estimation error')
axes[0].set_title('Sample Efficiency by Stability\n(New listing characterization speed)')
axes[0].legend()
axes[0].set_xticks(sample_sizes)
axes[0].set_xticklabels([f'{n}\n({contexts[n]})' for n in sample_sizes], fontsize=7)

# Distribution of time-to-characterize across all stocks
# Proxy: how many obs until centroid error < 0.02
print(f"\n  Time-to-characterize distribution (target error < 0.02):")
ttc_by_stability = []
for ticker in all_stable_tickers[:200]:
    mask  = tickers_fold == ticker
    z_t   = Z_fold[mask]
    stab  = stock_policies[ticker]['stability']
    if len(z_t) < 30:
        continue
    true_c = z_t.mean(axis=0)
    for n in range(5, min(len(z_t) // 2, 252), 5):
        errs = [np.linalg.norm(z_t[rng2.choice(len(z_t), n, replace=False)].mean(axis=0) - true_c)
                for _ in range(10)]
        if np.mean(errs) < 0.02:
            ttc_by_stability.append((stab, n))
            break

if ttc_by_stability:
    stab_ttc, n_ttc = zip(*ttc_by_stability)
    axes[1].scatter(stab_ttc, n_ttc, alpha=0.4, s=15, color='purple')
    r_ttc, p_ttc = pearsonr(stab_ttc, n_ttc)
    axes[1].set_xlabel('Manifold Stability')
    axes[1].set_ylabel('Trading days to characterize')
    axes[1].set_title(f'Time-to-Characterize vs Stability\nr={r_ttc:.3f}, p={p_ttc:.4f}')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'sample_efficiency_extended.png', dpi=150)
plt.close()
print(f"  Plot saved: sample_efficiency_extended.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Temporal Stability of Constraint Profiles
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 4: TEMPORAL STABILITY OF CONSTRAINT PROFILES")
print("=" * 80)
print("Is the alpha constraint profile stable across market regimes?")

# Define eras
eras = [
    ('Pre-GFC',    '2000-01-01', '2007-12-31'),
    ('GFC',        '2008-01-01', '2009-12-31'),
    ('Recovery',   '2010-01-01', '2014-12-31'),
    ('Bull',       '2015-01-01', '2019-12-31'),
    ('COVID',      '2020-01-01', '2020-12-31'),
    ('Post-COVID', '2021-01-01', '2022-12-31'),
    ('Rate-hike',  '2023-01-01', '2024-12-31'),
]

era_profiles = {}
print(f"\n  {'Era':<14} {'Dates':>7} {'Diag axes':>10}  Diagnostic axes")
print(f"  {'─'*65}")

for era_name, start, end in eras:
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)
    mask_era = (dates_all >= ts_start) & (dates_all <= ts_end)

    if mask_era.sum() < 5000:
        print(f"  {era_name:<14} insufficient data ({mask_era.sum():,} obs)")
        continue

    result = constraint_profile(Z_all[mask_era], dates_all[mask_era],
                                y_alpha_all[mask_era])
    if result is None:
        print(f"  {era_name:<14} insufficient dates")
        continue

    era_profiles[era_name] = result
    sc   = result['sign_consistency']
    diag = result['diagnostic']
    diag_str = ', '.join(f'z{i}({"↑" if sc[i]>0.5 else "↓"})' for i in np.where(diag)[0])
    print(f"  {era_name:<14} {result['n_dates']:>7,} {result['n_diagnostic']:>10}/12  {diag_str}")

# Cross-era profile similarity
if len(era_profiles) >= 3:
    era_names = list(era_profiles.keys())
    era_vecs  = {n: (era_profiles[n]['sign_consistency'] - 0.5) *
                    era_profiles[n]['effect_size']
                 for n in era_names}

    print(f"\n  Cross-era profile similarity (cosine):")
    sim_era = np.zeros((len(era_names), len(era_names)))
    for i, n1 in enumerate(era_names):
        for j, n2 in enumerate(era_names):
            v1, v2 = era_vecs[n1], era_vecs[n2]
            denom  = np.linalg.norm(v1) * np.linalg.norm(v2)
            sim_era[i, j] = np.dot(v1, v2) / denom if denom > 0 else 0

    mean_sim = (sim_era.sum() - np.trace(sim_era)) / (len(era_names) * (len(era_names) - 1))
    print(f"  Mean cross-era similarity: {mean_sim:.3f}")
    if mean_sim > 0.5:
        print(f"  → STABLE constraint profile across regimes")
    elif mean_sim > 0.2:
        print(f"  → MODERATELY stable — some regime dependence")
    else:
        print(f"  → UNSTABLE — constraint profile shifts significantly across regimes")

    # Plot era profiles heatmap
    sc_matrix_era = np.array([era_profiles[n]['sign_consistency'] for n in era_names])
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(era_names) * 0.8 + 2)))

    sns.heatmap(sc_matrix_era, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0.5, vmin=0, vmax=1,
                xticklabels=[f'z{i}' for i in range(latent_dim)],
                yticklabels=era_names, ax=axes[0])
    axes[0].set_title('Alpha Constraint Profile by Era\n(sign consistency — red=Q5 high, blue=Q5 low)')

    sns.heatmap(sim_era, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1,
                xticklabels=[n[:8] for n in era_names],
                yticklabels=[n[:8] for n in era_names], ax=axes[1])
    axes[1].set_title(f'Cross-Era Similarity\nmean={mean_sim:.3f}')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'constraint_temporal_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: constraint_temporal_stability.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Daily Manifold Dispersion as Regime Indicator
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 5: DAILY MANIFOLD DISPERSION")
print("=" * 80)
print("Does the cross-sectional cloud compress during crises?")

# Compute daily cross-sectional dispersion metrics
unique_dates_sorted = np.sort(np.unique(dates_all))
daily_stats = []

print(f"  Computing daily dispersion over {len(unique_dates_sorted):,} dates...")
for d in unique_dates_sorted:
    mask_d = dates_all == d
    if mask_d.sum() < 20:
        continue
    Z_d = Z_all[mask_d]

    # Mean pairwise distance (sample for speed)
    if len(Z_d) > 200:
        idx_s = np.random.choice(len(Z_d), 200, replace=False)
        Z_s   = Z_d[idx_s]
    else:
        Z_s = Z_d

    diffs      = Z_s[:, None, :] - Z_s[None, :, :]
    pair_dists = np.sqrt((diffs ** 2).sum(axis=2))
    mean_dist  = pair_dists[np.triu_indices(len(Z_s), k=1)].mean()

    daily_stats.append({
        'date':          pd.Timestamp(d),
        'n_stocks':      int(mask_d.sum()),
        'mean_dist':     float(mean_dist),
        'var_pc1':       float(Z_d[:, 0].var()),
        'centroid_norm': float(np.linalg.norm(Z_d.mean(axis=0))),
    })

daily_df = pd.DataFrame(daily_stats).set_index('date').sort_index()

# Rolling mean for smoothing
daily_df['mean_dist_30d'] = daily_df['mean_dist'].rolling(21).mean()

print(f"  Daily dispersion computed: {len(daily_df)} trading days")
print(f"  Mean dispersion: {daily_df['mean_dist'].mean():.4f}")
print(f"  Std dispersion:  {daily_df['mean_dist'].std():.4f}")

# Crisis period dispersion vs normal
crisis_periods = {
    'GFC_2008':   ('2008-09-01', '2009-03-31'),
    'COVID_2020': ('2020-02-01', '2020-06-30'),
    'Rate_2022':  ('2022-01-01', '2022-12-31'),
}

normal_disp = daily_df['mean_dist'].median()
print(f"\n  Dispersion by period (vs normal median={normal_disp:.4f}):")
for period, (start, end) in crisis_periods.items():
    mask_p = (daily_df.index >= start) & (daily_df.index <= end)
    if mask_p.sum() < 10:
        continue
    crisis_disp = daily_df.loc[mask_p, 'mean_dist'].mean()
    pct_change  = (crisis_disp - normal_disp) / normal_disp * 100
    direction   = '▼ compressed' if pct_change < 0 else '▲ expanded'
    print(f"    {period:<15} mean={crisis_disp:.4f}  "
          f"change={pct_change:+.1f}%  {direction}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(daily_df.index, daily_df['mean_dist'], alpha=0.3, color='steelblue', lw=0.5)
axes[0].plot(daily_df.index, daily_df['mean_dist_30d'], color='steelblue', lw=1.5,
             label='21d rolling mean')
axes[0].set_ylabel('Mean pairwise distance')
axes[0].set_title('Daily Cross-Sectional Manifold Dispersion')
axes[0].legend(fontsize=8)

# Shade crisis periods
for period, (start, end) in crisis_periods.items():
    for ax in axes:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color='red',
                   label=period if ax == axes[0] else '')
axes[0].legend(fontsize=7)

axes[1].plot(daily_df.index, daily_df['centroid_norm'], color='darkorange', lw=1, alpha=0.7)
axes[1].set_ylabel('||centroid||')
axes[1].set_title('Daily Centroid Norm (distance from origin)')

axes[2].plot(daily_df.index, daily_df['n_stocks'], color='gray', lw=0.8, alpha=0.6)
axes[2].set_ylabel('Stocks in cross-section')
axes[2].set_xlabel('Date')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'manifold_dispersion.png', dpi=150)
plt.close()
print(f"  Plot saved: manifold_dispersion.png")

# Save daily dispersion as potential signal
daily_df.to_parquet(DATA_DIR / 'manifold_dispersion_daily.parquet')
print(f"  Daily dispersion saved: manifold_dispersion_daily.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Finance ↔ F1 Axis Analog Mapping
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6: FINANCE ↔ F1 AXIS ANALOG MAPPING")
print("=" * 80)
print("Do the same geometric primitives appear in both domains?")

# From eval script section 5 we know:
# z5 ≈ volatility (r=-0.735)
# z7 ≈ price position / drawdown (r=0.617 drawdown)
# z0 ≈ momentum acceleration (r=-0.633)
# z8 ≈ alpha/return predictive
# z9, z10 ≈ also volatility-correlated

# F1 analogs from the papers:
# Primary latent dim Z1 ≈ within-team performance (r=-0.675)
# Stability ≈ consistency axis (independent of performance)
# Latent space captures: braking style, throttle application, gear selection

print("""
  Finance axis → F1 analog mapping (based on probe results):

  Finance z5  (volatility regime, r=-0.735)
    ↔ F1 pace consistency axis (σδ, η²=0.843 in clustering paper)
    Both: primary discriminant of behavioral regularity

  Finance z7  (price position + drawdown, r=0.617)
    ↔ F1 primary latent dim Z1 (within-team performance, r=-0.675)
    Both: position in competitive hierarchy, recovered without supervision

  Finance z0  (momentum acceleration, r=-0.633)
    ↔ F1 tyre management axis (η²=0.538)
    Both: rate-of-change / resource-depletion axis

  Finance z8  (alpha predictive, best r for alpha_resid)
    ↔ No direct F1 analog — alpha is already factor-stripped
    Closest: F1 manifold stability metric (predicts CI width)

  Finance manifold stability → alpha CI width (r=0.619)
    ↔ F1 manifold stability → pace CI width (r=0.771)
    DIRECT REPLICATION: same property, different domain
""")

# Quantify the analog more formally:
# For each finance axis, compute its correlation with financial analogs of F1 features
# volatility = pace consistency proxy
# pos_252d = competitive position proxy
# mom_accel = momentum/resource proxy

print("  Formal analog correlations (finance axes vs F1-analog features):")
analog_map = {
    'vol_21d':   ('Pace consistency proxy', 'z5'),
    'pos_252d':  ('Competitive position proxy', 'z7'),
    'mom_accel': ('Resource/momentum proxy', 'z0'),
    'drawdown':  ('Performance trajectory', 'z7'),
}

print(f"\n  {'Feature':<20} {'F1 analog':<30} {'Predicted axis':<15} {'Actual top axis'}")
print(f"  {'─'*80}")
for feat, (analog, predicted_axis) in analog_map.items():
    if feat not in feature_cols:
        continue
    idx = feature_cols.index(feat)
    feat_vals = X_all[:, idx]   # X_all = temporal pre-2023
    valid = ~np.isnan(feat_vals)
    axis_corrs = [abs(pearsonr(Z_temp[valid, i], feat_vals[valid])[0])
                  for i in range(latent_dim)]
    top_axis = f"z{np.argmax(axis_corrs)}"
    top_r    = max(axis_corrs)
    match    = '✓' if top_axis == predicted_axis else '✗'
    print(f"  {feat:<20} {analog:<30} {predicted_axis:<15} {top_axis} (r={top_r:.3f}) {match}")

# Domain comparison table
print(f"""
  Summary comparison table:

  Property                      F1 (2018-2021)      Finance (2000-2024)
  ─────────────────────────────────────────────────────────────────────
  Manifold stability → CI width  r=0.771 p<0.0001    r=0.619 p<0.0001
  Performance prediction (R²)    r=0.808 MC pace      R²=0.055 alpha
  Active latent dims             12/12 (16D model)    12/12
  Unsupervised structure         teammate pairs       regime recovery
  Sample efficiency ratio        1.9× at n=10         4.7× at n=10
  Constraint profiles            N/A (new finding)    8/12 diagnostic
""")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EXTENDED ANALYSIS COMPLETE")
print("=" * 80)

learnable_relations = [n for n, r in profiles.items() if r.get('learnable', False)]
print(f"\nSection 1 — Multi-relation constraint profiles:")
print(f"  Learnable relations: {len(learnable_relations)}/{len(profiles)} "
      f"({', '.join(learnable_relations)})")
if 'mean_off_diag' in dir():
    print(f"  Cross-relation orthogonality: {mean_off_diag:.3f}")

print(f"\nSection 2 — Policy distance matrix:")
print(f"  Stocks analyzed: {sample_n}")
print(f"  Hierarchical clusters: 8")

print(f"\nSection 3 — Sample efficiency (new listing analog):")
if stable_errs and unstable_errs:
    ratio_10 = unstable_errs[1] / stable_errs[1] if stable_errs[1] > 0 else np.nan
    print(f"  Stable vs unstable ratio at n=10: {ratio_10:.2f}x")

print(f"\nSection 4 — Temporal constraint stability:")
if era_profiles:
    print(f"  Eras analyzed: {len(era_profiles)}")
    if 'mean_sim' in dir():
        print(f"  Mean cross-era similarity: {mean_sim:.3f}")

print(f"\nSection 5 — Manifold dispersion:")
print(f"  Daily dispersion time series: {len(daily_df)} days")
print(f"  Saved: manifold_dispersion_daily.parquet")

print(f"\nAll plots: {PLOT_DIR}")

# Save summary
summary = {
    'constraint_profiles': {
        n: {
            'n_dates':      r['n_dates'],
            'n_diagnostic': r['n_diagnostic'],
            'learnable':    r['learnable'],
            'desc':         r.get('desc', ''),
        }
        for n, r in profiles.items()
    },
    'era_profiles': {
        n: {
            'n_dates':      r['n_dates'],
            'n_diagnostic': r['n_diagnostic'],
        }
        for n, r in era_profiles.items()
    } if era_profiles else {},
}

with open(DATA_DIR / 'ae_extended_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults: ae_extended_results.json")
print("Done.")