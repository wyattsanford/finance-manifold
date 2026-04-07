# -*- coding: utf-8 -*-
"""
stock_ae_eval.py
Evaluate trained autoencoder — linear probes, latent geometry,
manifold stability, constraint profiles, and dimensional analysis.
Run after all 5 CV folds + stock_ae_temporal.py finish.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR  = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
PLOT_DIR  = DATA_DIR / 'plots'
PLOT_DIR.mkdir(exist_ok=True)
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PCA_FORWARD_BASELINE = 0.0039   # from pca_baseline_forward.json — the real bar

EXCLUDE_FROM_AE = {
    'ticker', 'open', 'high', 'low', 'close', 'volume',
    'Mkt_RF', 'SMB', 'HML', 'RF',
    'high_252d', 'low_252d', 'high_63d', 'low_63d',
    'sma_21', 'sma_63', 'sma_252', 'vol_ma21',
    'up_vol_ratio', 'kurt_63d',
    'alpha_resid', 'ret_1d', 'excess_ret',
    'beta_mkt_rf', 'beta_smb', 'beta_hml',
}

# ── Model — must match training architecture exactly ──────────────────────────
# Trained as input→32→16→12 (not 64/32 as in the wider variant)
class StockAE(nn.Module):
    def __init__(self, input_dim, latent_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 16),        nn.LayerNorm(16), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.LayerNorm(16), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(16, 32),         nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, input_dim),
        )

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class StockDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


def encode_dataset(model, loader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for X_batch in loader:
            _, z = model(X_batch.to(device))
            latents.append(z.cpu().numpy())
    return np.vstack(latents)


def load_stock_split(tickers, feature_cols, date_filter=None):
    """
    Load clean stock data. date_filter = ('before'|'after', 'YYYY-MM-DD') or None.
    Returns X, y_ret, y_alpha, dates, ticker_labels.
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
                cutoff_ts = pd.Timestamp(cutoff)
                mask = (dates < cutoff_ts) if direction == 'before' else (dates >= cutoff_ts)
                df    = df[mask]
                dates = dates[mask]

            if len(df) < 63:
                continue

            X = df[feature_cols].values.astype(np.float32)
            X = np.where(np.isinf(X), np.nan, X)
            col_meds = np.nanmedian(X, axis=0)
            for j in range(X.shape[1]):
                m = np.isnan(X[:, j])
                X[m, j] = col_meds[j]
            X = np.nan_to_num(X, nan=0.0)

            y_ret   = df['ret_1d'].values    if 'ret_1d'    in df.columns else np.zeros(len(df))
            y_alpha = df['alpha_resid'].values if 'alpha_resid' in df.columns else np.zeros(len(df))

            valid = ~np.isnan(y_ret) & ~np.isnan(y_alpha)
            Xs.append(X[valid])
            y_rets.append(y_ret[valid])
            y_alphas.append(y_alpha[valid])
            all_dates.append(dates[valid])
            all_tickers.extend([ticker] * valid.sum())
        except Exception as e:
            print(f"  Skip {ticker}: {e}")

    return (np.vstack(Xs),
            np.concatenate(y_rets),
            np.concatenate(y_alphas),
            np.concatenate(all_dates),
            np.array(all_tickers))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — 5-Fold CV
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("SECTION 1: 5-FOLD CROSS-VALIDATION")
print("=" * 80)

with open(DATA_DIR / 'ae_training_results.json') as f:
    training_results = json.load(f)

cv_results   = training_results['cv_results']
input_dim    = training_results['architecture']['input_dim']
latent_dim   = training_results['architecture']['latent_dim']
eval_results = []

for fold_info in cv_results:
    fold = fold_info['fold']
    print(f"\n{'─'*60}")
    print(f"FOLD {fold}/5")
    print(f"{'─'*60}")

    ckpt = torch.load(DATA_DIR / f'ae_fold{fold}_best.pt', map_location=DEVICE, weights_only=False)
    model = StockAE(input_dim=input_dim, latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    scaler       = ckpt['scaler']
    feature_cols = ckpt['feature_cols']

    val_tickers   = fold_info['val_tickers']
    train_tickers = fold_info['train_tickers']

    X_tr, y_ret_tr, y_alpha_tr, _, _ = load_stock_split(train_tickers, feature_cols)
    X_vl, y_ret_vl, y_alpha_vl, _, _ = load_stock_split(val_tickers,   feature_cols)

    X_tr_s = scaler.transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    tr_loader = DataLoader(StockDataset(X_tr_s), batch_size=4096, shuffle=False)
    vl_loader = DataLoader(StockDataset(X_vl_s), batch_size=4096, shuffle=False)

    z_tr = encode_dataset(model, tr_loader, DEVICE)
    z_vl = encode_dataset(model, vl_loader, DEVICE)

    # ── Linear probes ─────────────────────────────────────────────────────────
    results = {}
    for label, y_tr, y_vl in [('ret',   y_ret_tr,   y_ret_vl),
                                ('alpha', y_alpha_tr, y_alpha_vl)]:
        ridge = Ridge(alpha=1.0)
        ridge.fit(z_tr, y_tr)
        pred       = ridge.predict(z_vl)
        results[label] = {
            'r2':   float(r2_score(y_vl, pred)),
            'mse':  float(mean_squared_error(y_vl, pred)),
            'corr': float(pearsonr(y_vl, pred)[0]),
        }
        print(f"  {label.upper():5s} → R²={results[label]['r2']:.4f}  "
              f"Corr={results[label]['corr']:.4f}")

    # ── Latent variance check ─────────────────────────────────────────────────
    dim_vars    = z_vl.var(axis=0)
    active_dims = int((dim_vars > dim_vars.mean() * 0.1).sum())
    print(f"  Active latent dims: {active_dims}/{latent_dim}")

    eval_results.append({
        'fold':         fold,
        'ret_r2':       results['ret']['r2'],
        'ret_corr':     results['ret']['corr'],
        'alpha_r2':     results['alpha']['r2'],
        'alpha_corr':   results['alpha']['corr'],
        'active_dims':  active_dims,
        'dim_variances': dim_vars.tolist(),
    })

mean_alpha_r2 = float(np.mean([r['alpha_r2'] for r in eval_results]))
std_alpha_r2  = float(np.std([r['alpha_r2']  for r in eval_results]))
mean_ret_r2   = float(np.mean([r['ret_r2']   for r in eval_results]))

print(f"\n{'─'*60}")
print(f"CV SUMMARY")
print(f"  Mean Ret R²:   {mean_ret_r2:.4f}")
print(f"  Mean Alpha R²: {mean_alpha_r2:.4f} ± {std_alpha_r2:.4f}")
print(f"  PCA baseline:  {PCA_FORWARD_BASELINE:.4f}")
if mean_alpha_r2 > PCA_FORWARD_BASELINE:
    print(f"  → AE WINS by {mean_alpha_r2 - PCA_FORWARD_BASELINE:.4f}")
else:
    print(f"  → PCA wins by {PCA_FORWARD_BASELINE - mean_alpha_r2:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Temporal holdout (2023-2024)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 2: TEMPORAL HOLDOUT (train <2023, val 2023-2024)")
print("=" * 80)

ckpt_temp    = torch.load(DATA_DIR / 'ae_temporal_best.pt', map_location=DEVICE, weights_only=False)

# Infer architecture directly from checkpoint weights — handles any hidden dim
_s = ckpt_temp['model_state']
_h1  = _s['encoder.0.weight'].shape[0]
_h2  = _s['encoder.4.weight'].shape[0]
_inp = _s['encoder.0.weight'].shape[1]
_lat = _s['encoder.8.weight'].shape[0]

class _TempAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(_inp, _h1), nn.LayerNorm(_h1), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(_h1, _h2),  nn.LayerNorm(_h2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(_h2, _lat),
        )
        self.decoder = nn.Sequential(
            nn.Linear(_lat, _h2), nn.LayerNorm(_h2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(_h2, _h1),  nn.LayerNorm(_h1), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(_h1, _inp),
        )
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

model_temp = _TempAE().to(DEVICE)
model_temp.load_state_dict(ckpt_temp['model_state'])
model_temp.eval()
scaler_temp       = ckpt_temp['scaler']
feature_cols_temp = ckpt_temp['feature_cols']

manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

X_pre,  y_ret_pre,  y_alpha_pre,  dates_pre,  _ = load_stock_split(
    all_tickers, feature_cols_temp, date_filter=('before', '2023-01-01'))
X_post, y_ret_post, y_alpha_post, dates_post, tickers_post = load_stock_split(
    all_tickers, feature_cols_temp, date_filter=('after',  '2023-01-01'))

print(f"Pre-2023:  {len(X_pre):,} obs")
print(f"Post-2023: {len(X_post):,} obs")

X_pre_s  = scaler_temp.transform(X_pre)
X_post_s = scaler_temp.transform(X_post)

pre_loader  = DataLoader(StockDataset(X_pre_s),  batch_size=4096, shuffle=False)
post_loader = DataLoader(StockDataset(X_post_s), batch_size=4096, shuffle=False)

z_pre  = encode_dataset(model_temp, pre_loader,  DEVICE)
z_post = encode_dataset(model_temp, post_loader, DEVICE)

temporal_results = {}
for label, y_tr, y_vl in [('ret',   y_ret_pre,   y_ret_post),
                            ('alpha', y_alpha_pre, y_alpha_post)]:
    ridge = Ridge(alpha=1.0)
    ridge.fit(z_pre, y_tr)
    pred = ridge.predict(z_post)
    temporal_results[label] = {
        'r2':   float(r2_score(y_vl, pred)),
        'corr': float(pearsonr(y_vl, pred)[0]),
    }
    print(f"  {label.upper():5s} → R²={temporal_results[label]['r2']:.4f}  "
          f"Corr={temporal_results[label]['corr']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Manifold Stability (the Atlas parallel)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 3: MANIFOLD STABILITY")
print("=" * 80)
print("Do stable stocks (low latent variance) have tighter alpha CIs?")
print("Analog: stable F1 drivers → narrower Monte Carlo performance intervals")

# Use temporal model, pre-2023 data, compute per-stock latent stats
X_pre_s_full = scaler_temp.transform(X_pre)
pre_loader_full = DataLoader(StockDataset(X_pre_s_full), batch_size=4096, shuffle=False)
z_pre_full = encode_dataset(model_temp, pre_loader_full, DEVICE)

# Rebuild ticker labels for pre-2023
_, _, _, _, tickers_pre = load_stock_split(
    all_tickers, feature_cols_temp, date_filter=('before', '2023-01-01'))

stock_stability = {}
for ticker in np.unique(tickers_pre):
    mask = tickers_pre == ticker
    if mask.sum() < 63:
        continue
    z_stock = z_pre_full[mask]
    # Manifold stability = mean per-dim variance (matches Atlas definition)
    stability = float(z_stock.var(axis=0).mean())
    centroid  = z_stock.mean(axis=0)
    stock_stability[ticker] = {
        'stability':  stability,
        'centroid':   centroid.tolist(),
        'n_obs':      int(mask.sum()),
    }

stabilities = np.array([v['stability'] for v in stock_stability.values()])
tickers_arr = np.array(list(stock_stability.keys()))

# Now compute per-stock forward alpha stats on post-2023
stock_alpha_stats = {}
for ticker in np.unique(tickers_post):
    if ticker not in stock_stability:
        continue
    mask = tickers_post == ticker
    if mask.sum() < 21:
        continue
    alphas = y_alpha_post[mask]
    stock_alpha_stats[ticker] = {
        'mean_alpha': float(np.nanmean(alphas)),
        'std_alpha':  float(np.nanstd(alphas)),
        'n_obs':      int(mask.sum()),
    }

# Merge stability + alpha stats
common = [t for t in stock_stability if t in stock_alpha_stats]
stab_vals  = np.array([stock_stability[t]['stability']       for t in common])
alpha_mean = np.array([stock_alpha_stats[t]['mean_alpha']    for t in common])
alpha_std  = np.array([stock_alpha_stats[t]['std_alpha']     for t in common])

r_stab_ci, p_stab_ci   = pearsonr(stab_vals, alpha_std)
r_stab_perf, p_stab_perf = pearsonr(stab_vals, alpha_mean)
r_sp_stab_ci, _         = spearmanr(stab_vals, alpha_std)

print(f"\n  Stocks with stability + forward alpha: {len(common)}")
print(f"  Stability → alpha CI width:  r={r_stab_ci:.3f}  p={p_stab_ci:.4f}  "
      f"(Spearman r={r_sp_stab_ci:.3f})")
print(f"  Stability → alpha mean:      r={r_stab_perf:.3f}  p={p_stab_perf:.4f}")
print(f"  (Atlas F1 result: stability→CI width r=0.771, p<0.0001)")

# Stability quintiles
quintile_labels = pd.qcut(stab_vals, 5, labels=['Q1\n(most stable)',
                                                  'Q2','Q3','Q4',
                                                  'Q5\n(most unstable)'])
print(f"\n  Alpha CI width by stability quintile:")
for q in quintile_labels.categories:
    mask_q = quintile_labels == q
    print(f"    {q.replace(chr(10),' '):<20} "
          f"mean CI={alpha_std[mask_q].mean():.5f}  "
          f"n={mask_q.sum()}")

# Plot: stability vs alpha CI width
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(stab_vals, alpha_std, alpha=0.4, s=15)
axes[0].set_xlabel('Manifold Stability (latent variance)')
axes[0].set_ylabel('Alpha CI Width (std)')
axes[0].set_title(f'Stability → Alpha CI Width\nr={r_stab_ci:.3f}, p={p_stab_ci:.4f}')

axes[1].scatter(stab_vals, alpha_mean, alpha=0.4, s=15, color='orange')
axes[1].set_xlabel('Manifold Stability (latent variance)')
axes[1].set_ylabel('Mean Forward Alpha')
axes[1].set_title(f'Stability → Alpha Performance\nr={r_stab_perf:.3f}, p={p_stab_perf:.4f}')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'stability_analysis.png', dpi=150)
plt.close()
print(f"  Plot saved: stability_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Latent Axis Structure
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 4: LATENT AXIS STRUCTURE")
print("=" * 80)
print("Do the 12 axes cluster into interpretable groups?")
print("Analog: Atlas 11 axes → Social-Power / Action / Contextual / Grounding")

# Use fold 1 model for consistency
ckpt1        = torch.load(DATA_DIR / 'ae_fold1_best.pt', map_location=DEVICE, weights_only=False)
model1       = StockAE(input_dim=input_dim, latent_dim=latent_dim).to(DEVICE)
model1.load_state_dict(ckpt1['model_state'])
model1.eval()
scaler1      = ckpt1['scaler']
feature_cols1= ckpt1['feature_cols']

val_tickers1 = cv_results[0]['val_tickers']
X_vl1, y_ret_vl1, y_alpha_vl1, dates_vl1, tickers_vl1 = load_stock_split(
    val_tickers1, feature_cols1)

X_vl1_s  = scaler1.transform(X_vl1)
vl1_loader = DataLoader(StockDataset(X_vl1_s), batch_size=4096, shuffle=False)
Z1 = encode_dataset(model1, vl1_loader, DEVICE)

# Inter-axis correlation matrix
corr_matrix = np.corrcoef(Z1.T)
print(f"\n  Inter-axis correlation matrix ({latent_dim}×{latent_dim}):")
print(f"  Mean |off-diagonal| corr: {np.abs(corr_matrix - np.eye(latent_dim)).mean():.3f}")
print(f"  Max  |off-diagonal| corr: {np.abs(corr_matrix - np.eye(latent_dim)).max():.3f}")

# Eigenvalue spectrum — how non-random is the structure?
eigenvalues = np.linalg.eigvalsh(corr_matrix)[::-1]
print(f"\n  Eigenvalue spectrum (top 5): "
      f"{', '.join(f'{e:.3f}' for e in eigenvalues[:5])}")
print(f"  Spectral gap (λ1-λ2): {eigenvalues[0]-eigenvalues[1]:.3f}")
print(f"  Variance in top 3 dims: {eigenvalues[:3].sum()/eigenvalues.sum()*100:.1f}%")

# Hierarchical clustering on axes
dist_matrix = 1 - np.abs(corr_matrix)
dist_matrix = (dist_matrix + dist_matrix.T) / 2   # force exact symmetry
np.fill_diagonal(dist_matrix, 0)
linkage_matrix = linkage(squareform(dist_matrix), method='ward')
clusters = fcluster(linkage_matrix, t=4, criterion='maxclust')
print(f"\n  Axis clusters (Ward, k=4):")
for c in range(1, 5):
    dims = [f"z{i}" for i, cl in enumerate(clusters) if cl == c]
    print(f"    Cluster {c}: {', '.join(dims)}")

# Dimensional variance
dim_vars = Z1.var(axis=0)
active   = int((dim_vars > dim_vars.mean() * 0.1).sum())
print(f"\n  Dimensional variance:")
for i, v in enumerate(dim_vars):
    bar = '█' * int(v / dim_vars.max() * 25)
    print(f"    z{i:<2} {v:.4f}  {bar}")
print(f"  Active dims (>10% mean): {active}/{latent_dim}")

# Plot correlation matrix
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1,
            xticklabels=[f'z{i}' for i in range(latent_dim)],
            yticklabels=[f'z{i}' for i in range(latent_dim)],
            ax=ax)
ax.set_title('Latent Axis Correlation Matrix')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'axis_correlation.png', dpi=150)
plt.close()
print(f"  Plot saved: axis_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Linear Probes: What Do the Axes Encode?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 5: LINEAR PROBES — AXIS INTERPRETABILITY")
print("=" * 80)
print("Which axes predict known financial quantities?")

probe_targets = {}

# Volatility (21d)
if 'vol_21d' in feature_cols1:
    idx = feature_cols1.index('vol_21d')
    probe_targets['vol_21d'] = X_vl1[:, idx]

# Momentum (63d return)
if 'ret_63d' in feature_cols1:
    idx = feature_cols1.index('ret_63d')
    probe_targets['ret_63d'] = X_vl1[:, idx]

# Price position (52w)
if 'pos_252d' in feature_cols1:
    idx = feature_cols1.index('pos_252d')
    probe_targets['pos_252d'] = X_vl1[:, idx]

# Momentum acceleration
if 'mom_accel' in feature_cols1:
    idx = feature_cols1.index('mom_accel')
    probe_targets['mom_accel'] = X_vl1[:, idx]

# Drawdown
if 'drawdown' in feature_cols1:
    idx = feature_cols1.index('drawdown')
    probe_targets['drawdown'] = X_vl1[:, idx]

# Volume ratio
if 'vol_ratio_v' in feature_cols1:
    idx = feature_cols1.index('vol_ratio_v')
    probe_targets['vol_ratio_v'] = X_vl1[:, idx]

# Alpha (forward)
probe_targets['alpha_resid'] = y_alpha_vl1
probe_targets['ret_1d']      = y_ret_vl1

print(f"\n  {'Target':<18} {'Best axis':>9} {'Best r':>8} {'Ridge R²':>9} "
      f"{'Top-3 axes'}")
print(f"  {'─'*70}")

probe_summary = {}
for name, y_target in probe_targets.items():
    valid = ~np.isnan(y_target)
    if valid.sum() < 1000:
        continue
    y = y_target[valid]
    Z = Z1[valid]

    # Per-axis correlation
    axis_corrs = [pearsonr(Z[:, i], y)[0] for i in range(latent_dim)]
    best_axis  = int(np.argmax(np.abs(axis_corrs)))
    best_r     = float(axis_corrs[best_axis])
    top3       = np.argsort(np.abs(axis_corrs))[::-1][:3]

    # Ridge from all axes
    ridge = Ridge(alpha=1.0)
    ridge.fit(Z, y)
    r2 = float(r2_score(y, ridge.predict(Z)))

    top3_str = ', '.join(f'z{i}(r={axis_corrs[i]:.2f})' for i in top3)
    print(f"  {name:<18} {'z'+str(best_axis):>9} {best_r:>8.3f} {r2:>9.4f}  {top3_str}")

    probe_summary[name] = {
        'best_axis': best_axis,
        'best_r':    best_r,
        'ridge_r2':  r2,
        'axis_corrs': [float(c) for c in axis_corrs],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Constraint Profile Analysis (Gärdenfors analog)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6: CONSTRAINT PROFILES")
print("=" * 80)
print("Do alpha quintile transitions show consistent axis-level sign patterns?")
print("Analog: 'causes' consistently decreases Agency+Intentionality in Atlas")

# Build daily cross-section: group by date, compute alpha quintile per day
dates_arr = np.array(dates_vl1)
unique_dates = np.unique(dates_arr)

# For each observation compute its cross-sectional alpha quintile that day
alpha_quintile = np.full(len(y_alpha_vl1), np.nan)
for d in unique_dates:
    mask_d = dates_arr == d
    if mask_d.sum() < 10:
        continue
    alphas_d = y_alpha_vl1[mask_d]
    valid_d  = ~np.isnan(alphas_d)
    if valid_d.sum() < 10:
        continue
    ranks = pd.Series(alphas_d).rank(pct=True)
    quintiles = pd.cut(ranks, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       labels=[1, 2, 3, 4, 5])
    alpha_quintile[mask_d] = quintiles.values.astype(float)

# Constraint profile: Q1→Q5 transitions (top alpha vs bottom alpha)
# For each pair of observations on the same date, compute delta Z
# Sign consistency per axis = fraction of pairs where delta > 0
print(f"\n  Computing cross-sectional constraint profiles per date...")

q1_mask = alpha_quintile == 1   # bottom alpha
q5_mask = alpha_quintile == 5   # top alpha

n_dates_used = 0
axis_deltas  = []   # list of (z_q5_mean - z_q1_mean) per date

for d in unique_dates:
    mask_d  = dates_arr == d
    q1_d    = mask_d & q1_mask
    q5_d    = mask_d & q5_mask
    if q1_d.sum() < 3 or q5_d.sum() < 3:
        continue
    delta = Z1[q5_d].mean(axis=0) - Z1[q1_d].mean(axis=0)
    axis_deltas.append(delta)
    n_dates_used += 1

if len(axis_deltas) > 0:
    axis_deltas = np.array(axis_deltas)  # (n_dates, latent_dim)

    # Sign consistency per axis
    sign_consistency = (axis_deltas > 0).mean(axis=0)
    effect_sizes     = np.abs(axis_deltas.mean(axis=0)) / (axis_deltas.std(axis=0) + 1e-8)

    # Diagnostic: axis is diagnostic if sign_consistency >= 0.65 AND effect >= 0.15
    diagnostic = (np.abs(sign_consistency - 0.5) >= 0.15) & (effect_sizes >= 0.15)

    print(f"\n  Dates used: {n_dates_used}")
    print(f"\n  {'Axis':<6} {'Sign Cons':>10} {'Effect':>8} {'Diagnostic':>11}  Direction")
    print(f"  {'─'*55}")
    for i in range(latent_dim):
        direction = '↑ (Q5>Q1)' if sign_consistency[i] > 0.5 else '↓ (Q5<Q1)'
        diag_str  = '  ✓ DIAGNOSTIC' if diagnostic[i] else ''
        print(f"  z{i:<4} {sign_consistency[i]:>10.3f} {effect_sizes[i]:>8.3f} "
              f"{diag_str:<13} {direction}")

    n_diagnostic = diagnostic.sum()
    print(f"\n  Diagnostic axes for Q5-vs-Q1 alpha: {n_diagnostic}/{latent_dim}")
    if n_diagnostic >= 3:
        print(f"  → LEARNABLE constraint profile (≥3 diagnostic axes)")
        print(f"  → Consistent with Gärdenfors constraint hypothesis")
    else:
        print(f"  → Weak constraint structure at this scale")

    # Plot constraint profile
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#2ecc71' if s > 0.65 else '#e74c3c' if s < 0.35 else '#95a5a6'
              for s in sign_consistency]
    bars = ax.bar(range(latent_dim), sign_consistency - 0.5, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(0.15,  color='green', linewidth=0.8, linestyle='--', alpha=0.5,
               label='Diagnostic threshold (±0.15)')
    ax.axhline(-0.15, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(range(latent_dim))
    ax.set_xticklabels([f'z{i}' for i in range(latent_dim)])
    ax.set_ylabel('Sign Consistency − 0.5')
    ax.set_title('Alpha Q5 vs Q1 Constraint Profile\n'
                 '(green = Q5 consistently higher, red = Q5 consistently lower)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'constraint_profile_alpha.png', dpi=150)
    plt.close()
    print(f"  Plot saved: constraint_profile_alpha.png")

    constraint_results = {
        'n_dates':           n_dates_used,
        'sign_consistency':  sign_consistency.tolist(),
        'effect_sizes':      effect_sizes.tolist(),
        'diagnostic_axes':   np.where(diagnostic)[0].tolist(),
        'n_diagnostic':      int(n_diagnostic),
    }
else:
    print("  Not enough cross-sectional data for constraint analysis")
    constraint_results = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Regime Recovery Without Supervision
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 7: UNSUPERVISED REGIME RECOVERY")
print("=" * 80)
print("Does the manifold separate 2008/2020/2022 without being told?")
print("Analog: teammate pairs clustering without team labels in F1")

# Use temporal model, full pre-2023 dataset, label known crisis periods
crisis_labels = {
    'Normal':       (None, None),
    'GFC_2008':     ('2008-09-01', '2009-03-31'),
    'COVID_2020':   ('2020-02-01', '2020-06-30'),
    'Rate_2022':    ('2022-01-01', '2022-12-31'),
}

# Build latent centroids per period
period_centroids = {}
for period_name, (start, end) in crisis_labels.items():
    if start is None:
        mask = np.ones(len(dates_pre), dtype=bool)
        # exclude crisis periods from normal
        for _, (cs, ce) in list(crisis_labels.items())[1:]:
            cs_ts, ce_ts = pd.Timestamp(cs), pd.Timestamp(ce)
            mask &= ~((dates_pre >= cs_ts) & (dates_pre <= ce_ts))
    else:
        cs_ts, ce_ts = pd.Timestamp(start), pd.Timestamp(end)
        mask = (dates_pre >= cs_ts) & (dates_pre <= ce_ts)

    if mask.sum() < 100:
        print(f"  {period_name}: insufficient data ({mask.sum()} obs)")
        continue

    centroid = z_pre_full[mask].mean(axis=0)
    period_centroids[period_name] = centroid
    print(f"  {period_name:<15} n={mask.sum():>7,}  "
          f"||centroid||={np.linalg.norm(centroid):.3f}")

# Pairwise distances between period centroids
if len(period_centroids) >= 3:
    periods = list(period_centroids.keys())
    print(f"\n  Pairwise centroid distances (Euclidean):")
    for i, p1 in enumerate(periods):
        for j, p2 in enumerate(periods):
            if j <= i:
                continue
            d = np.linalg.norm(period_centroids[p1] - period_centroids[p2])
            print(f"    {p1} ↔ {p2}: {d:.3f}")

    # PCA visualization of period centroids
    all_z_sample = z_pre_full[::10]   # every 10th for speed
    pca = PCA(n_components=2)
    pca.fit(all_z_sample)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors_map = {'Normal': 'steelblue', 'GFC_2008': 'crimson',
                  'COVID_2020': 'darkorange', 'Rate_2022': 'purple'}

    for period_name, (start, end) in crisis_labels.items():
        if period_name not in period_centroids:
            continue
        if start is None:
            mask = np.ones(len(dates_pre), dtype=bool)
            for _, (cs, ce) in list(crisis_labels.items())[1:]:
                mask &= ~((dates_pre >= pd.Timestamp(cs)) &
                           (dates_pre <= pd.Timestamp(ce)))
        else:
            mask = ((dates_pre >= pd.Timestamp(start)) &
                    (dates_pre <= pd.Timestamp(end)))
        sample = z_pre_full[mask][::20]   # subsample for plot
        proj   = pca.transform(sample)
        ax.scatter(proj[:, 0], proj[:, 1], alpha=0.15, s=4,
                   color=colors_map[period_name], label=period_name)

    # Plot centroids
    for period_name, centroid in period_centroids.items():
        proj = pca.transform(centroid.reshape(1, -1))
        ax.scatter(proj[0, 0], proj[0, 1], s=200, marker='*',
                   color=colors_map[period_name], edgecolors='black', zorder=5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ax.set_title('Latent Space — Market Regime Recovery (unsupervised)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'regime_recovery.png', dpi=150)
    plt.close()
    print(f"  Plot saved: regime_recovery.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Sample Efficiency by Stability
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 8: SAMPLE EFFICIENCY BY STABILITY")
print("=" * 80)
print("Stable stocks: centroid estimable from fewer observations?")
print("Analog: Kubica characterizable at n=10 laps vs Ricciardo needing 30+")

if len(stock_stability) > 20:
    stab_arr   = np.array([stock_stability[t]['stability'] for t in common])
    stable_q   = np.percentile(stab_arr, 25)
    unstable_q = np.percentile(stab_arr, 75)

    stable_tickers   = [t for t in common if stock_stability[t]['stability'] <= stable_q]
    unstable_tickers = [t for t in common if stock_stability[t]['stability'] >= unstable_q]

    sample_sizes = [10, 25, 50, 100, 200, 500]
    stable_errors, unstable_errors = [], []

    rng = np.random.RandomState(42)

    for n in sample_sizes:
        s_errs, u_errs = [], []
        for ticker in stable_tickers[:30]:
            mask = tickers_pre == ticker
            z_t  = z_pre_full[mask]
            if len(z_t) < n * 2:
                continue
            true_centroid = z_t.mean(axis=0)
            # Bootstrap centroid error at n samples
            errors = []
            for _ in range(20):
                idx = rng.choice(len(z_t), n, replace=False)
                err = np.linalg.norm(z_t[idx].mean(axis=0) - true_centroid)
                errors.append(err)
            s_errs.append(np.mean(errors))

        for ticker in unstable_tickers[:30]:
            mask = tickers_pre == ticker
            z_t  = z_pre_full[mask]
            if len(z_t) < n * 2:
                continue
            true_centroid = z_t.mean(axis=0)
            errors = []
            for _ in range(20):
                idx = rng.choice(len(z_t), n, replace=False)
                err = np.linalg.norm(z_t[idx].mean(axis=0) - true_centroid)
                errors.append(err)
            u_errs.append(np.mean(errors))

        stable_errors.append(np.mean(s_errs)   if s_errs   else np.nan)
        unstable_errors.append(np.mean(u_errs) if u_errs   else np.nan)

    print(f"\n  {'n obs':<8} {'Stable err':>12} {'Unstable err':>14} {'Ratio':>8}")
    print(f"  {'─'*46}")
    for n, se, ue in zip(sample_sizes, stable_errors, unstable_errors):
        ratio = ue / se if se > 0 else np.nan
        print(f"  {n:<8} {se:>12.4f} {ue:>14.4f} {ratio:>8.2f}x")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sample_sizes, stable_errors,   'o-', label='Stable stocks (Q1)',   color='steelblue')
    ax.plot(sample_sizes, unstable_errors, 's--', label='Unstable stocks (Q4)', color='crimson')
    ax.set_xlabel('Number of observations')
    ax.set_ylabel('Centroid estimation error')
    ax.set_title('Sample Efficiency by Manifold Stability\n'
                 '(Analog: F1 driver characterization from n laps)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'sample_efficiency.png', dpi=150)
    plt.close()
    print(f"  Plot saved: sample_efficiency.png")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\n5-Fold CV:")
for r in eval_results:
    print(f"  Fold {r['fold']}: Alpha R²={r['alpha_r2']:.4f}  "
          f"Active dims={r['active_dims']}/{latent_dim}")
print(f"  Mean Alpha R²: {mean_alpha_r2:.4f} ± {std_alpha_r2:.4f}")
print(f"  PCA baseline:  {PCA_FORWARD_BASELINE:.4f}")
beat = mean_alpha_r2 > PCA_FORWARD_BASELINE
print(f"  AE {'BEATS' if beat else 'LOSES TO'} PCA baseline")

print(f"\nTemporal holdout (2023-2024):")
print(f"  Alpha R²={temporal_results['alpha']['r2']:.4f}  "
      f"Corr={temporal_results['alpha']['corr']:.4f}")

print(f"\nManifold stability → alpha CI width: "
      f"r={r_stab_ci:.3f}  p={p_stab_ci:.4f}")
print(f"Diagnostic axes (constraint profile): "
      f"{constraint_results.get('n_diagnostic', 'N/A')}/{latent_dim}")
print(f"Active latent dims: {active}/{latent_dim}")

summary = {
    'cv_results':          eval_results,
    'cv_summary': {
        'mean_ret_r2':     float(mean_ret_r2),
        'mean_alpha_r2':   float(mean_alpha_r2),
        'std_alpha_r2':    float(std_alpha_r2),
        'pca_baseline':    PCA_FORWARD_BASELINE,
        'ae_beats_pca':    bool(beat),
    },
    'temporal_holdout':    temporal_results,
    'stability': {
        'r_stability_ci':   float(r_stab_ci),
        'p_stability_ci':   float(p_stab_ci),
        'r_stability_perf': float(r_stab_perf),
        'n_stocks':         len(common),
    },
    'axis_structure': {
        'active_dims':      int(active),
        'eigenvalues':      eigenvalues.tolist(),
        'probe_summary':    probe_summary,
    },
    'constraint_profiles':  constraint_results,
}

with open(DATA_DIR / 'ae_eval_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nFull results: ae_eval_results.json")
print(f"Plots:        {PLOT_DIR}")
print("\nDone.")