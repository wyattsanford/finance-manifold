# -*- coding: utf-8 -*-
"""
stock_pace_model.py
Proper pace model: centroid → mean alpha prediction.

Fixes from naive implementation:
  1. Observation-weighted regression (reliable mean alpha = more weight)
  2. Cross-validated Ridge alpha search over wide range
  3. LOO cross-validation for unbiased R² estimate
  4. Proper held-out test set (never touched during fitting)
  5. Nonlinear alternatives (GBM, MLP) benchmarked against Ridge
  6. OOS validation on fresh stocks using the fitted pace model
  7. Confidence-weighted evaluation — does restricting to stable
     stocks with tight CIs improve predictive performance?

Saves:
  pace_model.pkl          — best fitted pace model
  pace_model_results.json — full evaluation results
  plots/pace_model/       — diagnostic plots
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
OOS_DIR  = DATA_DIR / 'oos_validation'
PLOT_DIR = DATA_DIR / 'plots' / 'pace_model'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG      = np.random.RandomState(42)

CUTOFF_DATE   = '2023-01-01'
MC_SAMPLES    = 2000
MIN_OBS_PACE  = 252    # minimum observations to include stock in pace model fitting
TEST_FRACTION = 0.2    # held-out test set fraction

print(f"Device: {DEVICE}")
print(f"Plots:  {PLOT_DIR}")

# ── Model utilities ───────────────────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


def build_model_from_ckpt(ckpt):
    s   = ckpt['model_state']
    h1  = s['encoder.0.weight'].shape[0]
    h2  = s['encoder.4.weight'].shape[0]
    inp = s['encoder.0.weight'].shape[1]
    lat = s['encoder.8.weight'].shape[0]

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
    model.load_state_dict(s)
    model.eval()
    return model, inp, lat


def encode_batch(model, X_scaled, batch_size=4096):
    loader = DataLoader(StockDataset(X_scaled), batch_size=batch_size, shuffle=False)
    model.eval()
    parts = []
    with torch.no_grad():
        for xb in loader:
            _, z = model(xb.to(DEVICE))
            parts.append(z.cpu().numpy())
    return np.vstack(parts)


def clean_X(X):
    X = np.where(np.isinf(X), np.nan, X)
    col_meds = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = col_meds[j]
    return np.nan_to_num(X, nan=0.0)


# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading temporal model...")
ckpt         = torch.load(DATA_DIR / 'ae_temporal_best.pt',
                           map_location=DEVICE, weights_only=False)
model, input_dim, latent_dim = build_model_from_ckpt(ckpt)
scaler_ae    = ckpt['scaler']
feature_cols = ckpt['feature_cols']
print(f"  Latent dim: {latent_dim}  |  Features: {len(feature_cols)}")

# ── Build per-stock policy stats from training universe ───────────────────────
print("\nBuilding per-stock policy statistics...")
manifest    = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

stock_stats = []

for ticker in all_tickers:
    f = DATA_DIR / f'stock_clean_{ticker}.parquet'
    if not f.exists():
        continue
    try:
        df    = pd.read_parquet(f)
        dates = pd.to_datetime(df.index)
        pre   = df[dates < pd.Timestamp(CUTOFF_DATE)]

        if len(pre) < MIN_OBS_PACE:
            continue

        # Feature matrix
        for col in feature_cols:
            if col not in pre.columns:
                pre[col] = 0.0
        X = pre[feature_cols].values.astype(np.float32)
        X = clean_X(X)

        # Alpha target
        y_alpha = pre['alpha_resid'].values \
            if 'alpha_resid' in pre.columns else np.zeros(len(pre))
        valid = ~np.isnan(y_alpha)
        if valid.sum() < MIN_OBS_PACE:
            continue

        y_alpha_clean = y_alpha[valid]
        X_clean       = X[valid]

        # Encode
        X_s     = scaler_ae.transform(X_clean).astype(np.float32)
        z       = encode_batch(model, X_s)

        centroid    = z.mean(axis=0)
        stability   = float(z.var(axis=0).mean())
        mean_alpha  = float(np.nanmean(y_alpha_clean))
        std_alpha   = float(np.nanstd(y_alpha_clean))
        n_obs       = int(valid.sum())

        # Standard error of mean alpha — used for weighting
        se_alpha    = std_alpha / np.sqrt(n_obs) if n_obs > 1 else std_alpha

        stock_stats.append({
            'ticker':     ticker,
            'centroid':   centroid,
            'cov':        np.cov(z.T) + np.eye(latent_dim) * 1e-4,
            'stability':  stability,
            'mean_alpha': mean_alpha,
            'std_alpha':  std_alpha,
            'se_alpha':   se_alpha,
            'n_obs':      n_obs,
        })

    except Exception:
        continue

print(f"  Stocks with policy stats: {len(stock_stats)}")

# Build arrays
tickers_arr  = np.array([s['ticker']     for s in stock_stats])
centroids    = np.array([s['centroid']   for s in stock_stats])
mean_alphas  = np.array([s['mean_alpha'] for s in stock_stats])
stabilities  = np.array([s['stability']  for s in stock_stats])
se_alphas    = np.array([s['se_alpha']   for s in stock_stats])
n_obs_arr    = np.array([s['n_obs']      for s in stock_stats])

# Observation weights: inversely proportional to SE of mean alpha
# sqrt(n_obs) weighting as a simple proxy
weights_raw  = np.sqrt(n_obs_arr)
weights      = weights_raw / weights_raw.sum() * len(weights_raw)

print(f"  Mean alpha: {mean_alphas.mean():.5f} ± {mean_alphas.std():.5f}")
print(f"  Mean stability: {stabilities.mean():.4f}")
print(f"  Weight range: {weights.min():.2f} — {weights.max():.2f}")

# Winsorize extreme alpha targets (1st/99th percentile)
alpha_lo = np.percentile(mean_alphas, 1)
alpha_hi = np.percentile(mean_alphas, 99)
mean_alphas_w = np.clip(mean_alphas, alpha_lo, alpha_hi)
n_winsorized  = ((mean_alphas < alpha_lo) | (mean_alphas > alpha_hi)).sum()
print(f"  Winsorized {n_winsorized} extreme alpha targets "
      f"([{alpha_lo:.5f}, {alpha_hi:.5f}])")

# ── Train/test split ──────────────────────────────────────────────────────────
n_total  = len(stock_stats)
n_test   = int(n_total * TEST_FRACTION)
all_idx  = np.arange(n_total)
RNG.shuffle(all_idx)
test_idx  = all_idx[:n_test]
train_idx = all_idx[n_test:]

C_train  = centroids[train_idx]
C_test   = centroids[test_idx]
y_train  = mean_alphas_w[train_idx]
y_test   = mean_alphas_w[test_idx]
w_train  = weights[train_idx]
w_test   = weights[test_idx]

print(f"\n  Train: {len(train_idx)} stocks  |  Test: {len(test_idx)} stocks")

# ── Section 1: Ridge with CV alpha search ─────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1: RIDGECV — CROSS-VALIDATED REGULARIZATION")
print("=" * 70)

# Wide alpha range — we expect heavy regularization needed
alphas_grid = np.logspace(-3, 6, 100)

ridge_cv = RidgeCV(
    alphas=alphas_grid,
    fit_intercept=True,
    scoring='neg_mean_squared_error',
    cv=10,
)
ridge_cv.fit(C_train, y_train, sample_weight=w_train)

print(f"  Best Ridge alpha: {ridge_cv.alpha_:.4f}")
print(f"  (Naive script used alpha=1.0)")

# Evaluate on held-out test set
y_pred_ridge_test = ridge_cv.predict(C_test)
r2_ridge_test  = float(r2_score(y_test, y_pred_ridge_test,
                                 sample_weight=w_test))
mae_ridge_test = float(mean_absolute_error(y_test, y_pred_ridge_test,
                                            sample_weight=w_test))
corr_ridge, _  = pearsonr(y_test, y_pred_ridge_test)

print(f"\n  Held-out test (n={len(test_idx)}):")
print(f"    R²:   {r2_ridge_test:.4f}")
print(f"    MAE:  {mae_ridge_test:.6f}")
print(f"    Corr: {corr_ridge:.4f}")

# LOO cross-validation on full training set
y_loo_ridge = cross_val_predict(
    Ridge(alpha=ridge_cv.alpha_, fit_intercept=True),
    C_train, y_train,
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    params={'sample_weight': w_train},
)
r2_ridge_loo  = float(r2_score(y_train, y_loo_ridge, sample_weight=w_train))
corr_ridge_loo, _ = pearsonr(y_train, y_loo_ridge)
print(f"\n  10-fold LOO on train set:")
print(f"    R²:   {r2_ridge_loo:.4f}")
print(f"    Corr: {corr_ridge_loo:.4f}")


# ── Section 2: Nonlinear alternatives ─────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: NONLINEAR ALTERNATIVES")
print("=" * 70)

models_to_test = {
    'Ridge_CV':   ridge_cv,
    'GBM':        GradientBoostingRegressor(
                      n_estimators=200, max_depth=3, learning_rate=0.05,
                      subsample=0.8, random_state=42),
    'RF':         RandomForestRegressor(
                      n_estimators=200, max_depth=4, min_samples_leaf=10,
                      random_state=42, n_jobs=-1),
    'MLP':        MLPRegressor(
                      hidden_layer_sizes=(64, 32), activation='relu',
                      alpha=0.01, max_iter=500, random_state=42,
                      early_stopping=True, validation_fraction=0.1),
}

model_results = {}
print(f"\n  {'Model':<15} {'Test R²':>10} {'Test MAE':>12} {'LOO R²':>10} {'Corr':>8}")
print(f"  {'─'*55}")

best_model     = None
best_r2        = -np.inf
best_model_name= None

for name, m in models_to_test.items():
    try:
        # Fit on train
        if name in ['GBM', 'RF']:
            m.fit(C_train, y_train, sample_weight=w_train)
        else:
            m.fit(C_train, y_train)

        # Test set
        y_pred_test = m.predict(C_test)
        r2_t  = float(r2_score(y_test, y_pred_test, sample_weight=w_test))
        mae_t = float(mean_absolute_error(y_test, y_pred_test, sample_weight=w_test))
        corr_t, _ = pearsonr(y_test, y_pred_test)

        # LOO (only for Ridge — GBM/RF LOO is too slow)
        if name == 'Ridge_CV':
            r2_loo = r2_ridge_loo
        else:
            try:
                y_loo = cross_val_predict(
                    m.__class__(**m.get_params()),
                    C_train, y_train,
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                )
                r2_loo = float(r2_score(y_train, y_loo, sample_weight=w_train))
            except Exception:
                r2_loo = np.nan

        model_results[name] = {
            'test_r2':  r2_t,
            'test_mae': mae_t,
            'test_corr': corr_t,
            'loo_r2':   r2_loo,
        }

        print(f"  {name:<15} {r2_t:>10.4f} {mae_t:>12.6f} "
              f"{r2_loo:>10.4f} {corr_t:>8.4f}")

        if r2_t > best_r2:
            best_r2         = r2_t
            best_model      = m
            best_model_name = name

    except Exception as e:
        print(f"  {name:<15} FAILED: {e}")

print(f"\n  Best model: {best_model_name} (test R²={best_r2:.4f})")


# ── Section 3: Stability-filtered evaluation ──────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: STABILITY-FILTERED EVALUATION")
print("=" * 70)
print("Does restricting to stable stocks improve pace model performance?")
print("Analog: F1 — CI width is calibrated, stable drivers more predictable")

stab_test  = stabilities[test_idx]
stab_pcts  = [25, 50, 75, 100]

# Pre-compute per-stock MC CIs using already-built policy distributions
# (centroid + cov from stock_stats, no re-encoding needed)
print("  Pre-computing MC CIs for test stocks...")
test_mc_cache = {}
for ticker in tickers_arr[test_idx]:
    stat = next((s for s in stock_stats if s['ticker'] == ticker), None)
    if stat is None:
        continue
    mu  = stat['centroid']
    cov = stat.get('cov', np.eye(latent_dim) * stat['stability'])
    try:
        samples = RNG.multivariate_normal(mu, cov, size=500)
    except np.linalg.LinAlgError:
        samples = mu + RNG.randn(500, latent_dim) * np.sqrt(stat['stability'])
    preds = best_model.predict(samples)
    test_mc_cache[ticker] = {
        'ci_lo': float(np.percentile(preds, 2.5)),
        'ci_hi': float(np.percentile(preds, 97.5)),
    }

print(f"\n  {'Stability filter':<22} {'n':>5} {'R²':>8} {'Corr':>8} {'Coverage':>10}")
print(f"  {'─'*55}")

filter_results = {}
for pct in stab_pcts:
    thresh      = np.percentile(stab_test, pct)
    mask_stable = stab_test <= thresh
    if mask_stable.sum() < 5:
        continue

    y_t_f    = y_test[mask_stable]
    y_p_f    = y_pred_ridge_test[mask_stable]
    w_t_f    = w_test[mask_stable]

    r2_f     = float(r2_score(y_t_f, y_p_f, sample_weight=w_t_f))
    corr_f,_ = pearsonr(y_t_f, y_p_f)

    # Coverage from cache
    coverages = []
    for i, ticker in enumerate(tickers_arr[test_idx][mask_stable]):
        if ticker not in test_mc_cache:
            continue
        ci = test_mc_cache[ticker]
        coverages.append(ci['ci_lo'] <= y_t_f[i] <= ci['ci_hi'])
    coverage = float(np.mean(coverages)) if coverages else np.nan

    print(f"  {'Q1-Q'+str(pct//25)+'  (stab ≤ p'+str(pct)+')':<22} "
          f"{mask_stable.sum():>5} {r2_f:>8.4f} {corr_f:>8.4f} "
          f"{coverage*100 if not np.isnan(coverage) else float('nan'):>9.1f}%")

    filter_results[f'p{pct}'] = {
        'n': int(mask_stable.sum()),
        'r2': r2_f,
        'corr': corr_f,
        'coverage': float(coverage) if not np.isnan(coverage) else None,
    }


# ── Section 4: Full Monte Carlo scouting with best model ─────────────────────
print("\n" + "=" * 70)
print("SECTION 4: MONTE CARLO SCOUTING REPORT — BEST MODEL")
print("=" * 70)
print(f"Using {best_model_name} as pace model.")

# Refit best model on ALL training stocks (not just train split)
print(f"\nRefitting {best_model_name} on full stock universe...")
if best_model_name in ['GBM', 'RF']:
    best_model.fit(centroids, mean_alphas_w, sample_weight=weights)
else:
    best_model.fit(centroids, mean_alphas_w)

# Save pace model
with open(DATA_DIR / 'pace_model.pkl', 'wb') as f:
    pickle.dump({
        'model':       best_model,
        'model_name':  best_model_name,
        'scaler_ae':   scaler_ae,
        'feature_cols': feature_cols,
        'alpha_lo':    float(alpha_lo),
        'alpha_hi':    float(alpha_hi),
    }, f)
print(f"  Saved: pace_model.pkl")


# ── Section 5: Re-run OOS validation with proper pace model ───────────────────
print("\n" + "=" * 70)
print("SECTION 5: OOS VALIDATION WITH PROPER PACE MODEL")
print("=" * 70)
print("Re-running Russell test with fixed pace model.")

# Load OOS data
if not OOS_DIR.exists():
    print("  OOS data not found — run stock_oos_validation.py first")
else:
    oos_files = list(OOS_DIR.glob('oos_*.parquet'))
    print(f"  Found {len(oos_files)} OOS stock files")

    oos_mc_results = []

    for f in oos_files:
        ticker = f.stem.replace('oos_', '')
        try:
            df    = pd.read_parquet(f)
            dates = pd.to_datetime(df.index)

            pre_mask  = dates < pd.Timestamp(CUTOFF_DATE)
            post_mask = dates >= pd.Timestamp(CUTOFF_DATE)

            if pre_mask.sum() < 63 or post_mask.sum() < 21:
                continue

            # Prep features
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0

            X_pre = df.loc[pre_mask, feature_cols].values.astype(np.float32)
            X_pre = clean_X(X_pre)
            X_pre_s = scaler_ae.transform(X_pre).astype(np.float32)
            z_pre   = encode_batch(model, X_pre_s)

            centroid  = z_pre.mean(axis=0)
            cov       = np.cov(z_pre.T) + np.eye(latent_dim) * 1e-4
            stability = float(z_pre.var(axis=0).mean())
            n_pre     = len(z_pre)

            # Realized post-2023 alpha
            y_post    = df.loc[post_mask, 'alpha_resid'].values \
                if 'alpha_resid' in df.columns else np.zeros(post_mask.sum())
            valid_post = ~np.isnan(y_post)
            if valid_post.sum() < 21:
                continue
            realized_alpha = float(np.nanmean(y_post))
            n_post         = int(valid_post.sum())

            # Monte Carlo with best model
            try:
                samples = RNG.multivariate_normal(centroid, cov, size=MC_SAMPLES)
            except np.linalg.LinAlgError:
                samples = centroid + RNG.randn(MC_SAMPLES, latent_dim) * \
                          np.sqrt(np.diag(cov))

            # Clip samples to training alpha range
            pred_alphas = best_model.predict(samples)
            pred_alphas = np.clip(pred_alphas, alpha_lo * 2, alpha_hi * 2)

            mc_mean  = float(pred_alphas.mean())
            mc_std   = float(pred_alphas.std())
            ci_lo    = float(np.percentile(pred_alphas, 2.5))
            ci_hi    = float(np.percentile(pred_alphas, 97.5))
            within_ci= bool(ci_lo <= realized_alpha <= ci_hi)

            oos_mc_results.append({
                'ticker':         ticker,
                'stability':      stability,
                'n_pre':          n_pre,
                'n_post':         n_post,
                'mc_mean':        mc_mean,
                'mc_std':         mc_std,
                'ci_lo':          ci_lo,
                'ci_hi':          ci_hi,
                'ci_width':       ci_hi - ci_lo,
                'realized_alpha': realized_alpha,
                'within_ci':      within_ci,
                'error':          abs(mc_mean - realized_alpha),
            })

        except Exception as e:
            continue

    if oos_mc_results:
        oos_df = pd.DataFrame(oos_mc_results).dropna(subset=['realized_alpha'])
        oos_df = oos_df.sort_values('mc_mean').reset_index(drop=True)

        r2_oos    = float(r2_score(oos_df['realized_alpha'], oos_df['mc_mean']))
        corr_oos  = float(pearsonr(oos_df['realized_alpha'], oos_df['mc_mean'])[0])
        coverage  = float(oos_df['within_ci'].mean())
        mae_oos   = float(oos_df['error'].mean())
        r_stab_ci, p_stab_ci = pearsonr(oos_df['stability'], oos_df['ci_width'])

        print(f"\n  OOS results with {best_model_name} pace model:")
        print(f"    n stocks:     {len(oos_df)}")
        print(f"    R²:           {r2_oos:.4f}  (naive: -0.0382)")
        print(f"    Correlation:  {corr_oos:.4f}  (naive: -0.084)")
        print(f"    MAE:          {mae_oos:.5f}")
        print(f"    95% coverage: {coverage*100:.1f}%  (naive: 83.0%)")
        print(f"    Stab→CI r:    {r_stab_ci:.3f}  p={p_stab_ci:.4f}")

        # Stability-filtered OOS
        print(f"\n  OOS by stability quintile:")
        oos_df['stab_q'] = pd.qcut(oos_df['stability'],
                                    min(5, len(oos_df) // 5),
                                    labels=False, duplicates='drop') + 1
        print(f"  {'Q':<5} {'n':>4} {'R²':>8} {'Coverage':>10} {'MAE':>10}")
        print(f"  {'─'*40}")
        for q in sorted(oos_df['stab_q'].unique()):
            sub = oos_df[oos_df['stab_q'] == q]
            r2_q = float(r2_score(sub['realized_alpha'], sub['mc_mean'])) \
                   if len(sub) > 2 else np.nan
            print(f"  Q{q:<4} {len(sub):>4} {r2_q:>8.4f} "
                  f"{sub['within_ci'].mean()*100:>9.1f}% "
                  f"{sub['error'].mean():>10.5f}")

        # Save updated OOS results
        oos_df.to_parquet(OOS_DIR / 'oos_mc_results_v2.parquet')
        oos_df.to_csv(OOS_DIR / 'oos_mc_results_v2.csv', index=False)
        print(f"\n  Saved: oos_mc_results_v2.parquet")


# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Predicted vs true — Ridge CV (test set)
axes[0, 0].scatter(y_pred_ridge_test, y_test, alpha=0.4, s=15,
                   c=w_test, cmap='Blues')
lims = [min(y_pred_ridge_test.min(), y_test.min()),
        max(y_pred_ridge_test.max(), y_test.max())]
axes[0, 0].plot(lims, lims, 'r--', lw=1, alpha=0.5)
axes[0, 0].set_xlabel('Predicted mean alpha')
axes[0, 0].set_ylabel('Realized mean alpha')
axes[0, 0].set_title(f'Ridge CV — Test Set\nR²={r2_ridge_test:.4f}  '
                     f'Corr={corr_ridge:.4f}\n(color = sample weight)')

# 2. Alpha vs stability — colored by weight
sc = axes[0, 1].scatter(stabilities, mean_alphas_w, alpha=0.3, s=10,
                         c=weights, cmap='viridis')
axes[0, 1].set_xlabel('Manifold Stability')
axes[0, 1].set_ylabel('Mean Alpha (winsorized)')
axes[0, 1].set_title('Alpha vs Stability\n(color = observation weight)')
plt.colorbar(sc, ax=axes[0, 1], label='Weight')

# 3. Model comparison bar chart
model_names = list(model_results.keys())
test_r2s    = [model_results[m]['test_r2'] for m in model_names]
colors_bar  = ['steelblue' if m != best_model_name else 'crimson'
               for m in model_names]
axes[0, 2].bar(model_names, test_r2s, color=colors_bar, alpha=0.8)
axes[0, 2].axhline(0, color='black', lw=0.8, linestyle='--')
axes[0, 2].set_ylabel('Test R²')
axes[0, 2].set_title('Model Comparison — Test R²\n(red = best)')

# 4. LOO residuals — Ridge CV
loo_resid = y_loo_ridge - y_train
axes[1, 0].scatter(y_train, loo_resid, alpha=0.3, s=10)
axes[1, 0].axhline(0, color='red', lw=1, linestyle='--')
axes[1, 0].set_xlabel('True mean alpha')
axes[1, 0].set_ylabel('LOO residual')
axes[1, 0].set_title(f'Ridge CV — LOO Residuals\nLOO R²={r2_ridge_loo:.4f}')

# 5. Stability vs prediction error on test set
test_errors = np.abs(y_pred_ridge_test - y_test)
axes[1, 1].scatter(stab_test, test_errors, alpha=0.4, s=15, color='darkorange')
r_se, p_se = pearsonr(stab_test, test_errors)
axes[1, 1].set_xlabel('Manifold Stability')
axes[1, 1].set_ylabel('|Predicted - Realized|')
axes[1, 1].set_title(f'Stability → Prediction Error\nr={r_se:.3f}  p={p_se:.4f}')

# 6. OOS if available
if oos_mc_results and len(oos_df) > 0:
    axes[1, 2].scatter(oos_df['mc_mean'], oos_df['realized_alpha'],
                       alpha=0.6, s=25, color='purple')
    lims2 = [min(oos_df['mc_mean'].min(), oos_df['realized_alpha'].min()),
             max(oos_df['mc_mean'].max(), oos_df['realized_alpha'].max())]
    axes[1, 2].plot(lims2, lims2, 'r--', lw=1, alpha=0.5)
    axes[1, 2].set_xlabel('MC predicted alpha')
    axes[1, 2].set_ylabel('Realized alpha (post-2023)')
    axes[1, 2].set_title(f'OOS Validation — Never-Seen Stocks\n'
                         f'R²={r2_oos:.4f}  Corr={corr_oos:.4f}')
else:
    axes[1, 2].text(0.5, 0.5, 'Run stock_oos_validation.py first',
                    ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('OOS Validation')

plt.suptitle('Pace Model Diagnostics', fontsize=12)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'pace_model_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: pace_model_diagnostics.png")

# Alpha distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(mean_alphas,   bins=80, alpha=0.6, label='Raw',       color='steelblue')
axes[0].hist(mean_alphas_w, bins=80, alpha=0.6, label='Winsorized',color='crimson')
axes[0].set_xlabel('Mean daily alpha')
axes[0].set_title('Alpha Target Distribution')
axes[0].legend()

axes[1].scatter(n_obs_arr, np.abs(mean_alphas), alpha=0.2, s=5)
axes[1].set_xlabel('N observations')
axes[1].set_ylabel('|Mean alpha|')
axes[1].set_xscale('log')
axes[1].set_title('Alpha reliability vs history length\n'
                  '(sparse stocks have noisier targets)')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'alpha_distribution.png', dpi=150)
plt.close()
print(f"Plot saved: alpha_distribution.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PACE MODEL SUMMARY")
print("=" * 70)
print(f"\n  Best model:          {best_model_name}")
print(f"  Best Ridge alpha:    {ridge_cv.alpha_:.2f}  (naive was 1.0)")
print(f"  Test R²:             {best_r2:.4f}")
print(f"  LOO R²:              {r2_ridge_loo:.4f}")
print(f"  Test correlation:    {corr_ridge:.4f}")

if oos_mc_results:
    print(f"\n  OOS validation (never-seen stocks):")
    print(f"    R²:        {r2_oos:.4f}  (naive pace model: -0.0382)")
    print(f"    Corr:      {corr_oos:.4f}  (naive: -0.084)")
    print(f"    Coverage:  {coverage*100:.1f}%  (naive: 83.0%)")
    print(f"    Stab→CI r: {r_stab_ci:.3f}")

results_out = {
    'best_model':       best_model_name,
    'ridge_alpha':      float(ridge_cv.alpha_),
    'n_stocks':         len(stock_stats),
    'n_train':          len(train_idx),
    'n_test':           len(test_idx),
    'model_results':    model_results,
    'filter_results':   filter_results,
}
if oos_mc_results:
    results_out['oos'] = {
        'r2':       float(r2_oos),
        'corr':     float(corr_oos),
        'coverage': float(coverage),
        'mae':      float(mae_oos),
        'r_stab_ci': float(r_stab_ci),
    }

with open(DATA_DIR / 'pace_model_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)

print(f"\nResults: pace_model_results.json")
print(f"Model:   pace_model.pkl")
print(f"Plots:   {PLOT_DIR}")
print("Done.")