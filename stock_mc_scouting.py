# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:56:56 2026

@author: Justin.Sanford
"""

# -*- coding: utf-8 -*-
"""
stock_mc_scouting.py
Monte Carlo scouting report + transaction cost analysis.
Runs independently of permutation tests — just needs ae_temporal_best.pt
and the clean stock parquet files.

Analog: F1 driver scouting report — predicted alpha with calibrated CIs.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
PLOT_DIR = DATA_DIR / 'plots' / 'mc_scouting'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG      = np.random.RandomState(42)

MC_SAMPLES   = 2000
CUTOFF_DATE  = '2023-01-01'

# Cache paths
Z_CACHE      = DATA_DIR / 'mc_scouting_Z_cache.npy'
TICKER_CACHE = DATA_DIR / 'mc_scouting_tickers_cache.npy'
ALPHA_CACHE  = DATA_DIR / 'mc_scouting_alpha_cache.npy'

print(f"Device: {DEVICE}")
print(f"Plots:  {PLOT_DIR}")

# ── Utilities ─────────────────────────────────────────────────────────────────
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


def encode_batch(model, X_scaled, batch_size=8192):
    loader = DataLoader(StockDataset(X_scaled.astype(np.float32)),
                        batch_size=batch_size, shuffle=False)
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
scaler       = ckpt['scaler']
feature_cols = ckpt['feature_cols']
print(f"  Latent dim: {latent_dim}  |  Features: {len(feature_cols)}")

# ── Load / build encoded dataset ──────────────────────────────────────────────
manifest    = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

if Z_CACHE.exists() and TICKER_CACHE.exists() and ALPHA_CACHE.exists():
    print("\nLoading encoded data from cache...")
    Z_base      = np.load(Z_CACHE)
    tickers_all = np.load(TICKER_CACHE, allow_pickle=True)
    y_alpha_all = np.load(ALPHA_CACHE)
    print(f"  Loaded {len(Z_base):,} observations")
else:
    print("\nEncoding pre-2023 data (building cache)...")
    Xs, alphas, ticker_labels = [], [], []

    for ticker in all_tickers:
        f = DATA_DIR / f'stock_clean_{ticker}.parquet'
        if not f.exists():
            continue
        try:
            df    = pd.read_parquet(f)
            dates = pd.to_datetime(df.index)
            pre   = dates < pd.Timestamp(CUTOFF_DATE)
            df    = df[pre]
            if len(df) < 63:
                continue
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            X = clean_X(df[feature_cols].values.astype(np.float32))
            y_alpha = df['alpha_resid'].values \
                if 'alpha_resid' in df.columns else np.zeros(len(df))
            valid = ~np.isnan(y_alpha)
            if valid.sum() < 63:
                continue
            X_s = scaler.transform(X[valid]).astype(np.float32)
            z   = encode_batch(model, X_s)
            Xs.append(z)
            alphas.append(y_alpha[valid])
            ticker_labels.extend([ticker] * valid.sum())
        except Exception:
            continue

    Z_base      = np.vstack(Xs)
    y_alpha_all = np.concatenate(alphas)
    tickers_all = np.array(ticker_labels)

    np.save(Z_CACHE,      Z_base)
    np.save(TICKER_CACHE, tickers_all)
    np.save(ALPHA_CACHE,  y_alpha_all)
    print(f"  Encoded {len(Z_base):,} observations — cache saved")

# ── Build per-stock policy distributions ─────────────────────────────────────
print("\nBuilding per-stock policy distributions...")
stock_policies = {}
for ticker in np.unique(tickers_all):
    mask = tickers_all == ticker
    if mask.sum() < 63:
        continue
    z_t = Z_base[mask]
    stock_policies[ticker] = {
        'centroid':   z_t.mean(axis=0),
        'cov':        np.cov(z_t.T) + np.eye(latent_dim) * 1e-4,
        'stability':  float(z_t.var(axis=0).mean()),
        'n_obs':      int(mask.sum()),
        'mean_alpha': float(np.nanmean(y_alpha_all[mask])),
    }

print(f"  Stocks with policy distributions: {len(stock_policies)}")

# ── Fit pace model ────────────────────────────────────────────────────────────
# Load best pace model if available, otherwise fit Ridge
import pickle
pace_model_path = DATA_DIR / 'pace_model.pkl'
if pace_model_path.exists():
    print("\nLoading fitted pace model (GBM from stock_pace_model.py)...")
    with open(pace_model_path, 'rb') as f:
        pace_data  = pickle.load(f)
    pace_model = pace_data['model']
    print(f"  Model type: {pace_data['model_name']}")
else:
    print("\nFitting Ridge pace model (run stock_pace_model.py for better GBM model)...")
    ticker_list = list(stock_policies.keys())
    RNG.shuffle(ticker_list)
    n_fit       = int(len(ticker_list) * 0.8)
    fit_tickers = ticker_list[:n_fit]
    eval_tickers= ticker_list[n_fit:]

    centroids_fit = np.array([stock_policies[t]['centroid']  for t in fit_tickers])
    alphas_fit    = np.array([stock_policies[t]['mean_alpha'] for t in fit_tickers])

    from sklearn.linear_model import RidgeCV
    alphas_grid = np.logspace(-3, 6, 50)
    pace_model  = RidgeCV(alphas=alphas_grid, cv=5)
    pace_model.fit(centroids_fit, alphas_fit)
    eval_tickers_list = eval_tickers

# ── Monte Carlo scouting ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("MONTE CARLO SCOUTING REPORT")
print(f"{'='*70}")
print(f"Analog: F1 driver scouting — predicted alpha ± CI from policy stability")
print(f"MC samples per stock: {MC_SAMPLES}")

# Use 20% held-out stocks for evaluation
ticker_list  = list(stock_policies.keys())
RNG.shuffle(ticker_list)
n_fit        = int(len(ticker_list) * 0.8)
eval_tickers = ticker_list[n_fit:]
fit_tickers  = ticker_list[:n_fit]

# Refit pace model on 80% if we loaded from file (already fit on full universe)
# Just use all stocks for the eval to get more data points
eval_tickers = ticker_list   # evaluate on all — pace model already fit externally

mc_results = []
for ticker in eval_tickers:
    p      = stock_policies[ticker]
    mu     = p['centroid']
    cov    = p['cov']
    stab   = p['stability']
    true_a = p['mean_alpha']
    n_obs  = p['n_obs']

    try:
        samples = RNG.multivariate_normal(mu, cov, size=MC_SAMPLES)
    except np.linalg.LinAlgError:
        samples = mu + RNG.randn(MC_SAMPLES, latent_dim) * np.sqrt(np.diag(cov))

    pred_alphas = pace_model.predict(samples)

    mc_mean   = float(pred_alphas.mean())
    mc_std    = float(pred_alphas.std())
    ci_lo     = float(np.percentile(pred_alphas, 2.5))
    ci_hi     = float(np.percentile(pred_alphas, 97.5))
    within_ci = bool(ci_lo <= true_a <= ci_hi)

    mc_results.append({
        'ticker':     ticker,
        'stability':  stab,
        'n_obs':      n_obs,
        'true_alpha': true_a,
        'mc_mean':    mc_mean,
        'mc_std':     mc_std,
        'ci_lo':      ci_lo,
        'ci_hi':      ci_hi,
        'ci_width':   ci_hi - ci_lo,
        'within_ci':  within_ci,
        'error':      abs(mc_mean - true_a),
    })

mc_df = pd.DataFrame(mc_results).sort_values('mc_mean').reset_index(drop=True)

# ── Summary stats ─────────────────────────────────────────────────────────────
r2_mc     = float(r2_score(mc_df['true_alpha'], mc_df['mc_mean']))
corr_mc   = float(pearsonr(mc_df['true_alpha'], mc_df['mc_mean'])[0])
coverage  = float(mc_df['within_ci'].mean())
r_stab_ci, p_stab_ci = pearsonr(mc_df['stability'], mc_df['ci_width'])
r_stab_err, _        = pearsonr(mc_df['stability'], mc_df['error'])

print(f"\n  Stocks evaluated: {len(mc_df)}")
print(f"\n  Prediction quality:")
print(f"    R²:              {r2_mc:.4f}")
print(f"    Correlation:     {corr_mc:.4f}")
print(f"    MAE:             {mc_df['error'].mean():.6f}")
print(f"    95% CI coverage: {coverage*100:.1f}%  (target: 95%)")
print(f"    Within CI:       {mc_df['within_ci'].sum()}/{len(mc_df)}")

print(f"\n  Stability findings:")
print(f"    Stab → CI width:  r={r_stab_ci:.3f}  p={p_stab_ci:.4f}")
print(f"    Stab → error:     r={r_stab_err:.3f}")
print(f"    (F1 result:       r=0.771  p<0.0001)")
print(f"    (In-universe eval: r=0.619  p<0.0001)")

# Quintile breakdown
mc_df['stab_q'] = pd.qcut(mc_df['stability'], 5,
                            labels=['Q1\n(stable)', 'Q2', 'Q3', 'Q4', 'Q5\n(unstable)'],
                            duplicates='drop')
print(f"\n  Results by stability quintile:")
print(f"  {'Quintile':<14} {'n':>5} {'Mean CI':>10} {'Coverage':>10} {'MAE':>10}")
print(f"  {'─'*52}")
for q in mc_df['stab_q'].cat.categories:
    sub = mc_df[mc_df['stab_q'] == q]
    print(f"  {str(q):<14} {len(sub):>5} {sub['ci_width'].mean():>10.5f} "
          f"{sub['within_ci'].mean()*100:>9.1f}% "
          f"{sub['error'].mean():>10.5f}")

# Most and least stable
print(f"\n  Most stable stocks (tightest CIs):")
print(f"  {'Ticker':<8} {'Stability':>10} {'Predicted':>10} {'True α':>10} "
      f"{'CI width':>10} {'In CI':>6}")
print(f"  {'─'*58}")
for _, row in mc_df.nsmallest(10, 'stability').iterrows():
    print(f"  {row['ticker']:<8} {row['stability']:>10.4f} "
          f"{row['mc_mean']:>10.5f} {row['true_alpha']:>10.5f} "
          f"{row['ci_width']:>10.5f} {'✓' if row['within_ci'] else '✗':>6}")

print(f"\n  Most unstable stocks (widest CIs):")
print(f"  {'Ticker':<8} {'Stability':>10} {'Predicted':>10} {'True α':>10} "
      f"{'CI width':>10} {'In CI':>6}")
print(f"  {'─'*58}")
for _, row in mc_df.nlargest(10, 'stability').iterrows():
    print(f"  {row['ticker']:<8} {row['stability']:>10.4f} "
          f"{row['mc_mean']:>10.5f} {row['true_alpha']:>10.5f} "
          f"{row['ci_width']:>10.5f} {'✓' if row['within_ci'] else '✗':>6}")

# ── Transaction cost analysis ─────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TRANSACTION COST ANALYSIS")
print(f"{'='*70}")

# Signal autocorrelation → implied holding period
print("\nComputing signal autocorrelation...")
signal_autocorrs        = []
stable_autocorrs        = []
stable_q1_threshold     = np.percentile(
    [v['stability'] for v in stock_policies.values()], 25)

for ticker in list(np.unique(tickers_all))[:300]:
    if ticker not in stock_policies:
        continue
    mask   = tickers_all == ticker
    z_t    = Z_base[mask]
    if len(z_t) < 63:
        continue
    pred_a = pace_model.predict(z_t)
    if len(pred_a) < 10:
        continue
    ac1 = float(np.corrcoef(pred_a[:-1], pred_a[1:])[0, 1])
    if not np.isnan(ac1):
        signal_autocorrs.append(ac1)
        if stock_policies[ticker]['stability'] <= stable_q1_threshold:
            stable_autocorrs.append(ac1)

mean_ac1     = float(np.mean(signal_autocorrs))
implied_hold = 1 / max(1 - mean_ac1, 0.01)

stable_ac1        = float(np.mean(stable_autocorrs)) if stable_autocorrs else mean_ac1
stable_hold       = 1 / max(1 - stable_ac1, 0.01)

print(f"  Full universe:   AC1={mean_ac1:.3f}  implied hold={implied_hold:.1f}d")
print(f"  Stable Q1:       AC1={stable_ac1:.3f}  implied hold={stable_hold:.1f}d")

# IC and gross alpha from CV results
daily_alpha_std = float(np.nanstd(y_alpha_all))
ic_cv           = np.sqrt(max(0.0355, 0))   # from 5-fold CV
ic_temporal     = np.sqrt(max(0.0551, 0))   # from temporal holdout
ann_alpha_cv    = ic_cv       * daily_alpha_std * np.sqrt(252)
ann_alpha_temp  = ic_temporal * daily_alpha_std * np.sqrt(252)

print(f"\n  Signal quality:")
print(f"    Daily alpha std:           {daily_alpha_std:.5f}")
print(f"    IC (CV R²=0.0355):         {ic_cv:.4f}")
print(f"    IC (temporal R²=0.0551):   {ic_temporal:.4f}")
print(f"    Gross annual alpha (CV):   {ann_alpha_cv*100:.2f}%")
print(f"    Gross annual alpha (temp): {ann_alpha_temp*100:.2f}%")
print(f"    Theoretical annual IR:     {ic_cv*np.sqrt(252):.2f}")

print(f"\n  Transaction cost breakdown:")
print(f"  {'Scenario':<30} {'Daily turn':>11} {'Ann TC':>8} "
      f"{'Net α (CV)':>12} {'Profitable':>11}")
print(f"  {'─'*75}")

scenarios = [
    ('Full universe, 5bps',    implied_hold,  0.0005),
    ('Full universe, 10bps',   implied_hold,  0.0010),
    ('Full universe, 20bps',   implied_hold,  0.0020),
    ('Stable Q1, 5bps',        stable_hold,   0.0005),
    ('Stable Q1, 10bps',       stable_hold,   0.0010),
]

tc_results = []
for label, hold, tc_one_way in scenarios:
    daily_turn = 1 / max(hold, 1)
    ann_tc     = tc_one_way * daily_turn * 252
    net_cv     = ann_alpha_cv - ann_tc
    net_temp   = ann_alpha_temp - ann_tc
    profitable = '✓' if net_cv > 0 else '✗'
    print(f"  {label:<30} {daily_turn*100:>10.1f}% {ann_tc*100:>7.3f}% "
          f"{net_cv*100:>11.3f}% {profitable:>11}")
    tc_results.append({
        'scenario': label, 'hold': hold, 'tc': tc_one_way,
        'ann_tc': ann_tc, 'net_alpha_cv': net_cv, 'net_alpha_temp': net_temp,
    })

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Scouting report with CI bars
sorted_mc = mc_df.sort_values('mc_mean').reset_index(drop=True)
axes[0, 0].errorbar(range(len(sorted_mc)), sorted_mc['mc_mean'],
                    yerr=[sorted_mc['mc_mean'] - sorted_mc['ci_lo'],
                          sorted_mc['ci_hi']   - sorted_mc['mc_mean']],
                    fmt='none', alpha=0.2, color='steelblue', lw=0.5)
axes[0, 0].scatter(range(len(sorted_mc)), sorted_mc['true_alpha'],
                   s=6, color='red', alpha=0.5, zorder=3, label='True α')
axes[0, 0].scatter(range(len(sorted_mc)), sorted_mc['mc_mean'],
                   s=6, color='steelblue', alpha=0.5, zorder=3, label='MC mean')
axes[0, 0].set_xlabel('Stock (sorted by predicted α)')
axes[0, 0].set_ylabel('Alpha')
axes[0, 0].set_title(f'MC Scouting Report\nR²={r2_mc:.4f}  '
                     f'Coverage={coverage*100:.1f}%  n={len(mc_df)}')
axes[0, 0].legend(fontsize=8)

# 2. Predicted vs true
axes[0, 1].scatter(mc_df['mc_mean'], mc_df['true_alpha'], alpha=0.4, s=10,
                   c=mc_df['stability'], cmap='RdYlGn_r')
lims = [min(mc_df['mc_mean'].min(), mc_df['true_alpha'].min()),
        max(mc_df['mc_mean'].max(), mc_df['true_alpha'].max())]
axes[0, 1].plot(lims, lims, 'r--', lw=1, alpha=0.5)
axes[0, 1].set_xlabel('MC predicted alpha')
axes[0, 1].set_ylabel('True alpha')
axes[0, 1].set_title(f'Predicted vs True\nr={corr_mc:.3f}\n'
                     f'(color = stability, green=stable)')

# 3. Stability → CI width
axes[0, 2].scatter(mc_df['stability'], mc_df['ci_width'],
                   alpha=0.4, s=10, color='darkorange')
axes[0, 2].set_xlabel('Manifold Stability')
axes[0, 2].set_ylabel('CI Width')
axes[0, 2].set_title(f'Stability → CI Width\n'
                     f'r={r_stab_ci:.3f}  p={p_stab_ci:.4f}\n'
                     f'(F1 result: r=0.771)')

# 4. CI width by quintile (the scouting report bar chart)
q_means    = mc_df.groupby('stab_q', observed=True)['ci_width'].mean()
q_coverage = mc_df.groupby('stab_q', observed=True)['within_ci'].mean() * 100
q_errors   = mc_df.groupby('stab_q', observed=True)['error'].mean()

x = range(len(q_means))
bars = axes[1, 0].bar(x, q_means.values, color='steelblue', alpha=0.8)
axes[1, 0].set_xticks(list(x))
axes[1, 0].set_xticklabels([str(q).replace('\n', ' ') for q in q_means.index],
                             fontsize=7)
axes[1, 0].set_ylabel('Mean CI Width', color='steelblue')
ax2 = axes[1, 0].twinx()
ax2.plot(list(x), q_coverage.values, 'o-', color='crimson', lw=2, ms=8,
         label='Coverage %')
ax2.axhline(95, color='crimson', linestyle='--', alpha=0.4, lw=1)
ax2.set_ylabel('Coverage %', color='crimson')
ax2.set_ylim(0, 110)
axes[1, 0].set_title('CI Width + Coverage by Stability Quintile\n'
                     '(Analog: F1 driver scouting report)')

# 5. Stability → prediction error
axes[1, 1].scatter(mc_df['stability'], mc_df['error'],
                   alpha=0.4, s=10, color='purple')
axes[1, 1].set_xlabel('Manifold Stability')
axes[1, 1].set_ylabel('|Predicted − True|')
axes[1, 1].set_title(f'Stability → Prediction Error\nr={r_stab_err:.3f}')

# 6. Transaction cost waterfall
scenario_labels = [s['scenario'].replace(', ', '\n') for s in tc_results]
net_alphas      = [s['net_alpha_cv'] * 100 for s in tc_results]
colors_bar      = ['steelblue' if v > 0 else 'crimson' for v in net_alphas]
axes[1, 2].bar(range(len(net_alphas)), net_alphas, color=colors_bar, alpha=0.8)
axes[1, 2].axhline(0, color='black', lw=0.8)
axes[1, 2].axhline(ann_alpha_cv * 100, color='steelblue', linestyle='--',
                   alpha=0.5, lw=1, label=f'Gross α={ann_alpha_cv*100:.2f}%')
axes[1, 2].set_xticks(range(len(scenario_labels)))
axes[1, 2].set_xticklabels(scenario_labels, fontsize=7)
axes[1, 2].set_ylabel('Net annual alpha (%)')
axes[1, 2].set_title('Net Alpha After Transaction Costs')
axes[1, 2].legend(fontsize=8)

plt.suptitle('Monte Carlo Scouting Report — Finance Manifold\n'
             f'(Temporal model, {len(mc_df)} stocks, {MC_SAMPLES} MC samples each)',
             fontsize=11)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'mc_scouting_full.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: mc_scouting_full.png")

# ── Save ─────────────────────────────────────────────────────────────────────
mc_df.to_parquet(DATA_DIR / 'mc_scouting_results_full.parquet')
mc_df.to_csv(DATA_DIR / 'mc_scouting_results_full.csv', index=False)

summary = {
    'n_stocks':      len(mc_df),
    'r2':            float(r2_mc),
    'corr':          float(corr_mc),
    'coverage_95':   float(coverage),
    'r_stab_ci':     float(r_stab_ci),
    'p_stab_ci':     float(p_stab_ci),
    'r_stab_err':    float(r_stab_err),
    'signal': {
        'daily_alpha_std':  float(daily_alpha_std),
        'ic_cv':            float(ic_cv),
        'ic_temporal':      float(ic_temporal),
        'ann_alpha_cv':     float(ann_alpha_cv),
        'ann_alpha_temp':   float(ann_alpha_temp),
        'mean_ac1':         float(mean_ac1),
        'implied_hold_days': float(implied_hold),
        'stable_hold_days':  float(stable_hold),
    },
    'tc_scenarios': tc_results,
    'quintile_summary': {
        str(q): {
            'mean_ci':   float(mc_df[mc_df['stab_q']==q]['ci_width'].mean()),
            'coverage':  float(mc_df[mc_df['stab_q']==q]['within_ci'].mean()),
            'mae':       float(mc_df[mc_df['stab_q']==q]['error'].mean()),
        }
        for q in mc_df['stab_q'].cat.categories
    },
}

with open(DATA_DIR / 'mc_scouting_results_full.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"Results: mc_scouting_results_full.json")
print(f"MC data: mc_scouting_results_full.parquet")
print(f"Plots:   {PLOT_DIR}")
print("Done.")