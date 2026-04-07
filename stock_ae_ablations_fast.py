# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:20:18 2026

@author: Justin.Sanford
"""

# -*- coding: utf-8 -*-
"""
stock_ae_ablations_fast.py
Sections 2-6 of the ablation suite — no permutation tests.
Runs immediately using cached Z_base.

Sections:
  2. Feature ablations
  3. Rolling-window alpha R²
  4. Monte Carlo scouting (Ridge pace model)
  5. Transaction cost analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
PLOT_DIR = DATA_DIR / 'plots' / 'ablations'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG      = np.random.RandomState(42)

MC_SAMPLES = 2000

print(f"Device: {DEVICE}")
print(f"Plots:  {PLOT_DIR}")

# ── Utilities ─────────────────────────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


def build_ae(input_dim, h1, h2, latent_dim):
    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(h1, h2),        nn.LayerNorm(h2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(h2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(h2, h1),         nn.LayerNorm(h1), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(h1, input_dim),
            )
        def encode(self, x): return self.encoder(x)
        def decode(self, z): return self.decoder(z)
        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z
    return _AE()


def build_model_from_ckpt(ckpt):
    s   = ckpt['model_state']
    h1  = s['encoder.0.weight'].shape[0]
    h2  = s['encoder.4.weight'].shape[0]
    inp = s['encoder.0.weight'].shape[1]
    lat = s['encoder.8.weight'].shape[0]
    model = build_ae(inp, h1, h2, lat).to(DEVICE)
    model.load_state_dict(s)
    model.eval()
    return model, inp, lat


def encode(model, X_scaled, batch_size=8192):
    loader = DataLoader(StockDataset(X_scaled), batch_size=batch_size, shuffle=False)
    model.eval()
    parts = []
    with torch.no_grad():
        for xb in loader:
            _, z = model(xb.to(DEVICE))
            parts.append(z.cpu().numpy())
    return np.vstack(parts)


def load_stocks(tickers, feature_cols, date_filter=None):
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
            valid   = ~np.isnan(y_ret) & ~np.isnan(y_alpha)
            if valid.sum() < 63:
                continue
            Xs.append(X[valid])
            y_rets.append(y_ret[valid])
            y_alphas.append(y_alpha[valid])
            all_dates.append(dates[valid])
            all_tickers.extend([ticker] * valid.sum())
        except Exception:
            pass
    return (np.vstack(Xs), np.concatenate(y_rets), np.concatenate(y_alphas),
            np.concatenate(all_dates), np.array(all_tickers))


# ── Load model + data ─────────────────────────────────────────────────────────
print("\nLoading temporal model + pre-2023 data...")
ckpt_temp         = torch.load(DATA_DIR / 'ae_temporal_best.pt',
                               map_location=DEVICE, weights_only=False)
model_base, input_dim, latent_dim = build_model_from_ckpt(ckpt_temp)
scaler_base       = ckpt_temp['scaler']
feature_cols      = ckpt_temp['feature_cols']

manifest    = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

X_all, y_ret_all, y_alpha_all, dates_all, tickers_all = load_stocks(
    all_tickers, feature_cols, date_filter=('before', '2023-01-01'))
dates_all = np.array(dates_all)

# Load Z_base from cache — built by ablations.py or ablations_fast.py
Z_CACHE_PATH = DATA_DIR / 'ablations_Z_base_cache.npy'
if Z_CACHE_PATH.exists():
    print("  Loading Z_base from cache...")
    Z_base = np.load(Z_CACHE_PATH)
else:
    print("  Cache not found — encoding now...")
    X_all_s = scaler_base.transform(X_all)
    Z_base  = encode(model_base, X_all_s)
    np.save(Z_CACHE_PATH, Z_base)
    print(f"  Saved: {Z_CACHE_PATH.name}")

print(f"  {len(X_all):,} obs  |  {len(np.unique(tickers_all))} stocks  |  "
      f"Z shape: {Z_base.shape}")

ablation_results = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Feature Ablations
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: FEATURE ABLATIONS")
print("=" * 70)

feature_groups = {
    'volatility':   [c for c in feature_cols if 'vol' in c and 'ratio' not in c],
    'momentum':     [c for c in feature_cols if any(k in c for k in
                     ['mom', 'ret_5', 'ret_21', 'ret_63', 'ret_252', 'up_days', 'skew'])],
    'price_pos':    [c for c in feature_cols if any(k in c for k in
                     ['pos_', 'sma', 'price_vs', 'high_', 'low_'])],
    'trend':        [c for c in feature_cols if any(k in c for k in
                     ['price_vs_sma', 'sma21_vs'])],
    'drawdown':     [c for c in feature_cols if any(k in c for k in
                     ['drawdown', 'dd_'])],
    'volume':       [c for c in feature_cols if 'vol_ratio' in c or 'vol_ma' in c
                     or 'vol_21d_chg' in c],
    'distribution': [c for c in feature_cols if any(k in c for k in
                     ['skew', 'kurt', 'up_vol'])],
}
feature_groups = {k: v for k, v in feature_groups.items() if v}

print(f"\n  Feature groups:")
for gname, gcols in feature_groups.items():
    print(f"    {gname:<15} {len(gcols)} features: {', '.join(gcols[:4])}"
          f"{'...' if len(gcols) > 4 else ''}")

def quick_alpha_r2(X, y_alpha, latent_d=12):
    if X.shape[1] < 3:
        return np.nan
    sc = StandardScaler()
    Xs = sc.fit_transform(X).astype(np.float32)
    n_train = int(len(Xs) * 0.8)
    Xtr, Xvl = Xs[:n_train], Xs[n_train:]
    ytr, yvl = y_alpha[:n_train], y_alpha[n_train:]
    h1 = min(32, X.shape[1] * 2)
    h2 = min(16, X.shape[1])
    ld = min(latent_d, X.shape[1] - 1)
    model_abl = build_ae(X.shape[1], h1, h2, ld).to(DEVICE)
    opt  = torch.optim.AdamW(model_abl.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.HuberLoss(delta=1.0)
    tr_loader = DataLoader(StockDataset(Xtr), batch_size=4096, shuffle=True)
    best_loss, patience = float('inf'), 0
    for epoch in range(50):
        model_abl.train()
        for xb in tr_loader:
            opt.zero_grad()
            xr, _ = model_abl(xb.to(DEVICE))
            loss = crit(xr, xb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model_abl.parameters(), 1.0)
            opt.step()
        model_abl.eval()
        with torch.no_grad():
            xvt = torch.FloatTensor(Xvl).to(DEVICE)
            vl_loss = crit(model_abl(xvt)[0], xvt).item()
        if vl_loss < best_loss:
            best_loss = vl_loss
            patience  = 0
        else:
            patience += 1
        if patience >= 5:
            break
    z_tr = encode(model_abl, Xtr)
    z_vl = encode(model_abl, Xvl)
    ridge = Ridge(alpha=1.0)
    ridge.fit(z_tr, ytr)
    return float(r2_score(yvl, ridge.predict(z_vl)))

abl_idx = RNG.choice(len(X_all), min(500_000, len(X_all)), replace=False)
X_abl   = X_all[abl_idx]
y_abl   = y_alpha_all[abl_idx]

print(f"\n  Baseline (all features): alpha R² ≈ 0.0355 (from CV)")
print(f"\n  {'Ablation':<25} {'Features':>8} {'Alpha R²':>10} {'vs baseline':>12}")
print(f"  {'─'*58}")

r2_full = quick_alpha_r2(X_abl, y_abl)
print(f"  {'full (retrained)':<25} {X_abl.shape[1]:>8} {r2_full:>10.4f}  {'baseline':>12}")
ablation_results['full'] = r2_full

for gname, gcols in feature_groups.items():
    drop_idx  = [feature_cols.index(c) for c in gcols if c in feature_cols]
    keep_idx  = [i for i in range(len(feature_cols)) if i not in drop_idx]
    X_dropped = X_abl[:, keep_idx]
    r2_d      = quick_alpha_r2(X_dropped, y_abl)
    delta     = r2_d - r2_full
    ablation_results[f'drop_{gname}'] = r2_d
    direction = '▼' if delta < -0.002 else '▲' if delta > 0.002 else '≈'
    print(f"  {'drop_'+gname:<25} {X_dropped.shape[1]:>8} {r2_d:>10.4f}  "
          f"{delta:>+10.4f} {direction}")

for gname, gcols in feature_groups.items():
    keep_idx = [feature_cols.index(c) for c in gcols if c in feature_cols]
    if len(keep_idx) < 2:
        continue
    X_only = X_abl[:, keep_idx]
    r2_o   = quick_alpha_r2(X_only, y_abl)
    ablation_results[f'only_{gname}'] = r2_o
    print(f"  {'only_'+gname:<25} {X_only.shape[1]:>8} {r2_o:>10.4f}  "
          f"{'(single group)':>12}")

drop_names  = [k for k in ablation_results if k.startswith('drop_')]
drop_deltas = [ablation_results[k] - r2_full for k in drop_names]
clean_names = [k.replace('drop_', '') for k in drop_names]

fig, ax = plt.subplots(figsize=(9, 4))
colors = ['crimson' if d < -0.002 else 'steelblue' for d in drop_deltas]
ax.barh(clean_names, drop_deltas, color=colors)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Change in alpha R² when group dropped')
ax.set_title('Feature Group Importance\n(negative = group is load-bearing)')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'feature_ablations.png', dpi=150)
plt.close()
print(f"\n  Plot saved: feature_ablations.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Rolling-Window Alpha R²
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: ROLLING-WINDOW ALPHA R²")
print("=" * 70)

WINDOW_YEARS = 3
STEP_MONTHS  = 6

unique_years = sorted(set(pd.Timestamp(d).year for d in dates_all))
min_year     = min(unique_years)
max_year     = max(unique_years)

print(f"\n  Window: {WINDOW_YEARS}yr train → 6mo test, stepping every {STEP_MONTHS} months")
print(f"\n  {'Train period':<25} {'Test period':<20} {'Alpha R²':>10} {'N test obs':>12}")
print(f"  {'─'*70}")

rolling_results = []
start_year = min_year

while True:
    train_start = pd.Timestamp(f'{start_year}-01-01')
    train_end   = train_start + pd.DateOffset(years=WINDOW_YEARS)
    test_start  = train_end
    test_end    = test_start + pd.DateOffset(months=6)

    if test_end > pd.Timestamp(f'{max_year}-12-31'):
        break

    mask_train = (dates_all >= train_start) & (dates_all < train_end)
    mask_test  = (dates_all >= test_start)  & (dates_all < test_end)

    if mask_train.sum() < 50_000 or mask_test.sum() < 5_000:
        start_year += 1
        continue

    z_tr = Z_base[mask_train]
    z_te = Z_base[mask_test]
    y_tr = y_alpha_all[mask_train]
    y_te = y_alpha_all[mask_test]

    ridge = Ridge(alpha=1.0)
    ridge.fit(z_tr, y_tr)
    r2_roll = float(r2_score(y_te, ridge.predict(z_te)))

    print(f"  {str(train_start.date())+' – '+str(train_end.date()):<25} "
          f"{str(test_start.date())+' – '+str(test_end.date()):<20} "
          f"{r2_roll:>10.4f} {mask_test.sum():>12,}")

    rolling_results.append({
        'train_start': str(train_start.date()),
        'test_start':  str(test_start.date()),
        'r2':          r2_roll,
        'n_test':      int(mask_test.sum()),
    })

    start_year_dt = train_start + pd.DateOffset(months=STEP_MONTHS)
    start_year    = start_year_dt.year
    if start_year_dt.month > 1:
        start_year = start_year_dt.year + (1 if start_year_dt.month > 6 else 0)

roll_r2s = [r['r2'] for r in rolling_results]
print(f"\n  Mean rolling R²:   {np.mean(roll_r2s):.4f}")
print(f"  Std rolling R²:    {np.std(roll_r2s):.4f}")
print(f"  Min / Max:         {np.min(roll_r2s):.4f} / {np.max(roll_r2s):.4f}")
print(f"  % windows > 0:     {(np.array(roll_r2s) > 0).mean()*100:.1f}%")
print(f"  % windows > PCA:   {(np.array(roll_r2s) > 0.0039).mean()*100:.1f}%")

fig, ax = plt.subplots(figsize=(12, 4))
test_dates = [pd.Timestamp(r['test_start']) for r in rolling_results]
ax.plot(test_dates, roll_r2s, 'o-', color='steelblue', lw=1.5, ms=5)
ax.axhline(0,      color='black', lw=0.8, linestyle='--', label='R²=0')
ax.axhline(0.0039, color='red',   lw=1,   linestyle='--', label='PCA baseline')
ax.axhline(np.mean(roll_r2s), color='steelblue', lw=1, linestyle=':',
           label=f'Mean={np.mean(roll_r2s):.4f}')
for name, start, end in [('GFC',   '2008-09-01', '2009-03-31'),
                          ('COVID', '2020-02-01', '2020-06-30'),
                          ('Rate',  '2022-01-01', '2022-12-31')]:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color='red')
ax.set_xlabel('Test period start')
ax.set_ylabel('Alpha R²')
ax.set_title('Rolling-Window Alpha R² (3yr train → 6mo test)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'rolling_alpha_r2.png', dpi=150)
plt.close()
print(f"\n  Plot saved: rolling_alpha_r2.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Monte Carlo Scouting
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: MONTE CARLO SCOUTING REPORT")
print("=" * 70)

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

print(f"  Stocks: {len(stock_policies)}")

ticker_list  = list(stock_policies.keys())
RNG.shuffle(ticker_list)
n_fit        = int(len(ticker_list) * 0.8)
fit_tickers  = ticker_list[:n_fit]
eval_tickers = ticker_list[n_fit:]

centroids_fit = np.array([stock_policies[t]['centroid']  for t in fit_tickers])
alphas_fit    = np.array([stock_policies[t]['mean_alpha'] for t in fit_tickers])

pace_model = Ridge(alpha=1.0)
pace_model.fit(centroids_fit, alphas_fit)

print(f"  Running MC ({MC_SAMPLES} samples × {len(eval_tickers)} eval stocks)...")
mc_results = []
for ticker in eval_tickers:
    p = stock_policies[ticker]
    try:
        samples = RNG.multivariate_normal(p['centroid'], p['cov'], size=MC_SAMPLES)
    except np.linalg.LinAlgError:
        samples = p['centroid'] + RNG.randn(MC_SAMPLES, latent_dim) * \
                  np.sqrt(np.diag(p['cov']))
    preds    = pace_model.predict(samples)
    mc_mean  = float(preds.mean())
    ci_lo    = float(np.percentile(preds, 2.5))
    ci_hi    = float(np.percentile(preds, 97.5))
    true_a   = p['mean_alpha']
    mc_results.append({
        'ticker':     ticker,
        'stability':  p['stability'],
        'n_obs':      p['n_obs'],
        'true_alpha': true_a,
        'mc_mean':    mc_mean,
        'ci_lo':      ci_lo,
        'ci_hi':      ci_hi,
        'ci_width':   ci_hi - ci_lo,
        'within_ci':  bool(ci_lo <= true_a <= ci_hi),
        'error':      abs(mc_mean - true_a),
    })

mc_df    = pd.DataFrame(mc_results)
r2_mc    = float(r2_score(mc_df['true_alpha'], mc_df['mc_mean']))
corr_mc  = float(pearsonr(mc_df['true_alpha'], mc_df['mc_mean'])[0])
coverage = float(mc_df['within_ci'].mean())
r_stab_ci, p_stab_ci = pearsonr(mc_df['stability'], mc_df['ci_width'])

print(f"\n  R²={r2_mc:.4f}  Corr={corr_mc:.4f}  Coverage={coverage*100:.1f}%")
print(f"  Stability → CI width: r={r_stab_ci:.3f}  p={p_stab_ci:.4f}")
print(f"  (F1 result: r=0.771  |  eval result: r=0.619)")

mc_df['stab_q'] = pd.qcut(mc_df['stability'], 5,
                            labels=['Q1\n(stable)', 'Q2', 'Q3', 'Q4', 'Q5\n(unstable)'],
                            duplicates='drop')
print(f"\n  {'Quintile':<14} {'Mean CI':>10} {'Coverage':>10} {'MAE':>10}")
print(f"  {'─'*46}")
for q in mc_df['stab_q'].cat.categories:
    sub = mc_df[mc_df['stab_q'] == q]
    print(f"  {str(q):<14} {sub['ci_width'].mean():>10.5f} "
          f"{sub['within_ci'].mean()*100:>9.1f}% "
          f"{sub['error'].mean():>10.5f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sorted_mc = mc_df.sort_values('mc_mean').reset_index(drop=True)
axes[0, 0].errorbar(range(len(sorted_mc)), sorted_mc['mc_mean'],
                    yerr=[sorted_mc['mc_mean'] - sorted_mc['ci_lo'],
                          sorted_mc['ci_hi']   - sorted_mc['mc_mean']],
                    fmt='none', alpha=0.3, color='steelblue', lw=0.5)
axes[0, 0].scatter(range(len(sorted_mc)), sorted_mc['true_alpha'],
                   s=8, color='red', alpha=0.5, zorder=3, label='True α')
axes[0, 0].scatter(range(len(sorted_mc)), sorted_mc['mc_mean'],
                   s=8, color='steelblue', alpha=0.5, zorder=3, label='MC mean')
axes[0, 0].set_title(f'MC Scouting Report\nR²={r2_mc:.4f}  Coverage={coverage*100:.1f}%')
axes[0, 0].legend(fontsize=8)

axes[0, 1].scatter(mc_df['mc_mean'], mc_df['true_alpha'], alpha=0.4, s=15)
lims = [min(mc_df['mc_mean'].min(), mc_df['true_alpha'].min()),
        max(mc_df['mc_mean'].max(), mc_df['true_alpha'].max())]
axes[0, 1].plot(lims, lims, 'r--', lw=1, alpha=0.5)
axes[0, 1].set_title(f'Predicted vs True\nr={corr_mc:.3f}')

axes[1, 0].scatter(mc_df['stability'], mc_df['ci_width'], alpha=0.4, s=15,
                   color='darkorange')
axes[1, 0].set_title(f'Stability → CI Width\nr={r_stab_ci:.3f}  p={p_stab_ci:.4f}')

q_means = mc_df.groupby('stab_q', observed=True)['ci_width'].mean()
axes[1, 1].bar(range(len(q_means)), q_means.values, color='steelblue', alpha=0.8)
axes[1, 1].set_xticks(range(len(q_means)))
axes[1, 1].set_xticklabels([str(q) for q in q_means.index], fontsize=8)
axes[1, 1].set_title('CI Width by Stability Quintile')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'monte_carlo_scouting.png', dpi=150)
plt.close()
print(f"\n  Plot saved: monte_carlo_scouting.png")
mc_df.to_parquet(DATA_DIR / 'mc_scouting_results.parquet')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Transaction Cost Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: TRANSACTION COST ANALYSIS")
print("=" * 70)

signal_autocorrs = []
for ticker in np.unique(tickers_all)[:200]:
    mask   = tickers_all == ticker
    z_t    = Z_base[mask]
    if len(z_t) < 63:
        continue
    pred_a = pace_model.predict(z_t)
    if len(pred_a) < 10:
        continue
    ac1 = np.corrcoef(pred_a[:-1], pred_a[1:])[0, 1]
    if not np.isnan(ac1):
        signal_autocorrs.append(ac1)

mean_ac1     = float(np.mean(signal_autocorrs))
implied_hold = 1 / max(1 - mean_ac1, 0.01)
print(f"  Signal AC1: {mean_ac1:.3f}  implied hold: {implied_hold:.1f}d")

daily_alpha_std = float(np.nanstd(y_alpha_all))
ic              = np.sqrt(max(0.0355, 0))
ann_alpha_gross = ic * daily_alpha_std * np.sqrt(252)

print(f"  Daily alpha std: {daily_alpha_std:.5f}")
print(f"  IC: {ic:.4f}  Gross annual alpha: {ann_alpha_gross*100:.2f}%")
print(f"  Theoretical annual IR: {ic*np.sqrt(252):.2f}")

print(f"\n  {'Scenario':<28} {'Daily turn':>11} {'Ann TC':>8} "
      f"{'Net α':>10} {'OK?':>5}")
print(f"  {'─'*65}")

daily_turn = 1 / max(implied_hold, 1)
for tc_bps, label in [(5, 'Institutional 5bps'), (10, 'Retail 10bps'),
                       (20, 'High 20bps')]:
    ann_tc   = (tc_bps / 10000) * daily_turn * 252
    net_a    = ann_alpha_gross - ann_tc
    ok       = '✓' if net_a > 0 else '✗'
    print(f"  {label:<28} {daily_turn*100:>10.1f}% {ann_tc*100:>7.3f}% "
          f"{net_a*100:>9.3f}% {ok:>5}")

# Stable-only
stable_ac_list = []
stable_q1 = [t for t in np.unique(tickers_all)
             if t in stock_policies and
             stock_policies[t]['stability'] <= np.percentile(
                 [stock_policies[tt]['stability'] for tt in stock_policies], 25)]
for ticker in stable_q1[:100]:
    mask   = tickers_all == ticker
    z_t    = Z_base[mask]
    if len(z_t) < 63:
        continue
    pred_a = pace_model.predict(z_t)
    if len(pred_a) < 10:
        continue
    ac1 = np.corrcoef(pred_a[:-1], pred_a[1:])[0, 1]
    if not np.isnan(ac1):
        stable_ac_list.append(ac1)

if stable_ac_list:
    s_ac1   = float(np.mean(stable_ac_list))
    s_hold  = 1 / max(1 - s_ac1, 0.01)
    s_turn  = 1 / max(s_hold, 1)
    s_tc    = 0.0005 * s_turn * 252
    print(f"\n  Stable Q1 (5bps):        "
          f"hold={s_hold:.1f}d  TC={s_tc*100:.3f}%  "
          f"net={((ann_alpha_gross - s_tc)*100):.3f}%  "
          f"{'✓' if ann_alpha_gross > s_tc else '✗'}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY + SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY (sections 2-5)")
print("=" * 70)

print(f"\nFeature ablations:")
for k, v in ablation_results.items():
    if k.startswith('drop_'):
        print(f"  {k:<25} {v - r2_full:+.4f}")

print(f"\nRolling alpha R²:")
print(f"  Mean={np.mean(roll_r2s):.4f}  Std={np.std(roll_r2s):.4f}  "
      f"% > PCA: {(np.array(roll_r2s)>0.0039).mean()*100:.0f}%")

print(f"\nMC scouting:")
print(f"  R²={r2_mc:.4f}  Corr={corr_mc:.4f}  Coverage={coverage*100:.1f}%")
print(f"  Stab→CI r={r_stab_ci:.3f}  p={p_stab_ci:.4f}")

summary = {
    'feature_ablations': ablation_results,
    'rolling_r2': {
        'mean':          float(np.mean(roll_r2s)),
        'std':           float(np.std(roll_r2s)),
        'pct_above_pca': float((np.array(roll_r2s) > 0.0039).mean()),
        'windows':       rolling_results,
    },
    'monte_carlo': {
        'r2':        float(r2_mc),
        'corr':      float(corr_mc),
        'coverage':  float(coverage),
        'r_stab_ci': float(r_stab_ci),
        'p_stab_ci': float(p_stab_ci),
    },
    'signal': {
        'mean_ac1':         float(mean_ac1),
        'implied_hold':     float(implied_hold),
        'gross_alpha_pct':  float(ann_alpha_gross * 100),
        'daily_alpha_std':  float(daily_alpha_std),
    },
}

with open(DATA_DIR / 'ae_ablations_fast_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults: ae_ablations_fast_results.json")
print(f"Plots:   {PLOT_DIR}")
print("Done.")