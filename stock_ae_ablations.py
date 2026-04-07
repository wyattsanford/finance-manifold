# -*- coding: utf-8 -*-
"""
stock_ae_ablations.py
Ablation tests + Monte Carlo scouting report for the finance manifold.
Run after stock_ae_eval.py and stock_ae_extended.py.

Sections:
  1. Permutation tests — constraint profiles, cross-era stability
  2. Feature ablations — which features are load-bearing
  3. Rolling-window alpha R² — is signal concentrated in one period?
  4. Monte Carlo scouting report — per-stock alpha prediction with CIs
  5. Transaction cost analysis — back-of-envelope tradability
  (Architecture ablations → stock_ae_arch_ablations.py — run separately)
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
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
PLOT_DIR = DATA_DIR / 'plots' / 'ablations'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG      = np.random.RandomState(42)

N_PERMUTATIONS  = 500
MC_SAMPLES      = 2000
DIAG_SIGN_THRESH   = 0.15
DIAG_EFFECT_THRESH = 0.15
MIN_DIAG_AXES      = 3

print(f"Device: {DEVICE}")
print(f"Plots:  {PLOT_DIR}")

# ── Shared utilities ──────────────────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


def build_ae(input_dim, h1, h2, latent_dim):
    """Build AE with explicit hidden dims."""
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
        except Exception as e:
            pass
    return (np.vstack(Xs), np.concatenate(y_rets), np.concatenate(y_alphas),
            np.concatenate(all_dates), np.array(all_tickers))


def constraint_profile_stat(Z, dates, signal, n_quantiles=5, min_per_group=3):
    """Returns (sign_consistency, effect_size, n_dates, n_diagnostic)."""
    unique_dates = np.unique(dates)
    deltas = []
    for d in unique_dates:
        mask_d = dates == d
        sig_d  = signal[mask_d]
        Z_d    = Z[mask_d]
        valid  = ~np.isnan(sig_d)
        if valid.sum() < n_quantiles * min_per_group:
            continue
        sig_v  = sig_d[valid]
        Z_v    = Z_d[valid]
        ranks  = pd.Series(sig_v).rank(pct=True)
        qlabels= pd.cut(ranks, bins=n_quantiles, labels=range(1, n_quantiles + 1))
        top    = qlabels == n_quantiles
        bot    = qlabels == 1
        if top.sum() < min_per_group or bot.sum() < min_per_group:
            continue
        deltas.append(Z_v[top.values].mean(axis=0) - Z_v[bot.values].mean(axis=0))
    if len(deltas) < 50:
        return None
    deltas      = np.array(deltas)
    sign_cons   = (deltas > 0).mean(axis=0)
    effect_size = np.abs(deltas.mean(axis=0)) / (deltas.std(axis=0) + 1e-8)
    diagnostic  = (np.abs(sign_cons - 0.5) >= DIAG_SIGN_THRESH) & \
                  (effect_size >= DIAG_EFFECT_THRESH)
    return sign_cons, effect_size, len(deltas), int(diagnostic.sum())


# ── Load base data ─────────────────────────────────────────────────────────────
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

# ── Encode with cache ──────────────────────────────────────────────────────────
Z_CACHE_PATH = DATA_DIR / 'ablations_Z_base_cache.npy'
if Z_CACHE_PATH.exists():
    print("  Loading Z_base from cache...")
    Z_base = np.load(Z_CACHE_PATH)
else:
    X_all_s = scaler_base.transform(X_all)
    Z_base  = encode(model_base, X_all_s)
    np.save(Z_CACHE_PATH, Z_base)
    print(f"  Saved Z_base cache: {Z_CACHE_PATH.name}")

print(f"  {len(X_all):,} obs  |  {len(np.unique(tickers_all))} stocks  |  "
      f"Z shape: {Z_base.shape}")

# Real constraint profile stats on FULL data — never subsampled
real_result = constraint_profile_stat(Z_base, dates_all, y_alpha_all)
real_sc, real_es, real_ndates, real_ndiag = real_result
real_profile_vec = (real_sc - 0.5) * real_es

# ── Permutation subsample — used ONLY in permutation loops ────────────────────
# Real results stay on full 6M obs. Null distribution built on 500k subsample.
# Covers ~90 stocks/day on average — more than enough for constraint profiles.
PERM_N        = 500_000
perm_idx      = RNG.choice(len(Z_base), PERM_N, replace=False)
Z_perm        = Z_base[perm_idx]
dates_perm    = dates_all[perm_idx]
alpha_perm    = y_alpha_all[perm_idx]
ret_perm      = y_ret_all[perm_idx]

# Subsample feature signals for cross-relation permutation test
relation_signals_perm = {'alpha': alpha_perm, 'ret': ret_perm}
if 'vol_21d' in feature_cols:
    relation_signals_perm['vol_low']  = -X_all[perm_idx, feature_cols.index('vol_21d')]
if 'ret_63d' in feature_cols:
    relation_signals_perm['momentum'] = X_all[perm_idx, feature_cols.index('ret_63d')]
if 'drawdown' in feature_cols:
    relation_signals_perm['drawdown'] = -X_all[perm_idx, feature_cols.index('drawdown')]

print(f"  Permutation subsample: {PERM_N:,} obs  "
      f"({len(np.unique(dates_perm))} unique dates)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Permutation Tests
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 1: PERMUTATION TESTS")
print("=" * 80)

# ── 1a: Permutation test on constraint profile ────────────────────────────────
print("\n1a. Constraint profile permutation test")
print(f"    Real n_diagnostic: {real_ndiag}/12")
print(f"    Running {N_PERMUTATIONS} permutations...")

perm_ndiag      = []
perm_max_sc_dev = []

for perm_i in range(N_PERMUTATIONS):
    shuffled_alpha = RNG.permutation(alpha_perm)
    result = constraint_profile_stat(Z_perm, dates_perm, shuffled_alpha)
    if result is None:
        continue
    sc, es, _, nd = result
    perm_ndiag.append(nd)
    perm_max_sc_dev.append(np.abs(sc - 0.5).max())
    if (perm_i + 1) % 100 == 0:
        print(f"    {perm_i+1}/{N_PERMUTATIONS} permutations done...")

perm_ndiag      = np.array(perm_ndiag)
perm_max_sc_dev = np.array(perm_max_sc_dev)

p_ndiag = (perm_ndiag >= real_ndiag).mean()
print(f"    Null mean n_diagnostic: {perm_ndiag.mean():.2f} ± {perm_ndiag.std():.2f}")
print(f"    Real n_diagnostic:      {real_ndiag}")
print(f"    p-value:                {p_ndiag:.4f}")
print(f"    Z-score:                {(real_ndiag - perm_ndiag.mean()) / perm_ndiag.std():.2f}σ")

real_max_sc_dev = np.abs(real_sc - 0.5).max()
p_sc_dev = (perm_max_sc_dev >= real_max_sc_dev).mean()
print(f"    Real max |SC-0.5|:      {real_max_sc_dev:.3f}")
print(f"    Null max |SC-0.5|:      {perm_max_sc_dev.mean():.3f} ± {perm_max_sc_dev.std():.3f}")
print(f"    p-value (max deviation): {p_sc_dev:.4f}")

# ── 1b: Permutation test on cross-era stability ───────────────────────────────
print("\n1b. Cross-era constraint stability permutation test")
print(f"    Real cross-era similarity: 0.940")
print(f"    Running {N_PERMUTATIONS} permutations (shuffle dates within era)...")

eras = [
    ('Pre-GFC',    '2000-01-01', '2007-12-31'),
    ('GFC',        '2008-01-01', '2009-12-31'),
    ('Recovery',   '2010-01-01', '2014-12-31'),
    ('Bull',       '2015-01-01', '2019-12-31'),
    ('COVID',      '2020-01-01', '2020-12-31'),
    ('Post-COVID', '2021-01-01', '2022-12-31'),
]

def cross_era_similarity(Z, dates, alpha, eras):
    """Compute mean cross-era cosine similarity of constraint profiles."""
    vecs = []
    for era_name, start, end in eras:
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        if mask.sum() < 5000:
            continue
        result = constraint_profile_stat(Z[mask], dates[mask], alpha[mask])
        if result is None:
            continue
        sc, es, _, _ = result
        vecs.append((sc - 0.5) * es)
    if len(vecs) < 3:
        return np.nan
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            v1, v2 = vecs[i], vecs[j]
            denom  = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom > 0:
                sims.append(np.dot(v1, v2) / denom)
    return np.mean(sims) if sims else np.nan

perm_era_sims = []
for perm_i in range(N_PERMUTATIONS):
    shuffled = RNG.permutation(alpha_perm)
    sim      = cross_era_similarity(Z_perm, dates_perm, shuffled, eras)
    if not np.isnan(sim):
        perm_era_sims.append(sim)
    if (perm_i + 1) % 100 == 0:
        print(f"    {perm_i+1}/{N_PERMUTATIONS} permutations done...")

perm_era_sims = np.array(perm_era_sims)
real_era_sim  = 0.940
p_era_sim     = (perm_era_sims >= real_era_sim).mean()
print(f"    Null mean cross-era sim: {perm_era_sims.mean():.3f} ± {perm_era_sims.std():.3f}")
print(f"    Real cross-era sim:      {real_era_sim:.3f}")
print(f"    p-value:                 {p_era_sim:.4f}")
print(f"    Z-score:                 "
      f"{(real_era_sim - perm_era_sims.mean()) / perm_era_sims.std():.2f}σ")

# ── 1c: Permutation test on cross-relation orthogonality ─────────────────────
print("\n1c. Cross-relation orthogonality permutation test")
print(f"    Real mean off-diagonal similarity: -0.018")

# Build constraint vectors for all relations using real signals
relation_signals = {'alpha': y_alpha_all, 'ret': y_ret_all}
if 'vol_21d' in feature_cols:
    relation_signals['vol_low']  = -X_all[:, feature_cols.index('vol_21d')]
if 'ret_63d' in feature_cols:
    relation_signals['momentum'] = X_all[:, feature_cols.index('ret_63d')]
if 'drawdown' in feature_cols:
    relation_signals['drawdown'] = -X_all[:, feature_cols.index('drawdown')]

def mean_cross_sim(Z, dates, signals_dict):
    vecs = {}
    for name, sig in signals_dict.items():
        result = constraint_profile_stat(Z, dates, sig)
        if result is None:
            continue
        sc, es, _, _ = result
        vecs[name] = (sc - 0.5) * es
    names = list(vecs.keys())
    if len(names) < 3:
        return np.nan
    sims = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v1, v2 = vecs[names[i]], vecs[names[j]]
            denom  = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom > 0:
                sims.append(np.dot(v1, v2) / denom)
    return np.mean(sims) if sims else np.nan

# Permute each signal independently — use subsampled arrays
perm_cross_sims = []
for perm_i in range(min(200, N_PERMUTATIONS)):
    shuffled = {name: RNG.permutation(sig)
                for name, sig in relation_signals_perm.items()}
    sim = mean_cross_sim(Z_perm, dates_perm, shuffled)
    if not np.isnan(sim):
        perm_cross_sims.append(sim)
    if (perm_i + 1) % 50 == 0:
        print(f"    {perm_i+1}/200 permutations done...")

perm_cross_sims = np.array(perm_cross_sims)
real_cross_sim  = -0.018
p_cross_sim     = (np.abs(perm_cross_sims) <= np.abs(real_cross_sim)).mean()
print(f"    Null mean cross-sim:  {perm_cross_sims.mean():.3f} ± {perm_cross_sims.std():.3f}")
print(f"    Real cross-sim:       {real_cross_sim:.3f}")
print(f"    p (|real| ≤ |null|):  {p_cross_sim:.4f}  "
      f"(low = real IS more orthogonal than chance)")

# Plot permutation distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(perm_ndiag, bins=20, color='steelblue', alpha=0.7, edgecolor='none')
axes[0].axvline(real_ndiag, color='red', lw=2, label=f'Real={real_ndiag}')
axes[0].set_xlabel('N diagnostic axes')
axes[0].set_title(f'Constraint Profile\np={p_ndiag:.4f}')
axes[0].legend()

axes[1].hist(perm_era_sims, bins=30, color='darkorange', alpha=0.7, edgecolor='none')
axes[1].axvline(real_era_sim, color='red', lw=2, label=f'Real={real_era_sim:.3f}')
axes[1].set_xlabel('Cross-era similarity')
axes[1].set_title(f'Temporal Stability\np={p_era_sim:.4f}')
axes[1].legend()

axes[2].hist(perm_cross_sims, bins=30, color='purple', alpha=0.7, edgecolor='none')
axes[2].axvline(real_cross_sim, color='red', lw=2, label=f'Real={real_cross_sim:.3f}')
axes[2].set_xlabel('Cross-relation similarity')
axes[2].set_title(f'Cross-Relation Orthogonality\np={p_cross_sim:.4f}')
axes[2].legend()

plt.tight_layout()
plt.savefig(PLOT_DIR / 'permutation_tests.png', dpi=150)
plt.close()
print(f"\n  Plot saved: permutation_tests.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Feature Ablations
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 2: FEATURE ABLATIONS")
print("=" * 80)
print("Which feature groups are load-bearing for the manifold structure?")

# Define feature groups
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

# Remove empty groups
feature_groups = {k: v for k, v in feature_groups.items() if v}

print(f"\n  Feature groups:")
for gname, gcols in feature_groups.items():
    print(f"    {gname:<15} {len(gcols)} features: {', '.join(gcols[:4])}"
          f"{'...' if len(gcols) > 4 else ''}")

def quick_alpha_r2(X, y_alpha, latent_d=12):
    """Train a small AE and probe alpha R² — used for ablations."""
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
    opt       = torch.optim.AdamW(model_abl.parameters(), lr=1e-3, weight_decay=1e-4)
    crit      = nn.HuberLoss(delta=1.0)

    tr_loader = DataLoader(StockDataset(Xtr), batch_size=4096, shuffle=True)
    best_loss = float('inf')
    patience  = 0

    for epoch in range(50):
        model_abl.train()
        for xb in tr_loader:
            opt.zero_grad()
            xr, _ = model_abl(xb.to(DEVICE))
            loss   = crit(xr, xb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model_abl.parameters(), 1.0)
            opt.step()

        model_abl.eval()
        with torch.no_grad():
            xvt      = torch.FloatTensor(Xvl).to(DEVICE)
            vl_loss  = crit(*[model_abl(xvt)[0], xvt]).item()
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

print(f"\n  Baseline (all features): alpha R² ≈ 0.0355 (from CV)")
print(f"\n  {'Ablation':<25} {'Features':>8} {'Alpha R²':>10} {'vs baseline':>12}")
print(f"  {'─'*58}")

ablation_results = {}

# Subsample for speed
abl_idx   = RNG.choice(len(X_all), min(500_000, len(X_all)), replace=False)
X_abl     = X_all[abl_idx]
y_abl     = y_alpha_all[abl_idx]

# Full feature baseline (quick retrain)
r2_full = quick_alpha_r2(X_abl, y_abl)
print(f"  {'full (retrained)':<25} {X_abl.shape[1]:>8} {r2_full:>10.4f}  {'baseline':>12}")
ablation_results['full'] = r2_full

# Leave-one-group-out
for gname, gcols in feature_groups.items():
    drop_idx   = [feature_cols.index(c) for c in gcols if c in feature_cols]
    keep_idx   = [i for i in range(len(feature_cols)) if i not in drop_idx]
    X_dropped  = X_abl[:, keep_idx]
    r2_dropped = quick_alpha_r2(X_dropped, y_abl)
    delta      = r2_dropped - r2_full
    ablation_results[f'drop_{gname}'] = r2_dropped
    direction  = '▼' if delta < -0.002 else '▲' if delta > 0.002 else '≈'
    print(f"  {'drop_'+gname:<25} {X_dropped.shape[1]:>8} {r2_dropped:>10.4f}  "
          f"{delta:>+10.4f} {direction}")

# Single-group-only
for gname, gcols in feature_groups.items():
    keep_idx   = [feature_cols.index(c) for c in gcols if c in feature_cols]
    if len(keep_idx) < 2:
        continue
    X_only     = X_abl[:, keep_idx]
    r2_only    = quick_alpha_r2(X_only, y_abl)
    ablation_results[f'only_{gname}'] = r2_only
    print(f"  {'only_'+gname:<25} {X_only.shape[1]:>8} {r2_only:>10.4f}  "
          f"{'(single group)':>12}")

# Plot feature importance
drop_names   = [k for k in ablation_results if k.startswith('drop_')]
drop_deltas  = [ablation_results[k] - r2_full for k in drop_names]
clean_names  = [k.replace('drop_', '') for k in drop_names]

fig, ax = plt.subplots(figsize=(9, 4))
colors   = ['crimson' if d < -0.002 else 'steelblue' for d in drop_deltas]
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
print("\n" + "=" * 80)
print("SECTION 3: ROLLING-WINDOW ALPHA R²")
print("=" * 80)
print("Is the signal concentrated in one period or consistent across time?")

WINDOW_YEARS = 3
STEP_MONTHS  = 6

unique_years = sorted(set(pd.Timestamp(d).year for d in dates_all))
min_year     = min(unique_years)
max_year     = max(unique_years)

print(f"\n  Window: {WINDOW_YEARS}yr train → 6mo test, stepping every {STEP_MONTHS} months")
print(f"  Date range: {min_year} — {max_year}")
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

    # Encode both windows with base model (already trained on pre-2023)
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

    # Step forward by STEP_MONTHS
    start_year_dt = train_start + pd.DateOffset(months=STEP_MONTHS)
    start_year    = start_year_dt.year
    if start_year_dt.month > 1:
        # approximate by moving to next half-year
        start_year = start_year_dt.year + (1 if start_year_dt.month > 6 else 0)

roll_r2s = [r['r2'] for r in rolling_results]
print(f"\n  Mean rolling R²:  {np.mean(roll_r2s):.4f}")
print(f"  Std rolling R²:   {np.std(roll_r2s):.4f}")
print(f"  Min rolling R²:   {np.min(roll_r2s):.4f}")
print(f"  Max rolling R²:   {np.max(roll_r2s):.4f}")
print(f"  % windows > 0:    {(np.array(roll_r2s) > 0).mean()*100:.1f}%")
print(f"  % windows > PCA:  {(np.array(roll_r2s) > 0.0039).mean()*100:.1f}%")

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
test_dates = [pd.Timestamp(r['test_start']) for r in rolling_results]
ax.plot(test_dates, roll_r2s, 'o-', color='steelblue', lw=1.5, ms=5)
ax.axhline(0,      color='black', lw=0.8, linestyle='--', label='R²=0')
ax.axhline(0.0039, color='red',   lw=1,   linestyle='--', label='PCA baseline')
ax.axhline(np.mean(roll_r2s), color='steelblue', lw=1, linestyle=':',
           label=f'Mean={np.mean(roll_r2s):.4f}')

# Shade crisis periods
for name, start, end in [('GFC', '2008-09-01', '2009-03-31'),
                          ('COVID', '2020-02-01', '2020-06-30'),
                          ('Rate', '2022-01-01', '2022-12-31')]:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.1, color='red')

ax.set_xlabel('Test period start')
ax.set_ylabel('Alpha R²')
ax.set_title('Rolling-Window Alpha R² (3yr train → 6mo test)\nRed shading = crisis periods')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'rolling_alpha_r2.png', dpi=150)
plt.close()
print(f"\n  Plot saved: rolling_alpha_r2.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Monte Carlo Scouting Report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 5: MONTE CARLO SCOUTING REPORT")
print("=" * 80)
print("Per-stock alpha prediction with calibrated confidence intervals.")
print("Analog: F1 driver scouting — predicted pace ± CI from policy stability.")

# Build per-stock policy distributions from pre-2023 data
print("\nBuilding per-stock policy distributions...")
stock_policies = {}
for ticker in np.unique(tickers_all):
    mask = tickers_all == ticker
    if mask.sum() < 63:
        continue
    z_t = Z_base[mask]
    stock_policies[ticker] = {
        'centroid':  z_t.mean(axis=0),
        'cov':       np.cov(z_t.T) + np.eye(latent_dim) * 1e-4,
        'stability': float(z_t.var(axis=0).mean()),
        'n_obs':     int(mask.sum()),
        'mean_alpha': float(np.nanmean(y_alpha_all[mask])),
    }

print(f"  Stocks with policy distributions: {len(stock_policies)}")

# Fit pace model: centroid → mean alpha (Ridge regression)
# Use 80% of stocks for fitting, 20% for evaluation
ticker_list  = list(stock_policies.keys())
RNG.shuffle(ticker_list)
n_fit        = int(len(ticker_list) * 0.8)
fit_tickers  = ticker_list[:n_fit]
eval_tickers = ticker_list[n_fit:]

centroids_fit  = np.array([stock_policies[t]['centroid']  for t in fit_tickers])
alphas_fit     = np.array([stock_policies[t]['mean_alpha'] for t in fit_tickers])

pace_model = Ridge(alpha=1.0)
pace_model.fit(centroids_fit, alphas_fit)

# Monte Carlo: sample from policy distribution, predict, get CI
print(f"\nRunning Monte Carlo ({MC_SAMPLES} samples per stock)...")

mc_results = []
for ticker in eval_tickers:
    p      = stock_policies[ticker]
    mu     = p['centroid']
    cov    = p['cov']
    stab   = p['stability']
    true_a = p['mean_alpha']
    n_obs  = p['n_obs']

    # Sample from policy distribution
    try:
        samples = RNG.multivariate_normal(mu, cov, size=MC_SAMPLES)
    except np.linalg.LinAlgError:
        samples = mu + RNG.randn(MC_SAMPLES, latent_dim) * np.sqrt(np.diag(cov))

    # Predict alpha for each sample
    pred_alphas = pace_model.predict(samples)

    mc_mean  = float(pred_alphas.mean())
    mc_std   = float(pred_alphas.std())
    mc_ci_lo = float(np.percentile(pred_alphas, 2.5))
    mc_ci_hi = float(np.percentile(pred_alphas, 97.5))
    within_ci= bool(mc_ci_lo <= true_a <= mc_ci_hi)

    mc_results.append({
        'ticker':    ticker,
        'stability': stab,
        'n_obs':     n_obs,
        'true_alpha': true_a,
        'mc_mean':   mc_mean,
        'mc_std':    mc_std,
        'ci_lo':     mc_ci_lo,
        'ci_hi':     mc_ci_hi,
        'ci_width':  mc_ci_hi - mc_ci_lo,
        'within_ci': within_ci,
        'error':     abs(mc_mean - true_alpha),
    })

mc_df = pd.DataFrame(mc_results)

# Summary stats
r2_mc    = float(r2_score(mc_df['true_alpha'], mc_df['mc_mean']))
corr_mc  = float(pearsonr(mc_df['true_alpha'], mc_df['mc_mean'])[0])
coverage = float(mc_df['within_ci'].mean())
r_stab_ci= float(pearsonr(mc_df['stability'], mc_df['ci_width'])[0])
p_stab_ci= float(pearsonr(mc_df['stability'], mc_df['ci_width'])[1])

print(f"\n  MC pace model (centroid → mean alpha):")
print(f"    R²:           {r2_mc:.4f}")
print(f"    Correlation:  {corr_mc:.4f}")
print(f"    95% CI coverage: {coverage*100:.1f}%  (target: 95%)")
print(f"    Stocks within CI: {mc_df['within_ci'].sum()}/{len(mc_df)}")
print(f"    Stability → CI width: r={r_stab_ci:.3f}  p={p_stab_ci:.4f}")
print(f"    (F1 result: r=0.771, p<0.0001)")

# Top 10 most stable — tightest CIs
print(f"\n  Most stable stocks (tightest CIs):")
top_stable = mc_df.nsmallest(10, 'stability')
print(f"  {'Ticker':<8} {'Stability':>10} {'MC mean':>10} {'True α':>10} "
      f"{'CI width':>10} {'Within?':>8}")
print(f"  {'─'*60}")
for _, row in top_stable.iterrows():
    print(f"  {row['ticker']:<8} {row['stability']:>10.4f} {row['mc_mean']:>10.5f} "
          f"{row['true_alpha']:>10.5f} {row['ci_width']:>10.5f} "
          f"{'✓' if row['within_ci'] else '✗':>8}")

# Top 10 most unstable — widest CIs
print(f"\n  Most unstable stocks (widest CIs):")
top_unstable = mc_df.nlargest(10, 'stability')
print(f"  {'Ticker':<8} {'Stability':>10} {'MC mean':>10} {'True α':>10} "
      f"{'CI width':>10} {'Within?':>8}")
print(f"  {'─'*60}")
for _, row in top_unstable.iterrows():
    print(f"  {row['ticker']:<8} {row['stability']:>10.4f} {row['mc_mean']:>10.5f} "
          f"{row['true_alpha']:>10.5f} {row['ci_width']:>10.5f} "
          f"{'✓' if row['within_ci'] else '✗':>8}")

# Stability quintile breakdown
mc_df['stab_q'] = pd.qcut(mc_df['stability'], 5,
                            labels=['Q1\n(stable)', 'Q2', 'Q3', 'Q4', 'Q5\n(unstable)'])
print(f"\n  MC results by stability quintile:")
print(f"  {'Quintile':<14} {'Mean CI':>10} {'Coverage':>10} {'Mean |error|':>14}")
print(f"  {'─'*50}")
for q in mc_df['stab_q'].cat.categories:
    sub = mc_df[mc_df['stab_q'] == q]
    print(f"  {str(q):<14} {sub['ci_width'].mean():>10.5f} "
          f"{sub['within_ci'].mean()*100:>9.1f}% "
          f"{sub['error'].mean():>14.5f}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# MC scouting report — predicted vs true with CI bars
sorted_mc = mc_df.sort_values('mc_mean').reset_index(drop=True)
axes[0, 0].errorbar(range(len(sorted_mc)), sorted_mc['mc_mean'],
                    yerr=[sorted_mc['mc_mean'] - sorted_mc['ci_lo'],
                          sorted_mc['ci_hi'] - sorted_mc['mc_mean']],
                    fmt='none', alpha=0.3, color='steelblue', lw=0.5)
axes[0, 0].scatter(range(len(sorted_mc)), sorted_mc['true_alpha'],
                   s=8, color='red', alpha=0.5, zorder=3, label='True α')
axes[0, 0].scatter(range(len(sorted_mc)), sorted_mc['mc_mean'],
                   s=8, color='steelblue', alpha=0.5, zorder=3, label='MC mean')
axes[0, 0].set_xlabel('Stock (sorted by predicted α)')
axes[0, 0].set_ylabel('Alpha')
axes[0, 0].set_title(f'MC Scouting Report\nR²={r2_mc:.4f}, coverage={coverage*100:.1f}%')
axes[0, 0].legend(fontsize=8)

# Predicted vs true scatter
axes[0, 1].scatter(mc_df['mc_mean'], mc_df['true_alpha'], alpha=0.4, s=15)
lims = [min(mc_df['mc_mean'].min(), mc_df['true_alpha'].min()),
        max(mc_df['mc_mean'].max(), mc_df['true_alpha'].max())]
axes[0, 1].plot(lims, lims, 'r--', lw=1, alpha=0.5)
axes[0, 1].set_xlabel('MC predicted alpha')
axes[0, 1].set_ylabel('True alpha')
axes[0, 1].set_title(f'Predicted vs True\nr={corr_mc:.3f}')

# Stability → CI width
axes[1, 0].scatter(mc_df['stability'], mc_df['ci_width'], alpha=0.4, s=15,
                   color='darkorange')
axes[1, 0].set_xlabel('Manifold Stability')
axes[1, 0].set_ylabel('CI Width')
axes[1, 0].set_title(f'Stability → CI Width\nr={r_stab_ci:.3f}, p={p_stab_ci:.4f}')

# CI width by quintile
q_means = mc_df.groupby('stab_q', observed=True)['ci_width'].mean()
axes[1, 1].bar(range(len(q_means)), q_means.values, color='steelblue', alpha=0.8)
axes[1, 1].set_xticks(range(len(q_means)))
axes[1, 1].set_xticklabels([str(q) for q in q_means.index], fontsize=8)
axes[1, 1].set_ylabel('Mean CI Width')
axes[1, 1].set_title('CI Width by Stability Quintile\n(Analog: F1 driver scouting report)')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'monte_carlo_scouting.png', dpi=150)
plt.close()
print(f"\n  Plot saved: monte_carlo_scouting.png")

mc_df.to_parquet(DATA_DIR / 'mc_scouting_results.parquet')
print(f"  MC results saved: mc_scouting_results.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Transaction Cost Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6: TRANSACTION COST ANALYSIS")
print("=" * 80)
print("Back-of-envelope: does the alpha signal survive trading costs?")

# Use rolling alpha R² and a simple signal → turnover → cost model
# Assumptions (conservative):
#   - Signal rebalanced daily
#   - One-way transaction cost: 5bps (institutional), 10bps (retail)
#   - Average holding period implied by autocorrelation of signal

# Compute signal autocorrelation
print("\nComputing signal autocorrelation...")
signal_autocorrs = []
for ticker in np.unique(tickers_all)[:200]:
    mask   = tickers_all == ticker
    z_t    = Z_base[mask]
    if len(z_t) < 63:
        continue
    # Project onto pace model to get daily predicted alpha
    pred_a = pace_model.predict(z_t)
    if len(pred_a) < 10:
        continue
    ac1 = np.corrcoef(pred_a[:-1], pred_a[1:])[0, 1]
    signal_autocorrs.append(ac1)

mean_ac1 = np.mean(signal_autocorrs)
# Implied average holding period from AR(1): 1 / (1 - ac1) days
implied_hold = 1 / max(1 - mean_ac1, 0.01)

print(f"  Mean signal lag-1 autocorrelation: {mean_ac1:.3f}")
print(f"  Implied avg holding period:        {implied_hold:.1f} days")

# Annual alpha from R²
# R² ≈ 0.035, daily alpha std ≈ 0.01, so IC ≈ sqrt(0.035) ≈ 0.187
# Annualized information ratio ≈ IC * sqrt(252) ≈ 2.97 (theoretical max)
daily_alpha_std = float(np.nanstd(y_alpha_all))
ic              = np.sqrt(max(0.0355, 0))   # information coefficient
ann_ir_theory   = ic * np.sqrt(252)
ann_alpha_gross = ic * daily_alpha_std * np.sqrt(252)

print(f"\n  Signal quality (from CV alpha R²=0.0355):")
print(f"    Daily alpha std:         {daily_alpha_std:.5f}")
print(f"    Information coefficient: {ic:.4f}  (= sqrt(R²))")
print(f"    Theoretical annual IR:   {ann_ir_theory:.2f}")
print(f"    Gross annual alpha:      {ann_alpha_gross*100:.2f}%")

# Transaction costs
for tc_bps, label in [(5, 'institutional (5bps)'), (10, 'retail (10bps)'),
                       (20, 'high (20bps)')]:
    tc_daily    = tc_bps / 10000
    # Daily turnover ≈ 1/holding_period (fraction of portfolio turned over)
    daily_turn  = 1 / max(implied_hold, 1)
    ann_tc      = tc_daily * daily_turn * 252
    net_alpha   = ann_alpha_gross - ann_tc
    print(f"\n  {label}:")
    print(f"    Daily turnover:    {daily_turn*100:.1f}%")
    print(f"    Annual TC drag:    {ann_tc*100:.3f}%")
    print(f"    Net annual alpha:  {net_alpha*100:.3f}%  "
          f"({'profitable' if net_alpha > 0 else 'not profitable'})")

# Stable-only portfolio (lower turnover due to lower signal variance)
print(f"\n  Stable-only portfolio (Q1 stability, tightest CIs):")
stable_mask = np.array([stock_policies[t]['stability']
                         for t in tickers_all
                         if t in stock_policies]) if False else None

stable_autocorrs = []
stable_stab_q1 = [t for t in np.unique(tickers_all)
                   if t in stock_policies and
                   stock_policies[t]['stability'] <= np.percentile(
                       [stock_policies[tt]['stability'] for tt in stock_policies], 25)]

for ticker in stable_stab_q1[:100]:
    mask   = tickers_all == ticker
    z_t    = Z_base[mask]
    if len(z_t) < 63:
        continue
    pred_a = pace_model.predict(z_t)
    if len(pred_a) < 10:
        continue
    ac1 = np.corrcoef(pred_a[:-1], pred_a[1:])[0, 1]
    stable_autocorrs.append(ac1)

if stable_autocorrs:
    stable_ac1    = np.mean(stable_autocorrs)
    stable_hold   = 1 / max(1 - stable_ac1, 0.01)
    stable_turn   = 1 / max(stable_hold, 1)
    stable_ann_tc = 0.0005 * stable_turn * 252   # 5bps institutional
    print(f"    Signal autocorrelation: {stable_ac1:.3f}  (vs {mean_ac1:.3f} full universe)")
    print(f"    Implied holding period: {stable_hold:.1f} days")
    print(f"    Annual TC drag (5bps):  {stable_ann_tc*100:.3f}%")
    print(f"    Net alpha estimate:     {(ann_alpha_gross - stable_ann_tc)*100:.3f}%")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ABLATION SUMMARY  (architecture ablations in stock_ae_arch_ablations.py)")
print("=" * 80)

print(f"\nPermutation tests:")
print(f"  Constraint profile:      p={p_ndiag:.4f}  Z={((real_ndiag-perm_ndiag.mean())/perm_ndiag.std()):.1f}σ")
print(f"  Temporal stability:      p={p_era_sim:.4f}")
print(f"  Cross-relation orthog:   p={p_cross_sim:.4f}")

print(f"\nFeature ablations (drop impact on alpha R²):")
for k, v in ablation_results.items():
    if k.startswith('drop_'):
        delta = v - r2_full
        print(f"  {k:<25} {delta:+.4f}")

print(f"\nArchitecture ablations: see stock_ae_arch_ablations.py")

print(f"\nMonte Carlo scouting:")
print(f"  R²={r2_mc:.4f}  corr={corr_mc:.4f}  coverage={coverage*100:.1f}%")
print(f"  Stability → CI width: r={r_stab_ci:.3f}  p={p_stab_ci:.4f}")

print(f"\nRolling alpha R²:")
print(f"  Mean={np.mean(roll_r2s):.4f}  Std={np.std(roll_r2s):.4f}  "
      f"% > PCA baseline: {(np.array(roll_r2s)>0.0039).mean()*100:.0f}%")

# Save
summary = {
    'permutation': {
        'constraint_p': float(p_ndiag),
        'era_sim_p':    float(p_era_sim),
        'cross_rel_p':  float(p_cross_sim),
    },
    'rolling_r2': {
        'mean': float(np.mean(roll_r2s)),
        'std':  float(np.std(roll_r2s)),
        'pct_above_pca': float((np.array(roll_r2s) > 0.0039).mean()),
        'windows': rolling_results,
    },
    'monte_carlo': {
        'r2':       float(r2_mc),
        'corr':     float(corr_mc),
        'coverage': float(coverage),
        'r_stab_ci': float(r_stab_ci),
        'p_stab_ci': float(p_stab_ci),
    },
}

with open(DATA_DIR / 'ae_ablation_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults: ae_ablation_results.json")
print(f"Plots:   {PLOT_DIR}")
print("Done.")