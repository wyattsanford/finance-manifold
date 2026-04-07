# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:08:50 2026

@author: Justin.Sanford
"""

# -*- coding: utf-8 -*-
"""
stock_manifold_curiosities.py
Exploratory geometry tests on the finance manifold.
None of these are required for the paper — just interesting questions
about the shape of behavioral space.

Sections:
  1. Bankruptcy / delisting signal — does latent instability increase
     before a stock dies?
  2. Sector emergence — does policy distance matrix recover GICS sectors
     without labels?
  3. Latent velocity — rate of change through manifold as a signal
  4. Twin stocks — what are the nearest-neighbor pairs actually doing?
  5. Earnings surprise geometry — do beats/misses push in consistent directions?
  6. Contagion ordering — which stocks moved first in 2008/2020?

Run time: a few hours depending on how many delisted tickers yfinance has.
Saves:    plots/curiosities/  +  manifold_curiosities_results.json
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr, ttest_ind
from scipy.spatial.distance import cdist
import yfinance as yf
import requests
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
PLOT_DIR = DATA_DIR / 'plots' / 'curiosities'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG      = np.random.RandomState(42)

CUTOFF_DATE  = '2023-01-01'
HEADERS      = {'User-Agent': 'Justin Sanford justin.sanford@dlhcorp.com'}

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


def load_stock(ticker, feature_cols, date_filter=None):
    """Load a single stock's clean parquet and return X, dates."""
    f = DATA_DIR / f'stock_clean_{ticker}.parquet'
    if not f.exists():
        return None, None, None, None
    try:
        df    = pd.read_parquet(f)
        dates = pd.to_datetime(df.index)
        if date_filter is not None:
            direction, cutoff = date_filter
            ts   = pd.Timestamp(cutoff)
            mask = (dates < ts) if direction == 'before' else (dates >= ts)
            df, dates = df[mask], dates[mask]
        if len(df) < 63:
            return None, None, None, None
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X       = clean_X(df[feature_cols].values.astype(np.float32))
        y_alpha = df['alpha_resid'].values \
            if 'alpha_resid' in df.columns else np.zeros(len(df))
        return X, y_alpha, dates.values, df
    except Exception:
        return None, None, None, None


# ── Load model + base data ────────────────────────────────────────────────────
print("\nLoading temporal model...")
ckpt         = torch.load(DATA_DIR / 'ae_temporal_best.pt',
                           map_location=DEVICE, weights_only=False)
model, input_dim, latent_dim = build_model_from_ckpt(ckpt)
scaler       = ckpt['scaler']
feature_cols = ckpt['feature_cols']

manifest    = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

# Build per-stock centroid cache (pre-2023)
print("Building centroid cache...")
centroid_cache = {}
for ticker in all_tickers:
    X, y_alpha, dates, df = load_stock(ticker, feature_cols,
                                        date_filter=('before', CUTOFF_DATE))
    if X is None or len(X) < 63:
        continue
    z = encode_batch(model, scaler.transform(X))
    centroid_cache[ticker] = {
        'centroid':  z.mean(axis=0),
        'cov':       np.cov(z.T) + np.eye(latent_dim) * 1e-4,
        'stability': float(z.var(axis=0).mean()),
        'z_series':  z,        # full time series of latent positions
        'dates':     dates,
        'n_obs':     len(z),
    }

print(f"  Cached {len(centroid_cache)} stocks")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Bankruptcy / Delisting Signal
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1: BANKRUPTCY / DELISTING SIGNAL")
print("=" * 70)
print("Does latent instability increase before a stock dies?")

# Get delisted tickers from yfinance — try a known list of bankruptcies
# and supplement with manifest stocks whose data ends before 2023
print("\nIdentifying delisted stocks from manifest...")

delisted_candidates = []
for ticker in all_tickers:
    info = manifest[manifest['ticker'] == ticker]
    if len(info) == 0:
        continue
    end_date = pd.Timestamp(info['end'].values[0])
    # If data ends before 2022 it likely delisted
    if end_date < pd.Timestamp('2022-01-01'):
        delisted_candidates.append({
            'ticker':   ticker,
            'end_date': end_date,
        })

# Control: stocks that survived to 2023+
survivor_candidates = [t for t in all_tickers
                       if t not in [d['ticker'] for d in delisted_candidates]]

print(f"  Delisted candidates: {len(delisted_candidates)}")
print(f"  Survivors:           {len(survivor_candidates)}")

# For each delisted stock: compute rolling 63-day stability in the
# 12 months before delisting vs 12-24 months before
pre_death_stab   = []  # 0-12mo before delisting
control_stab     = []  # 12-24mo before delisting
survivor_stab    = []  # same window for survivors

WINDOW = 63
LOOKBACK_NEAR = 252    # 0-12mo before end
LOOKBACK_FAR  = 504    # 12-24mo before end

for item in delisted_candidates[:200]:
    ticker   = item['ticker']
    end_date = item['end_date']
    if ticker not in centroid_cache:
        continue

    z_series = centroid_cache[ticker]['z_series']
    dates    = centroid_cache[ticker]['dates']
    dates_ts = pd.DatetimeIndex(dates)

    near_mask = (dates_ts >= end_date - pd.DateOffset(days=LOOKBACK_NEAR)) & \
                (dates_ts < end_date)
    far_mask  = (dates_ts >= end_date - pd.DateOffset(days=LOOKBACK_FAR)) & \
                (dates_ts < end_date - pd.DateOffset(days=LOOKBACK_NEAR))

    if near_mask.sum() < WINDOW or far_mask.sum() < WINDOW:
        continue

    # Rolling stability: mean per-dim variance over window
    z_near = z_series[near_mask]
    z_far  = z_series[far_mask]

    stab_near = float(z_near.var(axis=0).mean())
    stab_far  = float(z_far.var(axis=0).mean())

    pre_death_stab.append(stab_near)
    control_stab.append(stab_far)

# Survivor baseline
for ticker in RNG.choice(survivor_candidates, min(200, len(survivor_candidates)),
                          replace=False):
    if ticker not in centroid_cache:
        continue
    z_series = centroid_cache[ticker]['z_series']
    if len(z_series) < LOOKBACK_NEAR:
        continue
    stab = float(z_series[-LOOKBACK_NEAR:].var(axis=0).mean())
    survivor_stab.append(stab)

pre_death_stab = np.array(pre_death_stab)
control_stab   = np.array(control_stab)
survivor_stab  = np.array(survivor_stab)

if len(pre_death_stab) > 10:
    t_stat, p_near_vs_far = ttest_ind(pre_death_stab, control_stab)
    t_stat2, p_near_vs_surv = ttest_ind(pre_death_stab, survivor_stab)

    print(f"\n  Delisted stocks (n={len(pre_death_stab)}):")
    print(f"    Stability 0-12mo before delisting:  {pre_death_stab.mean():.5f} ± {pre_death_stab.std():.5f}")
    print(f"    Stability 12-24mo before delisting: {control_stab.mean():.5f} ± {control_stab.std():.5f}")
    print(f"    p (near vs far):                    {p_near_vs_far:.4f}")
    print(f"\n  Survivor baseline (n={len(survivor_stab)}):")
    print(f"    Stability (same window):             {survivor_stab.mean():.5f} ± {survivor_stab.std():.5f}")
    print(f"    p (delisted near vs survivors):      {p_near_vs_surv:.4f}")

    # Does stability increase monotonically as delisting approaches?
    # Compute stability in 6 windows: 24-18mo, 18-12mo, 12-6mo, 6-0mo
    print(f"\n  Stability trajectory approaching delisting:")
    window_labels = ['24-18mo', '18-12mo', '12-6mo', '6-0mo']
    window_bounds = [(504, 378), (378, 252), (252, 126), (126, 0)]
    window_means  = []

    for label, (far, near) in zip(window_labels, window_bounds):
        stabs = []
        for item in delisted_candidates[:200]:
            ticker   = item['ticker']
            end_date = item['end_date']
            if ticker not in centroid_cache:
                continue
            z_series = centroid_cache[ticker]['z_series']
            dates    = centroid_cache[ticker]['dates']
            dates_ts = pd.DatetimeIndex(dates)
            mask = (dates_ts >= end_date - pd.DateOffset(days=far)) & \
                   (dates_ts <  end_date - pd.DateOffset(days=near))
            if mask.sum() < 21:
                continue
            stabs.append(z_series[mask].var(axis=0).mean())
        if stabs:
            m = np.mean(stabs)
            window_means.append(m)
            print(f"    {label}: {m:.5f}  (n={len(stabs)})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(pre_death_stab, bins=30, alpha=0.6, label='0-12mo before delisting',
                 color='crimson')
    axes[0].hist(survivor_stab,  bins=30, alpha=0.6, label='Survivors (same window)',
                 color='steelblue')
    axes[0].set_xlabel('Manifold Stability')
    axes[0].set_title(f'Stability Before Delisting\np={p_near_vs_surv:.4f}')
    axes[0].legend()

    if window_means:
        axes[1].plot(range(len(window_means)), window_means, 'o-',
                     color='crimson', lw=2, ms=8)
        axes[1].set_xticks(range(len(window_labels)))
        axes[1].set_xticklabels(window_labels)
        axes[1].set_ylabel('Mean stability')
        axes[1].set_title('Stability Trajectory → Delisting\n'
                          '(rising = increasing erratic behavior)')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'bankruptcy_signal.png', dpi=150)
    plt.close()
    print(f"  Plot saved: bankruptcy_signal.png")

    bankruptcy_results = {
        'n_delisted':           len(pre_death_stab),
        'n_survivors':          len(survivor_stab),
        'mean_stab_near':       float(pre_death_stab.mean()),
        'mean_stab_far':        float(control_stab.mean()),
        'mean_stab_survivor':   float(survivor_stab.mean()),
        'p_near_vs_far':        float(p_near_vs_far),
        'p_near_vs_survivor':   float(p_near_vs_surv),
    }
else:
    print("  Insufficient delisted stocks for analysis")
    bankruptcy_results = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Sector Emergence Without Labels
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: SECTOR EMERGENCE WITHOUT LABELS")
print("=" * 70)
print("Does policy distance matrix recover GICS sectors unsupervised?")
print("Analog: F1 teammate pairs clustering without team labels")

# Fetch sector info from yfinance for a sample of stocks
print("\nFetching sector labels from yfinance (this takes a few minutes)...")
sector_data = {}
sample_tickers = list(centroid_cache.keys())
RNG.shuffle(sample_tickers)

for ticker in sample_tickers[:400]:
    try:
        info   = yf.Ticker(ticker).info
        sector = info.get('sector', None)
        industry = info.get('industry', None)
        if sector and sector not in ['N/A', '', 'None']:
            sector_data[ticker] = {
                'sector':   sector,
                'industry': industry or sector,
            }
        time.sleep(0.05)
    except Exception:
        continue

print(f"  Sectors retrieved: {len(sector_data)}")

if len(sector_data) >= 50:
    sectors_found = {}
    for t, d in sector_data.items():
        s = d['sector']
        sectors_found[s] = sectors_found.get(s, 0) + 1

    print(f"  Sector distribution:")
    for s, n in sorted(sectors_found.items(), key=lambda x: -x[1]):
        print(f"    {s:<35} n={n}")

    # Build centroid matrix for labeled stocks
    labeled_tickers  = [t for t in sector_data if t in centroid_cache]
    labeled_centroids= np.array([centroid_cache[t]['centroid']
                                  for t in labeled_tickers])
    labeled_sectors  = [sector_data[t]['sector'] for t in labeled_tickers]

    # Encode sectors
    le             = LabelEncoder()
    sector_encoded = le.fit_transform(labeled_sectors)
    n_sectors      = len(le.classes_)

    # Test 1: KNN sector prediction from latent position
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    cv_scores = cross_val_score(knn, labeled_centroids, sector_encoded,
                                cv=5, scoring='accuracy')
    baseline  = 1 / n_sectors

    print(f"\n  KNN sector prediction (k=5, 5-fold CV):")
    print(f"    Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"    Baseline: {baseline:.3f}  ({n_sectors} sectors)")
    print(f"    Lift:     {cv_scores.mean()/baseline:.2f}x")

    # Test 2: Within-sector vs between-sector centroid distances
    sector_dists_within  = []
    sector_dists_between = []

    for i in range(len(labeled_tickers)):
        for j in range(i + 1, len(labeled_tickers)):
            d = np.linalg.norm(labeled_centroids[i] - labeled_centroids[j])
            if labeled_sectors[i] == labeled_sectors[j]:
                sector_dists_within.append(d)
            else:
                sector_dists_between.append(d)

    sector_dists_within  = np.array(sector_dists_within)
    sector_dists_between = np.array(sector_dists_between)

    t_stat, p_sector = ttest_ind(sector_dists_within, sector_dists_between)
    print(f"\n  Within-sector vs between-sector distances:")
    print(f"    Within:  {sector_dists_within.mean():.4f} ± {sector_dists_within.std():.4f}")
    print(f"    Between: {sector_dists_between.mean():.4f} ± {sector_dists_between.std():.4f}")
    print(f"    p-value: {p_sector:.4f}")
    print(f"    Ratio:   {sector_dists_between.mean()/sector_dists_within.mean():.3f}x")

    # Test 3: PCA visualization colored by sector
    pca = PCA(n_components=2)
    proj = pca.fit_transform(labeled_centroids)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = cm.tab10(np.linspace(0, 1, n_sectors))
    for i, sector in enumerate(le.classes_):
        mask = np.array(labeled_sectors) == sector
        axes[0].scatter(proj[mask, 0], proj[mask, 1],
                        color=colors[i], alpha=0.6, s=20, label=sector)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('Latent Space by Sector\n(no sector labels used in training)')
    axes[0].legend(fontsize=6, bbox_to_anchor=(1.05, 1))

    # Within vs between distance distribution
    axes[1].hist(sector_dists_within,  bins=50, alpha=0.6,
                 label='Within sector', color='steelblue', density=True)
    axes[1].hist(sector_dists_between, bins=50, alpha=0.6,
                 label='Between sector', color='crimson', density=True)
    axes[1].set_xlabel('Centroid distance')
    axes[1].set_title(f'Sector Separation\np={p_sector:.4f}  '
                      f'ratio={sector_dists_between.mean()/sector_dists_within.mean():.2f}x')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'sector_emergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: sector_emergence.png")

    sector_results = {
        'n_labeled':            len(labeled_tickers),
        'n_sectors':            n_sectors,
        'knn_accuracy':         float(cv_scores.mean()),
        'knn_baseline':         float(baseline),
        'knn_lift':             float(cv_scores.mean() / baseline),
        'within_dist_mean':     float(sector_dists_within.mean()),
        'between_dist_mean':    float(sector_dists_between.mean()),
        'p_sector':             float(p_sector),
        'dist_ratio':           float(sector_dists_between.mean() /
                                      sector_dists_within.mean()),
    }
else:
    print("  Insufficient sector data")
    sector_results = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Latent Velocity
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: LATENT VELOCITY")
print("=" * 70)
print("Rate of change through manifold — what does it predict?")

# Compute rolling 21-day latent velocity for each stock
# Velocity = mean displacement per day in latent space
print("\nComputing latent velocity time series...")

velocity_results = []
VEL_WINDOW = 21

for ticker in list(centroid_cache.keys())[:500]:
    cache  = centroid_cache[ticker]
    z      = cache['z_series']
    dates  = pd.DatetimeIndex(cache['dates'])

    if len(z) < VEL_WINDOW * 3:
        continue

    # Load alpha for this stock
    X, y_alpha, _, df = load_stock(ticker, feature_cols,
                                    date_filter=('before', CUTOFF_DATE))
    if X is None:
        continue
    y_alpha = y_alpha[:len(z)]

    # Rolling velocity: mean step size over window
    step_sizes = np.linalg.norm(np.diff(z, axis=0), axis=1)

    for i in range(VEL_WINDOW, len(z) - VEL_WINDOW):
        vel_now  = step_sizes[i-VEL_WINDOW:i].mean()
        vel_prev = step_sizes[max(0, i-VEL_WINDOW*2):i-VEL_WINDOW].mean() \
                   if i >= VEL_WINDOW * 2 else vel_now
        vel_accel = vel_now - vel_prev   # is velocity increasing?

        # Forward alpha (next 21 days)
        fwd_alpha = np.nanmean(y_alpha[i:i+VEL_WINDOW]) \
                    if i + VEL_WINDOW <= len(y_alpha) else np.nan

        # Forward volatility (realized vol next 21 days)
        if df is not None and 'ret_1d' in df.columns:
            ret_slice = df['ret_1d'].values[:len(z)]
            fwd_vol = np.nanstd(ret_slice[i:i+VEL_WINDOW]) \
                      if i + VEL_WINDOW <= len(ret_slice) else np.nan
        else:
            fwd_vol = np.nan

        velocity_results.append({
            'ticker':     ticker,
            'date':       dates[i],
            'velocity':   float(vel_now),
            'vel_accel':  float(vel_accel),
            'fwd_alpha':  fwd_alpha,
            'fwd_vol':    fwd_vol,
        })

vel_df = pd.DataFrame(velocity_results).dropna()
print(f"  Velocity observations: {len(vel_df):,}")

if len(vel_df) > 1000:
    # Velocity → forward volatility
    r_vel_vol, p_vel_vol = pearsonr(vel_df['velocity'], vel_df['fwd_vol'])
    # Velocity → forward alpha
    r_vel_alpha, p_vel_alpha = pearsonr(vel_df['velocity'], vel_df['fwd_alpha'])
    # Velocity acceleration → mean reversion
    r_acc_alpha, p_acc_alpha = pearsonr(vel_df['vel_accel'], vel_df['fwd_alpha'])

    print(f"\n  Velocity → forward volatility: r={r_vel_vol:.3f}  p={p_vel_vol:.4f}")
    print(f"  Velocity → forward alpha:      r={r_vel_alpha:.3f}  p={p_vel_alpha:.4f}")
    print(f"  Accel    → forward alpha:      r={r_acc_alpha:.3f}  p={p_acc_alpha:.4f}")

    # Velocity quintile breakdown
    vel_df['vel_q'] = pd.qcut(vel_df['velocity'], 5, labels=False,
                               duplicates='drop') + 1
    print(f"\n  Forward metrics by velocity quintile:")
    print(f"  {'Q':<4} {'n':>8} {'Fwd vol':>10} {'Fwd alpha':>12}")
    print(f"  {'─'*36}")
    for q in sorted(vel_df['vel_q'].unique()):
        sub = vel_df[vel_df['vel_q'] == q]
        print(f"  Q{q:<3} {len(sub):>8,} {sub['fwd_vol'].mean():>10.5f} "
              f"{sub['fwd_alpha'].mean():>12.6f}")

    # High velocity → mean reversion or momentum?
    high_vel = vel_df[vel_df['vel_q'] == 5]['fwd_alpha'].mean()
    low_vel  = vel_df[vel_df['vel_q'] == 1]['fwd_alpha'].mean()
    print(f"\n  High velocity future alpha: {high_vel:.6f}")
    print(f"  Low velocity future alpha:  {low_vel:.6f}")
    if high_vel < low_vel:
        print(f"  → HIGH VELOCITY PREDICTS MEAN REVERSION")
    else:
        print(f"  → HIGH VELOCITY PREDICTS CONTINUATION")

    # Daily cross-sectional velocity distribution
    daily_vel = vel_df.groupby('date')['velocity'].mean()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].scatter(vel_df['velocity'].values[::50],
                       vel_df['fwd_vol'].values[::50],
                       alpha=0.2, s=5, color='steelblue')
    axes[0, 0].set_xlabel('Latent velocity (21d mean step size)')
    axes[0, 0].set_ylabel('Forward realized volatility (21d)')
    axes[0, 0].set_title(f'Velocity → Forward Volatility\n'
                         f'r={r_vel_vol:.3f}  p={p_vel_vol:.4f}')

    axes[0, 1].scatter(vel_df['velocity'].values[::50],
                       vel_df['fwd_alpha'].values[::50],
                       alpha=0.2, s=5, color='darkorange')
    axes[0, 1].set_xlabel('Latent velocity')
    axes[0, 1].set_ylabel('Forward alpha (21d)')
    axes[0, 1].set_title(f'Velocity → Forward Alpha\n'
                         f'r={r_vel_alpha:.3f}  p={p_vel_alpha:.4f}')

    axes[1, 0].plot(daily_vel.index, daily_vel.values,
                    lw=0.8, color='steelblue', alpha=0.7)
    daily_vel_smooth = pd.Series(daily_vel.values,
                                  index=daily_vel.index).rolling(21).mean()
    axes[1, 0].plot(daily_vel_smooth.index, daily_vel_smooth.values,
                    lw=2, color='crimson', label='21d MA')
    for name, start, end in [('GFC', '2008-09-01', '2009-03-31'),
                              ('COVID', '2020-02-01', '2020-06-30'),
                              ('Rate', '2022-01-01', '2022-12-31')]:
        axes[1, 0].axvspan(pd.Timestamp(start), pd.Timestamp(end),
                           alpha=0.1, color='red')
    axes[1, 0].set_ylabel('Mean cross-sectional velocity')
    axes[1, 0].set_title('Daily Latent Velocity\n(red = crisis periods)')
    axes[1, 0].legend()

    q_means_vol   = vel_df.groupby('vel_q', observed=True)['fwd_vol'].mean()
    q_means_alpha = vel_df.groupby('vel_q', observed=True)['fwd_alpha'].mean()
    x = range(len(q_means_vol))
    axes[1, 1].bar([i - 0.2 for i in x], q_means_vol.values,
                   width=0.35, label='Fwd vol', color='steelblue', alpha=0.8)
    ax2 = axes[1, 1].twinx()
    ax2.bar([i + 0.2 for i in x], q_means_alpha.values,
            width=0.35, label='Fwd alpha', color='darkorange', alpha=0.8)
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels([f'Q{q}' for q in q_means_vol.index])
    axes[1, 1].set_ylabel('Forward volatility', color='steelblue')
    ax2.set_ylabel('Forward alpha', color='darkorange')
    axes[1, 1].set_title('Forward Metrics by Velocity Quintile')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'latent_velocity.png', dpi=150)
    plt.close()
    print(f"  Plot saved: latent_velocity.png")

    vel_df[['ticker', 'date', 'velocity', 'vel_accel',
            'fwd_alpha', 'fwd_vol']].to_parquet(
        DATA_DIR / 'latent_velocity.parquet')
    print(f"  Saved: latent_velocity.parquet")

    velocity_stats = {
        'r_vel_vol':   float(r_vel_vol),
        'p_vel_vol':   float(p_vel_vol),
        'r_vel_alpha': float(r_vel_alpha),
        'p_vel_alpha': float(p_vel_alpha),
        'r_acc_alpha': float(r_acc_alpha),
        'p_acc_alpha': float(p_acc_alpha),
        'high_vel_alpha': float(high_vel),
        'low_vel_alpha':  float(low_vel),
    }
else:
    print("  Insufficient data")
    velocity_stats = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Twin Stocks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: TWIN STOCKS")
print("=" * 70)
print("What are the nearest-neighbor pairs actually doing?")
print("Known from policy distance matrix: ISD↔GHY (d=0.003), BGT↔JFR (d=0.007)")

# Compute full pairwise centroid distances for all stocks
print("\nComputing pairwise centroid distances...")
cached_tickers  = list(centroid_cache.keys())
all_centroids   = np.array([centroid_cache[t]['centroid'] for t in cached_tickers])

# Use cdist for speed
D = cdist(all_centroids, all_centroids, metric='euclidean')
np.fill_diagonal(D, np.inf)

# Find top 20 nearest pairs
upper_i, upper_j = np.triu_indices(len(cached_tickers), k=1)
distances        = D[upper_i, upper_j]
sorted_idx       = np.argsort(distances)

print(f"\n  Top 20 nearest pairs (Euclidean centroid distance):")
print(f"  {'Rank':<6} {'Ticker 1':<10} {'Ticker 2':<10} {'Distance':>10}  "
      f"Sector 1 / Sector 2")
print(f"  {'─'*70}")

twin_pairs = []
for rank, idx in enumerate(sorted_idx[:20]):
    i, j = upper_i[idx], upper_j[idx]
    t1, t2 = cached_tickers[i], cached_tickers[j]
    d  = distances[idx]
    s1 = sector_data.get(t1, {}).get('sector', '?')
    s2 = sector_data.get(t2, {}).get('sector', '?')
    print(f"  {rank+1:<6} {t1:<10} {t2:<10} {d:>10.4f}  {s1} / {s2}")
    twin_pairs.append({
        'rank': rank + 1, 't1': t1, 't2': t2,
        'distance': float(d), 'sector1': s1, 'sector2': s2,
    })

# Fetch detailed info for top 10 twin pairs
print(f"\n  Fetching info for top 10 twin pairs...")
for pair in twin_pairs[:10]:
    t1, t2 = pair['t1'], pair['t2']
    infos = {}
    for t in [t1, t2]:
        try:
            info = yf.Ticker(t).info
            infos[t] = {
                'name':     info.get('longName', t),
                'sector':   info.get('sector', '?'),
                'industry': info.get('industry', '?'),
                'mktcap':   info.get('marketCap', 0),
            }
            time.sleep(0.05)
        except Exception:
            infos[t] = {'name': t, 'sector': '?', 'industry': '?', 'mktcap': 0}

    print(f"\n  Pair #{pair['rank']}  d={pair['distance']:.4f}")
    for t in [t1, t2]:
        i = infos[t]
        mc = f"${i['mktcap']/1e9:.1f}B" if i['mktcap'] > 0 else "N/A"
        print(f"    {t:<8} {i['name'][:40]:<40} {i['sector']:<25} {mc}")

# Same-sector rate at different distance percentiles
if sector_data:
    print(f"\n  Same-sector rate by distance percentile:")
    for pct in [1, 5, 10, 25, 50]:
        thresh = np.percentile(distances, pct)
        mask   = distances <= thresh
        pairs_in = [(cached_tickers[upper_i[k]], cached_tickers[upper_j[k]])
                    for k in np.where(mask)[0]]
        same_sector = sum(1 for t1, t2 in pairs_in
                          if t1 in sector_data and t2 in sector_data
                          and sector_data[t1]['sector'] == sector_data[t2]['sector'])
        n_labeled_pairs = sum(1 for t1, t2 in pairs_in
                              if t1 in sector_data and t2 in sector_data)
        if n_labeled_pairs > 0:
            rate = same_sector / n_labeled_pairs
            print(f"    p{pct:<3}: {rate*100:.1f}% same sector  "
                  f"(n={n_labeled_pairs} labeled pairs, thresh={thresh:.4f})")

    # Plot: distance distribution colored by same/different sector
    has_labels = np.array([(upper_i[k] < len(cached_tickers) and
                            upper_j[k] < len(cached_tickers) and
                            cached_tickers[upper_i[k]] in sector_data and
                            cached_tickers[upper_j[k]] in sector_data)
                           for k in range(len(distances))])

    if has_labels.sum() > 100:
        same_mask = np.array([
            sector_data.get(cached_tickers[upper_i[k]], {}).get('sector') ==
            sector_data.get(cached_tickers[upper_j[k]], {}).get('sector')
            for k in np.where(has_labels)[0]
        ])
        d_same = distances[has_labels][same_mask]
        d_diff = distances[has_labels][~same_mask]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(d_same, bins=60, alpha=0.6, density=True,
                label='Same sector', color='steelblue')
        ax.hist(d_diff, bins=60, alpha=0.6, density=True,
                label='Different sector', color='crimson')
        t_tw, p_tw = ttest_ind(d_same, d_diff)
        ax.set_xlabel('Centroid distance')
        ax.set_title(f'Centroid Distance: Same vs Different Sector\n'
                     f'p={p_tw:.4f}  '
                     f'(same={d_same.mean():.4f}, diff={d_diff.mean():.4f})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'twin_stocks.png', dpi=150)
        plt.close()
        print(f"\n  Plot saved: twin_stocks.png")

twin_results = {'top_pairs': twin_pairs[:20]}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Contagion Ordering
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: CONTAGION ORDERING")
print("=" * 70)
print("Which stocks moved first in 2008 and 2020?")
print("Latent displacement from pre-crisis centroid by week")

crisis_events = {
    'GFC_2008':   ('2007-07-01', '2008-09-15', '2009-06-30'),
    'COVID_2020': ('2019-10-01', '2020-02-20', '2020-06-30'),
}

contagion_results = {}
for crisis_name, (pre_start, crisis_start, crisis_end) in crisis_events.items():
    print(f"\n  {crisis_name}:")
    pre_ts     = pd.Timestamp(pre_start)
    crisis_ts  = pd.Timestamp(crisis_start)
    end_ts     = pd.Timestamp(crisis_end)

    # Per-stock pre-crisis centroid
    early_movers = []
    for ticker in list(centroid_cache.keys()):
        cache  = centroid_cache[ticker]
        z      = cache['z_series']
        dates  = pd.DatetimeIndex(cache['dates'])

        pre_mask    = (dates >= pre_ts)    & (dates < crisis_ts)
        crisis_mask = (dates >= crisis_ts) & (dates <= end_ts)

        if pre_mask.sum() < 21 or crisis_mask.sum() < 21:
            continue

        pre_centroid = z[pre_mask].mean(axis=0)

        # Weekly displacement from pre-crisis centroid
        # Find first week where displacement exceeds 2x pre-crisis variance
        pre_var    = float(z[pre_mask].var(axis=0).mean())
        threshold  = pre_var * 2

        crisis_dates = dates[crisis_mask]
        crisis_z     = z[crisis_mask]
        weekly_disp  = []
        for week_start in pd.date_range(crisis_ts, end_ts, freq='W'):
            week_end  = week_start + pd.Timedelta(weeks=1)
            week_mask = (crisis_dates >= week_start) & (crisis_dates < week_end)
            if week_mask.sum() < 3:
                continue
            disp = float(np.linalg.norm(
                crisis_z[week_mask].mean(axis=0) - pre_centroid))
            weekly_disp.append((week_start, disp))

        if not weekly_disp:
            continue

        # First week exceeding threshold
        first_breach = None
        for week_start, disp in weekly_disp:
            if disp > threshold:
                first_breach = week_start
                break

        max_disp = max(d for _, d in weekly_disp)
        sector   = sector_data.get(ticker, {}).get('sector', '?')

        early_movers.append({
            'ticker':       ticker,
            'first_breach': first_breach,
            'max_disp':     max_disp,
            'pre_var':      pre_var,
            'sector':       sector,
        })

    early_movers = [e for e in early_movers if e['first_breach'] is not None]
    early_movers.sort(key=lambda x: x['first_breach'])

    print(f"    Stocks with breach data: {len(early_movers)}")
    if early_movers:
        print(f"    First 10 to breach threshold:")
        for e in early_movers[:10]:
            print(f"      {e['ticker']:<8} first moved: "
                  f"{str(e['first_breach'])[:10]}  "
                  f"max disp: {e['max_disp']:.4f}  "
                  f"sector: {e['sector']}")

        # Sector of early movers vs late movers
        n_early = max(10, len(early_movers) // 4)
        early_sectors = [e['sector'] for e in early_movers[:n_early]
                         if e['sector'] != '?']
        late_sectors  = [e['sector'] for e in early_movers[-n_early:]
                         if e['sector'] != '?']

        from collections import Counter
        print(f"\n    Early mover sectors (top {n_early}):")
        for s, n in Counter(early_sectors).most_common(5):
            print(f"      {s}: {n}")
        print(f"    Late mover sectors (bottom {n_early}):")
        for s, n in Counter(late_sectors).most_common(5):
            print(f"      {s}: {n}")

    contagion_results[crisis_name] = {
        'n_stocks':    len(early_movers),
        'early_movers': [{'ticker': e['ticker'],
                          'first_breach': str(e['first_breach'])[:10],
                          'sector': e['sector']}
                         for e in early_movers[:20]],
    }

    # Plot: histogram of first breach dates
    if early_movers:
        breach_dates = [e['first_breach'] for e in early_movers]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(breach_dates, bins=20, color='crimson', alpha=0.7, edgecolor='none')
        ax.axvline(crisis_ts, color='black', lw=2, linestyle='--',
                   label='Crisis start')
        ax.set_xlabel('Date of first latent displacement breach')
        ax.set_title(f'{crisis_name}: Contagion Ordering\n'
                     f'When did each stock first show behavioral shift?')
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f'contagion_{crisis_name}.png', dpi=150)
        plt.close()
        print(f"  Plot saved: contagion_{crisis_name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Earnings Surprise Geometry
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6: EARNINGS SURPRISE GEOMETRY")
print("=" * 70)
print("Do beats/misses push latent position in consistent directions?")

# Get earnings surprise data from yfinance for a sample
print("\nFetching earnings history from yfinance...")
earnings_data = []

for ticker in list(centroid_cache.keys())[:300]:
    if ticker not in centroid_cache:
        continue
    try:
        t_obj = yf.Ticker(ticker)
        hist  = t_obj.earnings_history
        if hist is None or len(hist) == 0:
            continue

        # earnings_history returns surprisePercent and quarter dates
        for _, row in hist.iterrows():
            surprise_pct = row.get('surprisePercent', None)
            q_date       = row.name if hasattr(row, 'name') else None

            if surprise_pct is None or q_date is None:
                continue
            if pd.isna(surprise_pct):
                continue

            q_ts = pd.Timestamp(q_date)
            earnings_data.append({
                'ticker':       ticker,
                'date':         q_ts,
                'surprise_pct': float(surprise_pct),
                'beat':         bool(surprise_pct > 0),
            })
        time.sleep(0.05)
    except Exception:
        continue

print(f"  Earnings observations: {len(earnings_data)}")

if len(earnings_data) >= 50:
    earnings_df = pd.DataFrame(earnings_data)

    # For each earnings event: compute latent displacement
    # in the 5 days before vs 5 days after
    EARNINGS_WINDOW = 10   # days before and after

    displacement_results = []
    for _, row in earnings_df.iterrows():
        ticker  = row['ticker']
        q_date  = row['date']
        beat    = row['beat']
        surp    = row['surprise_pct']

        if ticker not in centroid_cache:
            continue

        cache  = centroid_cache[ticker]
        z      = cache['z_series']
        dates  = pd.DatetimeIndex(cache['dates'])

        pre_mask  = (dates >= q_date - pd.Timedelta(days=EARNINGS_WINDOW)) & \
                    (dates < q_date)
        post_mask = (dates >= q_date) & \
                    (dates < q_date + pd.Timedelta(days=EARNINGS_WINDOW))

        if pre_mask.sum() < 3 or post_mask.sum() < 3:
            continue

        pre_pos  = z[pre_mask].mean(axis=0)
        post_pos = z[post_mask].mean(axis=0)
        delta    = post_pos - pre_pos
        magnitude= float(np.linalg.norm(delta))

        displacement_results.append({
            'ticker':     ticker,
            'date':       q_date,
            'beat':       beat,
            'surprise':   surp,
            'delta':      delta,
            'magnitude':  magnitude,
        })

    print(f"  Events with latent displacement: {len(displacement_results)}")

    if len(displacement_results) >= 20:
        beats   = [r for r in displacement_results if r['beat']]
        misses  = [r for r in displacement_results if not r['beat']]

        print(f"  Beats: {len(beats)}  Misses: {len(misses)}")

        if len(beats) >= 10 and len(misses) >= 10:
            # Mean displacement direction for beats vs misses
            beat_deltas  = np.array([r['delta'] for r in beats])
            miss_deltas  = np.array([r['delta'] for r in misses])

            beat_mean    = beat_deltas.mean(axis=0)
            miss_mean    = miss_deltas.mean(axis=0)

            # Cosine similarity between beat and miss mean directions
            cos_sim = np.dot(beat_mean, miss_mean) / \
                      (np.linalg.norm(beat_mean) * np.linalg.norm(miss_mean) + 1e-8)

            # Magnitude difference
            beat_mags  = np.array([r['magnitude'] for r in beats])
            miss_mags  = np.array([r['magnitude'] for r in misses])
            t_mag, p_mag = ttest_ind(beat_mags, miss_mags)

            print(f"\n  Beat vs miss displacement analysis:")
            print(f"    Beat mean magnitude:   {beat_mags.mean():.5f}")
            print(f"    Miss mean magnitude:   {miss_mags.mean():.5f}")
            print(f"    p (magnitude):         {p_mag:.4f}")
            print(f"    Cosine sim (directions): {cos_sim:.3f}")
            if cos_sim < -0.3:
                print(f"    → BEATS AND MISSES PUSH IN OPPOSITE DIRECTIONS")
            elif cos_sim < 0.3:
                print(f"    → BEATS AND MISSES PUSH IN ORTHOGONAL DIRECTIONS")
            else:
                print(f"    → BEATS AND MISSES PUSH IN SIMILAR DIRECTIONS")

            # Which axes are most affected by beats vs misses?
            print(f"\n  Per-axis displacement (beat mean - miss mean):")
            axis_diff = beat_mean - miss_mean
            for i in range(latent_dim):
                bar = '█' * int(abs(axis_diff[i]) / np.abs(axis_diff).max() * 15)
                direction = '↑' if axis_diff[i] > 0 else '↓'
                print(f"    z{i:<2} {axis_diff[i]:>+8.5f}  {direction} {bar}")

            # Surprise magnitude → displacement magnitude
            surp_vals = np.array([r['surprise'] for r in displacement_results])
            disp_vals = np.array([r['magnitude'] for r in displacement_results])
            r_surp, p_surp = pearsonr(np.abs(surp_vals), disp_vals)
            print(f"\n  |Surprise| → displacement magnitude: "
                  f"r={r_surp:.3f}  p={p_surp:.4f}")

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].hist(beat_mags,  bins=30, alpha=0.6, label='Beat', color='steelblue')
            axes[0].hist(miss_mags,  bins=30, alpha=0.6, label='Miss', color='crimson')
            axes[0].set_xlabel('Latent displacement magnitude')
            axes[0].set_title(f'Displacement by Surprise Direction\np={p_mag:.4f}')
            axes[0].legend()

            axes[1].bar(range(latent_dim), axis_diff,
                        color=['steelblue' if v > 0 else 'crimson'
                               for v in axis_diff])
            axes[1].axhline(0, color='black', lw=0.8)
            axes[1].set_xticks(range(latent_dim))
            axes[1].set_xticklabels([f'z{i}' for i in range(latent_dim)], fontsize=7)
            axes[1].set_ylabel('Beat mean − Miss mean')
            axes[1].set_title('Per-Axis: Beat vs Miss Direction\n'
                              '(blue = beats push higher, red = misses push higher)')

            axes[2].scatter(np.abs(surp_vals), disp_vals, alpha=0.3, s=10,
                            color='darkorange')
            axes[2].set_xlabel('|Earnings surprise %|')
            axes[2].set_ylabel('Latent displacement magnitude')
            axes[2].set_title(f'Surprise Magnitude → Displacement\n'
                              f'r={r_surp:.3f}  p={p_surp:.4f}')

            plt.tight_layout()
            plt.savefig(PLOT_DIR / 'earnings_geometry.png', dpi=150)
            plt.close()
            print(f"\n  Plot saved: earnings_geometry.png")

            earnings_results = {
                'n_beats':     len(beats),
                'n_misses':    len(misses),
                'cos_sim':     float(cos_sim),
                'p_magnitude': float(p_mag),
                'r_surprise':  float(r_surp),
                'p_surprise':  float(p_surp),
            }
        else:
            earnings_results = {'insufficient_data': True}
    else:
        earnings_results = {'insufficient_data': True}
else:
    print("  Insufficient earnings data from yfinance")
    earnings_results = {}


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CURIOSITIES SUMMARY")
print("=" * 70)

if bankruptcy_results:
    print(f"\nBankruptcy signal:")
    print(f"  Delisted near vs survivor:  p={bankruptcy_results.get('p_near_vs_survivor', 'N/A'):.4f}")
    print(f"  Delisted near vs far:       p={bankruptcy_results.get('p_near_vs_far', 'N/A'):.4f}")

if sector_results:
    print(f"\nSector emergence:")
    print(f"  KNN accuracy: {sector_results.get('knn_accuracy', 0):.3f}  "
          f"({sector_results.get('knn_lift', 0):.2f}x baseline)")
    print(f"  Within vs between dist ratio: "
          f"{sector_results.get('dist_ratio', 0):.3f}x")

if velocity_stats:
    print(f"\nLatent velocity:")
    print(f"  Velocity → fwd vol:   r={velocity_stats.get('r_vel_vol', 0):.3f}")
    print(f"  Velocity → fwd alpha: r={velocity_stats.get('r_vel_alpha', 0):.3f}")

if earnings_results and 'cos_sim' in earnings_results:
    print(f"\nEarnings geometry:")
    print(f"  Beat/miss direction cosine sim: {earnings_results['cos_sim']:.3f}")
    print(f"  |Surprise| → displacement r:   {earnings_results['r_surprise']:.3f}")

print(f"\nPlots: {PLOT_DIR}")

results = {
    'bankruptcy':  bankruptcy_results,
    'sectors':     sector_results,
    'velocity':    velocity_stats,
    'twins':       twin_results,
    'contagion':   contagion_results,
    'earnings':    earnings_results,
}

with open(DATA_DIR / 'manifold_curiosities_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"Results: manifold_curiosities_results.json")
print("Done.")