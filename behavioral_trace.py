# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:42:31 2026

@author: Justin.Sanford
"""

# behavioral_trace.py
# Build behavioral trace from 13F position changes
# Each manager-quarter = vector of portfolio decisions
# This is the telemetry equivalent — what did they do, not how did they perform

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import json

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data')

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading filings...")
df = pd.read_parquet(DATA_DIR / 'all_filings.parquet')
df['manager'] = df['manager'].replace('Appaloosa_old', 'Appaloosa')
df['date']    = pd.to_datetime(df['date'])
df['quarter'] = df['date'].dt.to_period('Q')

with open(DATA_DIR / 'cusip_ticker_mapping.json') as f:
    cusip_ticker = json.load(f)

df['ticker'] = df['cusip'].map(cusip_ticker)

alpha_df = pd.read_parquet(DATA_DIR / 'alpha_residuals.parquet')
alpha_df['quarter'] = pd.to_datetime(
    alpha_df['quarter_end']).dt.to_period('Q')

print(f"Filings: {len(df):,}")
print(f"Managers: {df['manager'].nunique()}")
print(f"Quarters: {df['quarter'].nunique()}")

# ── Build position-level panel ────────────────────────────────────────────────
print("\nBuilding position panel...")

# For each manager-quarter, compute portfolio weights
df['total_value'] = df.groupby(
    ['manager', 'quarter'])['value'].transform('sum')
df['weight'] = df['value'] / df['total_value']

# Sort for diff computation
df = df.sort_values(['manager', 'ticker', 'quarter'])

# ── Behavioral features per manager-quarter ───────────────────────────────────
print("Computing behavioral features...")

# For each manager-quarter we compute:
# 1. Position concentration features
# 2. Turnover features (what changed vs last quarter)
# 3. Sector allocation features
# 4. Position sizing distribution features

def herfindahl(weights):
    """Concentration index — 1/n for equal weight, 1 for single position."""
    return (weights ** 2).sum()

def gini(weights):
    """Gini coefficient of position weights."""
    w = np.sort(weights)
    n = len(w)
    if n == 0:
        return 0
    cumw = np.cumsum(w)
    return (n + 1 - 2 * cumw.sum() / cumw[-1]) / n if cumw[-1] > 0 else 0

rows = []
managers = df['manager'].unique()

for manager in managers:
    mgr_df = df[df['manager'] == manager].copy()
    quarters = sorted(mgr_df['quarter'].unique())

    prev_positions = {}  # ticker → weight in previous quarter

    for q in quarters:
        q_df = mgr_df[mgr_df['quarter'] == q].copy()

        if len(q_df) == 0:
            continue

        # Current positions
        curr_positions = dict(zip(q_df['ticker'].fillna('UNKNOWN'),
                                  q_df['weight']))

        # ── Concentration features ────────────────────────────────────────────
        weights = q_df['weight'].values
        n_pos   = len(weights)

        hhi    = herfindahl(weights)
        gini_c = gini(weights)
        top1   = weights.max() if n_pos > 0 else 0
        top5   = np.sort(weights)[-5:].sum() if n_pos >= 5 else weights.sum()
        top10  = np.sort(weights)[-10:].sum() if n_pos >= 10 else weights.sum()
        weight_std = weights.std() if n_pos > 1 else 0
        weight_skew = (pd.Series(weights).skew()
                       if n_pos > 2 else 0)

        # ── Turnover features ─────────────────────────────────────────────────
        if prev_positions:
            all_tickers = set(curr_positions) | set(prev_positions)

            # Weight changes
            deltas = []
            for t in all_tickers:
                curr_w = curr_positions.get(t, 0)
                prev_w = prev_positions.get(t, 0)
                deltas.append(curr_w - prev_w)

            deltas = np.array(deltas)

            # Turnover = sum of absolute weight changes / 2
            turnover     = np.abs(deltas).sum() / 2
            n_new        = sum(1 for t in curr_positions
                               if t not in prev_positions)
            n_exited     = sum(1 for t in prev_positions
                               if t not in curr_positions)
            n_increased  = sum(1 for t in curr_positions
                               if curr_positions[t] >
                               prev_positions.get(t, 0) + 0.001)
            n_decreased  = sum(1 for t in curr_positions
                               if curr_positions[t] <
                               prev_positions.get(t, 0) - 0.001)
            frac_new     = n_new / max(len(curr_positions), 1)
            frac_exited  = n_exited / max(len(prev_positions), 1)
            # Conviction: did they add to winners or losers?
            # (positive weight change = added to position)
            net_buying   = max(0, deltas.sum())
            net_selling  = max(0, -deltas.sum())

        else:
            turnover    = 1.0   # first quarter = full turnover
            n_new       = n_pos
            n_exited    = 0
            n_increased = 0
            n_decreased = 0
            frac_new    = 1.0
            frac_exited = 0.0
            net_buying  = 1.0
            net_selling = 0.0

        # ── Position size distribution ────────────────────────────────────────
        pct_above_5  = (weights > 0.05).mean()
        pct_above_10 = (weights > 0.10).mean()
        pct_below_1  = (weights < 0.01).mean()
        median_w     = np.median(weights)

        # ── Effective N (inverse HHI) ─────────────────────────────────────────
        eff_n = 1 / hhi if hhi > 0 else n_pos

        rows.append({
            'manager':      manager,
            'quarter':      q,
            # Concentration
            'n_positions':  n_pos,
            'hhi':          hhi,
            'gini':         gini_c,
            'top1_weight':  top1,
            'top5_weight':  top5,
            'top10_weight': top10,
            'weight_std':   weight_std,
            'weight_skew':  weight_skew,
            'eff_n':        eff_n,
            # Turnover
            'turnover':     turnover,
            'n_new':        n_new,
            'n_exited':     n_exited,
            'n_increased':  n_increased,
            'n_decreased':  n_decreased,
            'frac_new':     frac_new,
            'frac_exited':  frac_exited,
            'net_buying':   net_buying,
            'net_selling':  net_selling,
            # Size distribution
            'pct_above_5':  pct_above_5,
            'pct_above_10': pct_above_10,
            'pct_below_1':  pct_below_1,
            'median_weight':median_w,
        })

        prev_positions = curr_positions

trace_df = pd.DataFrame(rows)
print(f"\nBehavioral trace shape: {trace_df.shape}")
print(f"Features: {len(trace_df.columns) - 2}")
print(f"Obs per manager:")
print(trace_df.groupby('manager').size().sort_values(ascending=False)
      .to_string())

# ── Merge with alpha residuals ────────────────────────────────────────────────
print("\nMerging with alpha residuals...")

merged = trace_df.merge(
    alpha_df[['manager', 'quarter', 'alpha_real',
              'excess_ret', 'port_ret']],
    on=['manager', 'quarter'],
    how='inner'
)

print(f"Merged obs: {len(merged)}")
print(f"Managers: {merged['manager'].nunique()}")

# ── Quick correlation check ───────────────────────────────────────────────────
print("\nCorrelation of behavioral features with next-quarter alpha:")
print("(Predictive — feature at t, alpha at t+1)\n")

feature_cols = [c for c in trace_df.columns
                if c not in ['manager', 'quarter']]

# Shift alpha forward by one quarter per manager
merged = merged.sort_values(['manager', 'quarter'])
merged['alpha_next'] = merged.groupby('manager')['alpha_real'].shift(-1)
merged_pred = merged.dropna(subset=['alpha_next'])

correlations = []
for col in feature_cols:
    r, p = [], []
    for manager in merged_pred['manager'].unique():
        sub = merged_pred[merged_pred['manager'] == manager]
        if len(sub) < 6:
            continue
        ri, pi = __import__('scipy').stats.pearsonr(
            sub[col], sub['alpha_next'])
        r.append(ri)
        p.append(pi)

    if r:
        correlations.append({
            'feature':  col,
            'mean_r':   np.mean(r),
            'pct_pos':  np.mean([ri > 0 for ri in r]),
            'mean_p':   np.mean(p),
        })

corr_df = (pd.DataFrame(correlations)
           .sort_values('mean_r', key=abs, ascending=False))

print(f"{'Feature':<20} {'Mean r':>8} {'% Pos':>8} {'Mean p':>8}")
print("-" * 48)
for _, row in corr_df.iterrows():
    print(f"{row['feature']:<20} {row['mean_r']:>+8.3f} "
          f"{row['pct_pos']*100:>7.0f}% {row['mean_p']:>8.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
trace_df.to_parquet(DATA_DIR / 'behavioral_trace.parquet')
merged.to_parquet(DATA_DIR / 'trace_with_alpha.parquet')
corr_df.to_parquet(DATA_DIR / 'feature_alpha_correlations.parquet')

# Scale features for AE training
feature_cols = [c for c in trace_df.columns
                if c not in ['manager', 'quarter']]
X = trace_df[feature_cols].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open(DATA_DIR / 'trace_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

np.save(DATA_DIR / 'trace_X.npy', X_scaled)

print(f"\nFeature matrix saved: {X_scaled.shape}")
print(f"\nDone — run finance_ae.py next")