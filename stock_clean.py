# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:09:56 2026

@author: Justin.Sanford
"""

# stock_clean.py
# Clean stock feature data using Isolation Forest
# Removes outlier observations before AE training
# Saves cleaned parquet per stock + combined clean matrix

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle
import random

DATA_DIR    = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
RANDOM_SEED = 42

# ── Feature columns ───────────────────────────────────────────────────────────
EXCLUDE_COLS = {'ticker', 'open', 'high', 'low', 'close', 'volume',
                'Mkt_RF', 'SMB', 'HML', 'RF',
                'high_252d', 'low_252d', 'high_63d', 'low_63d',
                'sma_21', 'sma_63', 'sma_252', 'vol_ma21',
                # Remove problematic ratio features with extreme variance
                'up_vol_ratio', 'kurt_63d'}

manifest    = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

sample_df    = pd.read_parquet(DATA_DIR / f"stock_{all_tickers[0]}.parquet")
feature_cols = [c for c in sample_df.columns
                if c not in EXCLUDE_COLS
                and sample_df[c].dtype in [np.float64, np.float32]]

print(f"Stocks: {len(all_tickers)}")
print(f"Features: {len(feature_cols)}")

# ── Step 1: Load all data and do basic cleaning ───────────────────────────────
print("\nStep 1: Loading and basic cleaning...")

all_X    = []
all_meta = []   # (ticker, row_indices) for tracking

n_raw = 0
n_inf = 0
n_nan = 0

for ticker in all_tickers:
    f = DATA_DIR / f'stock_{ticker}.parquet'
    if not f.exists():
        continue
    try:
        df = pd.read_parquet(f)
        X  = df[feature_cols].values.astype(np.float32)

        # Replace inf with nan
        n_inf += np.isinf(X).sum()
        X      = np.where(np.isinf(X), np.nan, X)

        # Drop rows with more than 30% nan
        nan_frac = np.isnan(X).mean(axis=1)
        keep     = nan_frac < 0.3
        X        = X[keep]

        if len(X) < 252:
            continue

        # Fill remaining nans with column median
        col_meds = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask      = np.isnan(X[:, j])
            n_nan    += mask.sum()
            X[mask, j] = col_meds[j]

        # Still has nan? fill with 0
        X = np.nan_to_num(X, nan=0.0)

        n_raw += len(X)
        all_X.append(X)
        all_meta.append(ticker)

    except Exception as e:
        print(f"  Failed {ticker}: {e}")
        continue

X_all = np.vstack(all_X)
print(f"Raw obs:    {n_raw:,}")
print(f"Inf values: {n_inf:,} (replaced with nan)")
print(f"Nan values: {n_nan:,} (filled with median)")
print(f"Shape:      {X_all.shape}")

# ── Step 2: Fit TEMPORARY scaler for Isolation Forest ─────────────────────────
print("\nStep 2: Fitting temporary scaler for outlier detection...")
scaler_temp = StandardScaler()
X_scaled    = scaler_temp.fit_transform(X_all).astype(np.float32)

# Hard clip at 10 sigma before isolation forest
# This removes the most extreme values that destabilize training
X_clipped = np.clip(X_scaled, -10, 10)

print(f"Values outside ±5σ:  {(np.abs(X_scaled) > 5).mean()*100:.2f}%")
print(f"Values outside ±10σ: {(np.abs(X_scaled) > 10).mean()*100:.2f}%")

# ── Step 3: Isolation Forest on a sample ─────────────────────────────────────
# Too many rows to run IF on all — sample 500k for fitting
print("\nStep 3: Fitting Isolation Forest...")
print("(Fitting on sample of 500k observations)")

rng        = np.random.RandomState(RANDOM_SEED)
sample_idx = rng.choice(len(X_clipped),
                         size=min(500_000, len(X_clipped)),
                         replace=False)
X_sample   = X_clipped[sample_idx]

iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # expect ~5% outliers
    max_samples=10_000,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1,
)
iso.fit(X_sample)
print("Isolation Forest fitted")

# ── Step 4: Score all data in chunks ─────────────────────────────────────────
print("\nStep 4: Scoring all observations...")
CHUNK    = 100_000
scores   = np.zeros(len(X_clipped), dtype=np.float32)

for i in range(0, len(X_clipped), CHUNK):
    chunk       = X_clipped[i:i+CHUNK]
    scores[i:i+CHUNK] = iso.score_samples(chunk)
    if (i // CHUNK + 1) % 10 == 0:
        print(f"  Scored {i+CHUNK:,}/{len(X_clipped):,}")

# Threshold: use the contamination-implied threshold
threshold = np.percentile(scores, 5)   # bottom 5% = outliers
mask_clean = scores >= threshold

print(f"\nOutliers removed: {(~mask_clean).sum():,} "
      f"({(~mask_clean).mean()*100:.1f}%)")
print(f"Clean obs:        {mask_clean.sum():,}")

# ── Step 5: Save clean data per stock ────────────────────────────────────────
print("\nStep 5: Saving clean data per stock...")

# Reconstruct per-stock indices
row_ptr = 0
clean_counts = {}

for ticker, X in zip(all_meta, all_X):
    n        = len(X)
    stock_mask = mask_clean[row_ptr:row_ptr + n]
    row_ptr  += n

    # Load original df and filter
    f  = DATA_DIR / f'stock_{ticker}.parquet'
    df = pd.read_parquet(f)

    # Match rows — same filtering as load above
    X_raw    = df[feature_cols].values.astype(np.float32)
    nan_frac = np.isnan(np.where(np.isinf(X_raw), np.nan, X_raw)).mean(axis=1)
    keep     = nan_frac < 0.3
    df_keep  = df[keep]

    if len(df_keep) != n:
        # Mismatch — skip this stock
        continue

    df_clean = df_keep[stock_mask]
    clean_counts[ticker] = len(df_clean)

    if len(df_clean) >= 252:
        out = DATA_DIR / f'stock_clean_{ticker}.parquet'
        df_clean.to_parquet(out)

print(f"Saved clean files for "
      f"{sum(1 for v in clean_counts.values() if v >= 252)} stocks")

# ── Step 6: Fit FINAL scaler on clean data only ───────────────────────────────
print("\nStep 6: Fitting final scaler on clean data...")
X_clean      = X_all[mask_clean]
scaler_final = StandardScaler()
scaler_final.fit(X_clean)

print(f"Clean data std range: "
      f"{scaler_final.scale_.min():.4f} — {scaler_final.scale_.max():.4f}")
print(f"Clean data mean range: "
      f"{scaler_final.mean_.min():.4f} — {scaler_final.mean_.max():.4f}")

# ── Step 7: Save final scaler and isolation forest ────────────────────────────
print("\nStep 7: Saving scaler and IsoForest...")

with open(DATA_DIR / 'stock_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_final, f)

with open(DATA_DIR / 'stock_iso.pkl', 'wb') as f:
    pickle.dump(iso, f)

print("Scaler (clean) and IsoForest saved")

# ── Step 8: Diagnostics ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)
print(f"Raw observations:   {n_raw:,}")
print(f"After IF cleaning:  {mask_clean.sum():,}")
print(f"Removed:            {(~mask_clean).sum():,} ({(~mask_clean).mean()*100:.1f}%)")

# Per-feature stats before and after
print(f"\nPer-feature std before/after cleaning (sample of 5 features):")
X_clean_scaled = scaler_final.transform(X_clean)
for j, col in enumerate(feature_cols[:5]):
    before = X_scaled[:, j].std()
    after  = X_clean_scaled[:, j].std()
    print(f"  {col:<25} before={before:.3f} after={after:.3f}")

print("\nDone — run stock_test_suite.py to validate, then stock_ae_train.py")