# pca_baseline_noret_forward.py
# Test forward prediction: predict alpha[t+1] from features[t]
# This is the real test of predictive power

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')

EXCLUDE_FROM_BASELINE = {
    'ticker', 'open', 'high', 'low', 'close', 'volume',
    'Mkt_RF', 'SMB', 'HML', 'RF',
    'high_252d', 'low_252d', 'high_63d', 'low_63d',
    'sma_21', 'sma_63', 'sma_252', 'vol_ma21',
    'up_vol_ratio', 'kurt_63d',
    # ALL RETURNS
    'ret_1d', 'ret_5d', 'ret_21d', 'ret_63d', 'ret_252d',
    # REDUNDANT
    'excess_ret', 'mom_1_12', 'up_days_63',
    # TARGET VARIABLES
    'alpha_resid',
}

print("="*80)
print("PCA BASELINE - FORWARD PREDICTION TEST")
print("Predicting alpha[t+1] from features[t]")
print("="*80)

# Load data
manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
tickers = manifest['ticker'].tolist()

sample_df = pd.read_parquet(DATA_DIR / f'stock_clean_{tickers[0]}.parquet')
feature_cols = [c for c in sample_df.columns 
                if c not in EXCLUDE_FROM_BASELINE
                and sample_df[c].dtype in [np.float64, np.float32]]

print(f"Features: {len(feature_cols)}")

# Load all stocks WITH TEMPORAL ORDERING
print(f"\nLoading {len(tickers)} stocks (keeping temporal order)...")
stock_data = {}

for ticker in tickers:
    f = DATA_DIR / f'stock_clean_{ticker}.parquet'
    if not f.exists():
        continue
    
    try:
        df = pd.read_parquet(f)
        X = df[feature_cols].values.astype(np.float32)
        X = np.where(np.isinf(X), np.nan, X)
        col_meds = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_meds[j]
        X = np.nan_to_num(X, nan=0.0)
        
        y_ret = df['ret_1d'].values if 'ret_1d' in df.columns else np.zeros(len(df))
        y_alpha = df['alpha_resid'].values if 'alpha_resid' in df.columns else np.zeros(len(df))
        
        valid = ~np.isnan(y_ret) & ~np.isnan(y_alpha)
        X = X[valid]
        y_ret = y_ret[valid]
        y_alpha = y_alpha[valid]
        
        if len(X) >= 252:
            stock_data[ticker] = {
                'X': X,
                'y_ret': y_ret,
                'y_alpha': y_alpha,
            }
    except Exception as e:
        continue

print(f"Loaded {len(stock_data)} stocks")

# 5-fold stock-level CV
tickers_list = list(stock_data.keys())
np.random.seed(42)
np.random.shuffle(tickers_list)

n_folds = 5
fold_size = len(tickers_list) // n_folds

results_contemp_6d = []
results_forward_6d = []
results_contemp_12d = []
results_forward_12d = []

for fold in range(n_folds):
    print(f"\n{'='*80}")
    print(f"FOLD {fold+1}/{n_folds}")
    print(f"{'='*80}")
    
    # Split
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(tickers_list)
    val_tickers = set(tickers_list[val_start:val_end])
    train_tickers = set(tickers_list) - val_tickers
    
    # Collect data
    X_train, y_ret_train, y_alpha_train = [], [], []
    X_val, y_ret_val, y_alpha_val = [], [], []
    
    for ticker in train_tickers:
        data = stock_data[ticker]
        X_train.append(data['X'])
        y_ret_train.append(data['y_ret'])
        y_alpha_train.append(data['y_alpha'])
    
    for ticker in val_tickers:
        data = stock_data[ticker]
        X_val.append(data['X'])
        y_ret_val.append(data['y_ret'])
        y_alpha_val.append(data['y_alpha'])
    
    X_train = np.vstack(X_train)
    y_ret_train = np.concatenate(y_ret_train)
    y_alpha_train = np.concatenate(y_alpha_train)
    
    X_val = np.vstack(X_val)
    y_ret_val = np.concatenate(y_ret_val)
    y_alpha_val = np.concatenate(y_alpha_val)
    
    print(f"Train: {len(X_train):,} obs")
    print(f"Val:   {len(X_val):,} obs")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # ── PCA 6D ────────────────────────────────────────────────────────────────
    pca_6 = PCA(n_components=6)
    z_train_6 = pca_6.fit_transform(X_train_scaled)
    z_val_6 = pca_6.transform(X_val_scaled)
    
    # Contemporaneous prediction (features[t] → alpha[t])
    ridge_alpha_contemp_6 = Ridge(alpha=1.0).fit(z_train_6, y_alpha_train)
    alpha_r2_contemp_6 = r2_score(y_alpha_val, ridge_alpha_contemp_6.predict(z_val_6))
    
    # Forward prediction (features[t] → alpha[t+1])
    z_train_6_fwd = z_train_6[:-1]
    y_alpha_train_fwd = y_alpha_train[1:]
    z_val_6_fwd = z_val_6[:-1]
    y_alpha_val_fwd = y_alpha_val[1:]
    
    ridge_alpha_fwd_6 = Ridge(alpha=1.0).fit(z_train_6_fwd, y_alpha_train_fwd)
    alpha_r2_fwd_6 = r2_score(y_alpha_val_fwd, ridge_alpha_fwd_6.predict(z_val_6_fwd))
    
    print(f"\nPCA 6D:")
    print(f"  Alpha R² (contemp): {alpha_r2_contemp_6:.4f}")
    print(f"  Alpha R² (forward): {alpha_r2_fwd_6:.4f}")
    
    results_contemp_6d.append({'fold': fold+1, 'alpha_r2': alpha_r2_contemp_6})
    results_forward_6d.append({'fold': fold+1, 'alpha_r2': alpha_r2_fwd_6})
    
    # ── PCA 12D ───────────────────────────────────────────────────────────────
    pca_12 = PCA(n_components=12)
    z_train_12 = pca_12.fit_transform(X_train_scaled)
    z_val_12 = pca_12.transform(X_val_scaled)
    
    # Contemporaneous
    ridge_alpha_contemp_12 = Ridge(alpha=1.0).fit(z_train_12, y_alpha_train)
    alpha_r2_contemp_12 = r2_score(y_alpha_val, ridge_alpha_contemp_12.predict(z_val_12))
    
    # Forward
    z_train_12_fwd = z_train_12[:-1]
    z_val_12_fwd = z_val_12[:-1]
    
    ridge_alpha_fwd_12 = Ridge(alpha=1.0).fit(z_train_12_fwd, y_alpha_train_fwd)
    alpha_r2_fwd_12 = r2_score(y_alpha_val_fwd, ridge_alpha_fwd_12.predict(z_val_12_fwd))
    
    print(f"\nPCA 12D:")
    print(f"  Alpha R² (contemp): {alpha_r2_contemp_12:.4f}")
    print(f"  Alpha R² (forward): {alpha_r2_fwd_12:.4f}")
    
    results_contemp_12d.append({'fold': fold+1, 'alpha_r2': alpha_r2_contemp_12})
    results_forward_12d.append({'fold': fold+1, 'alpha_r2': alpha_r2_fwd_12})

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\nPCA 6D - Contemporaneous (features[t] → alpha[t]):")
for r in results_contemp_6d:
    print(f"  Fold {r['fold']}: Alpha R²={r['alpha_r2']:.4f}")
mean_contemp_6 = np.mean([r['alpha_r2'] for r in results_contemp_6d])
print(f"Mean: {mean_contemp_6:.4f}")

print("\nPCA 6D - Forward (features[t] → alpha[t+1]):")
for r in results_forward_6d:
    print(f"  Fold {r['fold']}: Alpha R²={r['alpha_r2']:.4f}")
mean_fwd_6 = np.mean([r['alpha_r2'] for r in results_forward_6d])
print(f"Mean: {mean_fwd_6:.4f}")

print("\nPCA 12D - Contemporaneous:")
for r in results_contemp_12d:
    print(f"  Fold {r['fold']}: Alpha R²={r['alpha_r2']:.4f}")
mean_contemp_12 = np.mean([r['alpha_r2'] for r in results_contemp_12d])
print(f"Mean: {mean_contemp_12:.4f}")

print("\nPCA 12D - Forward:")
for r in results_forward_12d:
    print(f"  Fold {r['fold']}: Alpha R²={r['alpha_r2']:.4f}")
mean_fwd_12 = np.mean([r['alpha_r2'] for r in results_forward_12d])
print(f"Mean: {mean_fwd_12:.4f}")

print(f"\n{'='*80}")
print("INTERPRETATION:")
print(f"{'='*80}")
print(f"Contemporaneous prediction (features[t] → alpha[t]):")
print(f"  - Measures feature-alpha correlation within same time period")
print(f"  - PCA 6D:  R²={mean_contemp_6:.4f}")
print(f"  - PCA 12D: R²={mean_contemp_12:.4f}")
print(f"\nForward prediction (features[t] → alpha[t+1]):")
print(f"  - Measures true predictive power")
print(f"  - PCA 6D:  R²={mean_fwd_6:.4f} ← Can you make money?")
print(f"  - PCA 12D: R²={mean_fwd_12:.4f} ← Can you make money?")
print(f"\nIf forward R² ≈ 0, markets are efficient (alpha unpredictable)")
print(f"If forward R² > 0.01, there's exploitable signal")

# Save
import json
with open(DATA_DIR / 'pca_baseline_forward.json', 'w') as f:
    json.dump({
        'contemp_6d': {'mean': mean_contemp_6, 'folds': results_contemp_6d},
        'forward_6d': {'mean': mean_fwd_6, 'folds': results_forward_6d},
        'contemp_12d': {'mean': mean_contemp_12, 'folds': results_contemp_12d},
        'forward_12d': {'mean': mean_fwd_12, 'folds': results_forward_12d},
    }, f, indent=2)

print(f"\nResults saved: pca_baseline_forward.json")