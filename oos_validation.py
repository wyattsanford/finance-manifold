# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:35:12 2026

@author: Justin.Sanford
"""

# -*- coding: utf-8 -*-
"""
stock_oos_validation.py
Out-of-universe validation — the Russell test for financial manifolds.

Pulls N fresh stocks that were NEVER in the original 2000-stock universe,
encodes them cold with the temporal model, generates Monte Carlo scouting
predictions, then validates against realized post-2023 alpha.

Pipeline:
  1. Build exclusion list from existing manifest
  2. Pull candidate tickers from SEC/EDGAR (same criteria as universe pull)
  3. Validate and compute features (same logic as stock_universe_pull.py)
  4. Encode with temporal model — model has never seen these stocks
  5. Monte Carlo scouting report — predicted alpha + CI per stock
  6. Validate predictions against realized post-2023 alpha
  7. Compare to manifold stability prediction (stable = tighter CI = more accurate)

Key result: does the manifold generalize to completely unseen stocks?
Analog: Russell at Mercedes — predicted cold, validated prospectively.
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import random
import zipfile
import io
import warnings
warnings.filterwarnings('ignore')

DATA_DIR    = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
OOS_DIR     = DATA_DIR / 'oos_validation'
PLOT_DIR    = OOS_DIR / 'plots'
OOS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG         = np.random.RandomState(42)

TARGET_OOS    = 100        # how many fresh stocks to validate on
MIN_DAYS      = 252        # minimum history requirement (same as universe pull)
START_DATE    = '2000-01-01'
END_DATE      = '2024-12-31'
MIN_START_DATE= '2020-01-01'  # must have pre-2023 history
CUTOFF_DATE   = '2023-01-01'  # encode on pre-cutoff, validate post-cutoff
MC_SAMPLES    = 2000

HEADERS = {'User-Agent': 'Justin Sanford justin.sanford@dlhcorp.com'}

print(f"Device: {DEVICE}")
print(f"OOS dir: {OOS_DIR}")
print(f"Target: {TARGET_OOS} never-seen stocks")

# ── Model ─────────────────────────────────────────────────────────────────────
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


# ── Feature engineering (identical to stock_universe_pull.py) ─────────────────
def compute_features(hist, ticker):
    df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.ffill().dropna()
    if len(df) < 60:
        return None

    df['ret_1d']   = df['close'].pct_change(1)
    df['ret_5d']   = df['close'].pct_change(5)
    df['ret_21d']  = df['close'].pct_change(21)
    df['ret_63d']  = df['close'].pct_change(63)
    df['ret_252d'] = df['close'].pct_change(252)

    df['vol_5d']   = df['ret_1d'].rolling(5).std()
    df['vol_21d']  = df['ret_1d'].rolling(21).std()
    df['vol_63d']  = df['ret_1d'].rolling(63).std()
    df['vol_252d'] = df['ret_1d'].rolling(252).std()
    df['vol_ratio']= df['vol_21d'] / (df['vol_252d'] + 0.01)

    df['vol_ma21']    = df['volume'].rolling(21).mean()
    df['vol_ratio_v'] = df['volume'] / (df['vol_ma21'] + 1)
    df['vol_21d_chg'] = df['vol_ma21'].pct_change(21)

    df['high_252d'] = df['close'].rolling(252).max()
    df['low_252d']  = df['close'].rolling(252).min()
    df['pos_252d']  = ((df['close'] - df['low_252d']) /
                       (df['high_252d'] - df['low_252d'] + 0.01))
    df['high_63d']  = df['close'].rolling(63).max()
    df['low_63d']   = df['close'].rolling(63).min()
    df['pos_63d']   = ((df['close'] - df['low_63d']) /
                       (df['high_63d'] - df['low_63d'] + 0.01))

    df['hl_range']    = (df['high'] - df['low']) / (df['close'] + 0.01)
    df['hl_range_ma'] = df['hl_range'].rolling(21).mean()

    df['mom_1_12'] = df['ret_252d'] - df['ret_21d']
    df['mom_accel']= df['ret_21d']  - df['ret_63d']

    df['sma_21']  = df['close'].rolling(21).mean()
    df['sma_63']  = df['close'].rolling(63).mean()
    df['sma_252'] = df['close'].rolling(252).mean()

    df['price_vs_sma21']  = df['close'] / (df['sma_21']  + 0.01) - 1
    df['price_vs_sma63']  = df['close'] / (df['sma_63']  + 0.01) - 1
    df['price_vs_sma252'] = df['close'] / (df['sma_252'] + 0.01) - 1
    df['sma21_vs_sma63']  = df['sma_21'] / (df['sma_63'] + 0.01) - 1

    df['skew_63d']  = df['ret_1d'].rolling(63).skew()
    df['kurt_63d']  = df['ret_1d'].rolling(63).kurt()
    df['skew_252d'] = df['ret_1d'].rolling(252).skew()

    rolling_max    = df['close'].rolling(252, min_periods=1).max()
    df['drawdown'] = df['close'] / rolling_max - 1
    df['dd_duration'] = (df['drawdown'] < -0.1).rolling(63).sum()

    df['up_days_21']  = (df['ret_1d'] > 0).rolling(21).sum() / 21
    df['up_days_63']  = (df['ret_1d'] > 0).rolling(63).sum() / 63
    df['up_vol_ratio']= (df['ret_1d'].clip(lower=0).rolling(21).std() /
                         (df['ret_1d'].clip(upper=0).abs().rolling(21).std() + 0.01))

    df['ticker'] = ticker
    df = df.iloc[252:].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_ff_daily():
    url = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
           'ftp/F-F_Research_Data_Factors_daily_CSV.zip')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        fname = [f for f in z.namelist() if f.lower().endswith('.csv')][0]
        with z.open(fname) as f:
            raw = f.read().decode('utf-8')
    lines = raw.split('\n')
    start = next(i for i, l in enumerate(lines)
                 if l.strip() and l.strip()[0].isdigit() and len(l.strip()) >= 8)
    end   = next((i for i, l in enumerate(lines[start:], start=start)
                  if l.strip() and not l.strip()[0].isdigit()), len(lines))
    ff = pd.read_csv(io.StringIO('\n'.join(lines[start:end])),
                     header=None, names=['date', 'Mkt_RF', 'SMB', 'HML', 'RF'])
    ff = ff.dropna()
    ff['date'] = pd.to_datetime(ff['date'].astype(str).str.strip(),
                                format='%Y%m%d', errors='coerce')
    ff = ff.dropna(subset=['date'])
    ff[['Mkt_RF', 'SMB', 'HML', 'RF']] = (
        ff[['Mkt_RF', 'SMB', 'HML', 'RF']]
        .apply(pd.to_numeric, errors='coerce') / 100)
    return ff.set_index('date').sort_index()


def add_alpha_residual(df, ff_daily):
    """Add FF3 alpha residual — identical to universe pull logic."""
    feat = df.join(ff_daily, how='left')
    for col in ['RF', 'Mkt_RF', 'SMB', 'HML']:
        feat[col] = feat[col].ffill()

    for factor in ['Mkt_RF', 'SMB', 'HML']:
        roll_cov = feat['ret_1d'].rolling(63).cov(feat[factor])
        roll_var = feat[factor].rolling(63).var()
        feat[f'beta_{factor.lower()}'] = roll_cov / (roll_var + 0.01)

    feat['excess_ret'] = feat['ret_1d'] - feat['RF']
    feat['alpha_resid'] = (
        feat['excess_ret']
        - feat['beta_mkt_rf'] * feat['Mkt_RF']
        - feat['beta_smb']    * feat['SMB']
        - feat['beta_hml']    * feat['HML']
    )
    return feat


def clean_X(X):
    X = np.where(np.isinf(X), np.nan, X)
    col_meds = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = col_meds[j]
    return np.nan_to_num(X, nan=0.0)


# ── Load model + feature spec ─────────────────────────────────────────────────
print("\nLoading temporal model...")
ckpt         = torch.load(DATA_DIR / 'ae_temporal_best.pt',
                           map_location=DEVICE, weights_only=False)
model, input_dim, latent_dim = build_model_from_ckpt(ckpt)
scaler       = ckpt['scaler']
feature_cols = ckpt['feature_cols']
print(f"  Input dim: {input_dim}  Latent dim: {latent_dim}")
print(f"  Features:  {len(feature_cols)}")

# ── Load existing universe (exclusion list) ───────────────────────────────────
print("\nLoading existing universe for exclusion...")
manifest      = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
existing_set  = set(manifest['ticker'].str.upper().tolist())
print(f"  Excluding {len(existing_set)} known tickers")

# ── Get FF factors ────────────────────────────────────────────────────────────
print("\nDownloading FF3 factors...")
ff_daily = get_ff_daily()
print(f"  FF3: {len(ff_daily)} days")

# ── Build candidate universe ──────────────────────────────────────────────────
print("\nFetching candidate tickers from SEC EDGAR...")
candidates = set()

try:
    r = requests.get('https://www.sec.gov/files/company_tickers_exchange.json',
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    df_sec = pd.DataFrame(data['data'], columns=data['fields'])
    candidates.update(df_sec['ticker'].dropna().str.upper().tolist())
    print(f"  SEC exchange: {len(candidates)}")
except Exception as e:
    print(f"  SEC exchange failed: {e}")

try:
    r = requests.get('https://www.sec.gov/files/company_tickers.json',
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    df_all = pd.DataFrame(data.values())
    candidates.update(df_all['ticker'].dropna().str.upper().tolist())
    print(f"  SEC all: {len(candidates)}")
except Exception as e:
    print(f"  SEC all failed: {e}")

# Clean candidates
candidates = {t for t in candidates
              if t and len(t) <= 6 and t.isalpha()
              and not any(c in t for c in ['.', '-', '^', '='])}

# Remove known universe
candidates -= existing_set
candidate_list = list(candidates)
random.seed(99)   # different seed from universe pull
random.shuffle(candidate_list)
print(f"  Fresh candidates after exclusion: {len(candidate_list)}")

# ── Pull and validate fresh stocks ───────────────────────────────────────────
print(f"\nPulling fresh stocks (target: {TARGET_OOS})...")
print("Using same validation criteria as universe pull.\n")

oos_stocks = {}
n_tried    = 0
checkpoint_file = OOS_DIR / 'oos_checkpoint.json'

# Resume if interrupted
if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        ckpt_data = json.load(f)
    validated_oos = set(ckpt_data.get('validated', []))
    failed_oos    = set(ckpt_data.get('failed', []))
    print(f"  Resuming: {len(validated_oos)} validated, {len(failed_oos)} failed")
else:
    validated_oos = set()
    failed_oos    = set()

# Load existing validated data
for ticker in list(validated_oos):
    f = OOS_DIR / f'oos_{ticker}.parquet'
    if f.exists():
        oos_stocks[ticker] = pd.read_parquet(f)

for ticker in candidate_list:
    if len(validated_oos) >= TARGET_OOS:
        break
    if ticker in validated_oos or ticker in failed_oos:
        continue

    n_tried += 1

    try:
        t    = yf.Ticker(ticker)
        hist = t.history(start=START_DATE, end=END_DATE, auto_adjust=True)

        if hist is None or len(hist) < MIN_DAYS:
            failed_oos.add(ticker)
            continue

        hist.index = pd.to_datetime(hist.index).tz_localize(None)

        if hist.index[0] > pd.Timestamp(MIN_START_DATE):
            failed_oos.add(ticker)
            continue

        if hist['Close'].std() < 1e-6:
            failed_oos.add(ticker)
            continue

        # Must have both pre and post cutoff data
        pre_mask  = hist.index < pd.Timestamp(CUTOFF_DATE)
        post_mask = hist.index >= pd.Timestamp(CUTOFF_DATE)
        if pre_mask.sum() < MIN_DAYS or post_mask.sum() < 21:
            failed_oos.add(ticker)
            continue

        features = compute_features(hist, ticker)
        if features is None or len(features) < MIN_DAYS:
            failed_oos.add(ticker)
            continue

        # Add alpha residual
        full_df = add_alpha_residual(features, ff_daily)
        feat_cols_check = [c for c in feature_cols if c in full_df.columns]
        if len(feat_cols_check) < len(feature_cols) * 0.9:
            failed_oos.add(ticker)
            continue

        # Drop high-NaN rows
        full_df = full_df.replace([np.inf, -np.inf], np.nan)
        nan_frac = full_df[feature_cols].isnull().mean(axis=1)
        full_df  = full_df[nan_frac < 0.3]

        if len(full_df) < MIN_DAYS:
            failed_oos.add(ticker)
            continue

        # Save
        out = OOS_DIR / f'oos_{ticker}.parquet'
        full_df.to_parquet(out)
        validated_oos.add(ticker)
        oos_stocks[ticker] = full_df

        if len(validated_oos) % 10 == 0 or len(validated_oos) == TARGET_OOS:
            with open(checkpoint_file, 'w') as f:
                json.dump({'validated': list(validated_oos),
                           'failed':    list(failed_oos)}, f)
            print(f"  Validated: {len(validated_oos):3d}/{TARGET_OOS} | "
                  f"Tried: {n_tried} | Failed: {len(failed_oos)} | "
                  f"Latest: {ticker}")

        time.sleep(0.1)

    except Exception as e:
        failed_oos.add(ticker)
        continue

print(f"\nFresh stocks validated: {len(oos_stocks)}")
print(f"Total tried: {n_tried}  |  Failed: {len(failed_oos)}")

if len(oos_stocks) == 0:
    raise RuntimeError("No OOS stocks validated — check network / API access")

# ── Encode fresh stocks ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ENCODING FRESH STOCKS — MODEL HAS NEVER SEEN THESE")
print("=" * 70)

def prep_X(df, feature_cols):
    """Extract and clean feature matrix from dataframe."""
    # Fill missing feature cols with zeros
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X = df[feature_cols].values.astype(np.float32)
    return clean_X(X)


# Split each stock into pre-cutoff (for centroid) and post-cutoff (for validation)
oos_pre  = {}   # ticker → (X, y_alpha, dates)
oos_post = {}   # ticker → (X, y_alpha, dates)

for ticker, df in oos_stocks.items():
    dates = pd.to_datetime(df.index)
    pre_mask  = dates < pd.Timestamp(CUTOFF_DATE)
    post_mask = dates >= pd.Timestamp(CUTOFF_DATE)

    if pre_mask.sum() < 63 or post_mask.sum() < 21:
        continue

    X_full  = prep_X(df.copy(), feature_cols)
    y_alpha = df['alpha_resid'].values if 'alpha_resid' in df.columns else np.zeros(len(df))

    oos_pre[ticker]  = (X_full[pre_mask],  y_alpha[pre_mask],  dates[pre_mask])
    oos_post[ticker] = (X_full[post_mask], y_alpha[post_mask], dates[post_mask])

print(f"  Stocks with pre+post data: {len(oos_pre)}")

# Encode pre-cutoff observations for each fresh stock
print("\nEncoding pre-cutoff observations...")
oos_latents_pre = {}
for ticker, (X_pre, y_pre, dates_pre) in oos_pre.items():
    X_pre_s = scaler.transform(X_pre).astype(np.float32)
    z_pre   = encode_batch(model, X_pre_s)
    oos_latents_pre[ticker] = {
        'z':          z_pre,
        'y_alpha':    y_pre,
        'dates':      dates_pre,
        'centroid':   z_pre.mean(axis=0),
        'cov':        np.cov(z_pre.T) + np.eye(latent_dim) * 1e-4,
        'stability':  float(z_pre.var(axis=0).mean()),
        'n_obs':      len(z_pre),
        'mean_alpha_pre': float(np.nanmean(y_pre)),
    }

print(f"  Encoded {len(oos_latents_pre)} stocks")

# ── Build pace model from TRAINING stocks ─────────────────────────────────────
print("\nBuilding pace model from training universe...")

# Load training stock centroids and mean alphas
train_centroids, train_alphas = [], []
for ticker in manifest['ticker'].tolist():
    f = DATA_DIR / f'stock_clean_{ticker}.parquet'
    if not f.exists():
        continue
    try:
        df    = pd.read_parquet(f)
        dates = pd.to_datetime(df.index)
        pre   = df[dates < pd.Timestamp(CUTOFF_DATE)]
        if len(pre) < 63:
            continue
        X_pre = prep_X(pre.copy(), feature_cols)
        X_pre_s = scaler.transform(X_pre).astype(np.float32)
        z_pre   = encode_batch(model, X_pre_s)
        y_alpha = pre['alpha_resid'].values if 'alpha_resid' in pre.columns \
                  else np.zeros(len(pre))
        valid = ~np.isnan(y_alpha)
        if valid.sum() < 21:
            continue
        train_centroids.append(z_pre.mean(axis=0))
        train_alphas.append(float(np.nanmean(y_alpha)))
    except Exception:
        continue

train_centroids = np.array(train_centroids)
train_alphas    = np.array(train_alphas)

pace_model = Ridge(alpha=1.0)
pace_model.fit(train_centroids, train_alphas)

# Pace model self-check
r2_pace = float(r2_score(train_alphas, pace_model.predict(train_centroids)))
print(f"  Training stocks for pace model: {len(train_centroids)}")
print(f"  Pace model train R²: {r2_pace:.4f}")


# ── Monte Carlo scouting — fresh stocks ───────────────────────────────────────
print("\n" + "=" * 70)
print("MONTE CARLO SCOUTING REPORT — FRESH STOCKS")
print("=" * 70)
print("Predicting alpha from pre-2023 behavioral fingerprint.")
print("Model has never seen these stocks during training.\n")

mc_results = []

for ticker, ldata in oos_latents_pre.items():
    mu   = ldata['centroid']
    cov  = ldata['cov']
    stab = ldata['stability']
    n    = ldata['n_obs']

    # Monte Carlo: sample from policy distribution
    try:
        samples = RNG.multivariate_normal(mu, cov, size=MC_SAMPLES)
    except np.linalg.LinAlgError:
        samples = mu + RNG.randn(MC_SAMPLES, latent_dim) * np.sqrt(np.diag(cov))

    pred_alphas = pace_model.predict(samples)
    mc_mean     = float(pred_alphas.mean())
    mc_std      = float(pred_alphas.std())
    ci_lo       = float(np.percentile(pred_alphas, 2.5))
    ci_hi       = float(np.percentile(pred_alphas, 97.5))

    # Realized post-2023 alpha
    if ticker in oos_post:
        X_post, y_post, dates_post = oos_post[ticker]
        realized_alpha = float(np.nanmean(y_post))
        n_post         = len(y_post)
        within_ci      = bool(ci_lo <= realized_alpha <= ci_hi)
    else:
        realized_alpha = np.nan
        n_post         = 0
        within_ci      = False

    mc_results.append({
        'ticker':          ticker,
        'stability':       stab,
        'n_pre':           n,
        'n_post':          n_post,
        'mc_mean':         mc_mean,
        'mc_std':          mc_std,
        'ci_lo':           ci_lo,
        'ci_hi':           ci_hi,
        'ci_width':        ci_hi - ci_lo,
        'realized_alpha':  realized_alpha,
        'within_ci':       within_ci,
        'error':           abs(mc_mean - realized_alpha) if not np.isnan(realized_alpha) else np.nan,
        'mean_alpha_pre':  ldata['mean_alpha_pre'],
    })

mc_df = pd.DataFrame(mc_results).dropna(subset=['realized_alpha'])
mc_df = mc_df.sort_values('mc_mean').reset_index(drop=True)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n  Fresh stocks with full pre+post data: {len(mc_df)}")

r2_oos    = float(r2_score(mc_df['realized_alpha'], mc_df['mc_mean']))
corr_oos  = float(pearsonr(mc_df['realized_alpha'], mc_df['mc_mean'])[0])
coverage  = float(mc_df['within_ci'].mean())
mae_oos   = float(mc_df['error'].mean())

r_stab_ci, p_stab_ci = pearsonr(mc_df['stability'], mc_df['ci_width'])
r_stab_err, _        = pearsonr(mc_df['stability'], mc_df['error'])

print(f"\n  Prediction quality (cold — never seen):")
print(f"    R²:              {r2_oos:.4f}  (in-universe: 0.0355)")
print(f"    Correlation:     {corr_oos:.4f}")
print(f"    MAE:             {mae_oos:.5f}")
print(f"    95% CI coverage: {coverage*100:.1f}%  (target: 95%)")
print(f"    Within CI:       {mc_df['within_ci'].sum()}/{len(mc_df)}")

print(f"\n  Stability findings (cold):")
print(f"    Stab → CI width: r={r_stab_ci:.3f}  p={p_stab_ci:.4f}")
print(f"    Stab → error:    r={r_stab_err:.3f}")
print(f"    (In-universe:    r=0.619  p<0.0001)")

# Full scouting table
print(f"\n  {'Ticker':<8} {'Stab':>7} {'MC mean':>9} {'Realized':>10} "
      f"{'Error':>8} {'CI width':>10} {'In CI':>6} {'n_pre':>6}")
print(f"  {'─'*75}")
for _, row in mc_df.iterrows():
    print(f"  {row['ticker']:<8} {row['stability']:>7.4f} "
          f"{row['mc_mean']:>9.5f} {row['realized_alpha']:>10.5f} "
          f"{row['error']:>8.5f} {row['ci_width']:>10.5f} "
          f"{'✓' if row['within_ci'] else '✗':>6} {row['n_pre']:>6}")

# Stability quintile breakdown
if len(mc_df) >= 10:
    mc_df['stab_q'] = pd.qcut(mc_df['stability'], min(5, len(mc_df) // 5),
                                labels=False, duplicates='drop') + 1
    print(f"\n  By stability quintile:")
    print(f"  {'Q':<5} {'n':>4} {'Mean CI':>10} {'Coverage':>10} {'Mean error':>12}")
    print(f"  {'─'*45}")
    for q in sorted(mc_df['stab_q'].unique()):
        sub = mc_df[mc_df['stab_q'] == q]
        print(f"  Q{q:<4} {len(sub):>4} {sub['ci_width'].mean():>10.5f} "
              f"{sub['within_ci'].mean()*100:>9.1f}% "
              f"{sub['error'].mean():>12.5f}")

# Most and least accurate predictions
print(f"\n  Best predictions (smallest error):")
for _, row in mc_df.nsmallest(5, 'error').iterrows():
    print(f"    {row['ticker']:<8} predicted={row['mc_mean']:+.5f}  "
          f"realized={row['realized_alpha']:+.5f}  "
          f"error={row['error']:.5f}  stab={row['stability']:.4f}")

print(f"\n  Worst predictions (largest error):")
for _, row in mc_df.nlargest(5, 'error').iterrows():
    print(f"    {row['ticker']:<8} predicted={row['mc_mean']:+.5f}  "
          f"realized={row['realized_alpha']:+.5f}  "
          f"error={row['error']:.5f}  stab={row['stability']:.4f}")

# ── Latent space position of fresh stocks ────────────────────────────────────
print(f"\n  Where do fresh stocks land in the latent space?")

# Load training centroids for comparison
all_train_centroids = np.array(train_centroids)
oos_centroids       = np.array([oos_latents_pre[t]['centroid']
                                 for t in mc_df['ticker']])

# PCA on training centroids, project OOS onto same space
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(all_train_centroids)

train_proj = pca.transform(all_train_centroids)
oos_proj   = pca.transform(oos_centroids)

# Distance of each OOS stock from training distribution center
train_center    = all_train_centroids.mean(axis=0)
oos_dist_center = np.array([np.linalg.norm(oos_latents_pre[t]['centroid'] - train_center)
                             for t in mc_df['ticker']])

print(f"    Mean distance from training centroid: {oos_dist_center.mean():.4f}")
print(f"    Std distance from training centroid:  {oos_dist_center.std():.4f}")
print(f"    Range: {oos_dist_center.min():.4f} — {oos_dist_center.max():.4f}")

r_dist_err, p_dist_err = pearsonr(oos_dist_center, mc_df['error'].values)
print(f"    Distance from center → prediction error: r={r_dist_err:.3f}  p={p_dist_err:.4f}")
print(f"    (Does interpolation work better than extrapolation?)")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Scouting report — predicted vs realized with CIs
axes[0, 0].errorbar(range(len(mc_df)), mc_df['mc_mean'],
                    yerr=[mc_df['mc_mean'] - mc_df['ci_lo'],
                          mc_df['ci_hi']   - mc_df['mc_mean']],
                    fmt='none', alpha=0.3, color='steelblue', lw=0.8)
axes[0, 0].scatter(range(len(mc_df)), mc_df['realized_alpha'],
                   s=20, color='red', alpha=0.7, zorder=3, label='Realized α')
axes[0, 0].scatter(range(len(mc_df)), mc_df['mc_mean'],
                   s=20, color='steelblue', alpha=0.7, zorder=3, label='MC prediction')
axes[0, 0].set_xlabel('Stock (sorted by predicted α)')
axes[0, 0].set_title(f'OOS Scouting Report\nR²={r2_oos:.4f}  coverage={coverage*100:.1f}%')
axes[0, 0].legend(fontsize=8)

# 2. Predicted vs realized scatter
axes[0, 1].scatter(mc_df['mc_mean'], mc_df['realized_alpha'], alpha=0.6, s=25)
lims = [min(mc_df['mc_mean'].min(), mc_df['realized_alpha'].min()) - 0.001,
        max(mc_df['mc_mean'].max(), mc_df['realized_alpha'].max()) + 0.001]
axes[0, 1].plot(lims, lims, 'r--', lw=1, alpha=0.5)
axes[0, 1].set_xlabel('MC predicted alpha')
axes[0, 1].set_ylabel('Realized alpha (post-2023)')
axes[0, 1].set_title(f'Predicted vs Realized\nr={corr_oos:.3f}  (never-seen stocks)')

# 3. Stability → CI width
axes[0, 2].scatter(mc_df['stability'], mc_df['ci_width'], alpha=0.6, s=25,
                   color='darkorange')
axes[0, 2].set_xlabel('Manifold Stability')
axes[0, 2].set_ylabel('CI Width')
axes[0, 2].set_title(f'Stability → CI Width (OOS)\nr={r_stab_ci:.3f}  p={p_stab_ci:.4f}')

# 4. Stability → prediction error
axes[1, 0].scatter(mc_df['stability'], mc_df['error'], alpha=0.6, s=25, color='purple')
axes[1, 0].set_xlabel('Manifold Stability')
axes[1, 0].set_ylabel('Prediction error |predicted - realized|')
axes[1, 0].set_title(f'Stability → Prediction Error\nr={r_stab_err:.3f}')

# 5. Latent space — training vs OOS
axes[1, 1].scatter(train_proj[:, 0], train_proj[:, 1], alpha=0.2, s=5,
                   color='steelblue', label='Training universe')
axes[1, 1].scatter(oos_proj[:, 0], oos_proj[:, 1], alpha=0.8, s=40,
                   color='red', zorder=3, label='OOS stocks (never seen)')
axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1, 1].set_title('Latent Space Position\nTraining vs Never-Seen Stocks')
axes[1, 1].legend(fontsize=8)

# 6. Distance from training center → error
axes[1, 2].scatter(oos_dist_center, mc_df['error'].values, alpha=0.6, s=25,
                   color='green')
axes[1, 2].set_xlabel('Distance from training centroid')
axes[1, 2].set_ylabel('Prediction error')
axes[1, 2].set_title(f'Extrapolation vs Interpolation\nr={r_dist_err:.3f}  '
                     f'p={p_dist_err:.4f}')

plt.suptitle('Out-of-Universe Validation — Never-Seen Stocks\n'
             '(Analog: Russell at Mercedes prediction)', fontsize=11)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'oos_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Plot saved: oos_validation.png")

# ── Save results ──────────────────────────────────────────────────────────────
mc_df.to_parquet(OOS_DIR / 'oos_mc_results.parquet')
mc_df.to_csv(OOS_DIR / 'oos_mc_results.csv', index=False)

summary = {
    'n_stocks':       len(mc_df),
    'r2':             float(r2_oos),
    'corr':           float(corr_oos),
    'mae':            float(mae_oos),
    'coverage_95':    float(coverage),
    'within_ci':      int(mc_df['within_ci'].sum()),
    'r_stab_ci':      float(r_stab_ci),
    'p_stab_ci':      float(p_stab_ci),
    'r_stab_err':     float(r_stab_err),
    'r_dist_err':     float(r_dist_err),
    'p_dist_err':     float(p_dist_err),
    'comparison': {
        'in_universe_r2':       0.0355,
        'in_universe_stab_ci':  0.619,
        'in_universe_coverage': 'N/A',
    }
}

with open(OOS_DIR / 'oos_validation_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*70}")
print(f"OOS VALIDATION COMPLETE")
print(f"{'='*70}")
print(f"  Stocks:       {len(mc_df)}")
print(f"  R²:           {r2_oos:.4f}  (in-universe: 0.0355)")
print(f"  Coverage:     {coverage*100:.1f}%")
print(f"  Stab→CI r:    {r_stab_ci:.3f}  (in-universe: 0.619)")
print(f"\n  Results: {OOS_DIR / 'oos_validation_results.json'}")
print(f"  MC data: {OOS_DIR / 'oos_mc_results.parquet'}")
print(f"  Plots:   {PLOT_DIR}")
print("\nDone.")