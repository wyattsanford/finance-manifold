# -*- coding: utf-8 -*-
"""
stock_velocity_backtest_clean.py
Walk-forward clean backtest of the latent velocity signal.

Architecture:
  - Semi-annual rolling windows
  - Each encoder trained ONLY on data up to window start
  - Velocity computed on that window using that encoder
  - Pace model also rolling — fit on same training window
  - 4 encoders trained in parallel (multiprocessing)
  - No lookahead anywhere

Timeline:
  Train encoder on years T-3 to T
  Compute velocity + trade on T to T+0.5
  Roll forward 6 months, repeat

Starts: first window with 3 years of training data (~2003)
Ends:   last full 6-month window before 2023 cutoff

Saves:
  rolling_backtest_results.parquet  — per-period returns
  rolling_backtest_summary.json     — performance stats
  plots/rolling_backtest/           — plots
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, rankdata
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
PLOT_DIR = DATA_DIR / 'plots' / 'rolling_backtest'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_YEARS    = 3       # encoder training window length
TRADE_MONTHS   = 6       # trading window length
N_PARALLEL     = 4       # parallel encoder training jobs
VEL_WINDOW     = 21      # days for velocity computation
LATENT_DIM     = 12
BATCH_SIZE     = 32768
LR             = 1e-3
MAX_EPOCHS     = 60      # fewer epochs per window — faster, still converges
PATIENCE       = 7
HUBER_DELTA    = 1.0
DATA_START     = '2000-01-01'
BACKTEST_START = '2003-07-01'   # first trade date (needs 3yr training)
BACKTEST_END   = '2022-12-31'   # last trade date
CUTOFF_DATE    = '2023-01-01'

ALPHAS_GRID = np.logspace(-2, 4, 30)   # for RidgeCV pace model

print(f"Config: {TRAIN_YEARS}yr train | {TRADE_MONTHS}mo trade | "
      f"{N_PARALLEL} parallel | latent={LATENT_DIM}D")

# ── Model definition (must be picklable for multiprocessing) ──────────────────
class StockDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


def make_ae(input_dim, latent_dim):
    h1 = max(32, input_dim * 2)
    h2 = max(16, input_dim)

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


def encode_data(model_state, input_dim, latent_dim, X_scaled, device_str='cuda'):
    """Encode a dataset using a saved model state dict."""
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    model  = make_ae(input_dim, latent_dim).to(device)
    model.load_state_dict(model_state)
    model.eval()

    loader = DataLoader(StockDataset(X_scaled.astype(np.float32)),
                        batch_size=8192, shuffle=False)
    parts = []
    with torch.no_grad():
        for xb in loader:
            _, z = model(xb.to(device))
            parts.append(z.cpu().numpy())
    return np.vstack(parts)


def clean_X(X):
    X = np.where(np.isinf(X), np.nan, X)
    col_meds = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = col_meds[j]
    return np.nan_to_num(X, nan=0.0)


# ── Worker function for parallel training ─────────────────────────────────────
def train_window_encoder(args):
    """
    Train one encoder for one time window.
    Runs in a subprocess — each gets its own CUDA context.
    Returns (window_id, model_state_dict, scaler, val_loss)
    """
    (window_id, X_train, input_dim, latent_dim,
     batch_size, lr, max_epochs, patience, huber_delta, gpu_id) = args

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)

    # Split train/val (80/20 by time — no shuffle to avoid lookahead)
    n_train  = int(len(X_scaled) * 0.85)
    Xtr      = X_scaled[:n_train]
    Xvl      = X_scaled[n_train:]

    model    = make_ae(input_dim, latent_dim).to(device)
    opt      = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit     = nn.HuberLoss(delta=huber_delta)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(
                   opt, T_max=max_epochs, eta_min=1e-5)

    tr_loader = DataLoader(StockDataset(Xtr), batch_size=batch_size,
                           shuffle=True, num_workers=0)

    best_val  = float('inf')
    pat_ctr   = 0
    best_state= None

    for epoch in range(max_epochs):
        model.train()
        for xb in tr_loader:
            xb = xb.to(device)
            opt.zero_grad()
            xr, _ = model(xb)
            loss   = crit(xr, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            xvt     = torch.FloatTensor(Xvl).to(device)
            val_loss= crit(model(xvt)[0], xvt).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_ctr    = 0
        else:
            pat_ctr += 1
        if pat_ctr >= patience:
            break

    return window_id, best_state, scaler, best_val


# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading feature spec from temporal checkpoint...")
ckpt_ref     = torch.load(DATA_DIR / 'ae_temporal_best.pt',
                           map_location='cpu', weights_only=False)
feature_cols = ckpt_ref['feature_cols']
input_dim    = len(feature_cols)
del ckpt_ref

manifest    = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

print(f"Loading all stock data ({DATA_START} — {BACKTEST_END})...")

# Load per-stock data into memory once
STOCK_CACHE = DATA_DIR / 'rolling_bt_stock_cache.parquet'
if STOCK_CACHE.exists():
    print("  Loading from cache...")
    stock_df = pd.read_parquet(STOCK_CACHE)
else:
    frames = []
    for ticker in all_tickers:
        f = DATA_DIR / f'stock_clean_{ticker}.parquet'
        if not f.exists():
            continue
        try:
            df    = pd.read_parquet(f)
            dates = pd.to_datetime(df.index)
            mask  = (dates >= pd.Timestamp(DATA_START)) & \
                    (dates <= pd.Timestamp(BACKTEST_END))
            df    = df[mask].copy()
            if len(df) < 252:
                continue
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            df['ticker'] = ticker
            frames.append(df[feature_cols + ['ticker', 'ret_1d', 'alpha_resid']])
        except Exception:
            continue

    stock_df = pd.concat(frames).sort_index()
    stock_df.to_parquet(STOCK_CACHE)
    print(f"  Saved cache: {STOCK_CACHE.name}")

stock_df.index = pd.to_datetime(stock_df.index)
print(f"  {len(stock_df):,} observations  |  "
      f"{stock_df['ticker'].nunique()} stocks")

# ── Build rolling windows ─────────────────────────────────────────────────────
print("\nBuilding rolling windows...")
windows = []
current = pd.Timestamp(BACKTEST_START)
end_ts  = pd.Timestamp(BACKTEST_END)

while current <= end_ts:
    train_start = current - pd.DateOffset(years=TRAIN_YEARS)
    train_end   = current
    trade_start = current
    trade_end   = current + pd.DateOffset(months=TRADE_MONTHS)

    if trade_end > end_ts + pd.DateOffset(months=1):
        break

    windows.append({
        'id':          len(windows),
        'train_start': train_start,
        'train_end':   train_end,
        'trade_start': trade_start,
        'trade_end':   trade_end,
    })
    current += pd.DateOffset(months=TRADE_MONTHS)

print(f"  Total windows: {len(windows)}")
print(f"  First window: train {windows[0]['train_start'].date()} — "
      f"{windows[0]['train_end'].date()}, "
      f"trade {windows[0]['trade_start'].date()} — "
      f"{windows[0]['trade_end'].date()}")
print(f"  Last window:  train {windows[-1]['train_start'].date()} — "
      f"{windows[-1]['train_end'].date()}, "
      f"trade {windows[-1]['trade_start'].date()} — "
      f"{windows[-1]['trade_end'].date()}")

# ── Check for existing window results (resume support) ────────────────────────
WINDOW_CACHE_DIR = DATA_DIR / 'rolling_bt_windows'
WINDOW_CACHE_DIR.mkdir(exist_ok=True)

completed_windows = {
    int(f.stem.split('_')[1])
    for f in WINDOW_CACHE_DIR.glob('window_*.pkl')
}
print(f"  Already completed: {len(completed_windows)}/{len(windows)} windows")

# ── Train encoders in parallel batches ───────────────────────────────────────
print(f"\nTraining {len(windows) - len(completed_windows)} remaining encoders "
      f"({N_PARALLEL} at a time)...")

pending = [w for w in windows if w['id'] not in completed_windows]

# Build training arrays for each pending window
def get_train_array(window):
    mask = (stock_df.index >= window['train_start']) & \
           (stock_df.index <  window['train_end'])
    sub  = stock_df[mask]
    if len(sub) < 10_000:
        return None
    X = clean_X(sub[feature_cols].values.astype(np.float32))
    return X

# Process in batches of N_PARALLEL
for batch_start in range(0, len(pending), N_PARALLEL):
    batch = pending[batch_start:batch_start + N_PARALLEL]

    # Build args for each window in batch
    args_list = []
    valid_batch = []
    for i, w in enumerate(batch):
        X_train = get_train_array(w)
        if X_train is None:
            print(f"  Window {w['id']}: insufficient data, skipping")
            continue
        gpu_id = i % N_PARALLEL   # assign to GPU 0 since single GPU
        # All on GPU 0 — PyTorch handles concurrent CUDA streams
        args_list.append((
            w['id'], X_train, input_dim, LATENT_DIM,
            BATCH_SIZE, LR, MAX_EPOCHS, PATIENCE, HUBER_DELTA, 0
        ))
        valid_batch.append(w)

    if not args_list:
        continue

    print(f"  Batch {batch_start//N_PARALLEL + 1}: "
          f"windows {[w['id'] for w in valid_batch]} "
          f"(train sizes: {[get_train_array(w).shape[0] if get_train_array(w) is not None else 0 for w in valid_batch]})")

    # Run sequentially on single GPU (multiprocessing with CUDA is complex)
    # Each window takes ~5-8 min, batches of 4 = ~25 min per batch
    # For true parallelism on single GPU, threads share GPU fine
    import concurrent.futures
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_PARALLEL) as executor:
        futures = {executor.submit(train_window_encoder, args): args[0]
                   for args in args_list}
        for future in concurrent.futures.as_completed(futures):
            window_id = futures[future]
            try:
                wid, state, scaler, val_loss = future.result()
                results.append((wid, state, scaler, val_loss))
                print(f"    Window {wid} done — val_loss={val_loss:.4f}")
            except Exception as e:
                print(f"    Window {window_id} failed: {e}")

    # Save each result
    for wid, state, scaler, val_loss in results:
        w = next(x for x in windows if x['id'] == wid)
        cache = {
            'window_id':   wid,
            'train_start': w['train_start'],
            'train_end':   w['train_end'],
            'trade_start': w['trade_start'],
            'trade_end':   w['trade_end'],
            'model_state': state,
            'scaler':      scaler,
            'val_loss':    val_loss,
        }
        with open(WINDOW_CACHE_DIR / f'window_{wid:03d}.pkl', 'wb') as f:
            pickle.dump(cache, f)

    print(f"  Batch complete — {len(completed_windows) + len(results)}/{len(windows)} total")
    completed_windows.update(wid for wid, *_ in results)

print(f"\nAll {len(windows)} encoders trained.")

# ── Compute velocity and run backtest ─────────────────────────────────────────
print("\nComputing velocity signals and running backtest...")

portfolio_returns = []
window_stats      = []

for w in windows:
    wid = w['id']
    cache_path = WINDOW_CACHE_DIR / f'window_{wid:03d}.pkl'
    if not cache_path.exists():
        print(f"  Window {wid}: no encoder, skipping")
        continue

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    model_state = cache['model_state']
    scaler      = cache['scaler']

    # ── Encode training window → build pace model ──────────────────────────
    train_mask = (stock_df.index >= w['train_start']) & \
                 (stock_df.index <  w['train_end'])
    train_sub  = stock_df[train_mask]

    if len(train_sub) < 10_000:
        continue

    X_train = clean_X(train_sub[feature_cols].values.astype(np.float32))
    X_train_s = scaler.transform(X_train).astype(np.float32)
    Z_train   = encode_data(model_state, input_dim, LATENT_DIM, X_train_s)

    # Per-stock centroids and mean alpha in training window
    train_tickers = train_sub['ticker'].values
    train_alphas  = train_sub['alpha_resid'].values \
        if 'alpha_resid' in train_sub.columns else np.zeros(len(train_sub))

    stock_centroids, stock_alphas = [], []
    for ticker in np.unique(train_tickers):
        mask_t = train_tickers == ticker
        if mask_t.sum() < 63:
            continue
        z_t     = Z_train[mask_t]
        alpha_t = train_alphas[mask_t]
        valid   = ~np.isnan(alpha_t)
        if valid.sum() < 21:
            continue
        stock_centroids.append(z_t.mean(axis=0))
        stock_alphas.append(float(np.nanmean(alpha_t[valid])))

    if len(stock_centroids) < 20:
        continue

    # Fit rolling pace model
    C = np.array(stock_centroids)
    a = np.array(stock_alphas)
    a_w = np.clip(a, np.percentile(a, 1), np.percentile(a, 99))
    pace_model = RidgeCV(alphas=ALPHAS_GRID, cv=5)
    pace_model.fit(C, a_w)

    # ── Encode trading window → compute velocity ───────────────────────────
    trade_mask = (stock_df.index >= w['trade_start']) & \
                 (stock_df.index <  w['trade_end'])
    trade_sub  = stock_df[trade_mask].copy()

    if len(trade_sub) < 1000:
        continue

    X_trade   = clean_X(trade_sub[feature_cols].values.astype(np.float32))
    X_trade_s = scaler.transform(X_trade).astype(np.float32)
    Z_trade   = encode_data(model_state, input_dim, LATENT_DIM, X_trade_s)

    trade_dates   = trade_sub.index
    trade_tickers = trade_sub['ticker'].values
    trade_rets    = trade_sub['ret_1d'].values \
        if 'ret_1d' in trade_sub.columns else np.zeros(len(trade_sub))

    # Compute rolling velocity per stock per date
    # Encode the full lookback+trade window at once, then split by ticker
    unique_trade_dates = np.sort(np.unique(trade_dates))
    stock_vel_by_date  = {}   # {date: {ticker: velocity}}

    lookback_start = w['trade_start'] - pd.DateOffset(days=VEL_WINDOW * 3)
    lb_mask = (stock_df.index >= lookback_start) & \
              (stock_df.index <  w['trade_end'])
    lb_sub  = stock_df[lb_mask].copy()

    if len(lb_sub) > 1000:
        # Encode entire window at once
        X_lb_all   = clean_X(lb_sub[feature_cols].values.astype(np.float32))
        X_lb_all_s = scaler.transform(X_lb_all).astype(np.float32)
        Z_lb_all   = encode_data(model_state, input_dim, LATENT_DIM, X_lb_all_s)

        lb_dates   = lb_sub.index
        lb_tickers = lb_sub['ticker'].values

        # Split by ticker and compute velocity
        for ticker in np.unique(lb_tickers):
            mask_t   = lb_tickers == ticker
            z_t      = Z_lb_all[mask_t]
            dates_t  = lb_dates[mask_t]

            if len(z_t) < VEL_WINDOW + 5:
                continue

            steps = np.linalg.norm(np.diff(z_t, axis=0), axis=1)

            for i_t in range(VEL_WINDOW, len(z_t)):
                d = dates_t[i_t]
                if d < w['trade_start'] or d >= w['trade_end']:
                    continue
                vel = float(steps[i_t-VEL_WINDOW:i_t].mean())
                if d not in stock_vel_by_date:
                    stock_vel_by_date[d] = {}
                stock_vel_by_date[d][ticker] = vel

    print(f"    Velocity dates populated: {len(stock_vel_by_date)}")

    # ── Monthly rebalancing within trade window ────────────────────────────
    rebal_dates = pd.date_range(w['trade_start'], w['trade_end'],
                                freq='21D')[:-1]

    # Convert stock_vel_by_date keys to pandas Timestamps for consistent comparison
    stock_vel_by_date = {pd.Timestamp(k): v for k, v in stock_vel_by_date.items()}
    unique_trade_dates_ts = sorted([pd.Timestamp(d) for d in unique_trade_dates])

    for rebal_date in rebal_dates:
        # Find nearest available date with velocity data
        available = [d for d in unique_trade_dates_ts
                     if d >= rebal_date and d in stock_vel_by_date]
        if not available:
            continue
        actual_date = available[0]

        vel_dict = stock_vel_by_date[actual_date]
        if len(vel_dict) < 20:
            continue

        tickers_d = list(vel_dict.keys())
        vels_d    = np.array([vel_dict[t] for t in tickers_d])

        # Cross-sectional rank
        vel_ranks = rankdata(vels_d) / len(vels_d)
        long_mask  = vel_ranks <= 0.20   # low velocity = long
        short_mask = vel_ranks >= 0.80   # high velocity = short

        # Forward 21-day returns
        fwd_end  = actual_date + pd.DateOffset(days=30)
        fwd_mask = (stock_df.index > actual_date) & \
                   (stock_df.index <= fwd_end)
        fwd_sub  = stock_df[fwd_mask]

        long_rets, short_rets = [], []
        long_alphas, short_alphas = [], []

        for i, ticker in enumerate(tickers_d):
            t_fwd = fwd_sub[fwd_sub['ticker'] == ticker]
            if len(t_fwd) < 5:
                continue
            fwd_ret   = float(t_fwd['ret_1d'].mean()) \
                if 'ret_1d' in t_fwd.columns else np.nan
            fwd_alpha = float(t_fwd['alpha_resid'].mean()) \
                if 'alpha_resid' in t_fwd.columns else np.nan

            if np.isnan(fwd_ret):
                continue

            if long_mask[i]:
                long_rets.append(fwd_ret)
                if not np.isnan(fwd_alpha):
                    long_alphas.append(fwd_alpha)
            elif short_mask[i]:
                short_rets.append(fwd_ret)
                if not np.isnan(fwd_alpha):
                    short_alphas.append(fwd_alpha)

        if not long_rets or not short_rets:
            continue

        ls_ret   = float(np.mean(long_rets))   - float(np.mean(short_rets))
        ls_alpha = float(np.mean(long_alphas)) - float(np.mean(short_alphas)) \
                   if long_alphas and short_alphas else np.nan

        portfolio_returns.append({
            'window_id':   wid,
            'rebal_date':  actual_date,
            'ls_ret':      ls_ret,
            'ls_alpha':    ls_alpha,
            'long_ret':    float(np.mean(long_rets)),
            'short_ret':   float(np.mean(short_rets)),
            'n_long':      int(long_mask.sum()),
            'n_short':     int(short_mask.sum()),
            'encoder_val_loss': cache['val_loss'],
        })

    window_stats.append({
        'window_id':    wid,
        'train_start':  str(w['train_start'].date()),
        'trade_start':  str(w['trade_start'].date()),
        'n_train_obs':  int(train_mask.sum()),
        'n_trade_obs':  int(trade_mask.sum()),
        'val_loss':     float(cache['val_loss']),
        'pace_alpha':   float(pace_model.alpha_),
    })

    print(f"  Window {wid} ({w['trade_start'].date()} — {w['trade_end'].date()}): "
          f"{len([r for r in portfolio_returns if r['window_id']==wid])} rebal periods")

# ── Performance analysis ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLEAN ROLLING BACKTEST RESULTS")
print("=" * 70)

port_df = pd.DataFrame(portfolio_returns)
port_df['rebal_date'] = pd.to_datetime(port_df['rebal_date'])
port_df = port_df.sort_values('rebal_date').reset_index(drop=True)

if len(port_df) == 0:
    print("No results — check window data coverage")
else:
    ls_rets   = port_df['ls_ret'].values
    ls_alphas = port_df['ls_alpha'].dropna().values

    mean_ls   = float(ls_rets.mean())
    std_ls    = float(ls_rets.std())
    ann_ret   = mean_ls * 252 / 21
    ann_std   = std_ls * np.sqrt(252 / 21)
    sharpe    = ann_ret / (ann_std + 1e-8)
    hit_rate  = float((ls_rets > 0).mean())

    from scipy.stats import ttest_1samp
    t_stat, p_val = ttest_1samp(ls_rets, 0)

    print(f"\n  Periods:           {len(port_df)}")
    print(f"  Date range:        {port_df['rebal_date'].min().date()} — "
          f"{port_df['rebal_date'].max().date()}")
    print(f"\n  Mean period return: {mean_ls*100:.4f}%")
    print(f"  Annualized return:  {ann_ret*100:.2f}%")
    print(f"  Annualized vol:     {ann_std*100:.2f}%")
    print(f"  Annualized Sharpe:  {sharpe:.3f}")
    print(f"  Hit rate:           {hit_rate*100:.1f}%")
    print(f"  p-value:            {p_val:.4f}")

    # Alpha-adjusted
    mean_ls_a = float(ls_alphas.mean()) if len(ls_alphas) > 0 else np.nan
    ann_alpha = mean_ls_a * 252 / 21 if not np.isnan(mean_ls_a) else np.nan
    print(f"\n  Alpha-adjusted:")
    print(f"  Mean period alpha:  {mean_ls_a*100:.4f}%")
    print(f"  Annualized alpha:   {ann_alpha*100:.2f}%")

    # Compare to original (contaminated) backtest
    print(f"\n  Comparison:")
    print(f"  {'Metric':<25} {'Clean (walk-fwd)':>18} {'Original (full AE)':>20}")
    print(f"  {'─'*65}")
    print(f"  {'Ann. return':<25} {ann_ret*100:>17.2f}% {'0.86%':>20}")
    print(f"  {'Sharpe':<25} {sharpe:>18.3f} {'1.836':>20}")
    print(f"  {'Hit rate':<25} {hit_rate*100:>17.1f}% {'76.8%':>20}")
    print(f"  {'p-value':<25} {p_val:>18.4f} {'0.0000':>20}")

    # Crisis performance
    print(f"\n  Performance by period:")
    for name, start, end in [('Pre-GFC (2003-07)', '2003-01-01', '2007-12-31'),
                              ('GFC (2008-09)',    '2008-01-01', '2009-12-31'),
                              ('Recovery',        '2010-01-01', '2014-12-31'),
                              ('Bull (2015-19)',   '2015-01-01', '2019-12-31'),
                              ('COVID (2020)',     '2020-01-01', '2020-12-31'),
                              ('Post-COVID',       '2021-01-01', '2022-12-31')]:
        mask_p = (port_df['rebal_date'] >= start) & (port_df['rebal_date'] <= end)
        if mask_p.sum() < 3:
            continue
        pr = port_df.loc[mask_p, 'ls_ret'].mean() * 252 / 21
        print(f"    {name:<22} {pr*100:+.2f}% annualized  "
              f"(n={mask_p.sum()})")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Cumulative return
    cumret = (1 + port_df['ls_ret']).cumprod()
    axes[0, 0].plot(port_df['rebal_date'], cumret.values,
                    lw=1.5, color='steelblue', label='Clean walk-forward')
    axes[0, 0].axhline(1, color='black', lw=0.8, linestyle=':')
    for name, start, end, color in [
            ('GFC', '2008-01-01', '2009-12-31', 'red'),
            ('COVID', '2020-01-01', '2020-12-31', 'orange')]:
        axes[0, 0].axvspan(pd.Timestamp(start), pd.Timestamp(end),
                           alpha=0.1, color=color)
    axes[0, 0].set_title(f'Cumulative L/S Return (Clean)\n'
                         f'Ann={ann_ret*100:.1f}%  Sharpe={sharpe:.2f}  '
                         f'Hit={hit_rate*100:.0f}%')
    axes[0, 0].set_ylabel('Cumulative return')
    axes[0, 0].legend(fontsize=8)

    # Period returns distribution
    axes[0, 1].hist(ls_rets * 100, bins=40, color='steelblue', alpha=0.7)
    axes[0, 1].axvline(0, color='black', lw=1)
    axes[0, 1].axvline(mean_ls * 100, color='crimson', lw=2, linestyle='--',
                       label=f'Mean={mean_ls*100:.3f}%')
    axes[0, 1].set_xlabel('Period L/S return (%)')
    axes[0, 1].set_title(f'Return Distribution\np={p_val:.4f}')
    axes[0, 1].legend(fontsize=8)

    # Annual performance
    port_df['year'] = port_df['rebal_date'].dt.year
    annual = port_df.groupby('year')['ls_ret'].mean() * 252 / 21
    colors_bar = ['steelblue' if r > 0 else 'crimson' for r in annual.values]
    axes[1, 0].bar(annual.index, annual.values * 100, color=colors_bar, alpha=0.8)
    axes[1, 0].axhline(0, color='black', lw=0.8)
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Annualized return (%)')
    axes[1, 0].set_title('Annual Performance (Clean Walk-Forward)')

    # Rolling Sharpe (2-year window)
    rolling_ret  = port_df.set_index('rebal_date')['ls_ret'].rolling(24)
    rolling_sharpe = (rolling_ret.mean() / (rolling_ret.std() + 1e-8)) * \
                     np.sqrt(252 / 21)
    axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values,
                    lw=1.5, color='darkorange')
    axes[1, 1].axhline(0, color='black', lw=0.8, linestyle=':')
    axes[1, 1].axhline(sharpe, color='darkorange', lw=1, linestyle='--',
                       label=f'Full period={sharpe:.2f}')
    axes[1, 1].set_ylabel('Rolling 2yr Sharpe')
    axes[1, 1].set_title('Rolling Sharpe (2yr window)')
    axes[1, 1].legend(fontsize=8)

    plt.suptitle('Latent Velocity L/S Backtest — Clean Walk-Forward\n'
                 '(Each encoder trained only on past data)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'rolling_backtest.png', dpi=150)
    plt.close()
    print(f"\n  Plot saved: rolling_backtest.png")

    # Save results
    port_df.to_parquet(DATA_DIR / 'rolling_backtest_results.parquet')

    summary = {
        'ann_return':    float(ann_ret),
        'ann_vol':       float(ann_std),
        'sharpe':        float(sharpe),
        'hit_rate':      float(hit_rate),
        'p_value':       float(p_val),
        'ann_alpha':     float(ann_alpha) if not np.isnan(ann_alpha) else None,
        'n_periods':     len(port_df),
        'date_range':    [str(port_df['rebal_date'].min().date()),
                          str(port_df['rebal_date'].max().date())],
        'n_windows':     len(window_stats),
        'window_stats':  window_stats,
        'comparison': {
            'original_sharpe':   1.836,
            'original_ann_ret':  0.0086,
            'original_hit_rate': 0.768,
        }
    }

    with open(DATA_DIR / 'rolling_backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: rolling_backtest_results.parquet")
    print(f"Summary: rolling_backtest_summary.json")

print("\nDone.")