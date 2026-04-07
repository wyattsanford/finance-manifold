# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:15:43 2026

@author: Justin.Sanford
"""

# stock_universe_pull.py
# Build a random sample of 2000 stocks from the broadest possible universe
# No quality filters — everything that has ever traded with sufficient history
# Failsafe: skip failed tickers and draw replacements until target is met
#
# Universe sources:
#   1. SEC EDGAR — all companies that have ever filed (broadest possible)
#   2. yfinance screener — currently active tickers
#   3. Historical S&P 500 / Russell 2000 constituents (Wikipedia)
#   4. Known delisted ticker lists
#
# Output: stock_universe.parquet — 2000 stocks with daily OHLCV + features

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import json
import time
import random
import zipfile
import io
import re
from datetime import datetime, timedelta

DATA_DIR    = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = DATA_DIR / 'universe_checkpoint.json'
HEADERS = {
    'User-Agent': 'Justin Sanford justin.sanford@dlhcorp.com',
}

TARGET_STOCKS  = 2000
MIN_DAYS       = 252        # at least 1 year of history
START_DATE     = '2000-01-01'
END_DATE       = '2024-12-31'
MIN_START_DATE = '2020-01-01'  # must have data before this date

# ── Source 1: SEC EDGAR company tickers ──────────────────────────────────────
def get_sec_tickers():
    print("  Fetching SEC EDGAR tickers...")
    tickers = set()
    try:
        r = requests.get(
            'https://www.sec.gov/files/company_tickers_exchange.json',
            headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data['data'], columns=data['fields'])
        tickers.update(df['ticker'].dropna().str.upper().tolist())
        print(f"    SEC exchange tickers: {len(tickers)}")
    except Exception as e:
        print(f"    SEC exchange failed: {e}")

    try:
        r = requests.get(
            'https://www.sec.gov/files/company_tickers.json',
            headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data.values())
        tickers.update(df['ticker'].dropna().str.upper().tolist())
        print(f"    SEC all tickers: {len(tickers)}")
    except Exception as e:
        print(f"    SEC all failed: {e}")

    return tickers

# ── Source 2: Wikipedia S&P 500 historical constituents ──────────────────────
def get_wikipedia_tickers():
    print("  Fetching Wikipedia constituent lists...")
    tickers = set()

    urls = [
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
    ]

    for url in urls:
        try:
            tables = pd.read_html(url)
            for tbl in tables:
                for col in tbl.columns:
                    if any(kw in str(col).upper()
                           for kw in ['TICKER', 'SYMBOL', 'TRADE']):
                        tickers.update(
                            tbl[col].dropna().str.upper()
                            .str.replace(r'[^A-Z]', '', regex=True)
                            .tolist()
                        )
            print(f"    Wikipedia {url.split('/')[-1][:30]}: "
                  f"{len(tickers)} tickers")
            time.sleep(0.5)
        except Exception as e:
            print(f"    Wikipedia failed: {e}")

    return tickers

# ── Source 3: yfinance screener ───────────────────────────────────────────────
def get_yfinance_tickers():
    print("  Fetching yfinance screener tickers...")
    tickers = set()
    try:
        # Most active, most gained, most lost — broad coverage
        for screen in ['most_actives', 'day_gainers', 'day_losers',
                        'growth_technology_stocks', 'undervalued_large_caps',
                        'undervalued_growth_stocks', 'aggressive_small_caps']:
            try:
                result = yf.screen(screen, size=100)
                if result and 'quotes' in result:
                    for q in result['quotes']:
                        sym = q.get('symbol', '')
                        if sym:
                            tickers.add(sym.upper())
                time.sleep(0.2)
            except Exception:
                pass
        print(f"    yfinance screener: {len(tickers)} tickers")
    except Exception as e:
        print(f"    yfinance screener failed: {e}")
    return tickers

# ── Source 4: Known delisted / historical tickers ─────────────────────────────
def get_delisted_tickers():
    """
    Pull from a few public lists of historically traded tickers.
    Stooq maintains a broad list of US tickers including historical ones.
    """
    print("  Fetching Stooq US ticker list...")
    tickers = set()
    try:
        # Stooq publishes daily data for many US stocks including delisted
        # Their ticker list is publicly accessible
        url = 'https://stooq.com/t/?i=513'  # US stocks list
        r   = requests.get(url, timeout=30,
                          headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code == 200:
            # Parse the HTML table
            tables = pd.read_html(r.text)
            for tbl in tables:
                for col in tbl.columns:
                    if 'symbol' in str(col).lower():
                        syms = (tbl[col].dropna()
                                .str.upper()
                                .str.replace('.US', '', regex=False)
                                .tolist())
                        tickers.update(syms)
            print(f"    Stooq tickers: {len(tickers)}")
    except Exception as e:
        print(f"    Stooq failed: {e}")

    return tickers

# ── Validate ticker ────────────────────────────────────────────────────────────
def validate_ticker(ticker):
    """
    Try to pull price history for a ticker.
    Returns (ticker, df) if valid, (ticker, None) if not.
    Valid = at least MIN_DAYS of data with first date before MIN_START_DATE.
    """
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(start=START_DATE, end=END_DATE,
                         auto_adjust=True)

        if hist is None or len(hist) < MIN_DAYS:
            return ticker, None

        hist.index = pd.to_datetime(hist.index).tz_localize(None)

        # Must have data before MIN_START_DATE to avoid very new stocks
        if hist.index[0] > pd.Timestamp(MIN_START_DATE):
            return ticker, None

        # Must have actual price variation (not a dead ticker)
        if hist['Close'].std() < 1e-6:
            return ticker, None

        return ticker, hist

    except Exception:
        return ticker, None

# ── Feature engineering per stock ─────────────────────────────────────────────
def compute_features(hist, ticker):
    """
    Compute daily feature vector for a stock from OHLCV history.
    Returns DataFrame with one row per trading day.
    No assumptions about what matters — compute everything.
    """
    df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.ffill().dropna()

    if len(df) < 60:
        return None

    # ── Returns ───────────────────────────────────────────────────────────────
    df['ret_1d']   = df['close'].pct_change(1)
    df['ret_5d']   = df['close'].pct_change(5)
    df['ret_21d']  = df['close'].pct_change(21)
    df['ret_63d']  = df['close'].pct_change(63)
    df['ret_252d'] = df['close'].pct_change(252)

    # ── Volatility ────────────────────────────────────────────────────────────
    df['vol_5d']   = df['ret_1d'].rolling(5).std()
    df['vol_21d']  = df['ret_1d'].rolling(21).std()
    df['vol_63d']  = df['ret_1d'].rolling(63).std()
    df['vol_252d'] = df['ret_1d'].rolling(252).std()

    # Vol ratio — current vs long-run (FIXED: 1e-8 → 0.01)
    df['vol_ratio'] = df['vol_21d'] / (df['vol_252d'] + 0.01)

    # ── Volume features ───────────────────────────────────────────────────────
    df['vol_ma21']   = df['volume'].rolling(21).mean()
    df['vol_ratio_v']= df['volume'] / (df['vol_ma21'] + 1)
    df['vol_21d_chg']= df['vol_ma21'].pct_change(21)

    # ── Price position ────────────────────────────────────────────────────────
    df['high_252d'] = df['close'].rolling(252).max()
    df['low_252d']  = df['close'].rolling(252).min()
    df['pos_252d']  = ((df['close'] - df['low_252d']) /
                       (df['high_252d'] - df['low_252d'] + 0.01))  # FIXED

    df['high_63d']  = df['close'].rolling(63).max()
    df['low_63d']   = df['close'].rolling(63).min()
    df['pos_63d']   = ((df['close'] - df['low_63d']) /
                       (df['high_63d'] - df['low_63d'] + 0.01))  # FIXED

    # ── Intraday range (volatility proxy) ─────────────────────────────────────
    df['hl_range']    = (df['high'] - df['low']) / (df['close'] + 0.01)  # FIXED
    df['hl_range_ma'] = df['hl_range'].rolling(21).mean()

    # ── Momentum signals ──────────────────────────────────────────────────────
    # Relative strength across timeframes
    df['mom_1_12'] = df['ret_252d'] - df['ret_21d']  # skip-month momentum
    df['mom_accel']= df['ret_21d'] - df['ret_63d']   # momentum acceleration

    # ── Trend features ────────────────────────────────────────────────────────
    df['sma_21']  = df['close'].rolling(21).mean()
    df['sma_63']  = df['close'].rolling(63).mean()
    df['sma_252'] = df['close'].rolling(252).mean()

    df['price_vs_sma21']  = df['close'] / (df['sma_21'] + 0.01) - 1   # FIXED
    df['price_vs_sma63']  = df['close'] / (df['sma_63'] + 0.01) - 1   # FIXED
    df['price_vs_sma252'] = df['close'] / (df['sma_252'] + 0.01) - 1  # FIXED
    df['sma21_vs_sma63']  = df['sma_21'] / (df['sma_63'] + 0.01) - 1  # FIXED

    # ── Return distribution features ──────────────────────────────────────────
    df['skew_63d']  = df['ret_1d'].rolling(63).skew()
    df['kurt_63d']  = df['ret_1d'].rolling(63).kurt()
    df['skew_252d'] = df['ret_1d'].rolling(252).skew()

    # ── Drawdown ──────────────────────────────────────────────────────────────
    rolling_max      = df['close'].rolling(252, min_periods=1).max()
    df['drawdown']   = df['close'] / rolling_max - 1
    df['dd_duration']= (df['drawdown'] < -0.1).rolling(63).sum()

    # ── Up/down ratios ────────────────────────────────────────────────────────
    df['up_days_21']  = (df['ret_1d'] > 0).rolling(21).sum() / 21
    df['up_days_63']  = (df['ret_1d'] > 0).rolling(63).sum() / 63
    df['up_vol_ratio']= (df['ret_1d'].clip(lower=0).rolling(21).std() /
                         (df['ret_1d'].clip(upper=0).abs().rolling(21).std()
                          + 0.01))  # FIXED

    # Add ticker
    df['ticker'] = ticker

    # Drop warmup period
    df = df.iloc[252:].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

# ── FF daily factors ──────────────────────────────────────────────────────────
def get_ff_daily():
    print("\nDownloading FF3 daily factors...")
    url = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
           'ftp/F-F_Research_Data_Factors_daily_CSV.zip')
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        fname = [f for f in z.namelist()
                 if f.lower().endswith('.csv')][0]
        with z.open(fname) as f:
            raw = f.read().decode('utf-8')

    lines = raw.split('\n')
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and len(stripped) >= 8:
            start = i
            break

    if start is None:
        raise ValueError("Could not find data start in FF file")

    end = len(lines)
    for i, line in enumerate(lines[start:], start=start):
        stripped = line.strip()
        if stripped and not stripped[0].isdigit():
            end = i
            break

    ff = pd.read_csv(
        io.StringIO('\n'.join(lines[start:end])),
        header=None,
        names=['date', 'Mkt_RF', 'SMB', 'HML', 'RF'],
    )
    ff = ff.dropna()
    ff['date'] = pd.to_datetime(ff['date'].astype(str).str.strip(),
                                format='%Y%m%d', errors='coerce')
    ff = ff.dropna(subset=['date'])
    ff[['Mkt_RF', 'SMB', 'HML', 'RF']] = (
        ff[['Mkt_RF', 'SMB', 'HML', 'RF']]
        .apply(pd.to_numeric, errors='coerce') / 100
    )
    ff = ff.set_index('date').sort_index()
    print(f"FF3 daily: {len(ff)} days "
          f"({ff.index[0].year}--{ff.index[-1].year})")
    return ff

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STOCK UNIVERSE CONSTRUCTION")
print(f"Target: {TARGET_STOCKS} stocks | "
      f"{START_DATE} — {END_DATE}")
print("No quality filters — random sample from full universe")
print("=" * 60)

# Load checkpoint
if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE) as f:
        ckpt = json.load(f)
    validated   = set(ckpt.get('validated', []))
    failed      = set(ckpt.get('failed', []))
    print(f"\nResuming — {len(validated)} validated, {len(failed)} failed")
else:
    ckpt      = {'validated': [], 'failed': []}
    validated = set()
    failed    = set()

def save_ckpt():
    ckpt['validated'] = list(validated)
    ckpt['failed']    = list(failed)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt, f, indent=2)

# Build universe
print("\nBuilding ticker universe...")
universe = set()
universe.update(get_sec_tickers())
universe.update(get_wikipedia_tickers())
universe.update(get_yfinance_tickers())
universe.update(get_delisted_tickers())

# Clean — remove obvious non-tickers
universe = {t for t in universe
            if t and len(t) <= 6
            and t.isalpha()
            and not any(c in t for c in ['.', '-', '^', '='])}

# Remove already processed
universe -= validated
universe -= failed

universe_list = list(universe)
random.seed(42)
random.shuffle(universe_list)

print(f"\nTotal universe: {len(universe_list) + len(validated) + len(failed)}")
print(f"Already validated: {len(validated)}")
print(f"Already failed: {len(failed)}")
print(f"Remaining to try: {len(universe_list)}")
print(f"Need {max(0, TARGET_STOCKS - len(validated))} more\n")

# ── FF factors ────────────────────────────────────────────────────────────────
ff_daily = get_ff_daily()
ff_daily.to_parquet(DATA_DIR / 'ff3_daily.parquet')

# ── Pull and validate ─────────────────────────────────────────────────────────
print("\nValidating and pulling stock data...")
print("Skipping failures, drawing replacements until target met\n")

stock_dfs  = {}
n_tried    = 0
batch_size = 50

# Load already validated stocks
existing_files = list(DATA_DIR.glob('stock_*.parquet'))
for f in existing_files:
    ticker = f.stem.replace('stock_', '')
    if ticker in validated:
        try:
            stock_dfs[ticker] = pd.read_parquet(f)
        except Exception:
            validated.discard(ticker)

print(f"Loaded {len(stock_dfs)} existing validated stocks")

for ticker in universe_list:
    if len(validated) >= TARGET_STOCKS:
        break

    if ticker in validated or ticker in failed:
        continue

    n_tried += 1
    ticker_clean = ticker.strip().upper()

    t, hist = validate_ticker(ticker_clean)
    time.sleep(0.1)

    if hist is None:
        failed.add(ticker_clean)
        if n_tried % batch_size == 0:
            save_ckpt()
            print(f"  Tried: {n_tried} | "
                  f"Validated: {len(validated)} | "
                  f"Failed: {len(failed)} | "
                  f"Need: {TARGET_STOCKS - len(validated)}")
        continue

    # Compute features
    features = compute_features(hist, ticker_clean)
    if features is None or len(features) < MIN_DAYS:
        failed.add(ticker_clean)
        continue

    # Residualize against FF factors
    feat_aligned = features.join(ff_daily, how='left')
    feat_aligned['RF']     = feat_aligned['RF'].ffill()
    feat_aligned['Mkt_RF'] = feat_aligned['Mkt_RF'].ffill()
    feat_aligned['SMB']    = feat_aligned['SMB'].ffill()
    feat_aligned['HML']    = feat_aligned['HML'].ffill()

    # Rolling 63-day beta to each factor (FIXED: 1e-8 → 0.01)
    for factor in ['Mkt_RF', 'SMB', 'HML']:
        roll_cov = (feat_aligned['ret_1d']
                    .rolling(63)
                    .cov(feat_aligned[factor]))
        roll_var = feat_aligned[factor].rolling(63).var()
        feat_aligned[f'beta_{factor.lower()}'] = (
            roll_cov / (roll_var + 0.01))

    # Alpha residual (daily, rolling 63-day regression)
    feat_aligned['excess_ret'] = (feat_aligned['ret_1d'] -
                                   feat_aligned['RF'])
    feat_aligned['alpha_resid'] = (
        feat_aligned['excess_ret']
        - feat_aligned['beta_mkt_rf'] * feat_aligned['Mkt_RF']
        - feat_aligned['beta_smb']    * feat_aligned['SMB']
        - feat_aligned['beta_hml']    * feat_aligned['HML']
    )

    # Drop rows with too many NaNs
    feat_cols = [c for c in feat_aligned.columns
                 if c not in ['ticker', 'open', 'high', 'low',
                               'close', 'volume']]
    feat_aligned = feat_aligned.dropna(
        subset=feat_cols, thresh=len(feat_cols) // 2)

    if len(feat_aligned) < MIN_DAYS:
        failed.add(ticker_clean)
        continue

    # Save individual stock file
    out = DATA_DIR / f'stock_{ticker_clean}.parquet'
    feat_aligned.to_parquet(out)

    validated.add(ticker_clean)
    stock_dfs[ticker_clean] = feat_aligned

    if n_tried % batch_size == 0 or len(validated) % 100 == 0:
        save_ckpt()
        print(f"  Tried: {n_tried:5d} | "
              f"Validated: {len(validated):4d}/{TARGET_STOCKS} | "
              f"Failed: {len(failed):5d} | "
              f"Latest: {ticker_clean}")

save_ckpt()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"UNIVERSE COMPLETE")
print(f"{'='*60}")
print(f"Validated stocks: {len(validated)}")
print(f"Failed tickers:   {len(failed)}")
print(f"Total tried:      {n_tried}")
print(f"Success rate:     {len(validated)/max(n_tried,1)*100:.1f}%")

if stock_dfs:
    sample = list(stock_dfs.values())[0]
    print(f"\nFeatures per stock: {len([c for c in sample.columns if c not in ['ticker']])}")
    print(f"Sample date range: {sample.index[0].date()} — {sample.index[-1].date()}")
    print(f"Sample obs: {len(sample)}")

    # Combine into master file — sample first 2000
    print(f"\nSaving manifest...")
    manifest = []
    for ticker, df in stock_dfs.items():
        manifest.append({
            'ticker':    ticker,
            'n_days':    len(df),
            'start':     str(df.index[0].date()),
            'end':       str(df.index[-1].date()),
            'mean_ret':  float(df['ret_1d'].mean()),
            'vol':       float(df['ret_1d'].std()),
            'file':      str(DATA_DIR / f'stock_{ticker}.parquet'),
        })

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_parquet(DATA_DIR / 'stock_manifest.parquet')
    manifest_df.to_csv(DATA_DIR / 'stock_manifest.csv', index=False)

    print(f"Manifest saved: {len(manifest_df)} stocks")
    print(f"\nDate coverage distribution:")
    print(manifest_df['n_days'].describe().round(0).to_string())

print(f"\nDone — run stock_clean.py next")