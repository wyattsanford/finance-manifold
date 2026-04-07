# prices_pull_v2.py
# Pull prices for mapped tickers and compute portfolio returns
# Uses existing cusip_ticker_mapping.json

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import json
import time

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data')

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA_DIR / 'all_filings.parquet')
df['manager'] = df['manager'].replace('Appaloosa_old', 'Appaloosa')
df['date']    = pd.to_datetime(df['date'])

with open(DATA_DIR / 'cusip_ticker_mapping.json') as f:
    cusip_ticker = json.load(f)

df['ticker'] = df['cusip'].map(cusip_ticker)

print(f"Filings: {len(df):,}")
print(f"Managers: {df['manager'].nunique()}")
print(f"Mapped rows: {df['ticker'].notna().mean()*100:.1f}%")

unique_tickers = df['ticker'].dropna().unique().tolist()
print(f"Unique tickers to pull: {len(unique_tickers)}")

# ── Pull prices ───────────────────────────────────────────────────────────────
print("\nPulling prices via yfinance...")

CACHE = DATA_DIR / 'prices_raw.parquet'
if CACHE.exists():
    prices_df = pd.read_parquet(CACHE)
    already   = set(prices_df.columns)
    print(f"  Loaded {len(already)} cached tickers")
else:
    prices_df = pd.DataFrame()
    already   = set()

todo = [t for t in unique_tickers if t not in already]
print(f"  Pulling {len(todo)} new tickers...")

BATCH = 100
new_prices = {}

for i in range(0, len(todo), BATCH):
    batch = todo[i:i+BATCH]
    try:
        raw = yf.download(
            batch,
            start='2012-01-01',
            end='2024-06-30',
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw['Close']
        else:
            close = raw
            close.columns = [batch[0]]

        for t in batch:
            if t in close.columns and close[t].notna().sum() > 10:
                new_prices[t] = close[t].dropna()

        print(f"  Batch {i//BATCH+1}/{len(todo)//BATCH+1} | "
              f"Got {len(new_prices)} tickers so far")
        time.sleep(0.5)

    except Exception as e:
        print(f"  Batch {i//BATCH+1} failed: {e}")
        # Individual fallback
        for t in batch:
            try:
                h = yf.Ticker(t).history(
                    start='2012-01-01', end='2024-06-30',
                    auto_adjust=True)
                if len(h) > 10:
                    new_prices[t] = h['Close'].dropna()
                time.sleep(0.1)
            except Exception:
                pass

# Combine with cached
if new_prices:
    new_df = pd.DataFrame(new_prices)
    new_df.index = pd.to_datetime(new_df.index).tz_localize(None)
    if not prices_df.empty:
        prices_df.index = pd.to_datetime(prices_df.index).tz_localize(None)
        prices_df = pd.concat([prices_df, new_df], axis=1)
    else:
        prices_df = new_df
    prices_df = prices_df.sort_index()
    prices_df.to_parquet(CACHE)
    print(f"\nPrice cache: {prices_df.shape[1]} tickers, "
          f"{prices_df.shape[0]} days")

# ── Quarterly returns ─────────────────────────────────────────────────────────
print("\nComputing quarterly returns...")

prices_df.index = pd.to_datetime(prices_df.index).tz_localize(None)
q_prices  = prices_df.resample('QE').last()
q_returns = q_prices.pct_change()
q_returns.to_parquet(DATA_DIR / 'returns_quarterly.parquet')
print(f"Quarterly returns: {q_returns.shape}")

# ── Portfolio returns ─────────────────────────────────────────────────────────
print("\nComputing manager portfolio returns...")

df['quarter_end'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

portfolio_rows = []

for (manager, qend), group in df.groupby(['manager', 'quarter_end']):
    group = group[group['ticker'].notna()].copy()
    if len(group) == 0:
        continue

    total_val = group['value'].sum()
    if total_val == 0:
        continue

    group['weight'] = group['value'] / total_val

    if qend not in q_returns.index:
        continue

    ret_row = q_returns.loc[qend]

    weighted = []
    covered_val = 0.0

    for _, row in group.iterrows():
        t = row['ticker']
        if t in ret_row.index and not pd.isna(ret_row[t]):
            weighted.append(row['weight'] * ret_row[t])
            covered_val += row['value']

    if not weighted:
        continue

    portfolio_rows.append({
        'manager':       manager,
        'quarter_end':   qend,
        'port_ret':      sum(weighted),
        'n_positions':   len(group),
        'coverage_pct':  covered_val / total_val,
        'total_value_k': total_val,
    })

port_df = pd.DataFrame(portfolio_rows).sort_values(
    ['manager', 'quarter_end'])
port_df.to_parquet(DATA_DIR / 'portfolio_returns.parquet')

print(f"\nPortfolio returns computed:")
print(f"  Total obs:   {len(port_df)}")
print(f"  Managers:    {port_df['manager'].nunique()}")
print(f"  Date range:  {port_df['quarter_end'].min()} "
      f"— {port_df['quarter_end'].max()}")

print(f"\nCoverage and returns by manager:")
summary = port_df.groupby('manager').agg(
    n_quarters    =('quarter_end',  'count'),
    mean_coverage =('coverage_pct', 'mean'),
    mean_ret      =('port_ret',     'mean'),
    ann_ret       =('port_ret',     lambda x: (1+x).prod()**(4/len(x))-1),
).sort_values('ann_ret', ascending=False)
summary['mean_coverage'] = summary['mean_coverage'].map('{:.1%}'.format)
summary['mean_ret']      = summary['mean_ret'].map('{:.2%}'.format)
summary['ann_ret']       = summary['ann_ret'].map('{:.2%}'.format)
print(summary.to_string())

print("\nDone — run ff_factors.py next")