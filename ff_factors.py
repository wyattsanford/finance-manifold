# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:38:07 2026

@author: Justin.Sanford
"""

# ff_factors.py
# Download Fama-French 3 factors and compute alpha for each manager
# Ken French data library — free, no auth

import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path
from scipy import stats

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data')

# ── Download FF3 factors ──────────────────────────────────────────────────────
print("Downloading Fama-French 3 factors...")

FF3_URL = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
           'ftp/F-F_Research_Data_Factors_CSV.zip')

r = requests.get(FF3_URL, timeout=30)
r.raise_for_status()

with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    fname = [f for f in z.namelist() if f.endswith('.csv')][0]
    with z.open(fname) as f:
        # Skip header rows — FF data has descriptive text before the data
        raw = f.read().decode('utf-8')

with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    print(f"Files in zip: {z.namelist()}")
    fname = [f for f in z.namelist() if f.lower().endswith('.csv')][0]

# Parse — find where the monthly data starts
lines = raw.split('\n')
start = None
for i, line in enumerate(lines):
    if line.strip().startswith('197') or line.strip().startswith('192'):
        start = i
        break

# Also find where annual data starts (to stop monthly read)
end = None
for i, line in enumerate(lines[start:], start=start):
    if line.strip() == '' and i > start + 10:
        end = i
        break

monthly_lines = lines[start:end]
ff_df = pd.read_csv(
    io.StringIO('\n'.join(monthly_lines)),
    header=None,
    names=['date', 'Mkt_RF', 'SMB', 'HML', 'RF'],
    skipinitialspace=True,
)

ff_df = ff_df.dropna()
ff_df['date'] = pd.to_datetime(ff_df['date'].astype(str).str.strip(),
                                format='%Y%m')
ff_df[['Mkt_RF', 'SMB', 'HML', 'RF']] = (
    ff_df[['Mkt_RF', 'SMB', 'HML', 'RF']].apply(
        pd.to_numeric, errors='coerce'
    ) / 100
)
ff_df = ff_df.dropna()
ff_df = ff_df.set_index('date').sort_index()

print(f"FF3 monthly factors: {len(ff_df)} months "
      f"({ff_df.index[0].year}--{ff_df.index[-1].year})")

# Aggregate to quarterly
ff_q = (ff_df[['Mkt_RF', 'SMB', 'HML', 'RF']]
        .resample('QE')
        .apply(lambda x: (1 + x).prod() - 1))

ff_q.to_parquet(DATA_DIR / 'ff3_quarterly.parquet')
print(f"FF3 quarterly factors: {len(ff_q)} quarters")

# ── Load portfolio returns ────────────────────────────────────────────────────
port_df = pd.read_parquet(DATA_DIR / 'portfolio_returns.parquet')
port_df['quarter_end'] = pd.to_datetime(
    port_df['quarter_end']).dt.tz_localize(None)

# ── Factor residualization ────────────────────────────────────────────────────
print("\nResidualizing portfolio returns against FF3 factors...")
print("This is the 'teammate normalization' for finance —")
print("stripping out market/size/value exposure, leaving alpha\n")

results = []

for manager in port_df['manager'].unique():
    mgr = port_df[port_df['manager'] == manager].copy()
    mgr = mgr.set_index('quarter_end').sort_index()

    # Align with FF factors
    aligned = mgr.join(ff_q, how='inner')
    if len(aligned) < 8:
        print(f"  {manager}: too few obs ({len(aligned)}), skipping")
        continue

    # Excess return = portfolio return - risk free rate
    aligned['excess_ret'] = aligned['port_ret'] - aligned['RF']

    # OLS regression on FF3 factors
    X = aligned[['Mkt_RF', 'SMB', 'HML']].values
    y = aligned['excess_ret'].values

    # Add intercept
    X_int = np.column_stack([np.ones(len(X)), X])
    beta, resid, _, _ = np.linalg.lstsq(X_int, y, rcond=None)

    alpha_q   = beta[0]                    # quarterly alpha (intercept)
    alpha_ann = (1 + alpha_q)**4 - 1       # annualized
    beta_mkt  = beta[1]
    beta_smb  = beta[2]
    beta_hml  = beta[3]

    # Residuals = alpha realizations (the driver signal)
    y_hat     = X_int @ beta
    residuals = y - y_hat
    r_squared = 1 - np.var(residuals) / np.var(y)

    # T-stat on alpha
    n        = len(y)
    se       = np.sqrt(np.var(residuals) / n)
    t_alpha  = alpha_q / se if se > 0 else 0
    p_alpha  = 2 * (1 - stats.t.cdf(abs(t_alpha), df=n-4))

    print(f"  {manager:15s} | "
          f"α={alpha_ann*100:+6.2f}%pa | "
          f"β_mkt={beta_mkt:.2f} | "
          f"R²={r_squared:.2f} | "
          f"p={p_alpha:.3f} | "
          f"n={n}")

    # Store alpha time series
    mgr_result = pd.DataFrame({
        'manager':    manager,
        'quarter_end': aligned.index,
        'port_ret':   aligned['port_ret'].values,
        'excess_ret': aligned['excess_ret'].values,
        'alpha_real': residuals,           # quarter-by-quarter alpha
        'mkt_ret':    aligned['Mkt_RF'].values,
        'smb':        aligned['SMB'].values,
        'hml':        aligned['HML'].values,
        'rf':         aligned['RF'].values,
    })
    results.append(mgr_result)

    # Store summary
    results[-1].attrs = {
        'alpha_q':   float(alpha_q),
        'alpha_ann': float(alpha_ann),
        'beta_mkt':  float(beta_mkt),
        'beta_smb':  float(beta_smb),
        'beta_hml':  float(beta_hml),
        'r_squared': float(r_squared),
        't_alpha':   float(t_alpha),
        'p_alpha':   float(p_alpha),
        'n_obs':     int(n),
    }

alpha_df = pd.concat(results, ignore_index=True)
alpha_df.to_parquet(DATA_DIR / 'alpha_residuals.parquet')

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FACTOR MODEL SUMMARY")
print(f"{'='*60}")
print(f"This is the 'car' removal step.")
print(f"Alpha residuals = what remains after stripping")
print(f"market/size/value exposure — pure manager signal\n")

summary_rows = []
for r in results:
    summary_rows.append({
        'manager':   r['manager'].iloc[0],
        'alpha_ann': r.attrs['alpha_ann'],
        'beta_mkt':  r.attrs['beta_mkt'],
        'r_squared': r.attrs['r_squared'],
        'p_alpha':   r.attrs['p_alpha'],
        'n_obs':     r.attrs['n_obs'],
        'alpha_std': r['alpha_real'].std(),
        'alpha_mean':r['alpha_real'].mean(),
    })

summ = (pd.DataFrame(summary_rows)
        .sort_values('alpha_ann', ascending=False))

print(f"{'Manager':<15} {'Ann Alpha':>10} {'Beta_Mkt':>9} "
      f"{'R²':>6} {'p':>7} {'α Std':>8}")
print("-" * 60)
for _, row in summ.iterrows():
    print(f"{row['manager']:<15} "
          f"{row['alpha_ann']*100:>+9.2f}% "
          f"{row['beta_mkt']:>9.2f} "
          f"{row['r_squared']:>6.2f} "
          f"{row['p_alpha']:>7.3f} "
          f"{row['alpha_std']*100:>7.2f}%")

summ.to_parquet(DATA_DIR / 'alpha_summary.parquet')

print(f"\nAlpha residuals saved: {len(alpha_df)} obs")
print("Done — run behavioral_trace.py next")