# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:04:59 2026

@author: Justin.Sanford
"""

# check_features.py
import pandas as pd
from pathlib import Path
import numpy as np

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')

EXCLUDE_FROM_BASELINE = {
    'ticker', 'open', 'high', 'low', 'close', 'volume',
    'Mkt_RF', 'SMB', 'HML', 'RF',
    'high_252d', 'low_252d', 'high_63d', 'low_63d',
    'sma_21', 'sma_63', 'sma_252', 'vol_ma21',
    'up_vol_ratio', 'kurt_63d',
    # ALL RETURNS (match AE)
    'ret_1d', 'ret_5d', 'ret_21d', 'ret_63d', 'ret_252d',
    # REDUNDANT FEATURES
    'excess_ret', 'mom_1_12', 'up_days_63',
}

manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
ticker = manifest.iloc[0]['ticker']

sample_df = pd.read_parquet(DATA_DIR / f'stock_clean_{ticker}.parquet')

feature_cols = [c for c in sample_df.columns 
                if c not in EXCLUDE_FROM_BASELINE
                and sample_df[c].dtype in [np.float64, np.float32]]

print("Features being used:")
print(feature_cols)
print(f"\nTotal: {len(feature_cols)}")
print(f"\nalpha_resid in features: {'alpha_resid' in feature_cols}")
print(f"beta_mkt_rf in features: {'beta_mkt_rf' in feature_cols}")
print(f"beta_smb in features: {'beta_smb' in feature_cols}")
print(f"beta_hml in features: {'beta_hml' in feature_cols}")