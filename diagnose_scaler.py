# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:05:04 2026

@author: Justin.Sanford
"""

# diagnose_scaler.py
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')

with open(DATA_DIR / 'stock_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

EXCLUDE_COLS = {'ticker', 'open', 'high', 'low', 'close', 'volume',
                'Mkt_RF', 'SMB', 'HML', 'RF',
                'high_252d', 'low_252d', 'high_63d', 'low_63d',
                'sma_21', 'sma_63', 'sma_252', 'vol_ma21',
                'up_vol_ratio', 'kurt_63d'}

manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
ticker = manifest.iloc[0]['ticker']
sample_df = pd.read_parquet(DATA_DIR / f'stock_{ticker}.parquet')
feature_cols = [c for c in sample_df.columns 
                if c not in EXCLUDE_COLS
                and sample_df[c].dtype in [np.float64, np.float32]]

print("Features with extreme scale values:")
for idx in range(len(scaler.scale_)):
    if scaler.scale_[idx] < 0.001 or scaler.scale_[idx] > 10:
        print(f"{feature_cols[idx]:30s} scale={scaler.scale_[idx]:12.6f}  mean={scaler.mean_[idx]:12.6f}")

print(f"\nScale range: {scaler.scale_.min():.6f} to {scaler.scale_.max():.6f}")