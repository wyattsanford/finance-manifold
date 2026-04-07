# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:42:18 2026

@author: Justin.Sanford
"""

# quick_diagnose.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')

# Load validation report
with open(DATA_DIR / 'validation_report.json') as f:
    report = json.load(f)

print("Cross-sectional stats:")
print(json.dumps(report['stats']['cross_sectional'], indent=2))

# Load scaler and inspect
with open(DATA_DIR / 'stock_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"\nScaler scale_ range: {scaler.scale_.min():.6f} — {scaler.scale_.max():.6f}")
print(f"Scaler mean range: {scaler.mean_.min():.6f} — {scaler.mean_.max():.6f}")

# Check if any scale_ values are extreme
extreme_scale = np.where((scaler.scale_ < 0.001) | (scaler.scale_ > 100))[0]
print(f"\nFeatures with extreme scale_ values: {len(extreme_scale)}")

# Load one stock and scale it
manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
ticker = manifest.iloc[0]['ticker']

EXCLUDE_COLS = {'ticker', 'open', 'high', 'low', 'close', 'volume',
                'Mkt_RF', 'SMB', 'HML', 'RF',
                'high_252d', 'low_252d', 'high_63d', 'low_63d',
                'sma_21', 'sma_63', 'sma_252', 'vol_ma21',
                # Ratio features with extreme values
                'up_vol_ratio', 'kurt_63d'}

sample_df = pd.read_parquet(DATA_DIR / f'stock_{ticker}.parquet')
feature_cols = [c for c in sample_df.columns 
                if c not in EXCLUDE_COLS
                and sample_df[c].dtype in [np.float64, np.float32]]

print(f"\nFeature columns in validation: {len(feature_cols)}")
print(f"Scaler dimensions: {len(scaler.scale_)}")

if len(feature_cols) != len(scaler.scale_):
    print("\n❌ DIMENSION MISMATCH!")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Scaler:   {len(scaler.scale_)}")
else:
    print("\n✓ Dimensions match")
    
    # Show features with extreme scale
    if len(extreme_scale) > 0:
        print("\nFeatures with extreme scale_ values:")
        for idx in extreme_scale[:10]:
            print(f"  {feature_cols[idx]}: scale_={scaler.scale_[idx]:.6e}")