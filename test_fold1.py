# test_fold1_simple.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StockAE(nn.Module):
    def __init__(self, input_dim, latent_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.LayerNorm(16), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.LayerNorm(16), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(16, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, input_dim),
        )
    def encode(self, x): return self.encoder(x)
    def forward(self, x): return self.decoder(self.encode(x)), self.encode(x)

class StockDataset(Dataset):
    def __init__(self, X): self.X = torch.FloatTensor(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]

# Load checkpoint
checkpoint = torch.load(DATA_DIR / 'ae_fold1_best.pt', weights_only=False)
input_dim = checkpoint['model_state']['encoder.0.weight'].shape[1]

model = StockAE(input_dim=input_dim, latent_dim=6).to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

scaler = checkpoint['scaler']
feature_cols = checkpoint['feature_cols']

print(f"Model: {input_dim}D input → 6D latent")
print(f"Features: {feature_cols[:5]}... (showing first 5)")

# Load a few random stocks to test
manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
test_tickers = np.random.choice(manifest['ticker'].values, 10, replace=False)

X_all, y_ret_all, y_alpha_all = [], [], []

for ticker in test_tickers:
    f = DATA_DIR / f'stock_clean_{ticker}.parquet'
    if not f.exists(): continue
    
    df = pd.read_parquet(f)
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(np.where(np.isinf(X), np.nan, X), nan=0.0)
    
    ret = df['ret_1d'].values if 'ret_1d' in df.columns else np.zeros(len(df))
    alpha = df['alpha_resid'].values if 'alpha_resid' in df.columns else np.zeros(len(df))
    
    valid = ~np.isnan(ret) & ~np.isnan(alpha)
    X_all.append(X[valid])
    y_ret_all.append(ret[valid])
    y_alpha_all.append(alpha[valid])

X_all = scaler.transform(np.vstack(X_all))
y_ret_all = np.concatenate(y_ret_all)
y_alpha_all = np.concatenate(y_alpha_all)

print(f"\nTest set: {len(X_all):,} observations from {len(test_tickers)} stocks")

# Encode
loader = DataLoader(StockDataset(X_all), batch_size=32768, shuffle=False)
z_all = []
recon_all = []

with torch.no_grad():
    for X_batch in loader:
        X_batch = X_batch.to(DEVICE)
        X_recon, z = model(X_batch)
        z_all.append(z.cpu().numpy())
        recon_all.append(X_recon.cpu().numpy())

z_all = np.vstack(z_all)
recon_all = np.vstack(recon_all)

# Reconstruction quality
recon_mse = np.mean((X_all - recon_all)**2)
print(f"\nReconstruction MSE: {recon_mse:.4f}")
print(f"Reconstruction RMSE: {np.sqrt(recon_mse):.4f}")

# Quick probe test (train and test on same data — just to see if there's signal)
ridge_ret = Ridge(alpha=1.0).fit(z_all, y_ret_all)
ridge_alpha = Ridge(alpha=1.0).fit(z_all, y_alpha_all)

ret_pred = ridge_ret.predict(z_all)
alpha_pred = ridge_alpha.predict(z_all)

print(f"\nLinear probes (same data, not proper CV):")
print(f"  Return R² (overfitted): {r2_score(y_ret_all, ret_pred):.4f}")
print(f"  Alpha R² (overfitted):  {r2_score(y_alpha_all, alpha_pred):.4f}")
print(f"\nNote: These are overfitted. Wait for full training to finish for real CV results.")