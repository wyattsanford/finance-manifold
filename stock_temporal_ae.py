# -*- coding: utf-8 -*-
"""
stock_ae_temporal.py
Train autoencoder on pre-2023 data, validate on 2023-2024.
Standalone temporal holdout — run after 5-fold CV finishes.
Architecture matches ae_fold runs exactly.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import pickle

DATA_DIR    = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

CUTOFF_DATE  = '2023-01-01'
LATENT_DIM   = 12
BATCH_SIZE   = 4096
LR           = 1e-3
MAX_EPOCHS   = 100
PATIENCE     = 10
HUBER_DELTA  = 1.0


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Model ─────────────────────────────────────────────────────────────────────
class StockAE(nn.Module):
    def __init__(self, input_dim, latent_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class StockDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("TEMPORAL HOLDOUT TRAINING")
print(f"Train: before {CUTOFF_DATE}  |  Val: {CUTOFF_DATE} onward")
print("=" * 70)

manifest     = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers  = manifest['ticker'].tolist()

# Load feature_cols directly from fold 1 checkpoint — guarantees exact match
# with the CV models so all 6 checkpoints use an identical feature set
print("Loading feature_cols from ae_fold1_best.pt...")
fold1_ckpt   = torch.load(DATA_DIR / 'ae_fold1_best.pt', map_location='cpu', weights_only= False)
feature_cols = fold1_ckpt['feature_cols']
del fold1_ckpt   # free memory

print(f"Features: {len(feature_cols)} (matched to fold CV models)")
print(f"Device:   {DEVICE}\n")


def load_split(tickers, before_cutoff):
    """Load observations before or after CUTOFF_DATE."""
    Xs, y_rets, y_alphas = [], [], []
    cutoff = pd.Timestamp(CUTOFF_DATE)

    for ticker in tickers:
        f = DATA_DIR / f'stock_clean_{ticker}.parquet'
        if not f.exists():
            continue
        try:
            df  = pd.read_parquet(f)
            idx = pd.to_datetime(df.index)

            mask = (idx < cutoff) if before_cutoff else (idx >= cutoff)
            df   = df[mask]
            if len(df) < 63:
                continue

            X = df[feature_cols].values.astype(np.float32)
            X = np.where(np.isinf(X), np.nan, X)
            col_meds = np.nanmedian(X, axis=0)
            for j in range(X.shape[1]):
                m = np.isnan(X[:, j])
                X[m, j] = col_meds[j]
            X = np.nan_to_num(X, nan=0.0)

            y_ret   = df['ret_1d'].values   if 'ret_1d'    in df.columns else np.zeros(len(df))
            y_alpha = df['alpha_resid'].values if 'alpha_resid' in df.columns else np.zeros(len(df))

            valid = ~np.isnan(y_ret) & ~np.isnan(y_alpha)
            if valid.sum() < 63:
                continue

            Xs.append(X[valid])
            y_rets.append(y_ret[valid])
            y_alphas.append(y_alpha[valid])

        except Exception as e:
            print(f"  Skip {ticker}: {e}")

    return (np.vstack(Xs),
            np.concatenate(y_rets),
            np.concatenate(y_alphas))


print("Loading pre-cutoff (train) data...")
X_train, y_ret_train, y_alpha_train = load_split(all_tickers, before_cutoff=True)
print(f"  Train obs: {len(X_train):,}")

print("Loading post-cutoff (val) data...")
X_val, y_ret_val, y_alpha_val = load_split(all_tickers, before_cutoff=False)
print(f"  Val obs:   {len(X_val):,}")

if len(X_val) == 0:
    raise RuntimeError(
        f"No validation data found after {CUTOFF_DATE}. "
        "Check that stock files cover 2023-2024."
    )

# ── Scale ─────────────────────────────────────────────────────────────────────
print("\nFitting scaler on train data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled   = scaler.transform(X_val).astype(np.float32)

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_loader = DataLoader(
    StockDataset(X_train_scaled),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True,
)
val_loader = DataLoader(
    StockDataset(X_val_scaled),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True,
)

# ── Train ─────────────────────────────────────────────────────────────────────
input_dim = X_train_scaled.shape[1]
model     = StockAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.HuberLoss(delta=HUBER_DELTA)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-5
)

best_val_loss = float('inf')
patience_ctr  = 0
history       = []

print(f"\nTraining  (input_dim={input_dim}, latent_dim={LATENT_DIM})")
print("-" * 70)

for epoch in range(1, MAX_EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    for X_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        optimizer.zero_grad()
        X_recon, _ = model(X_batch)
        loss = criterion(X_recon, X_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * len(X_batch)
    train_loss /= len(X_train_scaled)

    # Val
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            X_recon, _ = model(X_batch)
            val_loss += criterion(X_recon, X_batch).item() * len(X_batch)
    val_loss /= len(X_val_scaled)

    scheduler.step()

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        patience_ctr  = 0
        torch.save({
            'epoch':        epoch,
            'model_state':  model.state_dict(),
            'val_loss':     best_val_loss,
            'scaler':       scaler,
            'feature_cols': feature_cols,
            'input_dim':    input_dim,
            'latent_dim':   LATENT_DIM,
            'cutoff_date':  CUTOFF_DATE,
        }, DATA_DIR / 'ae_temporal_best.pt')
    else:
        patience_ctr += 1

    history.append({
        'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss
    })

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"Best: {best_val_loss:.4f} | "
              f"Patience: {patience_ctr}/{PATIENCE}")

    if patience_ctr >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(no improvement for {PATIENCE} epochs)")
        break

# ── Latent variance check ─────────────────────────────────────────────────────
print("\nLatent dimension variance check (val set, best model)...")
ckpt = torch.load(DATA_DIR / 'ae_temporal_best.pt')
model.load_state_dict(ckpt['model_state'])
model.eval()

latents = []
with torch.no_grad():
    for X_batch in val_loader:
        _, z = model(X_batch.to(DEVICE))
        latents.append(z.cpu().numpy())
Z = np.vstack(latents)

dim_vars = Z.var(axis=0)
print(f"\n{'Dim':<6} {'Variance':>10}")
print("-" * 18)
for i, v in enumerate(dim_vars):
    bar = "█" * int(v / dim_vars.max() * 20)
    print(f"  z{i:<3} {v:>10.4f}  {bar}")

active_dims = (dim_vars > dim_vars.mean() * 0.1).sum()
print(f"\nActive dims (>10% of mean var): {active_dims}/12")
if active_dims < 8:
    print("  ⚠  Partial collapse — consider wider encoder for scale-up run")
else:
    print("  ✓  Variance spread looks healthy")

# ── Save summary ──────────────────────────────────────────────────────────────
summary = {
    'cutoff_date':    CUTOFF_DATE,
    'train_obs':      int(len(X_train)),
    'val_obs':        int(len(X_val)),
    'input_dim':      int(input_dim),
    'latent_dim':     LATENT_DIM,
    'best_val_loss':  float(best_val_loss),
    'epochs_trained': len(history),
    'latent_variances': [float(v) for v in dim_vars],
    'active_dims':    int(active_dims),
    'history':        history,
}

with open(DATA_DIR / 'ae_temporal_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nBest val loss:   {best_val_loss:.4f}")
print(f"Checkpoint:      ae_temporal_best.pt")
print(f"Summary:         ae_temporal_results.json")
print("\nDone — run stock_ae_eval.py")