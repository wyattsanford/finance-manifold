# stock_ae_train.py
# Autoencoder training on stock features (NO RETURNS)
# 5-fold stock-level CV + 2023-2024 temporal holdout
# Architecture: 23 → 64 → 32 → 12 → 32 → 64 → 23
# Loss: Huber (δ=1.0)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Feature definitions ───────────────────────────────────────────────────────
EXCLUDE_FROM_AE = {
    'ticker', 'open', 'high', 'low', 'close', 'volume',
    'Mkt_RF', 'SMB', 'HML', 'RF',
    'high_252d', 'low_252d', 'high_63d', 'low_63d',
    'sma_21', 'sma_63', 'sma_252', 'vol_ma21',
    'up_vol_ratio', 'kurt_63d',
    # ALL RETURNS (no contamination)
    'ret_1d', 'ret_5d', 'ret_21d', 'ret_63d', 'ret_252d',
    # REDUNDANT FEATURES
    'excess_ret', 'mom_1_12', 'up_days_63',
    # TARGET VARIABLES (not features!)
    'alpha_resid',  # ← ADD THIS
}

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM = 12
BATCH_SIZE = 32768
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 10
HUBER_DELTA = 1.0

# ── Dataset ───────────────────────────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X, y_ret, y_alpha, dates):
        self.X = torch.FloatTensor(X)
        self.y_ret = torch.FloatTensor(y_ret)
        self.y_alpha = torch.FloatTensor(y_alpha)
        self.dates = dates
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_ret[idx], self.y_alpha[idx]

# ── Autoencoder ───────────────────────────────────────────────────────────────
class StockAE(nn.Module):
    def __init__(self, input_dim, latent_dim=12):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),  # 23 → 32
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),  # 32 → 16
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, latent_dim),  # 16 → 6
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),  # 6 → 16
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 32),  # 16 → 32
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, input_dim),  # 32 → 23
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

# ── Training functions ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for X_batch, _, _ in loader:
        X_batch = X_batch.to(device)
        
        optimizer.zero_grad()
        X_recon, _ = model(X_batch)
        
        # Huber loss
        loss = nn.HuberLoss(delta=HUBER_DELTA)(X_recon, X_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(loader.dataset)

def validate_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, _, _ in loader:
            X_batch = X_batch.to(device)
            X_recon, _ = model(X_batch)
            loss = nn.HuberLoss(delta=HUBER_DELTA)(X_recon, X_batch)
            total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(loader.dataset)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
tickers = manifest['ticker'].tolist()

with open(DATA_DIR / 'stock_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Get feature list
sample_df = pd.read_parquet(DATA_DIR / f'stock_clean_{tickers[0]}.parquet')
all_cols = sample_df.columns.tolist()
feature_cols_ae = [c for c in all_cols 
                   if c not in EXCLUDE_FROM_AE
                   and sample_df[c].dtype in [np.float64, np.float32]]

print(f"Total features available: {len(all_cols)}")
print(f"Excluded from AE: {len(EXCLUDE_FROM_AE)}")
print(f"Features for AE training: {len(feature_cols_ae)}")
print(f"Latent dimension: {LATENT_DIM}")

# Load all stocks
print(f"\nLoading {len(tickers)} stocks...")
stock_data = {}

for ticker in tqdm(tickers, desc="Loading"):
    f = DATA_DIR / f'stock_clean_{ticker}.parquet'
    if not f.exists():
        continue
    
    try:
        df = pd.read_parquet(f)
        
        # Extract features for AE
        X = df[feature_cols_ae].values.astype(np.float32)
        
        # Extract targets (not used in training, only for evaluation)
        y_ret = df['ret_1d'].values if 'ret_1d' in df.columns else np.zeros(len(df))
        y_alpha = df['alpha_resid'].values if 'alpha_resid' in df.columns else np.zeros(len(df))
        dates = pd.to_datetime(df.index)
        
        # Basic cleaning
        X = np.where(np.isinf(X), np.nan, X)
        col_meds = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_meds[j]
        X = np.nan_to_num(X, nan=0.0)
        
        # Filter valid returns/alpha
        valid = ~np.isnan(y_ret) & ~np.isnan(y_alpha)
        X = X[valid]
        y_ret = y_ret[valid]
        y_alpha = y_alpha[valid]
        dates = dates[valid]
        
        if len(X) >= 252:
            stock_data[ticker] = {
                'X': X,
                'y_ret': y_ret,
                'y_alpha': y_alpha,
                'dates': dates,
            }
    except Exception as e:
        print(f"Failed {ticker}: {e}")
        continue

print(f"Loaded {len(stock_data)} stocks with sufficient data")

# ── 5-Fold Stock-Level Cross-Validation ──────────────────────────────────────
print("\n" + "="*80)
print("5-FOLD STOCK-LEVEL CROSS-VALIDATION")
print("="*80)

tickers_list = list(stock_data.keys())
np.random.shuffle(tickers_list)
n_folds = 5
fold_size = len(tickers_list) // n_folds

fold_results = []

for fold in range(n_folds):
    print(f"\n{'='*80}")
    print(f"FOLD {fold+1}/{n_folds}")
    print(f"{'='*80}")
    
    # Split stocks
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(tickers_list)
    val_tickers = set(tickers_list[val_start:val_end])
    train_tickers = set(tickers_list) - val_tickers
    
    print(f"Train stocks: {len(train_tickers)}")
    print(f"Val stocks:   {len(val_tickers)}")
    
    # Collect data
    X_train, y_ret_train, y_alpha_train = [], [], []
    X_val, y_ret_val, y_alpha_val = [], [], []
    dates_train, dates_val = [], []
    
    for ticker in train_tickers:
        data = stock_data[ticker]
        X_train.append(data['X'])
        y_ret_train.append(data['y_ret'])
        y_alpha_train.append(data['y_alpha'])
        dates_train.extend(data['dates'])
    
    for ticker in val_tickers:
        data = stock_data[ticker]
        X_val.append(data['X'])
        y_ret_val.append(data['y_ret'])
        y_alpha_val.append(data['y_alpha'])
        dates_val.extend(data['dates'])
    
    X_train = np.vstack(X_train)
    y_ret_train = np.concatenate(y_ret_train)
    y_alpha_train = np.concatenate(y_alpha_train)
    
    X_val = np.vstack(X_val)
    y_ret_val = np.concatenate(y_ret_val)
    y_alpha_val = np.concatenate(y_alpha_val)
    
    print(f"Train obs: {len(X_train):,}")
    print(f"Val obs:   {len(X_val):,}")
    
    # Scale (fit on train only)
    fold_scaler = StandardScaler()
    X_train_scaled = fold_scaler.fit_transform(X_train)
    X_val_scaled = fold_scaler.transform(X_val)
    
    # Datasets
    train_dataset = StockDataset(X_train_scaled, y_ret_train, y_alpha_train, dates_train)
    val_dataset = StockDataset(X_val_scaled, y_ret_val, y_alpha_val, dates_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = StockAE(input_dim=len(feature_cols_ae), latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = validate_epoch(model, val_loader, DEVICE)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state': model.state_dict(),
                'scaler': fold_scaler,
                'feature_cols': feature_cols_ae,
            }, DATA_DIR / f'ae_fold{fold+1}_best.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Best: {best_val_loss:.4f} | Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    fold_results.append({
        'fold': fold + 1,
        'train_tickers': list(train_tickers),
        'val_tickers': list(val_tickers),
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(train_losses[-1]),
        'epochs_trained': len(train_losses),
        'train_losses': train_losses,
        'val_losses': val_losses,
    })

# ── Temporal Holdout (2023-2024) ──────────────────────────────────────────────
print("\n" + "="*80)
print("TEMPORAL HOLDOUT: 2023-2024")
print("="*80)

X_pre2023, y_ret_pre2023, y_alpha_pre2023 = [], [], []
X_post2023, y_ret_post2023, y_alpha_post2023 = [], [], []

for ticker, data in stock_data.items():
    dates = data['dates']
    mask_pre = dates < pd.Timestamp('2023-01-01')
    mask_post = dates >= pd.Timestamp('2023-01-01')
    
    if mask_pre.sum() > 0:
        X_pre2023.append(data['X'][mask_pre])
        y_ret_pre2023.append(data['y_ret'][mask_pre])
        y_alpha_pre2023.append(data['y_alpha'][mask_pre])
    
    if mask_post.sum() > 0:
        X_post2023.append(data['X'][mask_post])
        y_ret_post2023.append(data['y_ret'][mask_post])
        y_alpha_post2023.append(data['y_alpha'][mask_post])

X_pre2023 = np.vstack(X_pre2023)
y_ret_pre2023 = np.concatenate(y_ret_pre2023)
y_alpha_pre2023 = np.concatenate(y_alpha_pre2023)

X_post2023 = np.vstack(X_post2023)
y_ret_post2023 = np.concatenate(y_ret_post2023)
y_alpha_post2023 = np.concatenate(y_alpha_post2023)

print(f"Pre-2023:  {len(X_pre2023):,} observations")
print(f"Post-2023: {len(X_post2023):,} observations")

# Scale
temporal_scaler = StandardScaler()
X_pre2023_scaled = temporal_scaler.fit_transform(X_pre2023)
X_post2023_scaled = temporal_scaler.transform(X_post2023)

# Train on pre-2023
train_dataset_temp = StockDataset(X_pre2023_scaled, y_ret_pre2023, y_alpha_pre2023, None)
val_dataset_temp = StockDataset(X_post2023_scaled, y_ret_post2023, y_alpha_post2023, None)

train_loader_temp = DataLoader(train_dataset_temp, batch_size=BATCH_SIZE, shuffle=True)
val_loader_temp = DataLoader(val_dataset_temp, batch_size=BATCH_SIZE, shuffle=False)

model_temp = StockAE(input_dim=len(feature_cols_ae), latent_dim=LATENT_DIM).to(DEVICE)
optimizer_temp = optim.AdamW(model_temp.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler_temp = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_temp, T_0=20, T_mult=2)

best_val_loss_temp = float('inf')
patience_counter_temp = 0
train_losses_temp = []
val_losses_temp = []

for epoch in range(EPOCHS):
    train_loss = train_epoch(model_temp, train_loader_temp, optimizer_temp, DEVICE)
    val_loss = validate_epoch(model_temp, val_loader_temp, DEVICE)
    scheduler_temp.step()
    
    train_losses_temp.append(train_loss)
    val_losses_temp.append(val_loss)
    
    if val_loss < best_val_loss_temp:
        best_val_loss_temp = val_loss
        patience_counter_temp = 0
        torch.save({
            'model_state': model_temp.state_dict(),
            'scaler': temporal_scaler,
            'feature_cols': feature_cols_ae,
        }, DATA_DIR / 'ae_temporal_best.pt')
    else:
        patience_counter_temp += 1
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Best: {best_val_loss_temp:.4f}")
    
    if patience_counter_temp >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("TRAINING COMPLETE - SUMMARY")
print("="*80)

print("\n5-Fold CV Results:")
for res in fold_results:
    print(f"  Fold {res['fold']}: Val Loss={res['best_val_loss']:.4f}, "
          f"Epochs={res['epochs_trained']}")

print(f"\nTemporal holdout:")
print(f"  Best val loss: {best_val_loss_temp:.4f}")
print(f"  Epochs: {len(train_losses_temp)}")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'architecture': {
        'input_dim': len(feature_cols_ae),
        'latent_dim': LATENT_DIM,
        'encoder': '23 → 64 → 32 → 12',
        'decoder': '12 → 32 → 64 → 23',
    },
    'training': {
        'loss': 'Huber',
        'huber_delta': HUBER_DELTA,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'max_epochs': EPOCHS,
        'patience': PATIENCE,
    },
    'cv_results': fold_results,
    'temporal_holdout': {
        'best_val_loss': float(best_val_loss_temp),
        'epochs_trained': len(train_losses_temp),
        'train_losses': train_losses_temp,
        'val_losses': val_losses_temp,
    },
    'features_used': feature_cols_ae,
}

with open(DATA_DIR / 'ae_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {DATA_DIR / 'ae_training_results.json'}")
print(f"Models saved:")
print(f"  - ae_fold1_best.pt through ae_fold5_best.pt")
print(f"  - ae_temporal_best.pt")
print("\nRun stock_ae_eval.py to evaluate with linear probes")