# -*- coding: utf-8 -*-
"""
stock_ae_arch_ablations.py
Architecture ablations — latent dim sensitivity.
Tests whether the key findings (alpha R², stability→CI, active dims)
hold across latent dimensions 4/6/8/12/16/24.

Run time: ~4-12 hours depending on GPU — queue overnight.
Each architecture trains on the FULL pre-2023 dataset (not a subsample)
for a fair comparison against the baseline 12D result.

Saves: ae_arch_ablation_results.json
       plots/ablations/architecture_ablations.png
       ae_arch_{ld}d_best.pt  for each latent dim
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
PLOT_DIR = DATA_DIR / 'plots' / 'ablations'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG      = np.random.RandomState(42)

# Architecture grid — test these latent dims against baseline 12D
LATENT_DIMS  = [4, 6, 8, 12, 16, 24]
HIDDEN_SCALE = 2   # h1 = input_dim * HIDDEN_SCALE, h2 = input_dim

# Training config — same as baseline
BATCH_SIZE   = 32768
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 100
PATIENCE     = 10
HUBER_DELTA  = 1.0

print(f"Device:      {DEVICE}")
print(f"Latent dims: {LATENT_DIMS}")
print(f"Plots:       {PLOT_DIR}")

# ── Utilities ─────────────────────────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


def build_ae(input_dim, latent_dim):
    """Build AE with fixed relative hidden dims."""
    h1 = max(32, input_dim * HIDDEN_SCALE)
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


def encode_all(model, X_scaled, batch_size=8192):
    loader = DataLoader(StockDataset(X_scaled), batch_size=batch_size, shuffle=False)
    model.eval()
    parts = []
    with torch.no_grad():
        for xb in loader:
            _, z = model(xb.to(DEVICE))
            parts.append(z.cpu().numpy())
    return np.vstack(parts)


def load_stocks(tickers, feature_cols, date_filter=None):
    Xs, y_rets, y_alphas, all_dates, all_tickers = [], [], [], [], []
    for ticker in tickers:
        f = DATA_DIR / f'stock_clean_{ticker}.parquet'
        if not f.exists():
            continue
        try:
            df    = pd.read_parquet(f)
            dates = pd.to_datetime(df.index)
            if date_filter is not None:
                direction, cutoff = date_filter
                ts   = pd.Timestamp(cutoff)
                mask = (dates < ts) if direction == 'before' else (dates >= ts)
                df, dates = df[mask], dates[mask]
            if len(df) < 63:
                continue
            X = df[feature_cols].values.astype(np.float32)
            X = np.where(np.isinf(X), np.nan, X)
            col_meds = np.nanmedian(X, axis=0)
            for j in range(X.shape[1]):
                m = np.isnan(X[:, j])
                X[m, j] = col_meds[j]
            X = np.nan_to_num(X, nan=0.0)
            y_ret   = df['ret_1d'].values      if 'ret_1d'      in df.columns else np.zeros(len(df))
            y_alpha = df['alpha_resid'].values  if 'alpha_resid' in df.columns else np.zeros(len(df))
            valid   = ~np.isnan(y_ret) & ~np.isnan(y_alpha)
            if valid.sum() < 63:
                continue
            Xs.append(X[valid])
            y_rets.append(y_ret[valid])
            y_alphas.append(y_alpha[valid])
            all_dates.append(dates[valid])
            all_tickers.extend([ticker] * valid.sum())
        except Exception:
            pass
    return (np.vstack(Xs), np.concatenate(y_rets), np.concatenate(y_alphas),
            np.concatenate(all_dates), np.array(all_tickers))


# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading feature cols from temporal checkpoint...")
ckpt_ref     = torch.load(DATA_DIR / 'ae_temporal_best.pt',
                           map_location='cpu', weights_only=False)
feature_cols = ckpt_ref['feature_cols']
input_dim    = len(feature_cols)
del ckpt_ref

manifest    = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

print("Loading pre-2023 data (full — no subsampling)...")
X_all, y_ret_all, y_alpha_all, dates_all, tickers_all = load_stocks(
    all_tickers, feature_cols, date_filter=('before', '2023-01-01'))
dates_all = np.array(dates_all)

print(f"  {len(X_all):,} observations  |  {len(np.unique(tickers_all))} stocks  |  "
      f"{input_dim} features")

# Stock-level train/val split (80/20) — consistent across all architectures
unique_t = np.unique(tickers_all)
RNG.shuffle(unique_t)
n_train_t    = int(len(unique_t) * 0.8)
train_tickers = set(unique_t[:n_train_t])
val_tickers   = set(unique_t[n_train_t:])

train_mask = np.array([t in train_tickers for t in tickers_all])
val_mask   = ~train_mask

X_tr, X_vl   = X_all[train_mask], X_all[val_mask]
y_tr, y_vl   = y_alpha_all[train_mask], y_alpha_all[val_mask]
dates_vl     = dates_all[val_mask]
tickers_vl   = tickers_all[val_mask]

# Fit scaler once on train — shared across all architectures
print("Fitting scaler on train split...")
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
X_vl_s = scaler.transform(X_vl).astype(np.float32)

print(f"  Train: {len(X_tr_s):,}  |  Val: {len(X_vl_s):,}")


# ── Train each architecture ───────────────────────────────────────────────────
arch_results = {}

for ld in LATENT_DIMS:
    print(f"\n{'='*70}")
    print(f"LATENT DIM: {ld}  ({input_dim}→{max(32,input_dim*2)}→{max(16,input_dim)}→{ld})")
    print(f"{'='*70}")
    t_start = datetime.now()

    model = build_ae(input_dim, ld).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit  = nn.HuberLoss(delta=HUBER_DELTA)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    tr_loader = DataLoader(StockDataset(X_tr_s), batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=0, pin_memory=True)

    best_val_loss = float('inf')
    patience_ctr  = 0
    best_state    = None
    history       = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb in tr_loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            xr, _ = model(xb)
            loss   = crit(xr, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_tr_s)

        # Val
        model.eval()
        with torch.no_grad():
            xvt      = torch.FloatTensor(X_vl_s).to(DEVICE)
            val_loss = crit(model(xvt)[0], xvt).item()

        sched.step()
        history.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Best: {best_val_loss:.4f} | Patience: {patience_ctr}/{PATIENCE}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = DATA_DIR / f'ae_arch_{ld}d_best.pt'
    torch.save({
        'model_state':  best_state,
        'scaler':       scaler,
        'feature_cols': feature_cols,
        'latent_dim':   ld,
        'input_dim':    input_dim,
        'val_loss':     best_val_loss,
    }, ckpt_path)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"  Evaluating...")
    z_tr_enc = encode_all(model, X_tr_s)
    z_vl_enc = encode_all(model, X_vl_s)

    # Alpha R²
    ridge = Ridge(alpha=1.0)
    ridge.fit(z_tr_enc, y_tr)
    r2_ld = float(r2_score(y_vl, ridge.predict(z_vl_enc)))

    # Active dims
    dim_vars = z_vl_enc.var(axis=0)
    active   = int((dim_vars > dim_vars.mean() * 0.1).sum())

    # Stability → CI width
    per_stock_stab, per_stock_ci = [], []
    for ticker in np.unique(tickers_vl):
        mask = tickers_vl == ticker
        if mask.sum() < 30:
            continue
        stab = float(z_vl_enc[mask].var(axis=0).mean())
        ci   = float(np.std(y_vl[mask]))
        per_stock_stab.append(stab)
        per_stock_ci.append(ci)

    r_stab, p_stab = pearsonr(per_stock_stab, per_stock_ci) \
        if len(per_stock_stab) > 10 else (np.nan, np.nan)

    # Val loss
    elapsed = (datetime.now() - t_start).seconds / 60

    arch_results[ld] = {
        'r2':          r2_ld,
        'active':      active,
        'active_frac': active / ld,
        'r_stab':      float(r_stab),
        'p_stab':      float(p_stab),
        'val_loss':    best_val_loss,
        'epochs':      len(history),
        'elapsed_min': elapsed,
        'dim_variances': dim_vars.tolist(),
    }

    print(f"  Alpha R²:    {r2_ld:.4f}")
    print(f"  Active dims: {active}/{ld}  ({active/ld*100:.0f}%)")
    print(f"  Stab→CI:     r={r_stab:.3f}  p={p_stab:.4f}")
    print(f"  Val loss:    {best_val_loss:.4f}")
    print(f"  Time:        {elapsed:.1f} min")


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ARCHITECTURE ABLATION SUMMARY")
print("=" * 70)
print(f"\n  {'Latent D':<10} {'Alpha R²':>10} {'Active':>8} {'Stab→CI r':>10} "
      f"{'Val loss':>10} {'Epochs':>8}")
print(f"  {'─'*58}")

for ld, res in arch_results.items():
    marker = ' ← baseline' if ld == 12 else ''
    print(f"  {ld:<10} {res['r2']:>10.4f} {res['active']:>5}/{ld:<2} "
          f"{res['r_stab']:>10.3f} {res['val_loss']:>10.4f} "
          f"{res['epochs']:>8}{marker}")

# Key questions
r2s    = [arch_results[d]['r2']     for d in LATENT_DIMS]
rstabs = [arch_results[d]['r_stab'] for d in LATENT_DIMS]
acts   = [arch_results[d]['active_frac'] for d in LATENT_DIMS]

best_r2_dim = LATENT_DIMS[int(np.argmax(r2s))]
print(f"\n  Best alpha R² at latent dim: {best_r2_dim}D ({max(r2s):.4f})")
print(f"  12D alpha R²:                {arch_results[12]['r2']:.4f}")
print(f"  Stability finding holds at all dims: "
      f"{'YES' if all(r > 0.3 for r in rstabs if not np.isnan(r)) else 'MIXED'}")
print(f"  Dimensional collapse (any dim <50% active): "
      f"{'YES' if any(a < 0.5 for a in acts) else 'NO'}")


# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Alpha R²
axes[0, 0].plot(LATENT_DIMS, r2s, 'o-', color='steelblue', lw=2, ms=8)
axes[0, 0].axvline(12, color='red', linestyle='--', alpha=0.5, label='Baseline (12D)')
axes[0, 0].axhline(0.0039, color='gray', linestyle=':', alpha=0.7, label='PCA baseline')
axes[0, 0].set_xlabel('Latent dimension')
axes[0, 0].set_ylabel('Alpha R²')
axes[0, 0].set_title('Forward Alpha R² vs Latent Dim')
axes[0, 0].legend(fontsize=8)

# Stability → CI width
axes[0, 1].plot(LATENT_DIMS, rstabs, '^-', color='purple', lw=2, ms=8)
axes[0, 1].axvline(12, color='red', linestyle='--', alpha=0.5)
axes[0, 1].axhline(0, color='black', lw=0.8, linestyle=':')
axes[0, 1].set_xlabel('Latent dimension')
axes[0, 1].set_ylabel('r(stability, CI width)')
axes[0, 1].set_title('Stability Finding vs Latent Dim\n(robust if consistently positive)')

# Active dimension fraction
axes[1, 0].plot(LATENT_DIMS, [a * 100 for a in acts], 's-', color='darkorange', lw=2, ms=8)
axes[1, 0].axvline(12, color='red', linestyle='--', alpha=0.5)
axes[1, 0].axhline(50, color='gray', linestyle=':', alpha=0.7, label='50% threshold')
axes[1, 0].set_xlabel('Latent dimension')
axes[1, 0].set_ylabel('% active dims (>10% mean var)')
axes[1, 0].set_title('Dimensional Utilization')
axes[1, 0].legend(fontsize=8)

# Val loss
val_losses = [arch_results[d]['val_loss'] for d in LATENT_DIMS]
axes[1, 1].plot(LATENT_DIMS, val_losses, 'D-', color='crimson', lw=2, ms=8)
axes[1, 1].axvline(12, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Latent dimension')
axes[1, 1].set_ylabel('Best val loss (Huber)')
axes[1, 1].set_title('Reconstruction Quality vs Latent Dim')

plt.suptitle('Architecture Ablations — Latent Dimension Sensitivity', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'architecture_ablations.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: architecture_ablations.png")

# Per-dim variance heatmap
fig, ax = plt.subplots(figsize=(max(8, len(LATENT_DIMS) * 1.2), 5))
var_matrix = np.zeros((len(LATENT_DIMS), max(LATENT_DIMS)))
for i, ld in enumerate(LATENT_DIMS):
    dvars = arch_results[ld]['dim_variances']
    # Normalize to [0,1] within each architecture
    dv    = np.array(dvars)
    if dv.max() > 0:
        dv = dv / dv.max()
    var_matrix[i, :ld] = dv

sns.heatmap(var_matrix,
            xticklabels=[f'z{i}' for i in range(max(LATENT_DIMS))],
            yticklabels=[f'{d}D' for d in LATENT_DIMS],
            cmap='Blues', ax=ax, vmin=0, vmax=1)
ax.set_title('Normalized Dimensional Variance by Architecture\n'
             '(dark = high variance dim, white = unused)')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'architecture_dim_variance.png', dpi=150)
plt.close()
print(f"Plot saved: architecture_dim_variance.png")

# ── Save ─────────────────────────────────────────────────────────────────────
results_out = {
    'latent_dims':   LATENT_DIMS,
    'input_dim':     input_dim,
    'n_train':       int(len(X_tr)),
    'n_val':         int(len(X_vl)),
    'results':       {str(k): v for k, v in arch_results.items()},
    'summary': {
        'best_r2_dim':    int(best_r2_dim),
        'best_r2':        float(max(r2s)),
        'baseline_12d_r2': float(arch_results[12]['r2']),
        'stability_robust': bool(all(r > 0.3 for r in rstabs if not np.isnan(r))),
    }
}

with open(DATA_DIR / 'ae_arch_ablation_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)

print(f"\nResults: ae_arch_ablation_results.json")
print(f"Checkpoints: ae_arch_{{ld}}d_best.pt for each latent dim")
print("Done.")