# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:47:56 2026

@author: Justin.Sanford
"""

# finance_ae.py
# AE on behavioral trace features
# Latent space = manager policy space
# LOO by manager + manifold stability + alpha prediction

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import json

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data')

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
trace_df = pd.read_parquet(DATA_DIR / 'behavioral_trace.parquet')
alpha_df = pd.read_parquet(DATA_DIR / 'alpha_residuals.parquet')
summ_df  = pd.read_parquet(DATA_DIR / 'alpha_summary.parquet')

alpha_df['quarter'] = pd.to_datetime(
    alpha_df['quarter_end']).dt.to_period('Q')

merged = trace_df.merge(
    alpha_df[['manager', 'quarter', 'alpha_real', 'excess_ret']],
    on=['manager', 'quarter'], how='inner'
)

managers = sorted(merged['manager'].unique())
print(f"Managers: {len(managers)}")
print(f"Obs: {len(merged)}")

# ── Features ──────────────────────────────────────────────────────────────────
feature_cols = [c for c in trace_df.columns
                if c not in ['manager', 'quarter']]

# Drop constant/NaN features
X_raw = merged[feature_cols].values.astype(np.float32)
valid_cols = []
for i, col in enumerate(feature_cols):
    col_data = X_raw[:, i]
    if np.isnan(col_data).any() or np.std(col_data) < 1e-8:
        print(f"  Dropping {col} (constant or NaN)")
        continue
    valid_cols.append(col)

X_raw = merged[valid_cols].values.astype(np.float32)
print(f"Features after cleaning: {len(valid_cols)}")

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

with open(DATA_DIR / 'finance_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

managers_arr = merged['manager'].values
quarters_arr = merged['quarter'].values
alpha_arr    = merged['alpha_real'].values

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIM   = X_scaled.shape[1]
LATENT_DIM  = 4
HIDDEN_DIMS = [32, 16]
EPOCHS      = 3000
LR          = 1e-3
WEIGHT_DECAY= 1e-4
BATCH_SIZE  = 16
PATIENCE    = 300
WARMUP      = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nInput dim:  {INPUT_DIM}")
print(f"Latent dim: {LATENT_DIM}")
print(f"Device:     {device}")

# ── Model ─────────────────────────────────────────────────────────────────────
class ManagerAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc = []
        in_d = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(in_d, h), nn.LayerNorm(h),
                    nn.GELU(), nn.Dropout(0.1)]
            in_d = h
        enc.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        dec.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def make_lr_lambda(epochs, warmup):
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return lr_lambda

def train_model(X_tr, X_vl, verbose=False):
    X_tr_t = torch.tensor(X_tr).to(device)
    X_vl_t = torch.tensor(X_vl).to(device)
    ds      = TensorDataset(X_tr_t, X_tr_t)
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = ManagerAE(INPUT_DIM, HIDDEN_DIMS, LATENT_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(EPOCHS, WARMUP))

    best_val   = float('inf')
    best_state = None
    best_epoch = 0
    patience   = 0

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            xh, z = model(xb)
            loss  = nn.functional.mse_loss(xh, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        scheduler.step()
        avg_tr = ep_loss / len(loader)

        model.eval()
        with torch.no_grad():
            xh_v, _ = model(X_vl_t)
            avg_vl  = nn.functional.mse_loss(xh_v, X_vl_t).item()

        if avg_vl < best_val:
            best_val   = avg_vl
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            break

        if verbose and (epoch + 1) % 300 == 0:
            print(f"    Epoch {epoch+1:4d} | "
                  f"Train: {avg_tr:.5f} | Val: {avg_vl:.5f} | "
                  f"Best: {best_val:.5f}")

    model.load_state_dict(best_state)
    return model, best_val, best_epoch

# ── Manager LOO ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MANAGER LOO VALIDATION")
print("=" * 60)

loo_results = []

for held_out in managers:
    tr_mask = managers_arr != held_out
    vl_mask = managers_arr == held_out

    model_loo, best_val, best_ep = train_model(
        X_scaled[tr_mask], X_scaled[vl_mask]
    )

    model_loo.eval()
    with torch.no_grad():
        X_tr_t = torch.tensor(X_scaled[tr_mask]).to(device)
        X_vl_t = torch.tensor(X_scaled[vl_mask]).to(device)
        xh_tr, _ = model_loo(X_tr_t)
        xh_vl, _ = model_loo(X_vl_t)
        r_tr = nn.functional.mse_loss(xh_tr, X_tr_t).item()
        r_vl = nn.functional.mse_loss(xh_vl, X_vl_t).item()

    mean_alpha = alpha_arr[vl_mask].mean()
    loo_results.append({
        'Manager':    held_out,
        'BestEpoch':  best_ep,
        'TrainRecon': r_tr,
        'ValRecon':   r_vl,
        'MeanAlpha':  mean_alpha,
        'AlphaAnn':   summ_df.set_index('manager').loc[
                          held_out, 'alpha_ann']
                      if held_out in summ_df['manager'].values else np.nan,
    })
    print(f"  {held_out:15s} | Epoch: {best_ep:4d} | "
          f"Train: {r_tr:.4f} | Val: {r_vl:.4f} | "
          f"Alpha: {mean_alpha*100:+.2f}%/q")

loo_df = pd.DataFrame(loo_results).sort_values('ValRecon')
print(f"\nMean train: {loo_df['TrainRecon'].mean():.4f}")
print(f"Mean val:   {loo_df['ValRecon'].mean():.4f}")
print(f"\nFull table:")
print(loo_df.to_string(index=False))
loo_df.to_csv(DATA_DIR / 'finance_loo_results.csv', index=False)

# ── Final model ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)

# Use smallest manager as val
val_manager = min(managers, key=lambda m: (managers_arr == m).sum())
tr_mask = managers_arr != val_manager
vl_mask = managers_arr == val_manager

print(f"Val manager: {val_manager}")
model_final, _, best_ep_f = train_model(
    X_scaled[tr_mask], X_scaled[vl_mask], verbose=True
)

model_final.eval()
with torch.no_grad():
    _, Z_all = model_final(torch.tensor(X_scaled).to(device))
Z_all = Z_all.cpu().numpy()

torch.save(model_final.state_dict(), DATA_DIR / 'finance_ae_final.pth')

# ── Analysis ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

# Manifold stability
latent_var = {}
for m in managers:
    mask = managers_arr == m
    if mask.sum() > 1:
        latent_var[m] = Z_all[mask].var(axis=0).mean()

loo_df['Stability'] = loo_df['Manager'].map(latent_var)
loo_clean = loo_df.dropna(subset=['Stability', 'AlphaAnn'])

r_stab, p_stab = stats.pearsonr(
    loo_clean['Stability'], loo_clean['ValRecon'])
print(f"\nManifold stability vs LOO recon: r={r_stab:.3f}, p={p_stab:.4f}")

# Alpha std vs stability
alpha_std = {m: alpha_arr[managers_arr == m].std()
             for m in managers}
loo_df['AlphaStd'] = loo_df['Manager'].map(alpha_std)
loo_clean2 = loo_df.dropna(subset=['Stability', 'AlphaStd'])

r_astab, p_astab = stats.pearsonr(
    loo_clean2['Stability'], loo_clean2['AlphaStd'])
print(f"Manifold stability vs alpha std: r={r_astab:.3f}, p={p_astab:.4f}")

# Centroid Z1 vs mean alpha
centroids = np.array([Z_all[managers_arr == m].mean(axis=0)
                      for m in managers])
mean_alphas = np.array([alpha_arr[managers_arr == m].mean()
                        for m in managers])

r_cent, p_cent = stats.pearsonr(centroids[:, 0], mean_alphas)
print(f"Centroid Z1 vs mean alpha:       r={r_cent:.3f}, p={p_cent:.4f}")

# LOO alpha prediction
loo_cv    = LeaveOneOut()
loo_preds = np.zeros(len(managers))
for tr_idx, te_idx in loo_cv.split(centroids):
    m = Ridge(alpha=1.0)
    m.fit(centroids[tr_idx], mean_alphas[tr_idx])
    loo_preds[te_idx] = m.predict(centroids[te_idx])

pace_mae  = np.abs(loo_preds - mean_alphas).mean()
pace_base = np.abs(mean_alphas - mean_alphas.mean()).mean()
print(f"\nLOO alpha model:")
print(f"  MAE:      {pace_mae*100:.3f}%/quarter")
print(f"  Baseline: {pace_base*100:.3f}%/quarter")
print(f"  {'BEATS BASELINE' if pace_mae < pace_base else 'DOES NOT BEAT BASELINE'}")

# Permutation test
null_rs = []
for _ in range(1000):
    shuf = np.random.permutation(mean_alphas)
    null_rs.append(stats.pearsonr(centroids[:, 0], shuf)[0])
p_perm = (np.abs(null_rs) >= abs(r_cent)).mean()
print(f"\nPermutation test on centroid-alpha correlation: p={p_perm:.4f}")

# Stability ranking
print(f"\nManager stability ranking:")
print(f"{'Manager':<15} {'Stability':>10} {'AlphaAnn':>10} "
      f"{'AlphaStd':>10} {'ValRecon':>10}")
print("-" * 60)
for _, row in loo_df.sort_values('Stability').iterrows():
    print(f"{row['Manager']:<15} "
          f"{row['Stability']:>10.4f} "
          f"{row.get('AlphaAnn', np.nan)*100:>+9.2f}% "
          f"{row.get('AlphaStd', np.nan)*100:>9.2f}% "
          f"{row['ValRecon']:>10.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
pca2 = PCA(n_components=2)
Z_2d = pca2.fit_transform(Z_all)

colors = plt.cm.tab20(np.linspace(0, 1, len(managers)))
m_colors = dict(zip(managers, colors))

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle('Finance Manager Policy Analysis\n'
             'Behavioral Trace AE | FF3 Factor-Adjusted Alpha',
             fontsize=13, fontweight='bold')

# Panel 1 — latent space
ax = axes[0, 0]
for m in managers:
    mask  = managers_arr == m
    color = m_colors[m]
    ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
              color=color, s=15, alpha=0.4)
    cx = Z_2d[mask, 0].mean()
    cy = Z_2d[mask, 1].mean()
    ax.scatter(cx, cy, color=color, s=150, zorder=4,
              edgecolors='black', linewidths=1.0)
    ax.annotate(m[:8], (cx, cy), textcoords='offset points',
               xytext=(4, 4), fontsize=7, fontweight='bold')
ax.set_title('Manager Policy Space\nBehavioral trace AE')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

# Panel 2 — colored by alpha
ax = axes[0, 1]
sc = ax.scatter(Z_2d[:, 0], Z_2d[:, 1],
                c=alpha_arr*100, cmap='RdYlGn',
                s=15, alpha=0.6, vmin=-10, vmax=10)
plt.colorbar(sc, ax=ax, label='Alpha Realization (%/quarter)')
ax.set_title('Latent Space — Alpha Realizations\nGreen = positive alpha')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

# Panel 3 — centroid Z1 vs mean alpha
ax = axes[0, 2]
for i, m in enumerate(managers):
    color = m_colors[m]
    ax.scatter(mean_alphas[i]*100, centroids[i, 0],
              color=color, s=100, zorder=3,
              edgecolors='black', linewidths=0.5)
    ax.annotate(m[:8], (mean_alphas[i]*100, centroids[i, 0]),
               textcoords='offset points', xytext=(4, 4), fontsize=7)

m2, b2 = np.polyfit(mean_alphas*100, centroids[:, 0], 1)
xl = np.linspace(mean_alphas.min()*100, mean_alphas.max()*100, 100)
ax.plot(xl, m2*xl+b2, 'k--', alpha=0.4,
        label=f'r={r_cent:.3f}, p={p_cent:.4f}')
ax.set_xlabel('Mean Quarterly Alpha (%)')
ax.set_ylabel('Z1 (first latent dim)')
ax.set_title('Style → Alpha\nDoes behavioral fingerprint predict alpha?')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4 — LOO reconstruction
ax = axes[1, 0]
loo_sorted  = loo_df.sort_values('ValRecon')
colors_bar  = [m_colors.get(m, (0.5,)*4)
               for m in loo_sorted['Manager']]
ax.barh(loo_sorted['Manager'], loo_sorted['ValRecon'],
        color=colors_bar, alpha=0.85,
        edgecolor='black', linewidth=0.5)
ax.axvline(1.0, color='red', linestyle='--', alpha=0.5,
           label='Baseline')
ax.set_xlabel('LOO Val Reconstruction MSE')
ax.set_title(f'LOO Reconstruction\n'
             f'Stability r={r_stab:.3f}, p={p_stab:.4f}')
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3)

# Panel 5 — stability vs alpha std
ax = axes[1, 1]
for _, row in loo_clean2.iterrows():
    color = m_colors.get(row['Manager'], (0.5,)*4)
    ax.scatter(row['Stability'], row['AlphaStd']*100,
              color=color, s=100, zorder=3,
              edgecolors='black', linewidths=0.5)
    ax.annotate(row['Manager'][:8],
               (row['Stability'], row['AlphaStd']*100),
               textcoords='offset points', xytext=(4, 4), fontsize=7)

x = loo_clean2['Stability'].values
y = loo_clean2['AlphaStd'].values * 100
m3, b3 = np.polyfit(x, y, 1)
xl2 = np.linspace(x.min(), x.max(), 100)
ax.plot(xl2, m3*xl2+b3, 'k--', alpha=0.4,
        label=f'r={r_astab:.3f}, p={p_astab:.4f}')
ax.set_xlabel('Behavioral Stability (lower = more stable)')
ax.set_ylabel('Alpha Std Dev (%/quarter)')
ax.set_title('Stability → Alpha Consistency\n'
             'Stable managers = more consistent alpha?')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 6 — alpha time series for top/bottom managers
ax = axes[1, 2]
top_managers = (loo_df.sort_values('Stability')
                ['Manager'].tolist()[:3])
bot_managers = (loo_df.sort_values('Stability', ascending=False)
                ['Manager'].tolist()[:3])

for m in top_managers + bot_managers:
    mask    = managers_arr == m
    qs      = [str(q) for q in quarters_arr[mask]]
    alphas  = alpha_arr[mask] * 100
    color   = m_colors[m]
    style   = '-' if m in top_managers else '--'
    label   = f"{m[:8]} ({'stable' if m in top_managers else 'unstable'})"
    ax.plot(range(len(alphas)), alphas,
           color=color, linestyle=style,
           linewidth=1.2, alpha=0.8, label=label)

ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Quarter')
ax.set_ylabel('Alpha Realization (%/quarter)')
ax.set_title('Alpha Time Series\nStable vs Unstable Managers')
ax.legend(fontsize=6)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / 'finance_ae_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: finance_ae_analysis.png")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    'r_stability_loo':   float(r_stab),
    'p_stability_loo':   float(p_stab),
    'r_stability_alpha': float(r_astab),
    'p_stability_alpha': float(p_astab),
    'r_centroid_alpha':  float(r_cent),
    'p_centroid_alpha':  float(p_cent),
    'loo_mae':           float(pace_mae),
    'baseline_mae':      float(pace_base),
    'permutation_p':     float(p_perm),
    'n_managers':        len(managers),
    'n_obs':             int(len(merged)),
}

with open(DATA_DIR / 'finance_ae_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved")
print("Done — paste output and upload finance_ae_analysis.png")