# -*- coding: utf-8 -*-
"""
stock_velocity_deepdive.py
Deep dive on the latent velocity → mean reversion finding.

Tests:
  1. Volatility control — does velocity predict alpha after partialing out vol?
  2. Direction vs magnitude — is the axis of movement more predictive than speed?
  3. Decay structure — at what horizon does the signal live?
  4. Cross-sectional vs time-series decomposition — where is the signal?
  5. Long-short portfolio backtest — is this tradeable?

Requires: latent_velocity.parquet (from stock_manifold_curiosities.py)
          ae_temporal_best.pt
          stock_clean_*.parquet files
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, ttest_ind
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')
PLOT_DIR = DATA_DIR / 'plots' / 'velocity_deepdive'
PLOT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RNG      = np.random.RandomState(42)

CUTOFF_DATE = '2023-01-01'
VEL_WINDOW  = 21

print(f"Device: {DEVICE}")
print(f"Plots:  {PLOT_DIR}")

# ── Utilities ─────────────────────────────────────────────────────────────────
class StockDataset(Dataset):
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx]


def build_model_from_ckpt(ckpt):
    s   = ckpt['model_state']
    h1  = s['encoder.0.weight'].shape[0]
    h2  = s['encoder.4.weight'].shape[0]
    inp = s['encoder.0.weight'].shape[1]
    lat = s['encoder.8.weight'].shape[0]

    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(inp, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(h1, h2),  nn.LayerNorm(h2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(h2, lat),
            )
            self.decoder = nn.Sequential(
                nn.Linear(lat, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(h2, h1),  nn.LayerNorm(h1), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(h1, inp),
            )
        def encode(self, x): return self.encoder(x)
        def decode(self, z): return self.decoder(z)
        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    model = _AE().to(DEVICE)
    model.load_state_dict(s)
    model.eval()
    return model, inp, lat


def encode_batch(model, X_scaled, batch_size=8192):
    loader = DataLoader(StockDataset(X_scaled.astype(np.float32)),
                        batch_size=batch_size, shuffle=False)
    model.eval()
    parts = []
    with torch.no_grad():
        for xb in loader:
            _, z = model(xb.to(DEVICE))
            parts.append(z.cpu().numpy())
    return np.vstack(parts)


def clean_X(X):
    X = np.where(np.isinf(X), np.nan, X)
    col_meds = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        m = np.isnan(X[:, j])
        X[m, j] = col_meds[j]
    return np.nan_to_num(X, nan=0.0)


def partial_correlation(x, y, z):
    """
    Partial correlation of x and y controlling for z.
    Drops NaN/inf rows before fitting.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    valid = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(z), axis=1)
    if valid.sum() < 10:
        return np.nan, np.nan
    x, y, z = x[valid], y[valid], z[valid]
    reg = LinearRegression()
    reg.fit(z, x)
    x_resid = x - reg.predict(z)
    reg.fit(z, y)
    y_resid = y - reg.predict(z)
    r, p = pearsonr(x_resid, y_resid)
    return float(r), float(p)

    y_resid = y - reg.predict(z)
    r, p = pearsonr(x_resid, y_resid)
    return float(r), float(p)


# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading temporal model...")
ckpt         = torch.load(DATA_DIR / 'ae_temporal_best.pt',
                           map_location=DEVICE, weights_only=False)
model, input_dim, latent_dim = build_model_from_ckpt(ckpt)
scaler       = ckpt['scaler']
feature_cols = ckpt['feature_cols']
print(f"  Latent dim: {latent_dim}  |  Features: {len(feature_cols)}")

# ── Build rich velocity dataset ───────────────────────────────────────────────
# We need more than what the curiosities script saved — specifically:
#   - velocity vector (not just magnitude)
#   - contemporaneous vol for control
#   - multiple forward horizons
#   - per-stock identifiers for CS/TS decomposition

print("\nBuilding rich velocity dataset (this takes a while)...")
manifest    = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
all_tickers = manifest['ticker'].tolist()

HORIZONS = [5, 10, 21, 42, 63, 126, 252]   # forward alpha at each horizon

all_obs = []

for ticker in all_tickers:
    f = DATA_DIR / f'stock_clean_{ticker}.parquet'
    if not f.exists():
        continue
    try:
        df    = pd.read_parquet(f)
        dates = pd.to_datetime(df.index)

        # Pre-cutoff only
        pre_mask = dates < pd.Timestamp(CUTOFF_DATE)
        df    = df[pre_mask]
        dates = dates[pre_mask]

        if len(df) < VEL_WINDOW * 4:
            continue

        # Feature matrix → encode
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X   = clean_X(df[feature_cols].values.astype(np.float32))
        X_s = scaler.transform(X).astype(np.float32)
        Z   = encode_batch(model, X_s)   # (T, latent_dim)

        y_alpha = df['alpha_resid'].values \
            if 'alpha_resid' in df.columns else np.zeros(len(df))
        y_ret   = df['ret_1d'].values \
            if 'ret_1d' in df.columns else np.zeros(len(df))

        # Vol from features
        vol_col = 'vol_21d'
        vol_vals= df[vol_col].values if vol_col in df.columns \
                  else np.full(len(df), np.nan)

        # Step size series (T-1,)
        steps = np.linalg.norm(np.diff(Z, axis=0), axis=1)
        # Step vectors (T-1, latent_dim) — direction of movement
        step_vecs = np.diff(Z, axis=0)   # unnormalized

        for i in range(VEL_WINDOW, len(Z) - max(HORIZONS)):
            # Scalar velocity: mean step size over window
            vel_mag = float(steps[i-VEL_WINDOW:i].mean())

            # Velocity vector: mean displacement direction over window
            vel_vec = step_vecs[i-VEL_WINDOW:i].mean(axis=0)

            # Velocity acceleration
            vel_prev = float(steps[max(0, i-VEL_WINDOW*2):i-VEL_WINDOW].mean()) \
                       if i >= VEL_WINDOW * 2 else vel_mag
            vel_accel = vel_mag - vel_prev

            # Current latent position
            z_now = Z[i]

            # Contemporaneous vol
            vol_now = float(vol_vals[i]) if not np.isnan(vol_vals[i]) else np.nan

            # Forward alpha at each horizon
            fwd_alphas = {}
            for h in HORIZONS:
                if i + h <= len(y_alpha):
                    fa = float(np.nanmean(y_alpha[i:i+h]))
                    fwd_alphas[h] = fa if not np.isnan(fa) else np.nan
                else:
                    fwd_alphas[h] = np.nan

            # Forward return at 21d (for portfolio backtest)
            fwd_ret_21 = float(np.nanmean(y_ret[i:i+21])) \
                         if i + 21 <= len(y_ret) else np.nan

            obs = {
                'ticker':     ticker,
                'date':       dates[i],
                'vel_mag':    vel_mag,
                'vel_accel':  vel_accel,
                'vol_21d':    vol_now,
                'z_now':      z_now,
                'vel_vec':    vel_vec,
                'fwd_ret_21': fwd_ret_21,
            }
            for h in HORIZONS:
                obs[f'fwd_alpha_{h}'] = fwd_alphas[h]

            all_obs.append(obs)

    except Exception as e:
        continue

print(f"  Total observations: {len(all_obs):,}")

# Extract arrays
dates_arr   = np.array([o['date']     for o in all_obs])
tickers_arr = np.array([o['ticker']   for o in all_obs])
vel_mag     = np.array([o['vel_mag']  for o in all_obs])
vel_accel   = np.array([o['vel_accel'] for o in all_obs])
vol_21d     = np.array([o['vol_21d']  for o in all_obs])
z_matrix    = np.array([o['z_now']    for o in all_obs])   # (N, latent_dim)
vel_vecs    = np.array([o['vel_vec']  for o in all_obs])   # (N, latent_dim)
fwd_ret_21  = np.array([o['fwd_ret_21'] for o in all_obs])

fwd_alphas  = {}
for h in HORIZONS:
    fwd_alphas[h] = np.array([o[f'fwd_alpha_{h}'] for o in all_obs])

# Build a DataFrame for easier manipulation
df_main = pd.DataFrame({
    'ticker':    tickers_arr,
    'date':      dates_arr,
    'vel_mag':   vel_mag,
    'vel_accel': vel_accel,
    'vol_21d':   vol_21d,
    'fwd_ret_21': fwd_ret_21,
})
for h in HORIZONS:
    df_main[f'fwd_alpha_{h}'] = fwd_alphas[h]

df_main = df_main.dropna(subset=['vel_mag', 'vol_21d', 'fwd_alpha_21'])
print(f"  Clean observations: {len(df_main):,}")

# Re-extract clean arrays
vel_mag_c  = df_main['vel_mag'].values
vol_21d_c  = df_main['vol_21d'].values
fa_21_c    = df_main['fwd_alpha_21'].values

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Volatility Control
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1: VOLATILITY CONTROL")
print("=" * 70)
print("Does velocity predict alpha after partialing out contemporaneous vol?")

# Raw correlations
r_vel_alpha, p_vel_alpha = pearsonr(vel_mag_c, fa_21_c)
r_vol_alpha, p_vol_alpha = pearsonr(vol_21d_c, fa_21_c)
r_vel_vol,   p_vel_vol   = pearsonr(vel_mag_c, vol_21d_c)

print(f"\n  Raw correlations:")
print(f"    vel_mag → fwd_alpha_21:  r={r_vel_alpha:.4f}  p={p_vel_alpha:.4f}")
print(f"    vol_21d → fwd_alpha_21:  r={r_vol_alpha:.4f}  p={p_vol_alpha:.4f}")
print(f"    vel_mag → vol_21d:       r={r_vel_vol:.4f}  p={p_vel_vol:.4f}")

# Partial correlation: vel_mag → fwd_alpha controlling for vol_21d
r_partial_vel, p_partial_vel = partial_correlation(
    vel_mag_c, fa_21_c, vol_21d_c)

# Partial correlation: vol_21d → fwd_alpha controlling for vel_mag
r_partial_vol, p_partial_vol = partial_correlation(
    vol_21d_c, fa_21_c, vel_mag_c)

print(f"\n  Partial correlations:")
print(f"    vel_mag → fwd_alpha | vol_21d:  r={r_partial_vel:.4f}  p={p_partial_vel:.4f}")
print(f"    vol_21d → fwd_alpha | vel_mag:  r={r_partial_vol:.4f}  p={p_partial_vol:.4f}")

if abs(r_partial_vel) > 0.01 and p_partial_vel < 0.05:
    print(f"\n  → VELOCITY SURVIVES VOL CONTROL — independent signal")
else:
    print(f"\n  → VELOCITY IS A VOL PROXY — signal absorbed by vol control")

# Multiple regression: alpha ~ vel + vol
X_reg = np.column_stack([vel_mag_c, vol_21d_c,
                          vel_mag_c**2, vol_21d_c**2,
                          vel_mag_c * vol_21d_c])
ridge = Ridge(alpha=1.0)
ridge.fit(X_reg, fa_21_c)
fa_pred = ridge.predict(X_reg)
r2_full = float(r2_score(fa_21_c, fa_pred))

# R² from vol alone
ridge_vol = Ridge(alpha=1.0)
ridge_vol.fit(vol_21d_c.reshape(-1, 1), fa_21_c)
r2_vol = float(r2_score(fa_21_c, ridge_vol.predict(vol_21d_c.reshape(-1, 1))))

# R² from vel alone
ridge_vel = Ridge(alpha=1.0)
ridge_vel.fit(vel_mag_c.reshape(-1, 1), fa_21_c)
r2_vel = float(r2_score(fa_21_c, ridge_vel.predict(vel_mag_c.reshape(-1, 1))))

print(f"\n  R² breakdown:")
print(f"    Vol only:     {r2_vol:.5f}")
print(f"    Vel only:     {r2_vel:.5f}")
print(f"    Vol + vel:    {r2_full:.5f}")
print(f"    Vel adds:     {r2_full - r2_vol:.5f}  ({(r2_full-r2_vol)/max(r2_vol,1e-8)*100:.1f}% gain)")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Raw velocity vs alpha
axes[0].scatter(vel_mag_c[::100], fa_21_c[::100], alpha=0.1, s=3, color='steelblue')
axes[0].set_xlabel('Latent velocity (mag)')
axes[0].set_ylabel('Forward alpha (21d)')
axes[0].set_title(f'Raw: Vel → Alpha\nr={r_vel_alpha:.3f}')

# Partial: velocity residual vs alpha residual
reg_temp = LinearRegression()
reg_temp.fit(vol_21d_c.reshape(-1, 1), vel_mag_c)
vel_resid = vel_mag_c - reg_temp.predict(vol_21d_c.reshape(-1, 1))
reg_temp.fit(vol_21d_c.reshape(-1, 1), fa_21_c)
alpha_resid = fa_21_c - reg_temp.predict(vol_21d_c.reshape(-1, 1))

axes[1].scatter(vel_resid[::100], alpha_resid[::100], alpha=0.1, s=3,
                color='darkorange')
axes[1].set_xlabel('Velocity residual (after vol)')
axes[1].set_ylabel('Alpha residual (after vol)')
axes[1].set_title(f'Partial: Vel → Alpha | Vol\nr={r_partial_vel:.3f}')

# Vol vs alpha
axes[2].scatter(vol_21d_c[::100], fa_21_c[::100], alpha=0.1, s=3, color='crimson')
axes[2].set_xlabel('Vol 21d')
axes[2].set_ylabel('Forward alpha (21d)')
axes[2].set_title(f'Raw: Vol → Alpha\nr={r_vol_alpha:.3f}')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'vol_control.png', dpi=150)
plt.close()
print(f"\n  Plot saved: vol_control.png")

vol_control_results = {
    'r_vel_alpha':      float(r_vel_alpha),
    'r_vol_alpha':      float(r_vol_alpha),
    'r_vel_vol':        float(r_vel_vol),
    'r_partial_vel':    float(r_partial_vel),
    'p_partial_vel':    float(p_partial_vel),
    'r_partial_vol':    float(r_partial_vol),
    'r2_vel_only':      float(r2_vel),
    'r2_vol_only':      float(r2_vol),
    'r2_combined':      float(r2_full),
    'incremental_r2':   float(r2_full - r2_vol),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Direction vs Magnitude
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: DIRECTION VS MAGNITUDE")
print("=" * 70)
print("Is the axis of movement more predictive than raw speed?")

# We know from section 5 of eval that:
# z5 ≈ volatility  z7 ≈ price position/drawdown  z0 ≈ momentum

# For each observation compute dot product of velocity vector with named axes
# This measures "how much is the stock moving toward/away from each axis"
vel_vecs_clean = vel_vecs[:len(df_main)]   # align with clean df

# Unit axis vectors (standard basis)
axis_projections = {}
for i in range(latent_dim):
    axis_vec = np.zeros(latent_dim)
    axis_vec[i] = 1.0
    # Projection of velocity vector onto axis i
    proj = vel_vecs_clean @ axis_vec   # (N,)
    axis_projections[f'z{i}'] = proj

# Correlation of each axis projection with forward alpha
print(f"\n  Per-axis velocity projection → forward alpha (21d):")
print(f"  {'Axis':<8} {'Raw r':>8} {'Partial r (|vol)':>18} {'Direction matters?'}")
print(f"  {'─'*55}")

axis_corrs = {}
for axis_name, proj in axis_projections.items():
    # Align with clean mask
    proj_clean = proj[:len(df_main)]
    r_raw, p_raw = pearsonr(proj_clean, fa_21_c)

    # Partial controlling for vol and velocity magnitude
    controls = np.column_stack([vol_21d_c, vel_mag_c])
    r_part, p_part = partial_correlation(proj_clean, fa_21_c, controls)

    axis_corrs[axis_name] = {
        'r_raw': float(r_raw),
        'p_raw': float(p_raw),
        'r_partial': float(r_part),
        'p_partial': float(p_part),
    }

    sig = '✓' if p_part < 0.05 and abs(r_part) > 0.01 else ' '
    print(f"  {axis_name:<8} {r_raw:>8.4f} {r_part:>18.4f}  {sig}")

# Best axis projections — stronger than raw magnitude?
best_axis = max(axis_corrs, key=lambda k: abs(axis_corrs[k]['r_partial']))
print(f"\n  Best axis by partial r: {best_axis} "
      f"(r_partial={axis_corrs[best_axis]['r_partial']:.4f})")
print(f"  Raw velocity magnitude partial r: {r_partial_vel:.4f}")

if abs(axis_corrs[best_axis]['r_partial']) > abs(r_partial_vel):
    print(f"  → DIRECTION MATTERS MORE THAN MAGNITUDE")
else:
    print(f"  → MAGNITUDE IS THE DOMINANT SIGNAL")

# Composite directional signal: weighted sum of axis projections by their r
weights = np.array([axis_corrs[f'z{i}']['r_raw'] for i in range(latent_dim)])
composite_dir = vel_vecs_clean @ weights   # (N,) weighted projection
r_composite, p_composite = pearsonr(composite_dir[:len(df_main)], fa_21_c)
print(f"  Composite directional signal r: {r_composite:.4f}  p={p_composite:.4f}")

# Plot: axis projection correlations
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axis_names = [f'z{i}' for i in range(latent_dim)]
r_raws     = [axis_corrs[a]['r_raw']     for a in axis_names]
r_parts    = [axis_corrs[a]['r_partial'] for a in axis_names]

x = range(latent_dim)
axes[0].bar(x, r_raws, color=['steelblue' if r > 0 else 'crimson' for r in r_raws],
            alpha=0.8)
axes[0].axhline(0, color='black', lw=0.8)
axes[0].axhline(r_vel_alpha, color='gray', linestyle='--', alpha=0.7,
                label=f'Raw vel mag r={r_vel_alpha:.3f}')
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(axis_names, fontsize=8)
axes[0].set_ylabel('r with forward alpha')
axes[0].set_title('Axis Projection → Alpha\n(raw correlation)')
axes[0].legend(fontsize=8)

axes[1].bar(x, r_parts, color=['steelblue' if r > 0 else 'crimson' for r in r_parts],
            alpha=0.8)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].axhline(r_partial_vel, color='gray', linestyle='--', alpha=0.7,
                label=f'Vel mag partial r={r_partial_vel:.3f}')
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(axis_names, fontsize=8)
axes[1].set_ylabel('Partial r with forward alpha | vol, vel_mag')
axes[1].set_title('Axis Projection → Alpha\n(partial correlation, controlling vol + vel_mag)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'direction_vs_magnitude.png', dpi=150)
plt.close()
print(f"\n  Plot saved: direction_vs_magnitude.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Signal Decay Structure
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3: SIGNAL DECAY STRUCTURE")
print("=" * 70)
print("At what horizon does the velocity → alpha signal live?")

horizons_results = {}
print(f"\n  {'Horizon':<10} {'Raw r':>8} {'Partial r':>12} {'p':>10}  Interpretation")
print(f"  {'─'*60}")

for h in HORIZONS:
    fa_h_full = fwd_alphas[h][:len(df_main)]
    common_idx = (
        np.isfinite(vel_mag_c) &
        np.isfinite(vol_21d_c) &
        np.isfinite(fa_h_full)
    )

    if common_idx.sum() < 1000:
        continue

    vel_h = vel_mag_c[common_idx]
    vol_h = vol_21d_c[common_idx]
    fa_h  = fa_h_full[common_idx]

    r_raw, p_raw   = pearsonr(vel_h, fa_h)
    r_part, p_part = partial_correlation(vel_h, fa_h, vol_h)

    if h <= 10:
        interp = 'microstructure'
    elif h <= 42:
        interp = 'short-term mean rev'
    elif h <= 126:
        interp = 'medium-term'
    else:
        interp = 'long-term / structural'

    horizons_results[h] = {
        'r_raw': float(r_raw),
        'r_partial': float(r_part),
        'p_partial': float(p_part),
        'n': int(common_idx.sum()),
    }
    sig = '✓' if p_part < 0.05 else ' '
    print(f"  {str(h)+'d':<10} {r_raw:>8.4f} {r_part:>12.4f} {p_part:>10.4f}  "
          f"{interp} {sig}")

# Plot decay curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

h_vals   = list(horizons_results.keys())
r_raws   = [horizons_results[h]['r_raw']     for h in h_vals]
r_parts  = [horizons_results[h]['r_partial'] for h in h_vals]

axes[0].plot(h_vals, r_raws,  'o-', color='steelblue', lw=2, ms=8, label='Raw r')
axes[0].plot(h_vals, r_parts, 's--', color='crimson',   lw=2, ms=8, label='Partial r | vol')
axes[0].axhline(0, color='black', lw=0.8, linestyle=':')
axes[0].set_xlabel('Forward horizon (days)')
axes[0].set_ylabel('Correlation with forward alpha')
axes[0].set_title('Signal Decay: Velocity → Alpha\nat different horizons')
axes[0].legend()
axes[0].set_xscale('log')

# Information coefficient per horizon
ics = []
for h in h_vals:
    fa_h_full = fwd_alphas[h][:len(df_main)]
    common = np.isfinite(vel_mag_c) & np.isfinite(fa_h_full)
    if common.sum() < 100:
        ics.append(np.nan)
        continue
    ic, _ = spearmanr(vel_mag_c[common], fa_h_full[common])
    ics.append(float(ic))

axes[1].bar(range(len(h_vals)), ics,
            color=['steelblue' if ic < 0 else 'crimson' for ic in ics],
            alpha=0.8)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_xticks(range(len(h_vals)))
axes[1].set_xticklabels([f'{h}d' for h in h_vals], fontsize=9)
axes[1].set_ylabel('Spearman IC (rank correlation)')
axes[1].set_title('Information Coefficient by Horizon\n'
                  '(negative IC = velocity predicts underperformance)')

plt.tight_layout()
plt.savefig(PLOT_DIR / 'signal_decay.png', dpi=150)
plt.close()
print(f"\n  Plot saved: signal_decay.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Cross-Sectional vs Time-Series Decomposition
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: CROSS-SECTIONAL VS TIME-SERIES DECOMPOSITION")
print("=" * 70)
print("Where does the signal live — between stocks or within stocks over time?")

df_cs = df_main[['ticker', 'date', 'vel_mag', 'vol_21d', 'fwd_alpha_21']].copy()

# ── Cross-sectional signal ────────────────────────────────────────────────────
# For each date, rank stocks by velocity, compute correlation with forward alpha
print("\nCross-sectional analysis (daily rank correlations)...")
cs_ics = []
unique_dates = np.unique(df_cs['date'].values)

for d in unique_dates:
    mask_d = df_cs['date'] == d
    if mask_d.sum() < 10:
        continue
    sub    = df_cs[mask_d]
    ic, _  = spearmanr(sub['vel_mag'], sub['fwd_alpha_21'])
    if not np.isnan(ic):
        cs_ics.append({'date': d, 'ic': float(ic), 'n': int(mask_d.sum())})

cs_ic_df  = pd.DataFrame(cs_ics)
mean_cs_ic = float(cs_ic_df['ic'].mean())
t_cs, p_cs = ttest_ind(cs_ic_df['ic'], np.zeros(len(cs_ic_df)))
ic_pct_neg = float((cs_ic_df['ic'] < 0).mean())

print(f"  Mean daily CS IC:    {mean_cs_ic:.4f}")
print(f"  p (IC ≠ 0):         {p_cs:.4f}")
print(f"  % days IC < 0:      {ic_pct_neg*100:.1f}%")
print(f"  (Negative IC = fast stocks underperform slow stocks on that day)")

# ── Time-series signal ────────────────────────────────────────────────────────
# For each stock, demean velocity and alpha (remove cross-sectional component)
# Then correlate within-stock velocity with within-stock alpha
print("\nTime-series analysis (within-stock correlations)...")
ts_ics = []
for ticker in np.unique(df_cs['ticker'].values):
    sub = df_cs[df_cs['ticker'] == ticker].copy()
    if len(sub) < 63:
        continue
    # Demean (remove stock fixed effect)
    sub['vel_dm']   = sub['vel_mag']     - sub['vel_mag'].mean()
    sub['alpha_dm'] = sub['fwd_alpha_21'] - sub['fwd_alpha_21'].mean()
    ic, _ = spearmanr(sub['vel_dm'], sub['alpha_dm'])
    if not np.isnan(ic):
        ts_ics.append({'ticker': ticker, 'ic': float(ic), 'n': len(sub)})

ts_ic_df   = pd.DataFrame(ts_ics)
mean_ts_ic  = float(ts_ic_df['ic'].mean())
t_ts, p_ts  = ttest_ind(ts_ic_df['ic'], np.zeros(len(ts_ic_df)))
ts_ic_neg   = float((ts_ic_df['ic'] < 0).mean())

print(f"  Mean stock-level TS IC:  {mean_ts_ic:.4f}")
print(f"  p (IC ≠ 0):              {p_ts:.4f}")
print(f"  % stocks IC < 0:         {ts_ic_neg*100:.1f}%")
print(f"  (Negative IC = when THIS stock moves fast it subsequently underperforms)")

print(f"\n  Decomposition summary:")
print(f"    Cross-sectional IC: {mean_cs_ic:.4f}  p={p_cs:.4f}")
print(f"    Time-series IC:     {mean_ts_ic:.4f}  p={p_ts:.4f}")

if abs(mean_cs_ic) > abs(mean_ts_ic):
    print(f"  → SIGNAL IS PRIMARILY CROSS-SECTIONAL")
    print(f"     Fast-moving stocks underperform slow-moving stocks on any given day")
else:
    print(f"  → SIGNAL IS PRIMARILY TIME-SERIES")
    print(f"     Each stock underperforms itself when it's moving fast")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cs_ic_series = pd.Series(cs_ic_df['ic'].values,
                          index=pd.DatetimeIndex(cs_ic_df['date']))
cs_ic_smooth = cs_ic_series.rolling(21).mean()

axes[0].plot(cs_ic_series.index, cs_ic_series.values,
             alpha=0.3, lw=0.5, color='steelblue')
axes[0].plot(cs_ic_smooth.index, cs_ic_smooth.values,
             lw=1.5, color='steelblue', label='21d MA')
axes[0].axhline(0, color='black', lw=0.8, linestyle='--')
axes[0].axhline(mean_cs_ic, color='crimson', lw=1, linestyle=':',
                label=f'Mean={mean_cs_ic:.3f}')
axes[0].set_title(f'Daily Cross-Sectional IC\np={p_cs:.4f}')
axes[0].set_ylabel('Spearman IC')
axes[0].legend(fontsize=8)

axes[1].hist(cs_ic_df['ic'], bins=50, color='steelblue', alpha=0.7)
axes[1].axvline(0, color='black', lw=1)
axes[1].axvline(mean_cs_ic, color='crimson', lw=2, linestyle='--',
                label=f'Mean={mean_cs_ic:.3f}')
axes[1].set_xlabel('Daily CS IC')
axes[1].set_title(f'CS IC Distribution\n{ic_pct_neg*100:.1f}% negative')
axes[1].legend(fontsize=8)

axes[2].hist(ts_ic_df['ic'], bins=50, color='darkorange', alpha=0.7)
axes[2].axvline(0, color='black', lw=1)
axes[2].axvline(mean_ts_ic, color='crimson', lw=2, linestyle='--',
                label=f'Mean={mean_ts_ic:.3f}')
axes[2].set_xlabel('Per-stock TS IC')
axes[2].set_title(f'TS IC Distribution\n{ts_ic_neg*100:.1f}% negative')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOT_DIR / 'cs_vs_ts.png', dpi=150)
plt.close()
print(f"\n  Plot saved: cs_vs_ts.png")

cs_ts_results = {
    'mean_cs_ic':  float(mean_cs_ic),
    'p_cs':        float(p_cs),
    'pct_cs_neg':  float(ic_pct_neg),
    'mean_ts_ic':  float(mean_ts_ic),
    'p_ts':        float(p_ts),
    'pct_ts_neg':  float(ts_ic_neg),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Long-Short Portfolio Backtest
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5: LONG-SHORT PORTFOLIO BACKTEST")
print("=" * 70)
print("Paper portfolio: short high-velocity stocks, long low-velocity stocks.")
print("Rebalanced monthly. Equal-weighted within quintiles.")

# For each month-end date:
#   - Rank all stocks by velocity
#   - Go long Q1 (low velocity), short Q5 (high velocity)
#   - Hold for 21 trading days
#   - Compute portfolio return

df_bt = df_main[['ticker', 'date', 'vel_mag', 'vol_21d',
                  'fwd_ret_21', 'fwd_alpha_21']].copy()
df_bt = df_bt.dropna(subset=['vel_mag', 'fwd_ret_21'])
df_bt['date'] = pd.to_datetime(df_bt['date'])

# Rebalance monthly — use approximately every 21st trading date
unique_dates_sorted = np.sort(df_bt['date'].unique())
rebal_dates = unique_dates_sorted[::21]   # every ~21 trading days

print(f"\n  Rebalancing dates: {len(rebal_dates)}")
print(f"  Date range: {str(rebal_dates[0])[:10]} — {str(rebal_dates[-1])[:10]}")

portfolio_returns = []

for rebal_date in rebal_dates:
    day_data = df_bt[df_bt['date'] == rebal_date].copy()
    if len(day_data) < 20:
        continue

    # Cross-sectional rank of velocity
    day_data['vel_rank'] = rankdata(day_data['vel_mag']) / len(day_data)

    # Q1 = lowest velocity (long), Q5 = highest velocity (short)
    long_mask  = day_data['vel_rank'] <= 0.20
    short_mask = day_data['vel_rank'] >= 0.80

    long_ret  = float(day_data.loc[long_mask,  'fwd_ret_21'].mean())
    short_ret = float(day_data.loc[short_mask, 'fwd_ret_21'].mean())
    ls_ret    = long_ret - short_ret

    long_alpha  = float(day_data.loc[long_mask,  'fwd_alpha_21'].mean())
    short_alpha = float(day_data.loc[short_mask, 'fwd_alpha_21'].mean())
    ls_alpha    = long_alpha - short_alpha

    portfolio_returns.append({
        'date':       rebal_date,
        'long_ret':   long_ret,
        'short_ret':  short_ret,
        'ls_ret':     ls_ret,
        'long_alpha': long_alpha,
        'short_alpha':short_alpha,
        'ls_alpha':   ls_alpha,
        'n_long':     int(long_mask.sum()),
        'n_short':    int(short_mask.sum()),
    })

port_df = pd.DataFrame(portfolio_returns)
port_df['date'] = pd.to_datetime(port_df['date'])
port_df = port_df.set_index('date').sort_index()

# Performance stats
mean_ls_ret   = float(port_df['ls_ret'].mean())
std_ls_ret    = float(port_df['ls_ret'].std())
sharpe_ret    = mean_ls_ret / (std_ls_ret + 1e-8) * np.sqrt(252 / 21)
ann_ls_ret    = mean_ls_ret * 252 / 21

mean_ls_alpha = float(port_df['ls_alpha'].mean())
std_ls_alpha  = float(port_df['ls_alpha'].std())
sharpe_alpha  = mean_ls_alpha / (std_ls_alpha + 1e-8) * np.sqrt(252 / 21)
ann_ls_alpha  = mean_ls_alpha * 252 / 21

hit_rate      = float((port_df['ls_ret'] > 0).mean())
t_stat, p_ls  = ttest_ind(port_df['ls_ret'], np.zeros(len(port_df)))

print(f"\n  Long-short (low vel long, high vel short):")
print(f"    Mean period return:   {mean_ls_ret*100:.4f}%")
print(f"    Annualized return:    {ann_ls_ret*100:.2f}%")
print(f"    Annualized Sharpe:    {sharpe_ret:.3f}")
print(f"    Hit rate:             {hit_rate*100:.1f}%")
print(f"    p-value:              {p_ls:.4f}")
print(f"\n  Alpha-adjusted L/S:")
print(f"    Mean period alpha:    {mean_ls_alpha*100:.4f}%")
print(f"    Annualized alpha:     {ann_ls_alpha*100:.2f}%")
print(f"    Annualized Sharpe:    {sharpe_alpha:.3f}")

# Crisis period performance
for name, start, end in [('GFC', '2008-09-01', '2009-03-31'),
                          ('COVID', '2020-02-01', '2020-06-30'),
                          ('Rate', '2022-01-01', '2022-12-31'),
                          ('Bull', '2013-01-01', '2019-12-31')]:
    mask = (port_df.index >= start) & (port_df.index <= end)
    if mask.sum() < 3:
        continue
    period_ret = port_df.loc[mask, 'ls_ret'].mean() * 252 / 21
    print(f"  {name:<10} annualized L/S: {period_ret*100:+.2f}%")

# Cumulative returns
cumret = (1 + port_df['ls_ret']).cumprod()
cumret_alpha = (1 + port_df['ls_alpha']).cumprod()

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Cumulative return
axes[0, 0].plot(cumret.index, cumret.values, lw=1.5, color='steelblue',
                label='L/S return')
axes[0, 0].plot(cumret_alpha.index, cumret_alpha.values, lw=1.5,
                color='darkorange', linestyle='--', label='L/S alpha')
axes[0, 0].axhline(1, color='black', lw=0.8, linestyle=':')
for name, start, end in [('GFC', '2008-09-01', '2009-03-31'),
                          ('COVID', '2020-02-01', '2020-06-30')]:
    axes[0, 0].axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.1, color='red')
axes[0, 0].set_title(f'Cumulative L/S Return\n'
                     f'Ann. Sharpe={sharpe_ret:.2f}  Hit={hit_rate*100:.0f}%')
axes[0, 0].set_ylabel('Cumulative return')
axes[0, 0].legend(fontsize=8)

# Period returns distribution
axes[0, 1].hist(port_df['ls_ret'] * 100, bins=40, color='steelblue', alpha=0.7)
axes[0, 1].axvline(0, color='black', lw=1)
axes[0, 1].axvline(mean_ls_ret * 100, color='crimson', lw=2, linestyle='--',
                   label=f'Mean={mean_ls_ret*100:.3f}%')
axes[0, 1].set_xlabel('Period L/S return (%)')
axes[0, 1].set_title(f'Return Distribution\np={p_ls:.4f}')
axes[0, 1].legend(fontsize=8)

# Long and short leg separately
axes[1, 0].plot(port_df.index, port_df['long_ret'].rolling(5).mean() * 100,
                lw=1, color='steelblue', label='Long (low vel)')
axes[1, 0].plot(port_df.index, port_df['short_ret'].rolling(5).mean() * 100,
                lw=1, color='crimson', label='Short (high vel)')
axes[1, 0].axhline(0, color='black', lw=0.8, linestyle=':')
axes[1, 0].set_ylabel('5-period MA return (%)')
axes[1, 0].set_title('Long vs Short Leg (5-period MA)')
axes[1, 0].legend(fontsize=8)

# Annual performance
port_df['year'] = port_df.index.year
annual = port_df.groupby('year')['ls_ret'].mean() * 252 / 21
colors_bar = ['steelblue' if r > 0 else 'crimson' for r in annual.values]
axes[1, 1].bar(annual.index, annual.values * 100, color=colors_bar, alpha=0.8)
axes[1, 1].axhline(0, color='black', lw=0.8)
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Annualized L/S return (%)')
axes[1, 1].set_title('Annual Performance')

plt.suptitle('Latent Velocity Long-Short Backtest\n'
             '(Long low-velocity, short high-velocity, rebal monthly)',
             fontsize=11)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'backtest.png', dpi=150)
plt.close()
print(f"\n  Plot saved: backtest.png")

backtest_results = {
    'mean_ls_ret':   float(mean_ls_ret),
    'ann_ls_ret':    float(ann_ls_ret),
    'sharpe_ret':    float(sharpe_ret),
    'ann_ls_alpha':  float(ann_ls_alpha),
    'sharpe_alpha':  float(sharpe_alpha),
    'hit_rate':      float(hit_rate),
    'p_value':       float(p_ls),
    'n_periods':     len(port_df),
}


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VELOCITY DEEP DIVE SUMMARY")
print("=" * 70)

print(f"\nSection 1 — Volatility control:")
print(f"  Partial r (vel | vol): {r_partial_vel:.4f}  p={p_partial_vel:.4f}")
print(f"  Incremental R²:        {r2_full - r2_vol:.5f}")

print(f"\nSection 2 — Direction vs magnitude:")
print(f"  Best axis partial r:   {axis_corrs[best_axis]['r_partial']:.4f}  ({best_axis})")
print(f"  Vel mag partial r:     {r_partial_vel:.4f}")
print(f"  Composite dir r:       {r_composite:.4f}")

print(f"\nSection 3 — Decay structure:")
for h, res in horizons_results.items():
    print(f"  {str(h)+'d':<6} raw r={res['r_raw']:.4f}  partial r={res['r_partial']:.4f}")

print(f"\nSection 4 — CS vs TS decomposition:")
print(f"  CS IC: {mean_cs_ic:.4f}  p={p_cs:.4f}")
print(f"  TS IC: {mean_ts_ic:.4f}  p={p_ts:.4f}")

print(f"\nSection 5 — Backtest:")
print(f"  Ann. return: {ann_ls_ret*100:.2f}%")
print(f"  Sharpe:      {sharpe_ret:.3f}")
print(f"  Hit rate:    {hit_rate*100:.1f}%")
print(f"  p-value:     {p_ls:.4f}")

results = {
    'vol_control':  vol_control_results,
    'axis_corrs':   axis_corrs,
    'direction': {
        'best_axis':      best_axis,
        'r_composite':    float(r_composite),
        'p_composite':    float(p_composite),
    },
    'decay':        horizons_results,
    'cs_ts':        cs_ts_results,
    'backtest':     backtest_results,
}

with open(DATA_DIR / 'velocity_deepdive_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults: velocity_deepdive_results.json")
print(f"Plots:   {PLOT_DIR}")
print("Done.")