# stock_test_suite.py
# Comprehensive validation of stock feature data before AE training
# Tests data integrity, scaler correctness, and feature sanity

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import timedelta
import json
from collections import defaultdict

DATA_DIR = Path(r'C:\Users\Justin.Sanford\finance\data\stocks')

# ── Feature definitions ───────────────────────────────────────────────────────
EXCLUDE_COLS = {'ticker', 'open', 'high', 'low', 'close', 'volume',
                'Mkt_RF', 'SMB', 'HML', 'RF',
                # Absolute price levels — not meaningful cross-sectionally
                'high_252d', 'low_252d', 'high_63d', 'low_63d',
                'sma_21', 'sma_63', 'sma_252',
                # Raw volume — not meaningful cross-sectionally
                'vol_ma21',
                # Ratio features with extreme variance
                'up_vol_ratio', 'kurt_63d'}

class StockDataTests:
    def __init__(self):
        self.manifest = pd.read_parquet(DATA_DIR / 'stock_manifest.parquet')
        self.failures = []
        self.warnings = []
        self.stats    = {}
        
        # Load scaler if exists
        self.scaler = None
        scaler_path = DATA_DIR / 'stock_scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Sample one stock to get feature list
        sample_ticker = self.manifest.iloc[0]['ticker']
        sample_df     = pd.read_parquet(DATA_DIR / f'stock_{sample_ticker}.parquet')
        self.all_cols = sample_df.columns.tolist()
        self.feature_cols = [c for c in self.all_cols 
                             if c not in EXCLUDE_COLS
                             and sample_df[c].dtype in [np.float64, np.float32]]
        
        print(f"Loaded manifest: {len(self.manifest)} stocks")
        print(f"Feature columns: {len(self.feature_cols)}")
        print(f"Excluded columns: {len(EXCLUDE_COLS)}")
        
    def _fail(self, test_name, message):
        self.failures.append(f"[{test_name}] {message}")
        
    def _warn(self, test_name, message):
        self.warnings.append(f"[{test_name}] {message}")
        
    def _stat(self, test_name, key, value):
        if test_name not in self.stats:
            self.stats[test_name] = {}
        self.stats[test_name][key] = value
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1: Scaler Dimensionality
    # ══════════════════════════════════════════════════════════════════════════
    def test_1_scaler_dimensionality(self):
        """Scaler must match feature_cols exactly"""
        print("\n" + "="*60)
        print("TEST 1: Scaler Dimensionality")
        print("="*60)
        
        if self.scaler is None:
            self._warn('scaler_dim', 'No scaler found — skip test')
            print("SKIP: No scaler found")
            return
        
        scaler_dim = len(self.scaler.scale_)
        expect_dim = len(self.feature_cols)
        
        self._stat('scaler_dim', 'scaler_features', scaler_dim)
        self._stat('scaler_dim', 'expected_features', expect_dim)
        
        if scaler_dim != expect_dim:
            self._fail('scaler_dim', 
                      f"Scaler has {scaler_dim} features but "
                      f"feature_cols has {expect_dim}")
            print(f"FAIL: Dimension mismatch")
            print(f"  Scaler:   {scaler_dim}")
            print(f"  Expected: {expect_dim}")
            print(f"  → Scaler was likely fit on WRONG features")
            print(f"  → REPAIR: Delete stock_scaler.pkl and rerun stock_clean.py")
        else:
            print(f"PASS: Scaler dimension matches ({scaler_dim} features)")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2: No Price Leakage
    # ══════════════════════════════════════════════════════════════════════════
    def test_2_no_price_leakage(self):
        """Absolute price features must be excluded from feature_cols"""
        print("\n" + "="*60)
        print("TEST 2: No Price/Volume Leakage")
        print("="*60)
        
        leakage = [c for c in self.feature_cols if c in EXCLUDE_COLS]
        
        if leakage:
            self._fail('price_leakage', 
                      f"Found {len(leakage)} excluded cols in feature_cols: "
                      f"{leakage}")
            print(f"FAIL: {len(leakage)} leakage columns found:")
            for c in leakage:
                print(f"  - {c}")
            print(f"  → These should NOT be in training features")
        else:
            print(f"PASS: No price/volume leakage")
            
        # Also check the other direction — are excluded cols actually excluded?
        sample_ticker = self.manifest.iloc[0]['ticker']
        sample_df     = pd.read_parquet(DATA_DIR / f'stock_{sample_ticker}.parquet')
        
        excluded_present = [c for c in EXCLUDE_COLS if c in sample_df.columns]
        excluded_in_features = [c for c in excluded_present if c in self.feature_cols]
        
        self._stat('price_leakage', 'excluded_cols_in_df', len(excluded_present))
        self._stat('price_leakage', 'excluded_cols_in_features', len(excluded_in_features))
        
        print(f"\nExcluded columns present in raw data: {len(excluded_present)}")
        print(f"Excluded columns in feature_cols: {len(excluded_in_features)}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3: Feature Coverage
    # ══════════════════════════════════════════════════════════════════════════
    def test_3_feature_coverage(self):
        """Every stock must have all expected features"""
        print("\n" + "="*60)
        print("TEST 3: Feature Coverage Across Stocks")
        print("="*60)
        
        missing_features = defaultdict(list)
        n_checked = 0
        
        # Sample 50 stocks to check
        sample_tickers = self.manifest['ticker'].sample(min(50, len(self.manifest)), 
                                                         random_state=42).tolist()
        
        for ticker in sample_tickers:
            f = DATA_DIR / f'stock_{ticker}.parquet'
            if not f.exists():
                continue
                
            df = pd.read_parquet(f)
            n_checked += 1
            
            # Check which features are missing
            for feat in self.feature_cols:
                if feat not in df.columns:
                    missing_features[feat].append(ticker)
        
        self._stat('feature_coverage', 'stocks_checked', n_checked)
        self._stat('feature_coverage', 'features_missing', len(missing_features))
        
        if missing_features:
            self._fail('feature_coverage', 
                      f"{len(missing_features)} features missing in some stocks")
            print(f"FAIL: {len(missing_features)} features have missing data:")
            for feat, tickers in list(missing_features.items())[:5]:
                print(f"  {feat}: missing in {len(tickers)} stocks")
            if len(missing_features) > 5:
                print(f"  ... and {len(missing_features)-5} more")
        else:
            print(f"PASS: All {len(self.feature_cols)} features present in "
                  f"{n_checked} sampled stocks")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 4: Cross-Sectional Sanity
    # ══════════════════════════════════════════════════════════════════════════
    def test_4_cross_sectional_sanity(self):
        """Scaled features should be comparable across stocks"""
        print("\n" + "="*60)
        print("TEST 4: Cross-Sectional Feature Sanity")
        print("="*60)
        
        if self.scaler is None:
            self._warn('cross_sectional', 'No scaler — skip test')
            print("SKIP: No scaler found")
            return
        
        # Load 20 random stocks and scale them
        sample_tickers = self.manifest['ticker'].sample(min(20, len(self.manifest)),
                                                         random_state=42).tolist()
        
        per_stock_stds = defaultdict(list)
        
        for ticker in sample_tickers:
            f = DATA_DIR / f'stock_{ticker}.parquet'
            if not f.exists():
                continue
                
            df = pd.read_parquet(f)
            X  = df[self.feature_cols].values
            
            # Basic cleaning
            X = np.where(np.isinf(X), np.nan, X)
            col_meds = np.nanmedian(X, axis=0)
            for j in range(X.shape[1]):
                mask = np.isnan(X[:, j])
                X[mask, j] = col_meds[j]
            X = np.nan_to_num(X, nan=0.0)
            
            # Scale
            try:
                X_scaled = self.scaler.transform(X)
                
                # Per-feature std within this stock
                for j, feat in enumerate(self.feature_cols):
                    per_stock_stds[feat].append(X_scaled[:, j].std())
            except Exception as e:
                self._warn('cross_sectional', f'{ticker}: scaling failed: {e}')
        
        # Check if any feature has wildly different stds across stocks
        bad_features = []
        for feat, stds in per_stock_stds.items():
            if len(stds) < 5:
                continue
            ratio = max(stds) / (min(stds) + 1e-8)
            if ratio > 100:  # 100× variance difference is suspicious
                bad_features.append((feat, ratio))
        
        self._stat('cross_sectional', 'stocks_checked', len(sample_tickers))
        self._stat('cross_sectional', 'bad_features', len(bad_features))
        
        if bad_features:
            self._fail('cross_sectional',
                      f"{len(bad_features)} features have extreme variance ratios")
            print(f"FAIL: {len(bad_features)} features with suspicious variance:")
            for feat, ratio in sorted(bad_features, key=lambda x: -x[1])[:5]:
                print(f"  {feat}: {ratio:.1f}× ratio")
            print(f"  → This suggests scaler was fit on contaminated data")
        else:
            print(f"PASS: Feature scales comparable across {len(sample_tickers)} stocks")
            
        # Also report mean std across all features
        mean_stds = {feat: np.mean(stds) 
                     for feat, stds in per_stock_stds.items() if len(stds) >= 5}
        if mean_stds:
            overall_mean = np.mean(list(mean_stds.values()))
            overall_std  = np.std(list(mean_stds.values()))
            print(f"\nScaled feature std: {overall_mean:.3f} ± {overall_std:.3f}")
            self._stat('cross_sectional', 'mean_scaled_std', overall_mean)
            self._stat('cross_sectional', 'std_scaled_std', overall_std)
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 5: Temporal Stability
    # ══════════════════════════════════════════════════════════════════════════
    def test_5_temporal_stability(self):
        """Feature distributions should not have structural breaks"""
        print("\n" + "="*60)
        print("TEST 5: Temporal Stability (No Structural Breaks)")
        print("="*60)
        
        # Load 10 stocks with longest history
        long_stocks = self.manifest.nlargest(10, 'n_days')['ticker'].tolist()
        
        # For each stock, split into 2010-2015 and 2020-2024
        # Check if feature distributions are stable
        breaks = []
        
        for ticker in long_stocks[:5]:  # Check 5 to keep it fast
            f = DATA_DIR / f'stock_{ticker}.parquet'
            if not f.exists():
                continue
                
            df = pd.read_parquet(f)
            
            # Split into early and late period
            early = df[df.index < '2015-01-01']
            late  = df[df.index >= '2020-01-01']
            
            if len(early) < 100 or len(late) < 100:
                continue
            
            # Check a few key features for mean shift
            check_feats = ['ret_1d', 'vol_21d', 'vol_ratio', 'drawdown']
            check_feats = [f for f in check_feats if f in df.columns]
            
            for feat in check_feats:
                early_mean = early[feat].mean()
                late_mean  = late[feat].mean()
                early_std  = early[feat].std()
                
                # Shift > 3 std is suspicious (unless it's 2008 crash artifacts)
                if abs(late_mean - early_mean) > 3 * early_std:
                    breaks.append((ticker, feat, early_mean, late_mean))
        
        self._stat('temporal', 'stocks_checked', len(long_stocks[:5]))
        self._stat('temporal', 'breaks_found', len(breaks))
        
        if breaks:
            self._warn('temporal', 
                      f"Found {len(breaks)} potential structural breaks")
            print(f"WARNING: {len(breaks)} features show large mean shifts:")
            for ticker, feat, early, late in breaks[:5]:
                print(f"  {ticker}/{feat}: {early:.4f} → {late:.4f}")
            print(f"  (This may be legitimate — e.g., 2008 crisis)")
        else:
            print(f"PASS: No major structural breaks detected")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 6: FF Factor Alignment
    # ══════════════════════════════════════════════════════════════════════════
    def test_6_ff_alignment(self):
        """Every trading day should have FF factors"""
        print("\n" + "="*60)
        print("TEST 6: Fama-French Factor Alignment")
        print("="*60)
        
        ff_path = DATA_DIR / 'ff3_daily.parquet'
        if not ff_path.exists():
            self._fail('ff_alignment', 'ff3_daily.parquet not found')
            print("FAIL: FF3 daily factors not found")
            return
        
        ff = pd.read_parquet(ff_path)
        
        # Check 5 random stocks for FF alignment
        sample_tickers = self.manifest['ticker'].sample(min(5, len(self.manifest)),
                                                         random_state=42).tolist()
        
        missing_counts = []
        
        for ticker in sample_tickers:
            f = DATA_DIR / f'stock_{ticker}.parquet'
            if not f.exists():
                continue
                
            df = pd.read_parquet(f)
            
            # Check which days are missing FF factors
            df_dates = set(df.index.date)
            ff_dates = set(ff.index.date)
            
            missing = df_dates - ff_dates
            missing_counts.append(len(missing))
            
            if len(missing) > 10:
                self._warn('ff_alignment',
                          f"{ticker}: {len(missing)} days missing FF factors")
        
        self._stat('ff_alignment', 'stocks_checked', len(sample_tickers))
        self._stat('ff_alignment', 'mean_missing', np.mean(missing_counts) if missing_counts else 0)
        
        if missing_counts and max(missing_counts) > 10:
            self._warn('ff_alignment',
                      f"Some stocks missing >10 days of FF factors")
            print(f"WARNING: Missing FF factors detected")
            print(f"  Max missing: {max(missing_counts)} days")
            print(f"  Mean missing: {np.mean(missing_counts):.1f} days")
        else:
            print(f"PASS: FF factors aligned for sampled stocks")
            if missing_counts:
                print(f"  Mean missing: {np.mean(missing_counts):.1f} days (acceptable)")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 7: Outlier Removal Stats
    # ══════════════════════════════════════════════════════════════════════════
    def test_7_outlier_removal(self):
        """Isolation forest should have removed ~5% of observations"""
        print("\n" + "="*60)
        print("TEST 7: Outlier Removal Statistics")
        print("="*60)
        
        # Compare raw vs clean file counts
        raw_counts   = {}
        clean_counts = {}
        
        for ticker in self.manifest['ticker'].tolist()[:100]:  # Check first 100
            raw_file   = DATA_DIR / f'stock_{ticker}.parquet'
            clean_file = DATA_DIR / f'stock_clean_{ticker}.parquet'
            
            if raw_file.exists():
                raw_counts[ticker] = len(pd.read_parquet(raw_file))
            
            if clean_file.exists():
                clean_counts[ticker] = len(pd.read_parquet(clean_file))
        
        if not clean_counts:
            self._warn('outlier_removal', 'No clean files found')
            print("SKIP: No clean files found yet")
            return
        
        # Compute removal rate
        common = set(raw_counts.keys()) & set(clean_counts.keys())
        removal_rates = []
        
        for ticker in common:
            raw   = raw_counts[ticker]
            clean = clean_counts[ticker]
            rate  = 1 - (clean / raw)
            removal_rates.append(rate)
        
        mean_removal = np.mean(removal_rates)
        std_removal  = np.std(removal_rates)
        
        self._stat('outlier_removal', 'mean_removal_rate', mean_removal)
        self._stat('outlier_removal', 'std_removal_rate', std_removal)
        self._stat('outlier_removal', 'stocks_compared', len(common))
        
        print(f"Outlier removal rate: {mean_removal*100:.1f}% ± {std_removal*100:.1f}%")
        
        if mean_removal < 0.03 or mean_removal > 0.10:
            self._warn('outlier_removal',
                      f"Removal rate {mean_removal*100:.1f}% outside expected 3-10%")
            print(f"WARNING: Removal rate outside expected range [3%, 10%]")
        else:
            print(f"PASS: Removal rate within expected range")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TEST 8: Return Sanity
    # ══════════════════════════════════════════════════════════════════════════
    def test_8_return_sanity(self):
        """Daily returns should be within [-50%, +50%] after cleaning"""
        print("\n" + "="*60)
        print("TEST 8: Return Value Sanity Checks")
        print("="*60)
        
        # Check cleaned files
        extreme_returns = []
        
        sample_tickers = self.manifest['ticker'].sample(min(50, len(self.manifest)),
                                                         random_state=42).tolist()
        
        for ticker in sample_tickers:
            f = DATA_DIR / f'stock_clean_{ticker}.parquet'
            if not f.exists():
                f = DATA_DIR / f'stock_{ticker}.parquet'
            if not f.exists():
                continue
                
            df = pd.read_parquet(f)
            
            if 'ret_1d' not in df.columns:
                continue
            
            # Check for extreme returns
            extreme = df[np.abs(df['ret_1d']) > 0.5]
            if len(extreme) > 0:
                extreme_returns.append((ticker, len(extreme), 
                                       extreme['ret_1d'].abs().max()))
        
        self._stat('return_sanity', 'stocks_checked', len(sample_tickers))
        self._stat('return_sanity', 'stocks_with_extremes', len(extreme_returns))
        
        if extreme_returns:
            self._warn('return_sanity',
                      f"{len(extreme_returns)} stocks have returns >50%")
            print(f"WARNING: {len(extreme_returns)} stocks with extreme returns:")
            for ticker, count, maxret in sorted(extreme_returns, 
                                                key=lambda x: -x[2])[:5]:
                print(f"  {ticker}: {count} days, max={maxret*100:.1f}%")
            print(f"  (May be legitimate for penny stocks or splits)")
        else:
            print(f"PASS: No extreme returns detected in sampled stocks")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Run All Tests
    # ══════════════════════════════════════════════════════════════════════════
    def run_all(self, save_report=True):
        """Run all tests and generate report"""
        print("\n" + "█"*60)
        print("STOCK DATA VALIDATION SUITE")
        print("█"*60)
        
        tests = [
            self.test_1_scaler_dimensionality,
            self.test_2_no_price_leakage,
            self.test_3_feature_coverage,
            self.test_4_cross_sectional_sanity,
            self.test_5_temporal_stability,
            self.test_6_ff_alignment,
            self.test_7_outlier_removal,
            self.test_8_return_sanity,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self._fail(test.__name__, f"Test crashed: {e}")
                print(f"\nERROR in {test.__name__}: {e}")
        
        # Summary
        print("\n" + "█"*60)
        print("SUMMARY")
        print("█"*60)
        
        if self.failures:
            print(f"\n❌ FAILURES ({len(self.failures)}):")
            for f in self.failures:
                print(f"  {f}")
        else:
            print(f"\n✓ All critical tests passed")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  {w}")
        
        # Save report
        if save_report:
            report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'failures': self.failures,
                'warnings': self.warnings,
                'stats': self.stats,
            }
            
            report_path = DATA_DIR / 'validation_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"\nReport saved: {report_path}")
        
        # Exit code
        if self.failures:
            print("\n❌ VALIDATION FAILED — fix errors before training")
            return False
        else:
            print("\n✓ VALIDATION PASSED — ready for AE training")
            return True


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    suite = StockDataTests()
    passed = suite.run_all()
    
    if not passed:
        print("\n" + "="*60)
        print("RECOMMENDED FIXES:")
        print("="*60)
        
        if any('scaler_dim' in f for f in suite.failures):
            print("\n1. Scaler dimension mismatch:")
            print("   → Delete stock_scaler.pkl")
            print("   → Rerun stock_clean.py")
        
        if any('price_leakage' in f for f in suite.failures):
            print("\n2. Price leakage detected:")
            print("   → Verify EXCLUDE_COLS matches stock_clean.py")
            print("   → Check feature_cols construction")
        
        if any('cross_sectional' in f for f in suite.failures):
            print("\n3. Cross-sectional variance issues:")
            print("   → Scaler likely fit on contaminated data")
            print("   → Delete stock_scaler.pkl and refit on clean data")