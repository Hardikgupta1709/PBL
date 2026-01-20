"""
Pair Quality Diagnostic Tool
Checks if a pair is actually suitable for pairs trading
Run: python check_pair_quality.py
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


def download_data(ticker_y, ticker_x, market_ticker='SPY', 
                 start_date='2018-01-01', end_date='2024-01-01'):
    """Download stock data"""
    print(f"📥 Downloading {ticker_y}, {ticker_x}, {market_ticker}...")
    
    data_y = yf.download(ticker_y, start=start_date, end=end_date, progress=False, auto_adjust=True)
    data_x = yf.download(ticker_x, start=start_date, end=end_date, progress=False, auto_adjust=True)
    data_market = yf.download(market_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    def extract_close(df):
        if df.empty:
            raise ValueError("Downloaded data is empty")
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                return df['Close'].iloc[:, 0].squeeze()
            else:
                return df.iloc[:, 0].squeeze()
        else:
            if 'Close' in df.columns:
                return df['Close'].squeeze()
            else:
                return df.iloc[:, 0].squeeze()
    
    stock_y = extract_close(data_y)
    stock_x = extract_close(data_x)
    market = extract_close(data_market)
    
    return pd.Series(stock_y), pd.Series(stock_x), pd.Series(market)


def check_pair_quality(ticker_y, ticker_x, start_date='2018-01-01', end_date='2024-01-01'):
    """
    Comprehensive pair quality check
    Returns quality score (0-100) and diagnostics
    """
    print("="*80)
    print(f"PAIR QUALITY DIAGNOSTIC: {ticker_y} vs {ticker_x}")
    print("="*80)
    
    # Download data
    try:
        y, x, _ = download_data(ticker_y, ticker_x, 'SPY', start_date, end_date)
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return 0, {}
    
    data = pd.DataFrame({'Y': y, 'X': x}).dropna()
    
    if len(data) < 252:
        print(f"❌ Insufficient data: {len(data)} days")
        return 0, {}
    
    print(f"✅ Downloaded {len(data)} days of data")
    print(f"   Period: {data.index[0].date()} to {data.index[-1].date()}\n")
    
    results = {}
    score = 0
    
    # ========================================================================
    # TEST 1: COINTEGRATION (Most Important)
    # ========================================================================
    print("-"*80)
    print("TEST 1: Cointegration Analysis")
    print("-"*80)
    
    try:
        _, coint_pvalue, _ = coint(data['Y'], data['X'])
        results['cointegration_pvalue'] = coint_pvalue
        
        print(f"Cointegration p-value: {coint_pvalue:.4f}")
        
        if coint_pvalue < 0.01:
            print("✅ EXCELLENT: Very strong cointegration (p < 0.01)")
            coint_score = 50
        elif coint_pvalue < 0.05:
            print("✅ GOOD: Significant cointegration (p < 0.05)")
            coint_score = 35
        elif coint_pvalue < 0.10:
            print("⚠️ WEAK: Marginal cointegration (p < 0.10)")
            coint_score = 15
        else:
            print("❌ FAIL: No cointegration (p > 0.10)")
            print("   This pair is NOT suitable for pairs trading!")
            coint_score = 0
        
        score += coint_score
        results['cointegration_score'] = coint_score
        
    except Exception as e:
        print(f"❌ Error in cointegration test: {e}")
        results['cointegration_score'] = 0
    
    # ========================================================================
    # TEST 2: CORRELATION
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Correlation Analysis")
    print("-"*80)
    
    correlation, _ = pearsonr(data['Y'], data['X'])
    results['correlation'] = correlation
    
    print(f"Correlation: {correlation:.3f}")
    
    if abs(correlation) > 0.80:
        print("✅ EXCELLENT: Very high correlation")
        corr_score = 15
    elif abs(correlation) > 0.60:
        print("✅ GOOD: Strong correlation")
        corr_score = 10
    elif abs(correlation) > 0.40:
        print("⚠️ WEAK: Moderate correlation")
        corr_score = 5
    else:
        print("❌ FAIL: Low correlation")
        corr_score = 0
    
    score += corr_score
    results['correlation_score'] = corr_score
    
    # ========================================================================
    # TEST 3: SPREAD STATIONARITY
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Spread Stationarity (Mean Reversion)")
    print("-"*80)
    
    try:
        # Calculate hedge ratio
        model = LinearRegression()
        model.fit(data['X'].values.reshape(-1, 1), data['Y'].values)
        hedge_ratio = model.coef_[0]
        results['hedge_ratio'] = hedge_ratio
        
        # Calculate spread
        spread = data['Y'] - hedge_ratio * data['X']
        results['spread'] = spread
        
        # ADF test on spread
        adf_result = adfuller(spread, maxlag=1)
        adf_pvalue = adf_result[1]
        results['adf_pvalue'] = adf_pvalue
        
        print(f"Hedge Ratio: {hedge_ratio:.4f}")
        print(f"ADF p-value: {adf_pvalue:.4f}")
        
        if adf_pvalue < 0.01:
            print("✅ EXCELLENT: Spread is very stationary (p < 0.01)")
            adf_score = 20
        elif adf_pvalue < 0.05:
            print("✅ GOOD: Spread is stationary (p < 0.05)")
            adf_score = 15
        elif adf_pvalue < 0.10:
            print("⚠️ WEAK: Spread shows some stationarity (p < 0.10)")
            adf_score = 8
        else:
            print("❌ FAIL: Spread is not stationary (p > 0.10)")
            print("   Mean reversion is unlikely!")
            adf_score = 0
        
        score += adf_score
        results['adf_score'] = adf_score
        
    except Exception as e:
        print(f"❌ Error in ADF test: {e}")
        results['adf_score'] = 0
    
    # ========================================================================
    # TEST 4: HALF-LIFE OF MEAN REVERSION
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Mean Reversion Speed (Half-Life)")
    print("-"*80)
    
    try:
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        spread_lag = spread_lag[spread_diff.index]
        
        model = LinearRegression()
        model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
        theta = model.coef_[0]
        
        if theta < 0:
            half_life = -np.log(2) / theta
            results['half_life'] = half_life
            
            print(f"Half-Life: {half_life:.1f} days")
            
            if 5 <= half_life <= 30:
                print("✅ EXCELLENT: Ideal mean reversion speed")
                print("   Spread reverts in 1-6 weeks")
                hl_score = 15
            elif 3 <= half_life <= 60:
                print("✅ GOOD: Acceptable mean reversion speed")
                hl_score = 10
            elif half_life < 3:
                print("⚠️ TOO FAST: Spread reverts too quickly")
                print("   May be mostly noise, hard to capture")
                hl_score = 5
            else:
                print("⚠️ TOO SLOW: Takes too long to revert")
                print("   May require long holding periods")
                hl_score = 5
        else:
            print("❌ FAIL: No mean reversion detected!")
            print("   Spread is trending, not reverting")
            half_life = 999
            hl_score = 0
            results['half_life'] = half_life
        
        score += hl_score
        results['halflife_score'] = hl_score
        
    except Exception as e:
        print(f"❌ Error in half-life calculation: {e}")
        results['halflife_score'] = 0
        results['half_life'] = 999
    
    # ========================================================================
    # TEST 5: CORRELATION STABILITY
    # ========================================================================
    print("\n" + "-"*80)
    print("TEST 5: Correlation Stability Over Time")
    print("-"*80)
    
    rolling_corr = data['Y'].rolling(60).corr(data['X'])
    corr_std = rolling_corr.std()
    corr_min = rolling_corr.min()
    
    results['correlation_std'] = corr_std
    results['correlation_min'] = corr_min
    
    print(f"Correlation Std Dev: {corr_std:.3f}")
    print(f"Min Rolling Correlation: {corr_min:.3f}")
    
    if corr_std < 0.10 and corr_min > 0.50:
        print("✅ EXCELLENT: Very stable correlation")
        stab_score = 10
    elif corr_std < 0.15 and corr_min > 0.30:
        print("✅ GOOD: Reasonably stable")
        stab_score = 5
    else:
        print("⚠️ UNSTABLE: Correlation changes significantly")
        print("   Relationship may break down unpredictably")
        stab_score = 0
    
    score += stab_score
    results['stability_score'] = stab_score
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    print(f"\n📊 Component Scores:")
    print(f"  Cointegration:.......... {results.get('cointegration_score', 0)}/50")
    print(f"  Correlation:............ {results.get('correlation_score', 0)}/15")
    print(f"  Spread Stationarity:.... {results.get('adf_score', 0)}/20")
    print(f"  Mean Reversion Speed:... {results.get('halflife_score', 0)}/15")
    print(f"  Correlation Stability:.. {results.get('stability_score', 0)}/10")
    print(f"\n  OVERALL QUALITY SCORE:.. {score}/110")
    
    # Normalize to 0-100
    normalized_score = (score / 110) * 100
    
    print(f"\n🎯 Normalized Score: {normalized_score:.1f}/100")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if normalized_score >= 70:
        print("\n✅ EXCELLENT PAIR - Highly Suitable for Pairs Trading")
        print("\nThis pair shows:")
        print("  • Strong cointegration")
        print("  • Stable relationship")
        print("  • Good mean reversion")
        print("\nProceed with strategy development!")
        
    elif normalized_score >= 50:
        print("\n⚠️ ACCEPTABLE PAIR - Can Work But Has Limitations")
        print("\nConsiderations:")
        print("  • Monitor correlation stability")
        print("  • Use conservative parameters")
        print("  • Expect moderate performance")
        print("\nProceed with caution.")
        
    elif normalized_score >= 30:
        print("\n⚠️ MARGINAL PAIR - Not Ideal for Pairs Trading")
        print("\nIssues detected:")
        if results.get('cointegration_score', 0) < 15:
            print("  • Weak or no cointegration")
        if results.get('adf_score', 0) < 8:
            print("  • Spread not stationary")
        if results.get('halflife_score', 0) < 5:
            print("  • Poor mean reversion characteristics")
        print("\nRecommendation: Try different pairs")
        
    else:
        print("\n❌ POOR PAIR - NOT SUITABLE for Pairs Trading")
        print("\nCritical failures:")
        if results.get('cointegration_pvalue', 1) > 0.10:
            print("  • No cointegration detected")
        if results.get('adf_pvalue', 1) > 0.10:
            print("  • Spread does not mean-revert")
        if results.get('half_life', 999) > 60:
            print("  • Mean reversion too slow")
        print("\n⚠️ DO NOT TRADE THIS PAIR")
        print("   Strategy will likely fail validation")
    
    # Suggest alternative pairs
    if normalized_score < 70:
        print("\n💡 Suggested Alternative Pairs:")
        print("   US Market:")
        print("     • GLD vs GDX (Gold - typically 85+ score)")
        print("     • XLE vs XOM (Energy - typically 80+ score)")
        print("     • JPM vs BAC (Banks - typically 75+ score)")
        print("   ")
        print("   Try: python check_pair_quality.py --pair GLD GDX")
    
    results['overall_score'] = normalized_score
    
    return normalized_score, results


def suggest_parameters(score, results):
    """
    Suggest strategy parameters based on pair quality
    """
    print("\n" + "="*80)
    print("SUGGESTED STRATEGY PARAMETERS")
    print("="*80)
    
    half_life = results.get('half_life', 20)
    corr = results.get('correlation', 0)
    
    if score >= 70:
        print("\n✅ High Quality Pair - Use Standard Parameters:")
        print(f"   entry_z_normal = 2.0")
        print(f"   exit_z_normal = 0.5")
        print(f"   entry_z_volatile = 2.5")
        print(f"   min_hold_days = {max(2, int(half_life * 0.3))}")
        
    elif score >= 50:
        print("\n⚠️ Moderate Quality Pair - Use Conservative Parameters:")
        print(f"   entry_z_normal = 2.5")
        print(f"   exit_z_normal = 0.3")
        print(f"   entry_z_volatile = 3.0")
        print(f"   min_hold_days = {max(3, int(half_life * 0.4))}")
        
    else:
        print("\n❌ Low Quality Pair - Even Aggressive Parameters Won't Help")
        print("   The fundamental pair relationship is too weak.")
        print("   No parameter tuning will make this work reliably.")
        print("\n   👉 RECOMMENDATION: Choose a different pair")


if __name__ == "__main__":
    import sys
    
    # Default pair
    ticker_y = 'PEP'
    ticker_x = 'KO'
    
    # Check for command line arguments
    if len(sys.argv) >= 3:
        if sys.argv[1] == '--pair':
            ticker_y = sys.argv[2]
            ticker_x = sys.argv[3]
    
    print("\n" + "="*80)
    print("PAIRS TRADING - PAIR QUALITY CHECKER")
    print("="*80)
    print("\nThis tool diagnoses if a stock pair is suitable for pairs trading.")
    print("It checks cointegration, stationarity, and mean reversion.\n")
    
    # Run check
    score, results = check_pair_quality(ticker_y, ticker_x, '2018-01-01', '2024-01-01')
    
    # Suggest parameters
    if score > 0:
        suggest_parameters(score, results)
    
    # Save results
    from datetime import datetime
    filename = f'pair_quality_{ticker_y}_{ticker_x}_{datetime.now().strftime("%Y%m%d")}.txt'
    
    with open(filename, 'w') as f:
        f.write(f"PAIR QUALITY ANALYSIS: {ticker_y} vs {ticker_x}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Score: {score:.1f}/100\n\n")
        f.write("Component Scores:\n")
        for key, value in results.items():
            if 'score' in key:
                f.write(f"  {key}: {value}\n")
        f.write("\nMetrics:\n")
        for key, value in results.items():
            if 'score' not in key and key != 'spread':
                f.write(f"  {key}: {value}\n")
    
    print(f"\n📄 Results saved to: {filename}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if score >= 50:
        print("\n1. Use suggested parameters above")
        print("2. Run: python test_ultra_conservative.py")
        print("3. Monitor Monte Carlo p-value")
    else:
        print("\n1. Test a different pair:")
        print("   python check_pair_quality.py --pair GLD GDX")
        print("2. Or try pairs from different sectors")
        print("3. Avoid pairs with score < 50")