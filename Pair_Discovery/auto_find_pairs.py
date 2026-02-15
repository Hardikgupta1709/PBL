import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
import yfinance as yf
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

STOCK_UNIVERSE = {
    'Banking': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC'],
    'Tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'ORCL', 'CRM'],
    'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'TGT', 'COST', 'MCD', 'SBUX'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'HAL', 'PSX'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY'],
    'Industrials': ['BA', 'HON', 'UPS', 'CAT', 'MMM', 'GE', 'DE', 'LMT'],
    'Materials': ['LIN', 'APD', 'ECL', 'DD', 'NEM', 'FCX', 'VMC', 'MLM'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL'],
    'Telecom': ['T', 'VZ', 'TMUS', 'CHTR', 'CMCSA'],
    'Retail': ['HD', 'LOW', 'TJX', 'ROST', 'DG', 'DLTR']
}


def download_prices(ticker, start_date, end_date, max_retries=2):
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, 
                             progress=False, auto_adjust=True)
            
            if data.empty:
                return None
            
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    return data['Close'].iloc[:, 0].squeeze()
                else:
                    return data.iloc[:, 0].squeeze()
            else:
                if 'Close' in data.columns:
                    return data['Close'].squeeze()
                else:
                    return data.iloc[:, 0].squeeze()
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            continue
    
    return None


def quick_pair_score(y, x):
    data = pd.DataFrame({'Y': y, 'X': x}).dropna()
    
    if len(data) < 252:
        return 0, {}
    
    score = 0
    metrics = {}
    
    try:
        # 1. Cointegration (50 points)
        _, coint_pval, _ = coint(data['Y'], data['X'])
        metrics['coint_pval'] = coint_pval
        
        if coint_pval < 0.01:
            score += 50
        elif coint_pval < 0.05:
            score += 35
        elif coint_pval < 0.10:
            score += 15
        
        # 2. Correlation (20 points)
        corr, _ = pearsonr(data['Y'], data['X'])
        metrics['correlation'] = corr
        
        if abs(corr) > 0.80:
            score += 20
        elif abs(corr) > 0.60:
            score += 12
        elif abs(corr) > 0.40:
            score += 5
        
        # 3. Spread Stationarity (30 points)
        model = LinearRegression()
        model.fit(data['X'].values.reshape(-1, 1), data['Y'].values)
        hedge_ratio = model.coef_[0]
        spread = data['Y'] - hedge_ratio * data['X']
        
        adf_result = adfuller(spread, maxlag=1)
        adf_pval = adf_result[1]
        metrics['adf_pval'] = adf_pval
        metrics['hedge_ratio'] = hedge_ratio
        
        if adf_pval < 0.01:
            score += 30
        elif adf_pval < 0.05:
            score += 20
        elif adf_pval < 0.10:
            score += 10
        
        # 4. Half-life check (bonus/penalty)
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        spread_lag = spread_lag[spread_diff.index]
        
        model = LinearRegression()
        model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
        theta = model.coef_[0]
        
        if theta < 0:
            half_life = -np.log(2) / theta
            metrics['half_life'] = half_life
            
            # Ideal: 5-30 days
            if 5 <= half_life <= 30:
                score += 10  # Bonus
            elif half_life > 100:
                score -= 10  # Penalty for too slow
        else:
            metrics['half_life'] = 999
            score -= 15  # Penalty for no mean reversion
        
        metrics['score'] = score
        return score, metrics
        
    except Exception as e:
        return 0, {'error': str(e)}


def scan_sector(sector_name, stocks, start_date, end_date, min_score=50, max_pairs=10):

    print(f"\n{'='*80}")
    print(f"Scanning {sector_name} Sector ({len(stocks)} stocks)")
    print(f"{'='*80}")
    
    print(f"Downloading data")
    stock_data = {}
    
    for ticker in stocks:
        print(f"  {ticker}", end='', flush=True)
        data = download_prices(ticker, start_date, end_date)
        if data is not None and len(data) >= 252:
            stock_data[ticker] = data
            print("Ok")
        else:
            print("No")
    
    if len(stock_data) < 2:
        print(f" Insufficient data in {sector_name}")
        return []
    
    print(f"\nSuccessfully downloaded {len(stock_data)}/{len(stocks)} stocks")
    
    print(f"\n Testing pairs")
    pairs = list(combinations(stock_data.keys(), 2))
    print(f"   Total combinations: {len(pairs)}")
    
    results = []
    
    for i, (ticker_y, ticker_x) in enumerate(pairs, 1):
        print(f"   [{i}/{len(pairs)}] {ticker_y} vs {ticker_x}...", end='', flush=True)
        
        score, metrics = quick_pair_score(stock_data[ticker_y], stock_data[ticker_x])
        
        if score >= min_score:
            results.append({
                'Ticker_Y': ticker_y,
                'Ticker_X': ticker_x,
                'Sector': sector_name,
                'Score': round(score, 1),
                'Coint_P': round(metrics.get('coint_pval', 1), 4),
                'Correlation': round(metrics.get('correlation', 0), 3),
                'ADF_P': round(metrics.get('adf_pval', 1), 4),
                'Half_Life': round(metrics.get('half_life', 999), 1),
                'Hedge_Ratio': round(metrics.get('hedge_ratio', 0), 4)
            })
            print(f"  {score:.0f}")
        else:
            print(f" ❌ {score:.0f}")
    
    # Sort by score
    results = sorted(results, key=lambda x: x['Score'], reverse=True)[:max_pairs]
    
    if results:
        print(f"\n Found {len(results)} good pairs in {sector_name}")
    else:
        print(f"\n No pairs with score >= {min_score} in {sector_name}")
    
    return results


def scan_all_sectors(min_score=50, max_pairs_per_sector=3, 
                     start_date='2020-01-01', end_date='2024-01-01'):
    print("\n")
    print("AUTOMATED PAIR SCANNER")
    print("\n")
    print(f"\nConfiguration:")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Min Score: {min_score}/100")
    print(f"  Max Pairs per Sector: {max_pairs_per_sector}")
    print(f"  Total Sectors: {len(STOCK_UNIVERSE)}")
    
    all_results = []
    
    for sector_name, stocks in STOCK_UNIVERSE.items():
        sector_results = scan_sector(
            sector_name, stocks, start_date, end_date,
            min_score=min_score, max_pairs=max_pairs_per_sector
        )
        all_results.extend(sector_results)
    
    return all_results


def display_results(results):
    if not results:
        print("\ NO GOOD PAIRS FOUND!")
        print("\nTry:")
        print("  1. Lower min_score (try 40 instead of 50)")
        print("  2. Use different date range")
        print("  3. Add more stocks to universe")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values('Score', ascending=False)
    
    print("\n")
    print("\n")
    print("TOP PAIRS FOUND")
    print("="*20)
    
    print(f"\n Total Good Pairs Found: {len(df)}")
    print(f"   Sectors Represented: {df['Sector'].nunique()}")
    
    # Group by quality tiers
    excellent = df[df['Score'] >= 70]
    good = df[(df['Score'] >= 60) & (df['Score'] < 70)]
    acceptable = df[(df['Score'] >= 50) & (df['Score'] < 60)]
    
    if len(excellent) > 0:
        print(f"\n EXCELLENT Pairs (Score ≥ 70): {len(excellent)}")
        print(excellent[['Ticker_Y', 'Ticker_X', 'Sector', 'Score', 'Coint_P', 'Half_Life']].to_string(index=False))
    
    if len(good) > 0:
        print(f"\n GOOD Pairs (Score 60-69): {len(good)}")
        print(good[['Ticker_Y', 'Ticker_X', 'Sector', 'Score', 'Coint_P', 'Half_Life']].to_string(index=False))
    
    if len(acceptable) > 0:
        print(f"\n ACCEPTABLE Pairs (Score 50-59): {len(acceptable)}")
        print(acceptable[['Ticker_Y', 'Ticker_X', 'Sector', 'Score', 'Coint_P', 'Half_Life']].head(10).to_string(index=False))
    
    # Recommendations
    print("\n")
    print("\n")
    print("RECOMMENDATIONS")
    print("="*20)
    
    if len(excellent) > 0:
        top_pair = excellent.iloc[0]
        print(f"\n BEST PAIR: {top_pair['Ticker_Y']} vs {top_pair['Ticker_X']}")
        print(f"   Sector: {top_pair['Sector']}")
        print(f"   Score: {top_pair['Score']}/100")
        print(f"   Cointegration p-value: {top_pair['Coint_P']}")
        print(f"   Half-life: {top_pair['Half_Life']:.1f} days")
        
        print(f"\n Suggested Parameters:")
        print(f"   entry_z_normal = 2.0")
        print(f"   exit_z_normal = 0.5")
        print(f"   min_hold_days = {max(2, int(top_pair['Half_Life'] * 0.3))}")
        
        print(f"\n Next Steps:")
        print(f"   1. Edit test_ultra_conservative.py:")
        print(f"      TICKER_Y = '{top_pair['Ticker_Y']}'")
        print(f"      TICKER_X = '{top_pair['Ticker_X']}'")
        print(f"   2. Run: python test_ultra_conservative.py")
        
    elif len(good) > 0:
        top_pair = good.iloc[0]
        print(f"\n BEST AVAILABLE: {top_pair['Ticker_Y']} vs {top_pair['Ticker_X']}")
        print(f"   Score: {top_pair['Score']}/100 (Good, not excellent)")
        print(f"\nUse conservative parameters:")
        print(f"   entry_z_normal = 2.5")
        print(f"   exit_z_normal = 0.3")
        
    else:
        print("\n Only marginal pairs found. Consider:")
        print("   1. Different time period")
        print("   2. International markets")
        print("   3. Sector-specific ETFs")
    
    return df


def save_results(df, filename='auto_found_pairs.csv'):
    if df is not None and len(df) > 0:
        df.to_csv(filename, index=False)
        print(f"\n Results saved to: {filename}")
        return filename
    return None


if __name__ == "__main__":
    import sys
    min_score = 50
    max_pairs = 3
    
    if '--min-score' in sys.argv:
        idx = sys.argv.index('--min-score')
        min_score = int(sys.argv[idx + 1])
    
    if '--max-pairs' in sys.argv:
        idx = sys.argv.index('--max-pairs')
        max_pairs = int(sys.argv[idx + 1])
    
    print("\n")
    print("\n")
    print(" AUTOMATED PAIR FINDER")
    print("="*30)
    print("\Scanning Large Cap Stockts")
    
    results = scan_all_sectors(
        min_score=min_score,
        max_pairs_per_sector=max_pairs,
        start_date='2020-01-01',
        end_date='2024-01-01'
    )

    df = display_results(results)

    if df is not None:
        filename = save_results(df)
        
        print("\n")
        print("\n")
        print("SCAN COMPLETE")
        print("\n")
        print("\n")
        print(f"\n Found {len(df)} tradeable pairs")
        print(f" Full results in: {filename}")
        print(f"\n To test the best pair:")
        print(f"   1. Note the top Ticker_Y and Ticker_X")
        print(f"   2. Edit test_conservative.py with those tickers")
        print(f"   3. Run: python test_conservative.py")
    
    else:
        print("\n")
        print("\n")
        print("NO GOOD PAIRS FOUND")
        print("\n")
        print("\n")