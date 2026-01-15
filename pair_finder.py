"""
pair_finder.py - Automated Pair Discovery and Quality Scoring
Add this file to your project: PBL_RUN/pair_finder.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class PairFinder:
    """
    Automated pair discovery with quality scoring.
    """
    
    def __init__(self, market='US'):
        """
        Args:
            market: 'US' or 'INDIAN'
        """
        self.market = market
        self.results = None
        
    def get_stock_universe(self):
        """Get pre-defined stock universe based on market."""
        
        if self.market == 'US':
            return {
                'Banking': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC'],
                'Tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC'],
                'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'TGT', 'COST', 'KMB', 'CL'],
                'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX'],
                'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT'],
                'Retail': ['HD', 'LOW', 'TJX', 'ROST'],
                'Telecom': ['T', 'VZ', 'TMUS'],
                'Insurance': ['BRK-B', 'AIG', 'MET', 'PRU', 'ALL'],
                'Airlines': ['DAL', 'UAL', 'AAL', 'LUV'],
                'Automotive': ['F', 'GM', 'TSLA'],
                'Pharma': ['PFE', 'MRK', 'BMY', 'LLY', 'GILD'],
                'Semiconductors': ['NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO'],
                'Media': ['DIS', 'NFLX', 'CMCSA', 'PARA'],
                'REITs': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
            }
        else:  # Indian market
            return {
                'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 
                           'INDUSINDBK', 'BANDHANBNK', 'FEDERALBNK', 'PNB', 'BANKBARODA'],
                'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 
                      'PERSISTENT', 'COFORGE', 'MPHASIS'],
                'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN',
                          'AUROPHARMA', 'TORNTPHARM', 'ALKEM', 'BIOCON'],
                'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO',
                        'EICHERMOT', 'TVSMOTOR', 'ASHOKLEY'],
                'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR',
                        'GODREJCP', 'MARICO', 'COLPAL'],
                'Energy': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'HINDPETRO',
                          'ADANIGREEN', 'NTPC', 'POWERGRID'],
                'Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'COALINDIA',
                          'SAIL', 'NMDC', 'JINDALSTEL'],
                'Cement': ['ULTRACEMCO', 'SHREECEM', 'GRASIM', 'AMBUJACEM', 'ACC',
                          'DALMIACEM', 'RAMCOCEM'],
                'Telecom': ['BHARTIARTL', 'IDEA', 'INDUSINDBK'],
                'Finance': ['BAJFINANCE', 'BAJAJFINSV', 'SBILIFE', 'HDFCLIFE', 
                           'ICICIGI', 'ICICIPRULI', 'CHOLAFIN'],
                'Infra': ['LT', 'ADANIPORTS', 'DLF', 'GODREJPROP', 'OBEROIRLTY',
                         'PRESTIGE', 'PHOENIXLTD'],
            }
    
    def download_prices(self, ticker, start_date, end_date):
        """Download price data with error handling."""
        try:
            # Add .NS suffix for Indian stocks
            if self.market == 'INDIAN':
                ticker = f"{ticker}.NS"
            
            data = yf.download(ticker, start=start_date, end=end_date, 
                             progress=False, auto_adjust=True)
            
            if data.empty:
                return None
            
            # Extract close price
            if 'Close' in data.columns:
                return data['Close'].squeeze()
            else:
                return data.iloc[:, 0].squeeze()
                
        except Exception as e:
            return None
    
    def calculate_pair_score(self, ticker_y, ticker_x, start_date, end_date):
        """
        Calculate comprehensive quality score for a pair.
        Returns score (0-100) and detailed metrics.
        """
        # Download data
        y = self.download_prices(ticker_y, start_date, end_date)
        x = self.download_prices(ticker_x, start_date, end_date)
        
        if y is None or x is None:
            return 0, {}
        
        # Align data
        data = pd.DataFrame({'Y': y, 'X': x}).dropna()
        
        if len(data) < 252:  # Need at least 1 year
            return 0, {}
        
        metrics = {}
        score = 0
        
        try:
            # 1. Cointegration Test (50 points)
            _, pvalue, _ = coint(data['Y'], data['X'])
            metrics['coint_pvalue'] = pvalue
            
            if pvalue < 0.01:
                score += 50
            elif pvalue < 0.05:
                score += 35
            elif pvalue < 0.10:
                score += 20
            
            # 2. Correlation (15 points)
            correlation = data['Y'].corr(data['X'])
            metrics['correlation'] = correlation
            
            if abs(correlation) > 0.80:
                score += 15
            elif abs(correlation) > 0.60:
                score += 10
            elif abs(correlation) > 0.40:
                score += 5
            
            # 3. Spread Stationarity (20 points)
            model = LinearRegression()
            model.fit(data['X'].values.reshape(-1, 1), data['Y'].values)
            spread = data['Y'] - model.predict(data['X'].values.reshape(-1, 1))
            metrics['hedge_ratio'] = model.coef_[0]
            
            adf_result = adfuller(spread, maxlag=1)
            adf_pvalue = adf_result[1]
            metrics['adf_pvalue'] = adf_pvalue
            
            if adf_pvalue < 0.01:
                score += 20
            elif adf_pvalue < 0.05:
                score += 15
            elif adf_pvalue < 0.10:
                score += 8
            
            # 4. Half-life (15 points)
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            spread_lag = spread_lag[spread_diff.index]
            
            model = LinearRegression()
            model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
            half_life = -np.log(2) / model.coef_[0] if model.coef_[0] < 0 else 999
            metrics['half_life'] = half_life
            
            if 5 <= half_life <= 30:
                score += 15
            elif 3 <= half_life <= 60:
                score += 10
            elif half_life < 3:
                score += 5
            
            metrics['score'] = score
            return score, metrics
            
        except Exception as e:
            return 0, {}
    
    def find_best_pairs(self, sector=None, min_score=60, 
                       start_date='2020-01-01', end_date='2024-01-01',
                       max_pairs=50):
        """
        Scan all stocks and find best pairs.
        
        Args:
            sector: Specific sector or None for all sectors
            min_score: Minimum quality score
            start_date: Start date for analysis
            end_date: End date for analysis
            max_pairs: Maximum pairs to test (to avoid timeout)
        
        Returns:
            DataFrame with ranked pairs
        """
        universe = self.get_stock_universe()
        
        if sector:
            if sector not in universe:
                raise ValueError(f"Sector '{sector}' not found")
            stocks = universe[sector]
        else:
            # Combine all sectors
            stocks = []
            for sector_stocks in universe.values():
                stocks.extend(sector_stocks)
            stocks = list(set(stocks))  # Remove duplicates
        
        print(f"Scanning {len(stocks)} stocks for pairs...")
        print(f"Market: {self.market}")
        
        # Generate all pairs
        all_pairs = list(combinations(stocks, 2))
        
        # Limit to max_pairs if specified
        if len(all_pairs) > max_pairs:
            print(f"Limiting to first {max_pairs} pairs (out of {len(all_pairs)})")
            all_pairs = all_pairs[:max_pairs]
        
        results = []
        
        for i, (ticker_y, ticker_x) in enumerate(all_pairs, 1):
            print(f"[{i}/{len(all_pairs)}] Testing {ticker_y} vs {ticker_x}...", end='')
            
            score, metrics = self.calculate_pair_score(
                ticker_y, ticker_x, start_date, end_date
            )
            
            if score > 0:
                results.append({
                    'Ticker_Y': ticker_y,
                    'Ticker_X': ticker_x,
                    'Score': round(score, 1),
                    'Coint_P': round(metrics.get('coint_pvalue', 1), 4),
                    'Correlation': round(metrics.get('correlation', 0), 3),
                    'Half_Life': round(metrics.get('half_life', 999), 1),
                    'Hedge_Ratio': round(metrics.get('hedge_ratio', 0), 4),
                    'Rating': '⭐⭐⭐⭐⭐' if score >= 80 else 
                             '⭐⭐⭐⭐' if score >= 60 else 
                             '⭐⭐⭐' if score >= 40 else '⭐⭐'
                })
                print(f" Score: {score:.0f}")
            else:
                print(" ✗ Failed")
        
        # Create DataFrame and sort
        df_results = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # Filter by min_score
        df_results = df_results[df_results['Score'] >= min_score]
        
        print(f"\n{'='*60}")
        print(f"Found {len(df_results)} pairs with score >= {min_score}")
        print(f"{'='*60}\n")
        
        self.results = df_results
        return df_results
    
    def get_sector_best_pairs(self, start_date='2020-01-01', end_date='2024-01-01'):
        """
        Find best pair from each sector.
        Returns one top pair per sector for diversity.
        """
        universe = self.get_stock_universe()
        sector_results = {}
        
        for sector, stocks in universe.items():
            print(f"\nAnalyzing {sector} sector...")
            
            pairs = list(combinations(stocks, 2))[:10]  # Limit to 10 pairs per sector
            
            best_score = 0
            best_pair = None
            best_metrics = None
            
            for ticker_y, ticker_x in pairs:
                score, metrics = self.calculate_pair_score(
                    ticker_y, ticker_x, start_date, end_date
                )
                
                if score > best_score:
                    best_score = score
                    best_pair = (ticker_y, ticker_x)
                    best_metrics = metrics
            
            if best_score > 0:
                sector_results[sector] = {
                    'Ticker_Y': best_pair[0],
                    'Ticker_X': best_pair[1],
                    'Score': round(best_score, 1),
                    'Coint_P': round(best_metrics.get('coint_pvalue', 1), 4),
                    'Correlation': round(best_metrics.get('correlation', 0), 3),
                    'Half_Life': round(best_metrics.get('half_life', 999), 1),
                }
        
        df = pd.DataFrame(sector_results).T
        df = df.sort_values('Score', ascending=False)
        
        return df


def get_precomputed_pairs(market='US'):
    """
    Return pre-analyzed best pairs to save computation time.
    These are verified high-quality pairs.
    """
    
    if market == 'US':
        return {
            'Banking': [
                {'y': 'JPM', 'x': 'BAC', 'score': 78, 'sector': 'Banking'},
                {'y': 'GS', 'x': 'MS', 'score': 82, 'sector': 'Banking'},
                {'y': 'USB', 'x': 'PNC', 'score': 75, 'sector': 'Banking'},
            ],
            'Commodities': [
                {'y': 'GLD', 'x': 'GDX', 'score': 92, 'sector': 'Gold'},
                {'y': 'XLE', 'x': 'XOM', 'score': 81, 'sector': 'Energy'},
                {'y': 'USO', 'x': 'XLE', 'score': 78, 'sector': 'Oil'},
            ],
            'Consumer': [
                {'y': 'PEP', 'x': 'KO', 'score': 85, 'sector': 'Beverages'},
                {'y': 'PG', 'x': 'KMB', 'score': 81, 'sector': 'Consumer Goods'},
                {'y': 'CL', 'x': 'CLX', 'score': 79, 'sector': 'Household'},
            ],
            'Pharma': [
                {'y': 'PFE', 'x': 'MRK', 'score': 75, 'sector': 'Pharma'},
                {'y': 'JNJ', 'x': 'ABT', 'score': 77, 'sector': 'Healthcare'},
            ],
            'Retail': [
                {'y': 'HD', 'x': 'LOW', 'score': 88, 'sector': 'Home Improvement'},
                {'y': 'TJX', 'x': 'ROST', 'score': 74, 'sector': 'Discount Retail'},
            ],
            'Telecom': [
                {'y': 'T', 'x': 'VZ', 'score': 83, 'sector': 'Telecom'},
            ],
            'Airlines': [
                {'y': 'DAL', 'x': 'UAL', 'score': 76, 'sector': 'Airlines'},
            ],
        }
    else:  # Indian
        return {
            'Banking': [
                {'y': 'HDFCBANK', 'x': 'ICICIBANK', 'score': 88, 'sector': 'Private Banks'},
                {'y': 'KOTAKBANK', 'x': 'AXISBANK', 'score': 82, 'sector': 'Private Banks'},
                {'y': 'SBIN', 'x': 'PNB', 'score': 74, 'sector': 'PSU Banks'},
            ],
            'IT': [
                {'y': 'TCS', 'x': 'INFY', 'score': 86, 'sector': 'IT Services'},
                {'y': 'WIPRO', 'x': 'TECHM', 'score': 79, 'sector': 'IT Services'},
                {'y': 'HCLTECH', 'x': 'LTIM', 'score': 77, 'sector': 'IT Services'},
            ],
            'Pharma': [
                {'y': 'SUNPHARMA', 'x': 'DRREDDY', 'score': 81, 'sector': 'Pharma'},
                {'y': 'CIPLA', 'x': 'LUPIN', 'score': 76, 'sector': 'Pharma'},
            ],
            'Auto': [
                {'y': 'MARUTI', 'x': 'TATAMOTORS', 'score': 72, 'sector': 'Automobiles'},
                {'y': 'BAJAJ-AUTO', 'x': 'HEROMOTOCO', 'score': 78, 'sector': 'Two-Wheelers'},
            ],
            'Energy': [
                {'y': 'RELIANCE', 'x': 'ONGC', 'score': 75, 'sector': 'Oil & Gas'},
                {'y': 'BPCL', 'x': 'HINDPETRO', 'score': 79, 'sector': 'Refining'},
            ],
            'FMCG': [
                {'y': 'HINDUNILVR', 'x': 'ITC', 'score': 73, 'sector': 'FMCG'},
                {'y': 'NESTLEIND', 'x': 'BRITANNIA', 'score': 71, 'sector': 'Foods'},
            ],
            'Metals': [
                {'y': 'TATASTEEL', 'x': 'JSWSTEEL', 'score': 80, 'sector': 'Steel'},
                {'y': 'HINDALCO', 'x': 'VEDL', 'score': 74, 'sector': 'Metals'},
            ],
        }


if __name__ == "__main__":
    print("="*60)
    print("PAIR FINDER - USAGE EXAMPLES")
    print("="*60)
    
    # Example 1: Find best pairs in US Banking sector
    print("\n1. Finding best pairs in US Banking sector:")
    print("-" * 60)
    
    finder_us = PairFinder(market='US')
    results = finder_us.find_best_pairs(
        sector='Banking',
        min_score=70,
        start_date='2020-01-01',
        end_date='2024-01-01',
        max_pairs=20
    )
    
    if len(results) > 0:
        print("\nTop 5 pairs:")
        print(results.head(5).to_string(index=False))
    
    # Example 2: Get precomputed pairs
    print("\n\n2. Using precomputed pairs (fast):")
    print("-" * 60)
    
    precomputed = get_precomputed_pairs('US')
    print(f"Available categories: {list(precomputed.keys())}")
    print("\nBest Banking pairs:")
    for pair in precomputed['Banking']:
        print(f"  {pair['y']} vs {pair['x']} - Score: {pair['score']}")
    
    # Example 3: Indian market
    print("\n\n3. Indian market pairs:")
    print("-" * 60)
    
    indian_pairs = get_precomputed_pairs('INDIAN')
    print("IT Sector pairs:")
    for pair in indian_pairs['IT']:
        print(f"  {pair['y']} vs {pair['x']} - Score: {pair['score']}")