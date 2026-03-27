import sys
import os 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core_Strategy.conservative_strategy import ConservativeSystem, download_data
from Core_Strategy.strategy_validator import StrategyValidator
import pandas as pd

TOP_PAIRS = [
    ('BAC', 'PNC', 110, {'entry_z': 1.8, 'exit_z': 0.5, 'min_hold': 2}),  
    ('WFC', 'MS', 100, {'entry_z': 2.0, 'exit_z': 0.5, 'min_hold': 3}),
    ('WFC', 'GS', 100, {'entry_z': 2.0, 'exit_z': 0.5, 'min_hold': 3}),
    ('VMC', 'MLM', 95, {'entry_z': 2.0, 'exit_z': 0.5, 'min_hold': 2}),
    ('UNH', 'ABBV', 95, {'entry_z': 2.0, 'exit_z': 0.5, 'min_hold': 3}),
    ('LIN', 'VMC', 95, {'entry_z': 2.0, 'exit_z': 0.5, 'min_hold': 3}),
    ('CVX', 'OXY', 85, {'entry_z': 2.0, 'exit_z': 0.5, 'min_hold': 3}),
    ('SO', 'SRE', 85, {'entry_z': 2.0, 'exit_z': 0.5, 'min_hold': 3}),
    ('BA', 'HON', 85, {'entry_z': 2.2, 'exit_z': 0.4, 'min_hold': 4}),
    ('BA', 'GE', 85, {'entry_z': 2.2, 'exit_z': 0.4, 'min_hold': 3}),
]


def test_single_pair(ticker_y, ticker_x, params, score, 
                     start_date='2020-01-01', end_date='2024-01-01',
                     full_validation=False):
    print(f"\n")
    print(f"Testing: {ticker_y} vs {ticker_x} (Quality Score: {score})")
    print("\n")
    
    print(f" Downloading data")
    try:
        stock_y, stock_x, market = download_data(ticker_y, ticker_x, 'SPY', start_date, end_date)
    except Exception as e:
        print(f" Download failed: {e}")
        return None
    
    print(f" Running backtest")
    try:
        system = ConservativeSystem()
        results = system.run_backtest(
            stock_y, stock_x, market,
            train_period=252,
            entry_z_normal=params['entry_z'],
            exit_z_normal=params['exit_z'],
            entry_z_volatile=params['entry_z'] + 0.5,
            exit_z_volatile=max(0.1, params['exit_z'] - 0.1),
            min_hold_days=params['min_hold'],
            z_score_window=60
        )
        
        metrics = system.get_performance_metrics()
        hybrid = metrics['Hybrid Strategy']
        
        # Extract metrics
        total_return = float(hybrid['Total Return'].rstrip('%'))
        sharpe = float(hybrid['Sharpe Ratio'])
        max_dd = float(hybrid['Max Drawdown'].rstrip('%'))
        win_rate = float(hybrid['Win Rate'].rstrip('%'))
        total_trades = int(hybrid['Total Trades'])
        
        print(f"\n Performance Summary:")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Max Drawdown: {max_dd:.2f}%")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total Trades: {total_trades}")
        
        result = {
            'Ticker_Y': ticker_y,
            'Ticker_X': ticker_x,
            'Quality_Score': score,
            'Total_Return': total_return,
            'Sharpe': sharpe,
            'Max_DD': max_dd,
            'Win_Rate': win_rate,
            'Total_Trades': total_trades,
            'MC_Score': 0,
            'MC_PValue': 1.0,
            'Status': 'Unknown'
        }
        
        # Monte Carlo test
        if sharpe > 0 and total_trades >= 10:
            print(f"\n Running Monte Carlo test...")
            
            validator = StrategyValidator(system, results, ticker_y, ticker_x)
            
            try:
                # Quick test (500 simulations)
                validator.monte_carlo_simulation(n_simulations=500)
                mc_pvalue = validator.validation_results['monte_carlo']['p_value']
                
                result['MC_PValue'] = mc_pvalue
                
                if mc_pvalue < 0.05:
                    print(f"    PASS: p-value = {mc_pvalue:.4f} < 0.05")
                    result['MC_Score'] = 100
                    result['Status'] = 'EXCELLENT'
                elif mc_pvalue < 0.10:
                    print(f"    MARGINAL: p-value = {mc_pvalue:.4f} < 0.10")
                    result['MC_Score'] = 60
                    result['Status'] = 'MARGINAL'
                else:
                    print(f"    FAIL: p-value = {mc_pvalue:.4f} > 0.10")
                    result['MC_Score'] = 0
                    result['Status'] = 'FAILED'
                
                # Full validation if requested and passed
                if full_validation and mc_pvalue < 0.05:
                    print(f"\n🔍 Running full validation suite...")
                    
                    # Stress test
                    stress = validator.stress_test_crisis_periods()
                    if stress is not None and len(stress) > 0:
                        avg_crisis = stress['Strategy_Return'].mean()
                        result['Crisis_Return'] = avg_crisis
                        print(f"   Crisis Return: {avg_crisis:.2f}%")
                    
                    # Transaction costs
                    validator.transaction_cost_analysis()
                    tc = validator.validation_results.get('transaction_costs', {})
                    if tc:
                        retention = tc['return_after_costs'] / tc['original_return'] if tc['original_return'] != 0 else 0
                        result['Cost_Retention'] = retention * 100
                        print(f"   Cost Retention: {retention*100:.1f}%")
                
            except Exception as e:
                print(f"    Validation error: {e}")
                result['Status'] = 'ERROR'
        
        else:
            print(f"\n Skipping validation (Sharpe={sharpe:.2f}, Trades={total_trades})")
            result['Status'] = 'SKIPPED'
        
        return result
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_all_top_pairs(max_pairs=5, full_validation=False):
    print("\n")
    print("\n")
    print(" TESTING TOP DISCOVERED PAIRS")
    print("\n")
    print("\n")
    print(f"\nTesting top {max_pairs} pairs from auto_find_pairs.py results")
    print("This takes 5-10 minutes...\n")
    
    results = []
    
    for i, (ticker_y, ticker_x, score, params) in enumerate(TOP_PAIRS[:max_pairs], 1):
        print(f"\n[{i}/{max_pairs}] Testing pair...")
        
        result = test_single_pair(ticker_y, ticker_x, params, score, 
                                  full_validation=full_validation)
        
        if result:
            results.append(result)
            if result['Status'] == 'EXCELLENT':
                print(f"\n WINNER! This pair passed Monte Carlo!")
            elif result['Status'] == 'MARGINAL':
                print(f"\n Marginal - may work with tuning")
            elif result['Status'] == 'FAILED':
                print(f"\n Failed Monte Carlo")
    
    return results


def display_final_results(results):
    if not results:
        print("\n No results to display")
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values(['MC_Score', 'Sharpe'], ascending=[False, False])
    
    print("\n")
    print("\n")
    print("FINAL RESULTS - RANKED BY VALIDATION")
    print("\n")
    print("\n")
    
    print("\n All Tested Pairs:")
    display_cols = ['Ticker_Y', 'Ticker_X', 'Quality_Score', 'Sharpe', 'Win_Rate', 
                    'MC_PValue', 'Status']
    print(df[display_cols].to_string(index=False))
    
    # Finding Winners
    excellent = df[df['Status'] == 'EXCELLENT']
    marginal = df[df['Status'] == 'MARGINAL']
    
    print("\n")
    print("RECOMMENDATION")
    print("\n")
    
    if len(excellent) > 0:
        print(f"\n FOUND {len(excellent)} WINNING PAIR(S)!")
        
        best = excellent.iloc[0]
        
        print(f"\n BEST VALIDATED PAIR:")
        print(f"   {best['Ticker_Y']} vs {best['Ticker_X']}")
        print(f"   Quality Score: {best['Quality_Score']}/110")
        print(f"   Sharpe Ratio: {best['Sharpe']:.2f}")
        print(f"   Win Rate: {best['Win_Rate']:.1f}%")
        print(f"   Total Return: {best['Total_Return']:.2f}%")
        print(f"   Monte Carlo p-value: {best['MC_PValue']:.4f} ")
        
        print(f"\n Parameters Used:")
        # Find params for this pair
        for y, x, score, params in TOP_PAIRS:
            if y == best['Ticker_Y'] and x == best['Ticker_X']:
                print(f"   entry_z_normal = {params['entry_z']}")
                print(f"   exit_z_normal = {params['exit_z']}")
                print(f"   min_hold_days = {params['min_hold']}")
                break
        
        print(f"\n NEXT STEPS:")
        print(f"   1. This pair has passed initial validation!")
        print(f"   2. Run full validation with more tests:")
        print(f"      python test_discovered_pairs.py --full-validation")
        print(f"   3. If passes all tests, start paper trading:")
        print(f"      - Use small position sizes")
        print(f"      - Monitor for 1-3 months")
        print(f"      - Track actual vs expected performance")
        print(f"   4. Only go live after successful paper trading")
        
        if len(excellent) > 1:
            print(f"\n Other Validated Pairs:")
            for idx, row in excellent.iloc[1:].iterrows():
                print(f"   • {row['Ticker_Y']} vs {row['Ticker_X']} (Sharpe: {row['Sharpe']:.2f}, p={row['MC_PValue']:.4f})")
        
    elif len(marginal) > 0:
        print(f"\n MARGINAL RESULTS")
        print(f"\nFound {len(marginal)} pairs with p-value < 0.10 (but > 0.05)")
        
        best = marginal.iloc[0]
        print(f"\n Best Available: {best['Ticker_Y']} vs {best['Ticker_X']}")
        print(f"   Sharpe: {best['Sharpe']:.2f}")
        print(f"   p-value: {best['MC_PValue']:.4f}")
        
        print(f"\n Options:")
        print(f"   1. Try stricter parameters (entry_z=2.5, exit_z=0.3)")
        print(f"   2. Test more pairs from the list")
        print(f"   3. Use different time period")
        
    else:
        print(f"\n NO PAIRS PASSED VALIDATION")
        
        if len(df) > 0:
            best_attempt = df.iloc[0]
            print(f"\n Best Attempt: {best_attempt['Ticker_Y']} vs {best_attempt['Ticker_X']}")
            print(f"   Sharpe: {best_attempt['Sharpe']:.2f}")
            print(f"   Status: {best_attempt['Status']}")
            
            if best_attempt['Sharpe'] < 0:
                print(f"\n Negative Sharpe suggests parameters are too conservative")
                print(f"   Try: entry_z=1.8, exit_z=0.6")
            elif best_attempt['MC_PValue'] > 0.10:
                print(f"\n Failed statistical significance")
                print(f"   Try: More selective entries (higher entry_z)")
        
        print(f"\n Recommendations:")
        print(f"   1. Test next batch: python test_discovered_pairs.py --start 5")
        print(f"   2. Adjust time period (maybe 2019-2024 instead of 2020-2024)")
        print(f"   3. Current market may not favor pairs trading")
    
    filename = 'validated_pairs_results.csv'
    df.to_csv(filename, index=False)
    print(f"\n Full results saved to: {filename}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-pairs', type=int, default=5, help='Number of pairs to test')
    parser.add_argument('--full-validation', action='store_true', help='Run full validation suite')
    parser.add_argument('--start', type=int, default=0, help='Start index for pairs')
    
    args = parser.parse_args()
    
    if args.start > 0:
        TOP_PAIRS = TOP_PAIRS[args.start:]
        print(f"Starting from pair #{args.start + 1}")
    
    print("\n")
    print("\n")
    print("PAIR VALIDATION TESTING")
    print("\n")
    print("\n")
    print(f"\nConfiguration:")
    print(f"   Testing: Top {args.max_pairs} pairs")
    print(f"   Full validation: {'Yes' if args.full_validation else 'No (quick test only)'}")
    print(f"   Period: 2020-01-01 to 2024-01-01")
    
    results = test_all_top_pairs(max_pairs=args.max_pairs, 
                                 full_validation=args.full_validation)
    
    df = display_final_results(results)
    
    print("\n")
    print("\n")
    print("TESTING COMPLETE!")
    print("\n")
    print("\n")
    
    if df is not None and len(df[df['Status'] == 'EXCELLENT']) > 0:
        print("\n validated pairs ready for paper trading!")
    else:
        print("\ Keep testing - the right pair is out there!")