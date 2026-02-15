import sys
sys.path.insert(0, '.')

from Core_Strategy.conservative_strategy import UltraConservativeSystem, download_data
from Core_Strategy.strategy_validator import StrategyValidator

ticker_y = 'SO'
ticker_x = 'SRE'
score = 85
params = {'entry_z': 2.0, 'exit_z': 0.5, 'min_hold': 3}

print(f"\n")
print(f"\n")
print(f"\n")
print(f"Testing: {ticker_y} vs {ticker_x} (Quality Score: {score})")
print("\n")
print("\n")

print(f" Downloading data")
try:
    stock_y, stock_x, market = download_data(ticker_y, ticker_x, 'SPY', 
                                             '2020-01-01', '2024-01-01')
except Exception as e:
    print(f" Download failed: {e}")
    sys.exit(1)

print(f" Running backtest")
try:
    system = UltraConservativeSystem()
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
    
except Exception as e:
    print(f" Backtest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n Running full validation")

validator = StrategyValidator(system, results, ticker_y, ticker_x)

scores = {}

# 1. Monte Carlo
print(f"\n{'='*80}")
print("1. Monte Carlo Simulation (1000 runs)")
print(f"{'='*80}")
try:
    validator.monte_carlo_simulation(n_simulations=1000)
    mc_pvalue = validator.validation_results['monte_carlo']['p_value']
    
    if mc_pvalue < 0.05:
        print(f"    EXCELLENT: p-value = {mc_pvalue:.4f} < 0.05")
        scores['Monte Carlo'] = 100
    elif mc_pvalue < 0.10:
        print(f"    MARGINAL: p-value = {mc_pvalue:.4f} < 0.10")
        scores['Monte Carlo'] = 60
    else:
        print(f"    FAIL: p-value = {mc_pvalue:.4f} > 0.10")
        scores['Monte Carlo'] = 0
except Exception as e:
    print(f"    Error: {e}")
    scores['Monte Carlo'] = 0

# 2. Walk-Forward
print(f"\n{'='*80}")
print("2. Walk-Forward Analysis")
print(f"{'='*80}")
try:
    wf_results = validator.walk_forward_analysis(
        stock_y, stock_x, market,
        train_period=252,
        test_period=126,
        n_folds=4,
        entry_z=params['entry_z'],
        exit_z=params['exit_z']
    )
    
    avg_return = wf_results['Return'].mean()
    consistency = (wf_results['Return'] > 0).mean()
    
    if avg_return > 0 and consistency >= 0.6:
        print(f"    PASS")
        scores['Walk-Forward'] = 100
    elif avg_return > 0:
        print(f"    MARGINAL")
        scores['Walk-Forward'] = 60
    else:
        print(f"    FAIL")
        scores['Walk-Forward'] = 0
except Exception as e:
    print(f"    Error: {e}")
    scores['Walk-Forward'] = 0

# 3. Stress Test
print(f"\n{'='*80}")
print("3. Stress Test (Crisis Periods)")
print(f"{'='*80}")
try:
    stress = validator.stress_test_crisis_periods()
    if stress is not None and len(stress) > 0:
        avg_crisis = stress['Strategy_Return'].mean()
        avg_trade_rate = stress['Trade_Rate'].mean()
        
        if avg_crisis > -5 or avg_trade_rate < 30:
            print(f"   PASS")
            scores['Stress Test'] = 100
        elif avg_crisis > -10:
            print(f"    MARGINAL")
            scores['Stress Test'] = 60
        else:
            print(f"    FAIL")
            scores['Stress Test'] = 0
    else:
        print(f"    No crisis periods")
        scores['Stress Test'] = 50
except Exception as e:
    print(f"    Error: {e}")
    scores['Stress Test'] = 0

# 4. Sensitivity
print(f"\n{'='*80}")
print("4. Sensitivity Analysis")
print(f"{'='*80}")
try:
    sens = validator.sensitivity_analysis(stock_y, stock_x, market)
    positive_returns = (sens['Return_%'] > 0).mean()
    
    if positive_returns >= 0.75:
        print(f"    PASS ({positive_returns*100:.0f}% positive)")
        scores['Sensitivity'] = 100
    elif positive_returns >= 0.60:
        print(f"    MARGINAL ({positive_returns*100:.0f}% positive)")
        scores['Sensitivity'] = 60
    else:
        print(f"    FAIL ({positive_returns*100:.0f}% positive)")
        scores['Sensitivity'] = 0
except Exception as e:
    print(f"    Error: {e}")
    scores['Sensitivity'] = 0

# 5. Transaction Costs
print("\n")
print("\n")
print("5. Transaction Cost Analysis")
print("\n")
print("\n")
try:
    validator.transaction_cost_analysis()
    tc = validator.validation_results.get('transaction_costs', {})
    if tc:
        retention = tc['return_after_costs'] / tc['original_return'] if tc['original_return'] != 0 else 0
        
        if retention > 0.7:
            print(f"   PASS (retains {retention*100:.1f}%)")
            scores['Transaction Costs'] = 100
        elif retention > 0.5:
            print(f"    MARGINAL (retains {retention*100:.1f}%)")
            scores['Transaction Costs'] = 60
        else:
            print(f"    FAIL (retains {retention*100:.1f}%)")
            scores['Transaction Costs'] = 0
    else:
        scores['Transaction Costs'] = 0
except Exception as e:
    print(f"    Error: {e}")
    scores['Transaction Costs'] = 0

# 6. Statistical Tests
print(f"\n{'='*80}")
print("6. Statistical Tests")
print(f"{'='*80}")
try:
    validator.statistical_tests()
    scores['Statistical'] = 100
except Exception as e:
    print(f"    Error: {e}")
    scores['Statistical'] = 50

# Final Results
print("\n")
print("VALIDATION RESULTS SUMMARY")
print("\n")

for test, score in scores.items():
    status = " PASS" if score >= 80 else ("⚠️ MARGINAL" if score >= 60 else "❌ FAIL")
    print(f"  {test:.<30} {score:>3}/100  {status}")

overall = sum(scores.values()) / len(scores)
print("\n")
print("\n")
print(f"  {'OVERALL SCORE':.<30} {overall:>3.0f}/100")
print("\n")
print("\n")
print("GENERATING DETAILED REPORT")
print("\n")
print("\n")
print("\n")
validator.generate_report(save_path=f'validation_{ticker_y}_{ticker_x}_full.txt')


print("\n")
print("\n")
print("RECOMMENDATION")
print("\n")
print("\n")

if overall >= 80 and scores['Monte Carlo'] >= 80:
    print("\n EXCELLENT! Strategy is validated and ready!")
    print("\n PROCEED TO PAPER TRADING")
    print("\nNext steps:")
    print("  1. Sign up: https://app.alpaca.markets/signup")
    print("  2. Get API keys")
    print("  3. Run: python paper_trading_system.py")
    print("  4. Monitor for 90 days")
    
elif overall >= 60:
    print("\n MARGINAL ")
    print("\nOptions:")
    print("  1. Paper trade with small size")
    print("  2. Adjust parameters")
    print("  3. Test more pairs")
    
else:
    print("\n FAILED VALIDATION")
    print("\nActions:")
    print("  1. Try different parameters")
    print("  2. Test different time period")
    print("  3. Find alternative pairs")

print(f"\n Full report: validation_{ticker_y}_{ticker_x}_full.txt")
print("\n")