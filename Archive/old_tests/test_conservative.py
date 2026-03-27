import sys
print("\n")
print("\n")
print("TESTING CONSERVATIVE STRATEGY")
print("\n")
print("\n")

from Core_Strategy.conservative_strategy import ConservativeSystem, download_data
from Core_Strategy.strategy_validator import StrategyValidator

TICKER_Y = 'PEP'
TICKER_X = 'KO'

print(f"\n Testing: {TICKER_Y} vs {TICKER_X}")
print("Strategy parameters:")
print("  - Entry Z-Score: 2.5 (normal), 3.0 (volatile)")
print("  - Exit Z-Score: 0.3 (normal), 0.2 (volatile)")
print("  - Minimum hold: 3 days")
print("  - Z-score window: 60 days\n")

print("\n")
print("\n")
print("STEP 1: Downloading Data")
print("\n")
print("\n")
stock_y, stock_x, market = download_data(TICKER_Y, TICKER_X, 'SPY', 
                                         '2018-01-01', '2024-01-01')

print("\n")
print("\n")
print("STEP 2: Running Backtest")
print("="*20)

system = ConservativeSystem()
results = system.run_backtest(
    stock_y, stock_x, market,
    train_period=252,
    entry_z_normal=3.0,     # Even higher (was 2.5)
    exit_z_normal=0.2,      # Even tighter (was 0.3)
    entry_z_volatile=3.5,   # Much higher (was 3.0)
    exit_z_volatile=0.15,   # Very tight (was 0.2)
    min_hold_days=5,        # Longer minimum (was 3)
    z_score_window=90       # Even longer window (was 60)
)

metrics = system.get_performance_metrics()

print("\n Performance Summary:")
hybrid = metrics['Hybrid Strategy']
print(f"  Total Return: {hybrid['Total Return']}")
print(f"  Sharpe Ratio: {hybrid['Sharpe Ratio']}")
print(f"  Max Drawdown: {hybrid['Max Drawdown']}")
print(f"  Win Rate: {hybrid['Win Rate']}")
print(f"  Total Trades: {hybrid['Total Trades']}")

# Step 3: Monte Carlo Test 
print("\n")
print("\n")
print("STEP 3: Monte Carlo Test")
print("\n")
print("\n")
print(" Running 1000 simulations\n")

validator = StrategyValidator(system, results, TICKER_Y, TICKER_X)

try:
    mc_results = validator.monte_carlo_simulation(n_simulations=1000, confidence_level=0.95)
    mc_data = validator.validation_results['monte_carlo']
    
    p_value = mc_data['p_value']
    actual_return = mc_data['actual_return'] * 100
    ci_lower = mc_data['ci_lower'] * 100
    ci_upper = mc_data['ci_upper'] * 100
    
    print("\n")
    print("\n")
    print("MONTE CARLO RESULTS")
    print("\n")
    print("\n")
    print(f"\n Statistics:")
    print(f"  Actual Return: {actual_return:.2f}%")
    print(f"  95% CI Range: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print(f"  P-Value: {p_value:.4f}")
    
    print(f"\n Interpretation:")
    if p_value < 0.05:
        print(f"   EXCELLENT: p = {p_value:.4f} < 0.05")
        print(f"  Returns are statistically significant!")
        print(f"  Only {p_value*100:.1f}% chance due to luck")
        mc_score = 100
    elif p_value < 0.10:
        print(f"   MARGINAL: p = {p_value:.4f} < 0.10")
        print(f"  Returns show some significance")
        print(f"  About {p_value*100:.1f}% chance due to luck")
        mc_score = 60
    else:
        print(f"   FAIL: p = {p_value:.4f} > 0.10")
        print(f"  Returns not statistically significant")
        print(f"  High chance ({p_value*100:.1f}%) due to random luck")
        mc_score = 0
    
    print(f"\n Your Strategy vs Random:")
    simulated_mean = mc_results.mean() * 100
    print(f"  Your Return: {actual_return:.2f}%")
    print(f"  Random Average: {simulated_mean:.2f}%")
    print(f"  Outperformance: {actual_return - simulated_mean:+.2f}%")
    
    if actual_return > ci_upper:
        print(f"\n   This strategy beats 97.5% of random permutations!")
    elif actual_return > simulated_mean:
        print(f"\n   This strategy beats average random performance")
    else:
        print(f"\n   This strategy underperforms random shuffling")
        
except Exception as e:
    print(f" Error in Monte Carlo: {e}")
    import traceback
    traceback.print_exc()
    mc_score = 0

# Quick other tests
print("\n")
print("\n")
print("STEP 4: Other Quick Tests")
print("="*10)

# Stress test
print("\n  Stress Test...")
try:
    stress_results = validator.stress_test_crisis_periods()
    if stress_results is not None and len(stress_results) > 0:
        avg_crisis = stress_results['Strategy_Return'].mean()
        if avg_crisis > -5:
            print(f"   PASS: Avg crisis return = {avg_crisis:.2f}%")
            stress_score = 100
        else:
            print(f"   FAIL: Avg crisis return = {avg_crisis:.2f}%")
            stress_score = 0
    else:
        print(f"   No crisis periods in data")
        stress_score = 50
except:
    stress_score = 0

# Transaction costs
print("\n Transaction Cost Test")
try:
    validator.transaction_cost_analysis(slippage_bps=5, commission_per_trade=1)
    tc_data = validator.validation_results['transaction_costs']
    retention = tc_data['return_after_costs'] / tc_data['original_return'] if tc_data['original_return'] != 0 else 0
    if retention > 0.7:
        print(f"   PASS: Retains {retention*100:.1f}% after costs")
        tc_score = 100
    elif retention > 0.5:
        print(f"   MARGINAL: Retains {retention*100:.1f}% after costs")
        tc_score = 60
    else:
        print(f"   FAIL: Retains only {retention*100:.1f}% after costs")
        tc_score = 0
except:
    tc_score = 0

# Final score
print("\n")
print("FINAL RESULTS")
print("="*10)

overall = (mc_score + stress_score + tc_score) / 3

print(f"\n Scores:")
print(f"  Monte Carlo:............ {mc_score}/100")
print(f"  Stress Test:............ {stress_score}/100")
print(f"  Transaction Costs:...... {tc_score}/100")
print(f"\n  OVERALL:................ {overall:.0f}/100")

print("\n")
print("\n")
print("\n")
print("RECOMMENDATION")
print("="*20)

if mc_score >= 80:
    print("\n SUCCESS! Monte Carlo test PASSED")
    print("\nYour strategy shows statistically significant returns.")
    print("This means the edge is real, not random luck.")
    
    if overall >= 80:
        print("\n ALL TESTS PASSED - Strategy is validated!")
        print("\nNext steps:")
        print("  1. Paper trade for 1-3 months")
        print("  2. Monitor actual performance vs backtest")
        print("  3. Start with small position sizes")
    else:
        print("\n Monte Carlo passed but other tests need work")
        print("Review stress test and transaction cost results")
        
elif mc_score >= 60:
    print("\ MARGINAL - Monte Carlo shows weak significance")
    print("\nConsider:")
    print("  1. Increase entry threshold to 2.7 or 3.0")
    print("  2. Test on different pairs")
    print("  3. Extend backtest period")
    
else:
    print("\n FAILED - Monte Carlo test failed")
    print("\nThe strategy's returns are not statistically significant.")
    print("\nOptions:")
    print("  1. Try even stricter parameters:")
    print("     entry_z_normal=3.0, exit_z_normal=0.2")
    print("  2. Test different stock pairs")
    print("  3. This pair may not be suitable for pairs trading")

from datetime import datetime
filename = f'ultraconservative_test_{TICKER_Y}_{TICKER_X}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

with open(filename, 'w') as f:
    f.write("CONSERVATIVE STRATEGY TEST RESULTS\n")
    f.write("="*20+ "\n\n")
    f.write(f"Pair: {TICKER_Y} vs {TICKER_X}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Performance:\n")
    for k, v in hybrid.items():
        f.write(f"  {k}: {v}\n")
    f.write(f"\nValidation:\n")
    f.write(f"  Monte Carlo: {mc_score}/100 (p={p_value:.4f})\n")
    f.write(f"  Stress Test: {stress_score}/100\n")
    f.write(f"  Trans Costs: {tc_score}/100\n")
    f.write(f"\nOverall: {overall:.0f}/100\n")

print(f"\n Results saved to: {filename}")
print("\n" )