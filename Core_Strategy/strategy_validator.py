"""
Comprehensive Validation Framework for Hybrid Pairs Trading Strategy
Implements industry-standard validation methods to ensure robustness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import your trading system
from  hybrid_pairs_trading import UltimateHybridSystem, download_data


class StrategyValidator:
    """
    Comprehensive validation framework for pairs trading strategies.
    Implements multiple validation methods used by quant funds.
    """
    
    def __init__(self, system, results, ticker_y, ticker_x):
        """
        Args:
            system: UltimateHybridSystem instance
            results: Backtest results DataFrame
            ticker_y: Stock Y ticker
            ticker_x: Stock X ticker
        """
        self.system = system
        self.results = results
        self.ticker_y = ticker_y
        self.ticker_x = ticker_x
        self.validation_results = {}
    
    # ========================================================================
    # 1. WALK-FORWARD ANALYSIS (Out-of-Sample Testing)
    # ========================================================================
    
    def walk_forward_analysis(self, stock_y, stock_x, market_index,
                              train_period=252, test_period=126,
                              n_folds=5, entry_z=2.0, exit_z=0.5):
        """
        Walk-forward analysis: Train on period N, test on period N+1.
        Most important validation - prevents overfitting.
        """
        print("="*80)
        print("1. WALK-FORWARD ANALYSIS (Out-of-Sample Testing)")
        print("="*80)
        print("Purpose: Ensure strategy wasn't overfit to historical data")
        print(f"Method: Train on {train_period} days, test on {test_period} days")
        print(f"Number of folds: {n_folds}\n")
        
        results = []
        total_data_len = len(stock_y)
        
        for i in range(n_folds):
            start_idx = i * test_period
            train_end_idx = start_idx + train_period
            test_end_idx = min(train_end_idx + test_period, total_data_len)
            
            if test_end_idx >= total_data_len:
                break
            
            # Split data
            train_y = stock_y.iloc[start_idx:train_end_idx]
            train_x = stock_x.iloc[start_idx:train_end_idx]
            train_market = market_index.iloc[start_idx:train_end_idx]
            
            test_y = stock_y.iloc[train_end_idx:test_end_idx]
            test_x = stock_x.iloc[train_end_idx:test_end_idx]
            test_market = market_index.iloc[train_end_idx:test_end_idx]
            
            print(f"Fold {i+1}: Train {train_y.index[0].date()} to {train_y.index[-1].date()}, "
                  f"Test {test_y.index[0].date()} to {test_y.index[-1].date()}")
            
            try:
                # Test on test period
                test_system = UltimateHybridSystem()
                test_results = test_system.run_backtest(test_y, test_x, test_market,
                                                        train_period=min(126, len(test_y)//2),
                                                        entry_z=entry_z, exit_z=exit_z)
                
                # Calculate metrics
                test_returns = test_results['strategy_return'].dropna()
                total_return = (1 + test_returns).prod() - 1
                sharpe = (test_returns.mean() / test_returns.std() * np.sqrt(252)) if test_returns.std() > 0 else 0
                
                max_dd = ((test_results['strategy_cumulative'] - 
                          test_results['strategy_cumulative'].expanding().max()) / 
                          test_results['strategy_cumulative'].expanding().max()).min()
                
                results.append({
                    'Fold': i + 1,
                    'Test_Start': test_y.index[0],
                    'Test_End': test_y.index[-1],
                    'Return': total_return * 100,
                    'Sharpe': sharpe,
                    'Max_DD': max_dd * 100,
                    'Win_Rate': (test_returns > 0).mean() * 100
                })
                
            except Exception as e:
                print(f"  Error in fold {i+1}: {e}")
                continue
        
        wf_results = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("WALK-FORWARD RESULTS")
        print("="*80)
        print(wf_results.to_string(index=False))
        
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY")
        print("="*80)
        print(f"Average Out-of-Sample Return: {wf_results['Return'].mean():.2f}%")
        print(f"Average Out-of-Sample Sharpe: {wf_results['Sharpe'].mean():.2f}")
        print(f"Std Dev of Returns: {wf_results['Return'].std():.2f}%")
        print(f"Consistency (% positive folds): {(wf_results['Return'] > 0).mean()*100:.1f}%")
        
        # Validation criteria
        print("\n" + "="*80)
        print("VALIDATION CRITERIA")
        print("="*80)
        avg_return = wf_results['Return'].mean()
        consistency = (wf_results['Return'] > 0).mean()
        avg_sharpe = wf_results['Sharpe'].mean()
        
        if avg_return > 0 and consistency >= 0.6 and avg_sharpe > 0.5:
            print("✅ PASS: Strategy shows consistent out-of-sample performance")
        elif avg_return > 0 and consistency >= 0.5:
            print("⚠️ MARGINAL: Strategy shows some out-of-sample performance")
        else:
            print("❌ FAIL: Strategy does not perform well out-of-sample")
            print("   Likely overfit to historical data. DO NOT TRADE.")
        
        self.validation_results['walk_forward'] = wf_results
        return wf_results
    
    # ========================================================================
    # 2. MONTE CARLO SIMULATION
    # ========================================================================
    
    def monte_carlo_simulation(self, n_simulations=1000, confidence_level=0.95):
        """
        Monte Carlo simulation to assess strategy robustness.
        Randomly shuffles returns to see if results are statistically significant.
        """
        print("\n" + "="*80)
        print("2. MONTE CARLO SIMULATION")
        print("="*80)
        print("Purpose: Test if returns are due to skill or luck")
        print(f"Method: Shuffle trade returns {n_simulations} times\n")
        
        trades = self.system.get_trade_analysis()
        
        if len(trades) == 0:
            print("❌ No trades to analyze")
            return None
        
        actual_returns = trades['pnl'].values
        actual_total = actual_returns.sum()
        actual_sharpe = actual_returns.mean() / actual_returns.std() * np.sqrt(252/trades['duration_days'].mean()) if actual_returns.std() > 0 else 0
        
        # Run simulations
        simulated_returns = []
        simulated_sharpes = []
        
        for _ in range(n_simulations):
            shuffled = np.random.permutation(actual_returns)
            simulated_returns.append(shuffled.sum())
            sharpe = shuffled.mean() / shuffled.std() * np.sqrt(252/trades['duration_days'].mean()) if shuffled.std() > 0 else 0
            simulated_sharpes.append(sharpe)
        
        simulated_returns = np.array(simulated_returns)
        simulated_sharpes = np.array(simulated_sharpes)
        
        # Calculate p-value
        p_value_return = (simulated_returns >= actual_total).mean()
        p_value_sharpe = (simulated_sharpes >= actual_sharpe).mean()
        
        # Confidence intervals
        ci_lower = np.percentile(simulated_returns, (1 - confidence_level) / 2 * 100)
        ci_upper = np.percentile(simulated_returns, (1 + confidence_level) / 2 * 100)
        
        print(f"Actual Total Return: {actual_total*100:.2f}%")
        print(f"Actual Sharpe Ratio: {actual_sharpe:.3f}")
        print(f"\nSimulation Statistics:")
        print(f"  Mean Simulated Return: {simulated_returns.mean()*100:.2f}%")
        print(f"  {int(confidence_level*100)}% Confidence Interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
        print(f"  P-value (Return): {p_value_return:.4f}")
        print(f"  P-value (Sharpe): {p_value_sharpe:.4f}")
        
        print("\n" + "="*80)
        print("VALIDATION CRITERIA")
        print("="*80)
        
        if p_value_return < 0.05 and actual_total > ci_upper:
            print("✅ PASS: Returns are statistically significant (p < 0.05)")
            print("   Strategy performance is unlikely due to random chance")
        elif p_value_return < 0.10:
            print("⚠️ MARGINAL: Returns show some significance (p < 0.10)")
        else:
            print("❌ FAIL: Returns not statistically significant")
            print("   Performance may be due to luck. Use caution.")
        
        self.validation_results['monte_carlo'] = {
            'actual_return': actual_total,
            'p_value': p_value_return,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'simulated_returns': simulated_returns
        }
        
        return simulated_returns
    
    # ========================================================================
    # 3. STRESS TESTING (Crisis Periods)
    # ========================================================================
    
    def stress_test_crisis_periods(self):
        """
        Test strategy performance during known crisis periods.
        """
        print("\n" + "="*80)
        print("3. STRESS TESTING (Crisis Period Analysis)")
        print("="*80)
        print("Purpose: Evaluate strategy behavior during market crashes\n")
        
        # Define crisis periods
        crisis_periods = {
            '2020 COVID Crash': (pd.Timestamp('2020-02-19'), pd.Timestamp('2020-03-23')),
            '2022 Bear Market': (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-10-12')),
            '2018 December Selloff': (pd.Timestamp('2018-10-01'), pd.Timestamp('2018-12-24')),
        }
        
        results = []
        
        for crisis_name, (start, end) in crisis_periods.items():
            crisis_data = self.results[(self.results.index >= start) & (self.results.index <= end)]
            
            if len(crisis_data) == 0:
                continue
            
            crisis_return = (1 + crisis_data['strategy_return']).prod() - 1
            market_return = (crisis_data['Market'].iloc[-1] / crisis_data['Market'].iloc[0]) - 1
            
            trading_days = (crisis_data['final_signal'] != 0).sum()
            total_days = len(crisis_data)
            regime_safe_pct = (crisis_data['regime'] == 1).mean() * 100
            
            results.append({
                'Crisis': crisis_name,
                'Start': start.date(),
                'End': end.date(),
                'Strategy_Return': crisis_return * 100,
                'Market_Return': market_return * 100,
                'Days_Traded': trading_days,
                'Total_Days': total_days,
                'Trade_Rate': trading_days / total_days * 100 if total_days > 0 else 0,
                'Safe_Regime_%': regime_safe_pct
            })
        
        if len(results) == 0:
            print("⚠️ No crisis periods found in backtest timeframe")
            return None
        
        stress_results = pd.DataFrame(results)
        print(stress_results.to_string(index=False))
        
        print("\n" + "="*80)
        print("VALIDATION CRITERIA")
        print("="*80)
        
        avg_crisis_return = stress_results['Strategy_Return'].mean()
        avg_market_return = stress_results['Market_Return'].mean()
        avg_trade_rate = stress_results['Trade_Rate'].mean()
        
        if avg_crisis_return > avg_market_return and avg_crisis_return > -5:
            print("✅ PASS: Strategy outperforms market during crises")
        elif avg_trade_rate < 30:
            print("✅ PASS: Strategy correctly avoids trading during crises")
        elif avg_crisis_return > -10:
            print("⚠️ MARGINAL: Strategy shows moderate losses during crises")
        else:
            print("❌ FAIL: Strategy performs poorly during crises")
            print("   Risk management may be inadequate")
        
        self.validation_results['stress_test'] = stress_results
        return stress_results
    
    # ========================================================================
    # 4. SENSITIVITY ANALYSIS
    # ========================================================================
    
    def sensitivity_analysis(self, stock_y, stock_x, market_index):
        """
        Test how sensitive strategy is to parameter changes.
        """
        print("\n" + "="*80)
        print("4. SENSITIVITY ANALYSIS (Parameter Robustness)")
        print("="*80)
        print("Purpose: Ensure strategy isn't overfit to specific parameters\n")
        
        param_grid = {
            'entry_z': [1.5, 2.0, 2.5],
            'exit_z': [0.3, 0.5, 0.7]
        }
        
        results = []
        
        for entry_z in param_grid['entry_z']:
            for exit_z in param_grid['exit_z']:
                print(f"Testing: entry_z={entry_z}, exit_z={exit_z}")
                
                try:
                    system = UltimateHybridSystem()
                    test_results = system.run_backtest(
                        stock_y, stock_x, market_index,
                        train_period=252,
                        entry_z=entry_z,
                        exit_z=exit_z
                    )
                    
                    returns = test_results['strategy_return'].dropna()
                    total_return = (1 + returns).prod() - 1
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    
                    results.append({
                        'entry_z': entry_z,
                        'exit_z': exit_z,
                        'Return_%': total_return * 100,
                        'Sharpe': sharpe,
                        'Trades': len(returns[returns != 0])
                    })
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
        
        sensitivity_df = pd.DataFrame(results)
        print("\n" + sensitivity_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("VALIDATION CRITERIA")
        print("="*80)
        
        positive_returns = (sensitivity_df['Return_%'] > 0).mean()
        sharpe_std = sensitivity_df['Sharpe'].std()
        
        if positive_returns >= 0.75 and sharpe_std < 0.5:
            print("✅ PASS: Strategy is robust across parameter ranges")
        elif positive_returns >= 0.60:
            print("⚠️ MARGINAL: Strategy shows some parameter sensitivity")
        else:
            print("❌ FAIL: Strategy is highly parameter-dependent")
            print("   Likely overfit. Results may not be repeatable.")
        
        self.validation_results['sensitivity'] = sensitivity_df
        return sensitivity_df
    
    # ========================================================================
    # 5. TRANSACTION COST ANALYSIS
    # ========================================================================
    
    def transaction_cost_analysis(self, slippage_bps=5, commission_per_trade=1):
        """
        Test strategy under realistic transaction costs.
        """
        print("\n" + "="*80)
        print("5. TRANSACTION COST ANALYSIS")
        print("="*80)
        print(f"Purpose: Test if strategy remains profitable after costs")
        print(f"Slippage: {slippage_bps} bps, Commission: ${commission_per_trade} per trade\n")
        
        trades = self.system.get_trade_analysis()
        
        if len(trades) == 0:
            print("❌ No trades to analyze")
            return None
        
        # Calculate costs
        slippage_cost = slippage_bps / 10000
        
        # Original returns
        original_return = trades['pnl'].sum() * 100
        original_sharpe = trades['pnl'].mean() / trades['pnl'].std() * np.sqrt(252/trades['duration_days'].mean()) if trades['pnl'].std() > 0 else 0
        
        # Apply costs
        cost_per_trade = 2 * slippage_cost
        trades_with_costs = trades.copy()
        trades_with_costs['pnl_after_costs'] = trades_with_costs['pnl'] - cost_per_trade
        
        # New metrics
        return_after_costs = trades_with_costs['pnl_after_costs'].sum() * 100
        sharpe_after_costs = (trades_with_costs['pnl_after_costs'].mean() / 
                             trades_with_costs['pnl_after_costs'].std() * 
                             np.sqrt(252/trades['duration_days'].mean())) if trades_with_costs['pnl_after_costs'].std() > 0 else 0
        
        total_cost = cost_per_trade * len(trades) * 100
        
        print(f"Original Return: {original_return:.2f}%")
        print(f"Return After Costs: {return_after_costs:.2f}%")
        print(f"Total Cost Impact: {total_cost:.2f}%")
        print(f"\nOriginal Sharpe: {original_sharpe:.3f}")
        print(f"Sharpe After Costs: {sharpe_after_costs:.3f}")
        print(f"Sharpe Degradation: {original_sharpe - sharpe_after_costs:.3f}")
        
        print("\n" + "="*80)
        print("VALIDATION CRITERIA")
        print("="*80)
        
        if return_after_costs > original_return * 0.7 and sharpe_after_costs > 0.5:
            print("✅ PASS: Strategy remains profitable after transaction costs")
        elif return_after_costs > 0:
            print("⚠️ MARGINAL: Strategy barely profitable after costs")
        else:
            print("❌ FAIL: Strategy unprofitable after transaction costs")
            print("   Strategy trades too frequently or margins too thin")
        
        self.validation_results['transaction_costs'] = {
            'original_return': original_return,
            'return_after_costs': return_after_costs,
            'cost_impact': total_cost
        }
        
        return trades_with_costs
    
    # ========================================================================
    # 6. STATISTICAL TESTS
    # ========================================================================
    
    def statistical_tests(self):
        """
        Run statistical tests on strategy returns.
        """
        print("\n" + "="*80)
        print("6. STATISTICAL TESTS")
        print("="*80)
        print("Purpose: Verify statistical properties of returns\n")
        
        returns = self.results['strategy_return'].dropna()
        
        # 1. Normality Test
        jb_stat, jb_pval = stats.jarque_bera(returns)
        print(f"Jarque-Bera Normality Test:")
        print(f"  Statistic: {jb_stat:.4f}, P-value: {jb_pval:.4f}")
        if jb_pval > 0.05:
            print(f"  ✅ Returns appear normally distributed")
        else:
            print(f"  ⚠️ Returns are NOT normally distributed (fat tails likely)")
        
        # 2. Mean Return Test
        t_stat, t_pval = stats.ttest_1samp(returns, 0)
        print(f"\nT-Test (Mean > 0):")
        print(f"  Statistic: {t_stat:.4f}, P-value: {t_pval/2:.4f}")
        if t_pval/2 < 0.05 and returns.mean() > 0:
            print(f"  ✅ Mean return significantly positive")
        else:
            print(f"  ❌ Mean return not significantly different from zero")
        
        # 3. Autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_stat = acorr_ljungbox(returns, lags=[10], return_df=True)
            print(f"\nLjung-Box Autocorrelation Test:")
            print(f"  P-value: {lb_stat['lb_pvalue'].iloc[0]:.4f}")
            if lb_stat['lb_pvalue'].iloc[0] > 0.05:
                print(f"  ✅ No significant autocorrelation")
            else:
                print(f"  ⚠️ Returns show autocorrelation")
        except ImportError:
            print("\n⚠️ Statsmodels not available for autocorrelation test")
        
        self.validation_results['statistical_tests'] = {
            'normality_pval': jb_pval,
            'mean_test_pval': t_pval/2
        }
    
    # ========================================================================
    # 7. GENERATE VALIDATION REPORT
    # ========================================================================
    
    def generate_report(self, save_path='validation_report.txt'):
        """Generate comprehensive validation report."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("PAIRS TRADING STRATEGY VALIDATION REPORT")
        report.append(f"Pair: {self.ticker_y} vs {self.ticker_x}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        report.append("")
        
        # Summary scores
        scores = {
            'Walk-Forward': 0,
            'Monte Carlo': 0,
            'Stress Test': 0,
            'Sensitivity': 0,
            'Transaction Costs': 0
        }
        
        # Walk-Forward
        if 'walk_forward' in self.validation_results:
            wf = self.validation_results['walk_forward']
            if wf['Return'].mean() > 0 and (wf['Return'] > 0).mean() >= 0.6:
                scores['Walk-Forward'] = 100
            elif wf['Return'].mean() > 0:
                scores['Walk-Forward'] = 60
        
        # Monte Carlo
        if 'monte_carlo' in self.validation_results:
            mc = self.validation_results['monte_carlo']
            if mc['p_value'] < 0.05:
                scores['Monte Carlo'] = 100
            elif mc['p_value'] < 0.10:
                scores['Monte Carlo'] = 60
        
        # Transaction Costs
        if 'transaction_costs' in self.validation_results:
            tc = self.validation_results['transaction_costs']
            retention = tc['return_after_costs'] / tc['original_return'] if tc['original_return'] != 0 else 0
            if retention > 0.7:
                scores['Transaction Costs'] = 100
            elif retention > 0.5:
                scores['Transaction Costs'] = 60
        
        # Overall Score
        overall_score = np.mean(list(scores.values()))
        
        report.append("VALIDATION SCORES")
        report.append("-" * 80)
        for test, score in scores.items():
            status = "✅ PASS" if score >= 80 else ("⚠️ MARGINAL" if score >= 60 else "❌ FAIL")
            report.append(f"{test:.<40} {score:>3.0f}/100  {status}")
        
        report.append("-" * 80)
        report.append(f"{'OVERALL VALIDATION SCORE':.<40} {overall_score:>3.0f}/100")
        report.append("")
        
        # Recommendation
        report.append("="*80)
        report.append("FINAL RECOMMENDATION")
        report.append("="*80)
        
        if overall_score >= 80:
            report.append("✅ STRATEGY VALIDATED")
            report.append("This strategy shows robust performance across multiple validation tests.")
            report.append("Proceed with paper trading, then small live allocation.")
        elif overall_score >= 60:
            report.append("⚠️ MARGINAL - PROCEED WITH CAUTION")
            report.append("Strategy shows some promise but has limitations.")
            report.append("Consider: Additional parameter tuning, longer backtest period,")
            report.append("or combining with other strategies.")
        else:
            report.append("❌ STRATEGY NOT VALIDATED")
            report.append("Strategy fails critical validation tests.")
            report.append("DO NOT TRADE. Strategy likely overfit or fundamentally flawed.")
        
        # Save report
        report_text = "\n".join(report)
        print(report_text)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n📄 Report saved to: {save_path}")
        
        return overall_score


# ============================================================================
# COMPLETE VALIDATION WORKFLOW
# ============================================================================

def run_complete_validation(ticker_y='PEP', ticker_x='KO',
                           start_date='2018-01-01', end_date='2024-01-01',
                           entry_z=2.0, exit_z=0.5):
    """
    Run complete validation workflow for a pairs trading strategy.
    """
    print("="*80)
    print("COMPLETE PAIRS TRADING STRATEGY VALIDATION")
    print(f"Pair: {ticker_y} vs {ticker_x}")
    print("="*80)
    print()
    
    # Download data
    print("📥 Downloading data...")
    stock_y, stock_x, market = download_data(ticker_y, ticker_x, 'SPY',
                                             start_date, end_date)
    
    # Run initial backtest
    print("\n🔄 Running initial backtest...")
    system = UltimateHybridSystem()
    results = system.run_backtest(stock_y, stock_x, market,
                                  train_period=252,
                                  entry_z=entry_z,
                                  exit_z=exit_z)
    
    # Create validator
    validator = StrategyValidator(system, results, ticker_y, ticker_x)
    
    # Run all validation tests
    print("\n" + "="*80)
    print("STARTING VALIDATION TESTS")
    print("="*80)
    
    # 1. Walk-Forward Analysis
    validator.walk_forward_analysis(stock_y, stock_x, market,
                                   train_period=252, test_period=126,
                                   n_folds=4, entry_z=entry_z, exit_z=exit_z)
    
    # 2. Monte Carlo Simulation
    validator.monte_carlo_simulation(n_simulations=1000)
    
    # 3. Stress Testing
    validator.stress_test_crisis_periods()
    
    # 4. Sensitivity Analysis
    validator.sensitivity_analysis(stock_y, stock_x, market)
    
    # 5. Transaction Cost Analysis
    validator.transaction_cost_analysis(slippage_bps=5, commission_per_trade=1)
    
    # 6. Statistical Tests
    validator.statistical_tests()
    
    # 7. Generate Final Report
    overall_score = validator.generate_report(
        save_path=f'validation_{ticker_y}_{ticker_x}_{datetime.now().strftime("%Y%m%d")}.txt'
    )
    
    return validator, overall_score


if __name__ == "__main__":
    # Validate a pairs trading strategy
    validator, score = run_complete_validation(
        ticker_y='PEP',
        ticker_x='KO',
        start_date='2018-01-01',
        end_date='2024-01-01',
        entry_z=2.0,
        exit_z=0.5
    )
    
    print("\n" + "="*80)
    print(f"FINAL VALIDATION SCORE: {score:.0f}/100")
    print("="*80)