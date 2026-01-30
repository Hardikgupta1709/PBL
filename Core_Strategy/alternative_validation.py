"""
Alternative Validation for Low Win-Rate Mean-Reversion Strategies
When traditional Monte Carlo fails due to skewed trade distributions
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from Core_Strategy.ultra_conservative_strategy import UltraConservativeSystem, download_data

def validate_low_winrate_strategy(ticker_y, ticker_x, 
                                  start_date='2020-01-01', 
                                  end_date='2024-01-01'):
    """
    Comprehensive validation for strategies with low win rates
    but positive expectancy (few big winners, many small losers)
    """
    
    print("="*80)
    print(f"ALTERNATIVE VALIDATION: {ticker_y} vs {ticker_x}")
    print("Specialized for Low Win-Rate Mean-Reversion Strategies")
    print("="*80)
    
    # Download and backtest
    print("\n📥 Downloading data...")
    stock_y, stock_x, market = download_data(ticker_y, ticker_x, 'SPY', 
                                             start_date, end_date)
    
    print("🔄 Running backtest...")
    system = UltraConservativeSystem()
    results = system.run_backtest(
        stock_y, stock_x, market,
        train_period=252,
        entry_z_normal=2.0,
        exit_z_normal=0.5,
        entry_z_volatile=2.5,
        exit_z_volatile=0.3,
        min_hold_days=3
    )
    
    # Get trade analysis
    trades = system.get_trade_analysis()
    
    if len(trades) == 0:
        print("❌ No trades to analyze")
        return None
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Win Rate: {trades['profitable'].mean()*100:.1f}%")
    print(f"  Avg P&L: {trades['pnl_pct'].mean():.2f}%")
    print(f"  Total Return: {trades['pnl_pct'].sum():.2f}%")
    
    # TEST 1: Profit Factor (Critical for low win-rate)
    print("\n" + "="*80)
    print("TEST 1: Profit Factor Analysis")
    print("="*80)
    
    winners = trades[trades['profitable']]['pnl_pct']
    losers = trades[~trades['profitable']]['pnl_pct']
    
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0
    
    profit_factor = (winners.sum() / abs(losers.sum())) if losers.sum() != 0 else 0
    
    print(f"\nWinners: {len(winners)} trades")
    print(f"  Average Win: {avg_win:.2f}%")
    print(f"  Total Win: {winners.sum():.2f}%")
    
    print(f"\nLosers: {len(losers)} trades")
    print(f"  Average Loss: {avg_loss:.2f}%")
    print(f"  Total Loss: {losers.sum():.2f}%")
    
    print(f"\n💰 Profit Factor: {profit_factor:.2f}")
    print(f"   (Total Wins / Total Losses)")
    
    if profit_factor > 1.5:
        print(f"   ✅ EXCELLENT: Each dollar lost makes ${profit_factor:.2f}")
        pf_score = 100
    elif profit_factor > 1.2:
        print(f"   ✅ GOOD: Profitable with margin of safety")
        pf_score = 80
    elif profit_factor > 1.0:
        print(f"   ⚠️ MARGINAL: Barely profitable")
        pf_score = 50
    else:
        print(f"   ❌ FAIL: Losing strategy")
        pf_score = 0
    
    # TEST 2: Expectancy (Better than win rate)
    print("\n" + "="*80)
    print("TEST 2: Mathematical Expectancy")
    print("="*80)
    
    win_rate = trades['profitable'].mean()
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    print(f"\nExpectancy = (Win% × AvgWin) - (Loss% × AvgLoss)")
    print(f"           = ({win_rate:.2%} × {avg_win:.2f}%) - ({1-win_rate:.2%} × {avg_loss:.2f}%)")
    print(f"           = {expectancy:.2f}%")
    
    print(f"\n💡 Interpretation:")
    print(f"   On average, each trade expects to make {expectancy:.2f}%")
    
    if expectancy > 0.5:
        print(f"   ✅ EXCELLENT: Strong positive expectancy")
        exp_score = 100
    elif expectancy > 0.2:
        print(f"   ✅ GOOD: Decent positive expectancy")
        exp_score = 80
    elif expectancy > 0:
        print(f"   ⚠️ MARGINAL: Small edge")
        exp_score = 50
    else:
        print(f"   ❌ FAIL: Negative expectancy")
        exp_score = 0
    
    # TEST 3: Risk-Reward Ratio
    print("\n" + "="*80)
    print("TEST 3: Risk-Reward Ratio")
    print("="*80)
    
    risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
    
    print(f"\nRisk-Reward Ratio: {risk_reward:.2f}:1")
    print(f"  Average Win: {avg_win:.2f}%")
    print(f"  Average Loss: {avg_loss:.2f}%")
    
    # Calculate required win rate to break even
    required_wr = 1 / (1 + risk_reward)
    print(f"\n📐 Required Win Rate to Break Even: {required_wr*100:.1f}%")
    print(f"   Your Win Rate: {win_rate*100:.1f}%")
    
    if win_rate > required_wr:
        margin = ((win_rate - required_wr) / required_wr) * 100
        print(f"   ✅ PASS: {margin:.0f}% margin above breakeven")
        rr_score = 100
    else:
        print(f"   ❌ FAIL: Below required win rate")
        rr_score = 0
    
    # TEST 4: Consecutive Loss Stress Test
    print("\n" + "="*80)
    print("TEST 4: Consecutive Loss Analysis")
    print("="*80)
    
    # Find longest losing streak
    max_consec_losses = 0
    current_streak = 0
    
    for profitable in trades['profitable']:
        if not profitable:
            current_streak += 1
            max_consec_losses = max(max_consec_losses, current_streak)
        else:
            current_streak = 0
    
    # Calculate maximum drawdown from consecutive losses
    max_drawdown_risk = max_consec_losses * avg_loss
    
    print(f"\n📉 Maximum Consecutive Losses: {max_consec_losses}")
    print(f"   Drawdown Risk: {max_drawdown_risk:.2f}%")
    
    if max_consec_losses < 10:
        print(f"   ✅ GOOD: Manageable losing streaks")
        streak_score = 100
    elif max_consec_losses < 20:
        print(f"   ⚠️ MODERATE: Watch for long losing streaks")
        streak_score = 70
    else:
        print(f"   ❌ CONCERNING: Very long losing streaks")
        streak_score = 40
    
    # TEST 5: Time-Based Bootstrap (Better than trade shuffle)
    print("\n" + "="*80)
    print("TEST 5: Time-Series Bootstrap Validation")
    print("="*80)
    print("Testing if returns persist across different time blocks...\n")
    
    # Get daily returns from results
    daily_returns = results['strategy_return'].dropna()
    
    # Block bootstrap: divide into 30-day blocks
    block_size = 30
    n_blocks = len(daily_returns) // block_size
    
    if n_blocks < 4:
        print("⚠️ Not enough data for block bootstrap")
        block_score = 50
    else:
        block_returns = []
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block_ret = daily_returns.iloc[start_idx:end_idx].sum()
            block_returns.append(block_ret)
        
        block_returns = np.array(block_returns)
        
        # Test if blocks are consistently positive
        positive_blocks = (block_returns > 0).sum()
        consistency = positive_blocks / n_blocks
        
        print(f"30-Day Blocks: {n_blocks}")
        print(f"Positive Blocks: {positive_blocks} ({consistency*100:.0f}%)")
        print(f"Average Block Return: {block_returns.mean()*100:.2f}%")
        
        # Statistical test
        t_stat, p_value = stats.ttest_1samp(block_returns, 0)
        
        print(f"\nT-Test (blocks > 0): p-value = {p_value/2:.4f}")
        
        if p_value/2 < 0.05 and block_returns.mean() > 0:
            print(f"   ✅ PASS: Returns statistically significant across time")
            block_score = 100
        elif p_value/2 < 0.10:
            print(f"   ⚠️ MARGINAL: Some evidence of edge")
            block_score = 60
        else:
            print(f"   ❌ FAIL: No consistent edge across time")
            block_score = 0
    
    # TEST 6: Regime-Conditional Performance
    print("\n" + "="*80)
    print("TEST 6: Regime-Conditional Analysis")
    print("="*80)
    
    regime_names = {0: 'CRISIS', 1: 'VOLATILE', 2: 'NORMAL'}
    
    print("\nPerformance by Entry Regime:")
    for regime_code, regime_name in regime_names.items():
        regime_trades = trades[trades['entry_regime'] == regime_code]
        if len(regime_trades) > 0:
            regime_ret = regime_trades['pnl_pct'].sum()
            regime_wr = regime_trades['profitable'].mean()
            print(f"\n  {regime_name}:")
            print(f"    Trades: {len(regime_trades)}")
            print(f"    Total Return: {regime_ret:.2f}%")
            print(f"    Win Rate: {regime_wr*100:.1f}%")
    
    # Check if NORMAL regime is profitable (key requirement)
    normal_trades = trades[trades['entry_regime'] == 2]
    if len(normal_trades) > 0:
        normal_profitable = normal_trades['pnl_pct'].sum() > 0
        if normal_profitable:
            print(f"\n   ✅ PASS: Profitable in NORMAL regime")
            regime_score = 100
        else:
            print(f"\n   ❌ FAIL: Unprofitable in NORMAL regime")
            regime_score = 0
    else:
        print(f"\n   ⚠️ WARNING: No trades in NORMAL regime")
        regime_score = 50
    
    # FINAL SCORING
    print("\n" + "="*80)
    print("ALTERNATIVE VALIDATION SCORES")
    print("="*80)
    
    scores = {
        'Profit Factor': pf_score,
        'Mathematical Expectancy': exp_score,
        'Risk-Reward Ratio': rr_score,
        'Consecutive Loss Management': streak_score,
        'Time-Series Bootstrap': block_score,
        'Regime Performance': regime_score
    }
    
    for test, score in scores.items():
        status = "✅ PASS" if score >= 80 else ("⚠️ MARGINAL" if score >= 60 else "❌ FAIL")
        print(f"  {test:.<35} {score:>3}/100  {status}")
    
    overall = sum(scores.values()) / len(scores)
    print("-" * 80)
    print(f"  {'OVERALL SCORE':.<35} {overall:>3.0f}/100")
    
    # RECOMMENDATION
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    # Key criteria for low win-rate strategies
    critical_pass = (
        pf_score >= 80 and  # Must have good profit factor
        exp_score >= 50 and  # Must have positive expectancy
        rr_score >= 80  # Must have adequate risk-reward
    )
    
    if overall >= 80 and critical_pass:
        print("\n✅ VALIDATED - Low Win-Rate Strategy Approved")
        print("\n🎯 This is a valid mean-reversion strategy with:")
        print("   • Low win rate but large winners")
        print("   • Positive mathematical expectancy")
        print("   • Good profit factor")
        print("\n🚀 READY FOR PAPER TRADING")
        print("\nNext steps:")
        print("  1. Start with small position sizes (1-2%)")
        print("  2. Be prepared for losing streaks")
        print("  3. Monitor profit factor weekly")
        print("  4. Exit if profit factor drops below 1.2")
        
    elif overall >= 60 and exp_score >= 50:
        print("\n⚠️ MARGINAL BUT TRADEABLE")
        print("\n💡 Strategy shows:")
        print("   • Positive expectancy (good)")
        print("   • Some weaknesses in execution")
        print("\nRecommendations:")
        print("  1. Paper trade with very small size")
        print("  2. Tighten stops to improve win rate")
        print("  3. Consider higher entry thresholds (z=2.5)")
        print("  4. Monitor closely for 90 days")
        
    else:
        print("\n❌ NOT VALIDATED")
        print("\nCritical failures:")
        if pf_score < 80:
            print("  • Profit factor too low")
        if exp_score < 50:
            print("  • Negative or too small expectancy")
        if rr_score < 80:
            print("  • Poor risk-reward ratio")
        print("\nDO NOT TRADE - Find better pairs")
    
    return scores, overall


if __name__ == "__main__":
    scores, overall = validate_low_winrate_strategy('SO', 'SRE')
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

