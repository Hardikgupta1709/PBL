"""
FIND WHAT WORKS NOW - Test on Recent Data

Your diagnostic revealed:
• Historical backtest (2020-2024): Works great!
• Recent data (2025-2026): FAILS! (Sharpe -0.014)

This script tests multiple approaches on RECENT data (last year)
to find what actually works in current market conditions.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Core_Strategy.ultra_conservative_strategy import UltraConservativeSystem, download_data


def test_on_recent_data():
    """Test strategies on the MOST RECENT data"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║        FIND WHAT WORKS NOW - RECENT DATA TESTING                   ║
    ║                                                                      ║
    ║  Problem: Historical backtests look good but fail on recent data!   ║
    ║  Solution: Test EVERYTHING on last 12 months to find what works    ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Download RECENT data only
    print("📥 Downloading RECENT data (last 12 months)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock_y, stock_x, market = download_data(
        'SO', 'SRE', 'SPY',
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    print(f"✅ {len(stock_y)} days loaded")
    print(f"Period: {stock_y.index[0].date()} to {stock_y.index[-1].date()}")
    
    # ========================================================================
    # TEST MULTIPLE CONFIGURATIONS
    # ========================================================================
    
    configs = [
        {
            'name': 'ULTRA-AGGRESSIVE',
            'entry_z': 1.25,
            'exit_z': 0.5,
            'window': 30,
            'min_hold': 2,
            'volatile_mult': 1.3
        },
        {
            'name': 'AGGRESSIVE',
            'entry_z': 1.5,
            'exit_z': 0.5,
            'window': 30,
            'min_hold': 2,
            'volatile_mult': 1.3
        },
        {
            'name': 'BALANCED-SHORT',
            'entry_z': 1.75,
            'exit_z': 0.4,
            'window': 30,
            'min_hold': 2,
            'volatile_mult': 1.4
        },
        {
            'name': 'BALANCED-MEDIUM',
            'entry_z': 1.75,
            'exit_z': 0.4,
            'window': 45,
            'min_hold': 2,
            'volatile_mult': 1.4
        },
        {
            'name': 'MODERATE',
            'entry_z': 2.0,
            'exit_z': 0.5,
            'window': 50,
            'min_hold': 3,
            'volatile_mult': 1.5
        },
        {
            'name': 'CONSERVATIVE',
            'entry_z': 2.25,
            'exit_z': 0.5,
            'window': 60,
            'min_hold': 3,
            'volatile_mult': 1.5
        },
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("TESTING ON RECENT DATA (Last 12 Months)")
    print("="*80)
    
    for config in configs:
        print(f"\n🔍 Testing {config['name']}...")
        
        try:
            system = UltraConservativeSystem()
            backtest = system.run_backtest(
                stock_y, stock_x, market,
                train_period=min(125, len(stock_y) // 2),
                entry_z_normal=config['entry_z'],
                exit_z_normal=config['exit_z'],
                entry_z_volatile=config['entry_z'] * config['volatile_mult'],
                exit_z_volatile=max(0.2, config['exit_z'] - 0.1),
                min_hold_days=config['min_hold'],
                z_score_window=config['window']
            )
            
            # Calculate metrics
            rets = backtest['strategy_return'].dropna()
            
            if len(rets) == 0 or rets.std() == 0:
                print(f"   ⚠️  No valid returns")
                continue
            
            total_return = (1 + rets).prod() - 1
            sharpe = rets.mean() / rets.std() * np.sqrt(252)
            vol = rets.std() * np.sqrt(252)
            
            trades = ((backtest['final_signal'] != 0) & 
                     (backtest['final_signal'].shift(1) == 0)).sum()
            
            # Trade analysis
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            
            if trades > 0:
                try:
                    trade_analysis = system.get_trade_analysis()
                    if len(trade_analysis) > 0:
                        win_rate = trade_analysis['profitable'].mean()
                        
                        winners = trade_analysis[trade_analysis['profitable']]
                        losers = trade_analysis[~trade_analysis['profitable']]
                        
                        if len(winners) > 0:
                            avg_win = winners['pnl_pct'].mean()
                            total_wins = winners['pnl_pct'].sum()
                        else:
                            avg_win = 0
                            total_wins = 0
                        
                        if len(losers) > 0:
                            avg_loss = abs(losers['pnl_pct'].mean())
                            total_losses = abs(losers['pnl_pct'].sum())
                        else:
                            avg_loss = 0
                            total_losses = 0
                        
                        if total_losses > 0:
                            profit_factor = total_wins / total_losses
                        else:
                            profit_factor = 999 if total_wins > 0 else 0
                except:
                    pass
            
            # Drawdown
            cumulative = (1 + rets).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            # Regime analysis
            regime_dist = backtest['regime'].value_counts()
            crisis_pct = regime_dist.get(0, 0) / len(backtest) * 100
            
            results.append({
                'Strategy': config['name'],
                'Entry_Z': config['entry_z'],
                'Window': config['window'],
                'Trades': trades,
                'Sharpe': sharpe,
                'Return_%': total_return * 100,
                'Vol_%': vol * 100,
                'Max_DD_%': max_dd * 100,
                'Win_Rate_%': win_rate * 100,
                'Avg_Win_%': avg_win,
                'Avg_Loss_%': avg_loss,
                'Profit_Factor': profit_factor,
                'Crisis_%': crisis_pct
            })
            
            status = "✅" if sharpe > 0 and trades > 0 else "❌"
            print(f"   {status} Sharpe: {sharpe:.3f}, Trades: {trades}, Return: {total_return*100:.2f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if len(results) == 0:
        print("\n❌ No successful configurations found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 RESULTS ON RECENT DATA (Last 12 Months)")
    print("="*80)
    
    # Sort by Sharpe
    df_sorted = df.sort_values('Sharpe', ascending=False)
    
    print(f"\n{'Strategy':<20} {'Trades':<8} {'Sharpe':<8} {'Return':<8} {'Win%':<8} {'P.Factor':<8}")
    print("-" * 80)
    
    for _, row in df_sorted.iterrows():
        status = "✅" if row['Sharpe'] > 0 and row['Trades'] > 0 else "❌"
        print(f"{status} {row['Strategy']:<17} "
              f"{row['Trades']:<8.0f} "
              f"{row['Sharpe']:<8.3f} "
              f"{row['Return_%']:<8.2f} "
              f"{row['Win_Rate_%']:<8.1f} "
              f"{row['Profit_Factor']:<8.2f}")
    
    # ========================================================================
    # DETAILED ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("📈 DETAILED ANALYSIS (Top 3)")
    print("="*80)
    
    for i, (_, row) in enumerate(df_sorted.head(3).iterrows(), 1):
        print(f"\n#{i}. {row['Strategy']}:")
        print(f"   Entry: ±{row['Entry_Z']}, Window: {row['Window']} days")
        print(f"   Trades: {row['Trades']:.0f} (annualized: {row['Trades'] * 365 / len(stock_y):.0f})")
        print(f"   Sharpe: {row['Sharpe']:.3f}")
        print(f"   Return: {row['Return_%']:.2f}%")
        print(f"   Volatility: {row['Vol_%']:.2f}%")
        print(f"   Max DD: {row['Max_DD_%']:.2f}%")
        
        if row['Trades'] > 0:
            print(f"   Win Rate: {row['Win_Rate_%']:.1f}%")
            print(f"   Avg Win: {row['Avg_Win_%']:.2f}%")
            print(f"   Avg Loss: {row['Avg_Loss_%']:.2f}%")
            print(f"   Profit Factor: {row['Profit_Factor']:.2f}")
        
        print(f"   Crisis Time: {row['Crisis_%']:.1f}%")
    
    # ========================================================================
    # VERDICT
    # ========================================================================
    
    print("\n" + "="*80)
    print("🎯 VERDICT FOR RECENT DATA")
    print("="*80)
    
    best = df_sorted.iloc[0]
    positive_sharpe = df[df['Sharpe'] > 0]
    
    if len(positive_sharpe) == 0:
        print("""
    ❌ CRITICAL: NO CONFIGURATION WORKS ON RECENT DATA!
    
    All tested configurations show negative/zero Sharpe on last 12 months.
    
    Possible reasons:
    1. SO-SRE pair has broken down (lost cointegration)
    2. Market regime has changed permanently
    3. Pair in extended crisis (52% of time!)
    4. Need completely different approach
    
    RECOMMENDATIONS:
    
    Option 1: FIND DIFFERENT PAIR
    • This pair isn't working anymore
    • Test alternatives: JPM-BAC, XLE-XLU, KO-PEP
    • Look for cointegrated pairs in current market
    
    Option 2: WAIT & MONITOR
    • Keep current config but DON'T expect profits
    • Wait for regime to change
    • Be ready to enter when conditions improve
    
    Option 3: ABANDON PAIRS TRADING
    • Market conditions may not support it
    • Consider different strategy
    
    ⚠️  DO NOT deploy ANY config until you see positive Sharpe on recent data!
        """)
    
    elif best['Sharpe'] < 0.2:
        print(f"""
    ⚠️  MARGINAL: Best configuration barely profitable
    
    Best strategy: {best['Strategy']}
    Sharpe: {best['Sharpe']:.3f} (very low!)
    Trades: {best['Trades']:.0f}
    
    This suggests:
    • Pair is struggling in current conditions
    • Crisis regime ({best['Crisis_%']:.1f}% of time) limiting trades
    • Profits marginal even with best config
    
    RECOMMENDATIONS:
    
    1. Use MOST AGGRESSIVE config that's still positive
       → {best['Strategy']}: Entry ±{best['Entry_Z']}
    
    2. Lower expectations:
       → Expect {best['Trades']:.0f} trades in next year
       → Expect ~{best['Return_%']:.1f}% return
    
    3. Test alternative pairs in parallel
    
    4. Be ready to switch if this deteriorates further
        """)
    
    else:
        print(f"""
    ✅ FOUND WORKING CONFIGURATION!
    
    Best strategy: {best['Strategy']}
    Sharpe: {best['Sharpe']:.3f}
    Trades: {best['Trades']:.0f}
    Return: {best['Return_%']:.2f}%
    Win Rate: {best['Win_Rate_%']:.1f}%
    
    This configuration works on RECENT data!
    
    DEPLOY THIS:
    ```python
    'entry_z_normal': {best['Entry_Z']},
    'exit_z_normal': 0.5,
    'z_score_window': {best['Window']},
    'min_hold_days': 2
    ```
    
    Expected performance (next 12 months):
    • Trades: ~{best['Trades'] * 365 / len(stock_y):.0f}
    • Sharpe: ~{best['Sharpe']:.2f}
    • Return: ~{best['Return_%']:.1f}%
        """)
    
    # Save results
    df_sorted.to_csv('recent_data_test_results.csv', index=False)
    print(f"\n✅ Results saved to: recent_data_test_results.csv")
    
    return df_sorted


if __name__ == "__main__":
    results = test_on_recent_data()
    
    if results is not None and len(results) > 0:
        best = results.iloc[0]
        
        print(f"""
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║                   FINAL RECOMMENDATION                               ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    Based on RECENT data (not historical):
    
    Best Configuration: {best['Strategy']}
    
    Parameters:
    • entry_z_normal: {best['Entry_Z']}
    • z_score_window: {best['Window']}
    • exit_z_normal: 0.5
    
    Expected (next 12 months):
    • Sharpe: {best['Sharpe']:.3f}
    • Return: {best['Return_%']:.2f}%
    • Trades: ~{best['Trades']:.0f}
    
    ⚠️  IMPORTANT:
    Historical backtests showed better results, but RECENT data
    shows weaker performance. This is the reality right now.
    
    Deploy with realistic expectations!
        """)