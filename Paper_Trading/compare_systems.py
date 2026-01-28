"""
System Comparison Tool - Optimized Version
Compare Ultra-Conservative vs Ultimate Hybrid systems
Run after 90 days to evaluate if upgrade is worthwhile
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def compare_systems(
    ticker_y: str = 'SO',
    ticker_x: str = 'SRE',
    start_date: str = '2020-01-01',
    end_date: str = '2024-01-01'
) -> pd.DataFrame:
    """
    Compare both systems side-by-side with comprehensive analysis
    
    Args:
        ticker_y: First ticker in pair
        ticker_x: Second ticker in pair
        start_date: Backtest start date
        end_date: Backtest end date
        
    Returns:
        DataFrame with comparison metrics
    """
    
    print("="*80)
    print("SYSTEM COMPARISON: Ultra-Conservative vs Ultimate Hybrid")
    print("="*80)
    print(f"\nPair: {ticker_y}/{ticker_x}")
    print(f"Period: {start_date} to {end_date}")
    
    # Import strategies
    try:
        from Core_Strategy.ultra_conservative_strategy import (
            UltraConservativeSystem,
            download_data
        )
    except ImportError:
        print("\n❌ Error: Cannot import Ultra-Conservative strategy")
        print("Make sure Core_Strategy module is in your path")
        return None
    
    try:
        from hybrid_pairs_trading import UltimateHybridSystem
        has_hybrid = True
    except ImportError:
        print("\n⚠️  Warning: Ultimate Hybrid system not found")
        print("Comparison will only show Ultra-Conservative results")
        has_hybrid = False
    
    # ========================================================================
    # Download Data
    # ========================================================================
    print("\n" + "="*80)
    print("📥 DOWNLOADING DATA")
    print("="*80)
    
    try:
        stock_y, stock_x, market = download_data(
            ticker_y, ticker_x, 'SPY',
            start_date, end_date
        )
        
        print(f"✅ Data downloaded successfully")
        print(f"   {ticker_y}: {len(stock_y)} days")
        print(f"   {ticker_x}: {len(stock_x)} days")
        print(f"   SPY: {len(market)} days")
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None
    
    # ========================================================================
    # Test Ultra-Conservative System
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: ULTRA-CONSERVATIVE SYSTEM")
    print("="*80)
    
    try:
        uc_system = UltraConservativeSystem()
        
        print("\nRunning backtest...")
        uc_results = uc_system.run_backtest(
            stock_y, stock_x, market,
            train_period=252,
            entry_z_normal=2.0,
            exit_z_normal=0.5,
            entry_z_volatile=2.5,
            exit_z_volatile=0.3,
            min_hold_days=3,
            z_score_window=60
        )
        
        print("✅ Backtest complete")
        
        uc_metrics = uc_system.get_performance_metrics()
        uc_trades = uc_system.get_trade_analysis()
        
        print("\n📊 Performance Metrics:")
        for key, value in uc_metrics['Hybrid Strategy'].items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error running Ultra-Conservative: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================================================
    # Test Ultimate Hybrid System (if available)
    # ========================================================================
    uh_metrics = None
    uh_trades = None
    uh_results = None
    
    if has_hybrid:
        print("\n" + "="*80)
        print("TEST 2: ULTIMATE HYBRID SYSTEM")
        print("="*80)
        
        try:
            uh_system = UltimateHybridSystem()
            
            print("\nRunning backtest...")
            uh_results = uh_system.run_backtest(
                stock_y, stock_x, market,
                train_period=252,
                entry_z_normal=2.0,
                exit_z_normal=0.5,
                entry_z_volatile=2.5,
                exit_z_volatile=0.3,
                min_hold_days=3,
                min_correlation=0.5,
                use_dynamic_sizing=True,
                use_multi_timeframe=True,
                use_coint_monitoring=True
            )
            
            print("✅ Backtest complete")
            
            uh_metrics = uh_system.get_performance_metrics()
            uh_trades = uh_system.get_trade_analysis()
            
            print("\n📊 Performance Metrics:")
            for key, value in uh_metrics['Ultimate Strategy'].items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
                    
        except Exception as e:
            print(f"❌ Error running Ultimate Hybrid: {e}")
            import traceback
            traceback.print_exc()
            has_hybrid = False
    
    # ========================================================================
    # Performance Comparison
    # ========================================================================
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON")
    print("="*80)
    
    # Build comparison table
    comparison_data = {
        'Ultra-Conservative': {
            'Total Return (%)': uc_metrics['Hybrid Strategy']['Total Return'],
            'Sharpe Ratio': uc_metrics['Hybrid Strategy']['Sharpe Ratio'],
            'Max Drawdown (%)': uc_metrics['Hybrid Strategy']['Max Drawdown'],
            'Win Rate (%)': uc_metrics['Hybrid Strategy']['Win Rate'],
            'Total Trades': uc_metrics['Hybrid Strategy']['Total Trades'],
            'Avg Trade Return (%)': uc_metrics['Hybrid Strategy'].get('Avg Trade Return', 0)
        }
    }
    
    if has_hybrid and uh_metrics:
        comparison_data['Ultimate Hybrid'] = {
            'Total Return (%)': uh_metrics['Ultimate Strategy']['Total Return'],
            'Sharpe Ratio': uh_metrics['Ultimate Strategy']['Sharpe Ratio'],
            'Max Drawdown (%)': uh_metrics['Ultimate Strategy']['Max Drawdown'],
            'Win Rate (%)': uh_metrics['Ultimate Strategy']['Win Rate'],
            'Total Trades': uh_metrics['Ultimate Strategy']['Total Trades'],
            'Avg Trade Return (%)': uh_metrics['Ultimate Strategy'].get('Avg Trade Return', 0)
        }
        
        # Calculate improvement
        comparison_data['Improvement (%)'] = {}
        for metric in comparison_data['Ultra-Conservative'].keys():
            uc_val = comparison_data['Ultra-Conservative'][metric]
            uh_val = comparison_data['Ultimate Hybrid'][metric]
            
            if metric == 'Max Drawdown (%)':
                # For drawdown, lower is better
                improvement = ((uc_val - uh_val) / abs(uc_val)) * 100 if uc_val != 0 else 0
            else:
                improvement = ((uh_val - uc_val) / abs(uc_val)) * 100 if uc_val != 0 else 0
            
            comparison_data['Improvement (%)'][metric] = improvement
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string())
    
    # ========================================================================
    # Trade Analysis
    # ========================================================================
    if len(uc_trades) > 0:
        print("\n" + "="*80)
        print("💼 TRADE ANALYSIS")
        print("="*80)
        
        print(f"\n📈 Ultra-Conservative:")
        print(f"   Total Trades: {len(uc_trades)}")
        print(f"   Win Rate: {uc_trades['profitable'].mean()*100:.1f}%")
        print(f"   Avg P&L: {uc_trades['pnl_pct'].mean():.2f}%")
        print(f"   Max Win: {uc_trades['pnl_pct'].max():.2f}%")
        print(f"   Max Loss: {uc_trades['pnl_pct'].min():.2f}%")
        print(f"   Max Consecutive Losses: {calculate_max_streak(uc_trades)}")
        
        if has_hybrid and uh_trades is not None and len(uh_trades) > 0:
            print(f"\n📈 Ultimate Hybrid:")
            print(f"   Total Trades: {len(uh_trades)}")
            print(f"   Win Rate: {uh_trades['profitable'].mean()*100:.1f}%")
            print(f"   Avg P&L: {uh_trades['pnl_pct'].mean():.2f}%")
            print(f"   Max Win: {uh_trades['pnl_pct'].max():.2f}%")
            print(f"   Max Loss: {uh_trades['pnl_pct'].min():.2f}%")
            print(f"   Max Consecutive Losses: {calculate_max_streak(uh_trades)}")
    
    # ========================================================================
    # Recommendation
    # ========================================================================
    print("\n" + "="*80)
    print("🎯 RECOMMENDATION")
    print("="*80)
    
    if not has_hybrid:
        print("\n✅ CONTINUE WITH ULTRA-CONSERVATIVE")
        print("\nUltimate Hybrid system not available for comparison")
        print("The Ultra-Conservative system is performing well")
    else:
        uc_sharpe = float(uc_metrics['Hybrid Strategy']['Sharpe Ratio'])
        uh_sharpe = float(uh_metrics['Ultimate Strategy']['Sharpe Ratio'])
        
        sharpe_improvement = ((uh_sharpe - uc_sharpe) / abs(uc_sharpe)) * 100 if uc_sharpe != 0 else 0
        
        print(f"\nSharpe Ratio Comparison:")
        print(f"   Ultra-Conservative: {uc_sharpe:.3f}")
        print(f"   Ultimate Hybrid: {uh_sharpe:.3f}")
        print(f"   Improvement: {sharpe_improvement:+.1f}%")
        
        # Decision logic
        if uh_sharpe > uc_sharpe * 1.2:  # 20% improvement
            print("\n✅ UPGRADE RECOMMENDED")
            print(f"\n🚀 Ultimate Hybrid shows significant improvement ({sharpe_improvement:.1f}%)")
            print("   Benefits:")
            print("   • Better risk-adjusted returns")
            print("   • More sophisticated signal generation")
            print("   • Dynamic position sizing")
            print("\n   Next Steps:")
            print("   1. Test Ultimate Hybrid in paper trading for 30 days")
            print("   2. Monitor performance closely")
            print("   3. Compare with current system results")
            
        elif uh_sharpe > uc_sharpe:
            print("\n⚠️  MARGINAL IMPROVEMENT")
            print(f"\n📊 Ultimate Hybrid shows modest improvement ({sharpe_improvement:.1f}%)")
            print("   Consideration:")
            print("   • Improvement may not justify added complexity")
            print("   • Consider testing for validation")
            print("   • Not urgent to switch")
            
        else:
            print("\n❌ STICK WITH ULTRA-CONSERVATIVE")
            print(f"\n📉 Ultimate Hybrid shows no improvement ({sharpe_improvement:.1f}%)")
            print("   Recommendation:")
            print("   • Continue with current system")
            print("   • The simpler system is performing better")
            print("   • No need to add complexity")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    if has_hybrid and uh_results is not None:
        try:
            plot_comparison(uc_results, uh_results, ticker_y, ticker_x)
        except Exception as e:
            print(f"\n⚠️  Could not generate comparison chart: {e}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    return comparison_df


def calculate_max_streak(trades: pd.DataFrame) -> int:
    """
    Calculate maximum consecutive losses
    
    Args:
        trades: DataFrame with trade results
        
    Returns:
        Maximum consecutive loss streak
    """
    max_streak = 0
    current_streak = 0
    
    for profitable in trades['profitable']:
        if not profitable:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def plot_comparison(
    uc_results: pd.DataFrame,
    uh_results: pd.DataFrame,
    ticker_y: str,
    ticker_x: str
):
    """
    Plot comparison charts
    
    Args:
        uc_results: Ultra-Conservative results
        uh_results: Ultimate Hybrid results
        ticker_y: First ticker
        ticker_x: Second ticker
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Cumulative Returns
    axes[0, 0].plot(uc_results.index, uc_results['strategy_cum_returns'], 
                   label='Ultra-Conservative', linewidth=2)
    axes[0, 0].plot(uh_results.index, uh_results['strategy_cum_returns'], 
                   label='Ultimate Hybrid', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Cumulative Returns Comparison')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Drawdown Comparison
    axes[0, 1].fill_between(uc_results.index, 0, uc_results['strategy_drawdown'], 
                           alpha=0.3, label='Ultra-Conservative')
    axes[0, 1].fill_between(uh_results.index, 0, uh_results['strategy_drawdown'], 
                           alpha=0.3, label='Ultimate Hybrid')
    axes[0, 1].set_title('Drawdown Comparison')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Rolling Sharpe
    if 'rolling_sharpe' in uc_results.columns:
        axes[1, 0].plot(uc_results.index, uc_results['rolling_sharpe'], 
                       label='Ultra-Conservative', linewidth=2)
        axes[1, 0].plot(uh_results.index, uh_results['rolling_sharpe'], 
                       label='Ultimate Hybrid', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Rolling Sharpe Ratio (60-day)')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Signal Distribution
    uc_signals = uc_results['final_signal'].value_counts()
    uh_signals = uh_results['final_signal'].value_counts()
    
    x = np.arange(3)
    width = 0.35
    
    axes[1, 1].bar(x - width/2, [uc_signals.get(-1, 0), uc_signals.get(0, 0), uc_signals.get(1, 0)], 
                  width, label='Ultra-Conservative')
    axes[1, 1].bar(x + width/2, [uh_signals.get(-1, 0), uh_signals.get(0, 0), uh_signals.get(1, 0)], 
                  width, label='Ultimate Hybrid')
    
    axes[1, 1].set_title('Signal Distribution')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Short', 'Flat', 'Long'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'System Comparison: {ticker_y}/{ticker_x}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig('system_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison chart saved: system_comparison.png")
    
    plt.show()


def main():
    """Main execution"""
    
    # Run comparison
    comparison = compare_systems('SO', 'SRE')
    
    if comparison is not None:
        # Save results
        comparison.to_csv('system_comparison_results.csv')
        print(f"\n💾 Results saved to: system_comparison_results.csv")


if __name__ == "__main__":
    main()