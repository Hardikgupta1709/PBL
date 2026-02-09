import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

class TradingMonitor:
    
    def __init__(self, log_file='paper_trades_log.csv', 
                 backtest_sharpe=0.30, backtest_return=6.24):
        self.log_file = log_file
        self.backtest_sharpe = backtest_sharpe
        self.backtest_return = backtest_return
    
    def load_trades(self):
        """Load trade log"""
        try:
            trades = pd.read_csv(self.log_file)
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            return trades
        except:
            print("No trades found yet")
            return None
    
    def calculate_metrics(self, trades):
        """Calculate current performance metrics"""
        if trades is None or len(trades) == 0:
            return None
        
        # Get unique days
        trades['date'] = trades['timestamp'].dt.date
        daily_signals = trades.groupby(['date', 'pair']).last()
        
        # Calculate daily returns (simplified - you'll need actual prices)
        # This is a placeholder - integrate with Alpaca account history
        
        metrics = {
            'days_active': (trades['timestamp'].max() - trades['timestamp'].min()).days,
            'total_signals': len(trades),
            'long_signals': (trades['signal'] == 1).sum(),
            'short_signals': (trades['signal'] == -1).sum(),
            'flat_signals': (trades['signal'] == 0).sum(),
            'avg_z_score': trades['z_score'].abs().mean(),
            'max_z_score': trades['z_score'].abs().max()
        }
        
        return metrics
    
    def generate_report(self):
        """Generate weekly monitoring report"""
        trades = self.load_trades()
        
        print("="*80)
        print(f"PAPER TRADING MONITOR - {datetime.now().strftime('%Y-%m-%d')}")
        print("="*80)
        
        if trades is None:
            print("\n⚠️ No trading data available yet")
            print("Paper trading system needs to run for at least 1 day")
            return
        
        metrics = self.calculate_metrics(trades)
        
        print(f"\n📅 Trading Period:")
        print(f"  Start: {trades['timestamp'].min()}")
        print(f"  End: {trades['timestamp'].max()}")
        print(f"  Days Active: {metrics['days_active']}")
        print(f"  Days Remaining: {90 - metrics['days_active']}")
        print(f"  Progress: {(metrics['days_active']/90)*100:.1f}%")
        
        print(f"\n📊 Signal Activity:")
        print(f"  Total Signals: {metrics['total_signals']}")
        print(f"  Long: {metrics['long_signals']} ({(metrics['long_signals']/metrics['total_signals'])*100:.1f}%)")
        print(f"  Short: {metrics['short_signals']} ({(metrics['short_signals']/metrics['total_signals'])*100:.1f}%)")
        print(f"  Flat: {metrics['flat_signals']} ({(metrics['flat_signals']/metrics['total_signals'])*100:.1f}%)")
        
        print(f"\n🎯 Entry Quality:")
        print(f"  Avg |Z-score|: {metrics['avg_z_score']:.2f}")
        print(f"  Max |Z-score|: {metrics['max_z_score']:.2f}")
        
        # By pair
        print(f"\n💼 By Pair:")
        for pair in trades['pair'].unique():
            pair_trades = trades[trades['pair'] == pair]
            print(f"\n  {pair}:")
            print(f"    Signals: {len(pair_trades)}")
            print(f"    Active: {(pair_trades['signal'] != 0).sum()}")
            print(f"    Avg Z: {pair_trades['z_score'].abs().mean():.2f}")
        
        # Regime distribution
        print(f"\n🌡️  Market Regime Distribution:")
        regime_counts = trades['regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} ({(count/len(trades))*100:.1f}%)")
        
        print(f"\n✅ Status: {'ON TRACK' if metrics['days_active'] >= 7 else 'JUST STARTED'}")
        
        # Warnings
        print(f"\n⚠️  Alerts:")
        if metrics['avg_z_score'] < 1.5:
            print("  - Warning: Low average Z-scores (entries may be too conservative)")
        if metrics['flat_signals'] / metrics['total_signals'] > 0.9:
            print("  - Warning: Too much time flat (strategy not trading enough)")
        if 'CRISIS' in regime_counts and regime_counts['CRISIS'] / len(trades) > 0.3:
            print("  - Info: High crisis regime % (this is normal defensive behavior)")
        
        print("\n" + "="*80)
    
    def plot_performance(self):
        """
        Plot performance charts
        """
        trades = self.load_trades()
        if trades is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Z-score over time
        axes[0, 0].plot(trades['timestamp'], trades['z_score'])
        axes[0, 0].axhline(y=2.0, color='r', linestyle='--', label='Entry threshold')
        axes[0, 0].axhline(y=-2.0, color='r', linestyle='--')
        axes[0, 0].axhline(y=0.5, color='g', linestyle='--', label='Exit threshold')
        axes[0, 0].axhline(y=-0.5, color='g', linestyle='--')
        axes[0, 0].set_title('Z-Score Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Signal distribution
        signal_counts = trades['signal'].value_counts()
        axes[0, 1].bar(signal_counts.index, signal_counts.values)
        axes[0, 1].set_title('Signal Distribution')
        axes[0, 1].set_xlabel('Signal (-1=Short, 0=Flat, 1=Long)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Regime distribution
        regime_counts = trades['regime'].value_counts()
        axes[1, 0].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Regime Distribution')
        
        # Z-score histogram
        axes[1, 1].hist(trades['z_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=2.0, color='r', linestyle='--', label='Entry threshold')
        axes[1, 1].axvline(x=-2.0, color='r', linestyle='--')
        axes[1, 1].set_title('Z-Score Distribution')
        axes[1, 1].set_xlabel('Z-Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('paper_trading_monitor.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Charts saved to: paper_trading_monitor.png")
        plt.show()


if __name__ == "__main__":
    monitor = TradingMonitor()
    monitor.generate_report()
    
    # Generate charts if data available
    try:
        monitor.plot_performance()
    except Exception as e:
        print(f"\nCouldn't generate charts yet: {e}")
        print("Charts will be available after 1+ week of trading data")


