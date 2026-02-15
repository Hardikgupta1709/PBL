import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import seaborn as sns
    sns.set_style('whitegrid')
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import config


class TradingMonitor:
    
    def __init__(
        self,
        log_file: str = None,
        backtest_sharpe: float = 0.30,
        backtest_return: float = 6.24
    ):
        self.log_file = Path(log_file or config.TRADE_LOG_FILE)
        self.backtest_sharpe = backtest_sharpe
        self.backtest_return = backtest_return
        
        # Get from config if available
        if config.VALIDATED_PAIRS:
            first_pair = list(config.VALIDATED_PAIRS.values())[0]
            self.backtest_sharpe = first_pair.get('backtest_sharpe', backtest_sharpe)
            self.backtest_return = first_pair.get('backtest_return', backtest_return)
    
    def load_trades(self) -> pd.DataFrame:
        try:
            if not self.log_file.exists():
                print(f" Trade log not found: {self.log_file}")
                return None
            
            trades = pd.read_csv(self.log_file)
            
            if len(trades) == 0:
                print(" Trade log is empty")
                return None
            
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            trades['date'] = trades['timestamp'].dt.date
            
            return trades
            
        except Exception as e:
            print(f" Error loading trades: {e}")
            return None
    
    def calculate_metrics(self, trades: pd.DataFrame) -> dict:

        if trades is None or len(trades) == 0:
            return None
        
        # Time metrics
        start_date = trades['timestamp'].min()
        end_date = trades['timestamp'].max()
        days_active = (end_date - start_date).days
        days_remaining = max(0, 90 - days_active)
        
        # Signal metrics
        total_signals = len(trades)
        long_signals = (trades['signal'] == 1).sum()
        short_signals = (trades['signal'] == -1).sum()
        flat_signals = (trades['signal'] == 0).sum()
        
        # Z-score metrics
        avg_z = trades['z_score'].abs().mean()
        max_z = trades['z_score'].abs().max()
        min_z = trades['z_score'].abs().min()
        
        # Confidence metrics
        if 'confidence' in trades.columns:
            avg_confidence = trades['confidence'].mean()
            high_confidence_pct = (trades['confidence'] > 1.5).sum() / len(trades)
        else:
            avg_confidence = None
            high_confidence_pct = None
        
        # Regime distribution
        regime_dist = trades['regime'].value_counts()
        
        # Trading frequency
        unique_dates = trades['date'].nunique()
        signals_per_day = total_signals / max(unique_dates, 1)
        
        # Active trading percentage
        active_pct = (trades['signal'] != 0).sum() / total_signals if total_signals > 0 else 0
        
        metrics = {
            'start_date': start_date,
            'end_date': end_date,
            'days_active': days_active,
            'days_remaining': days_remaining,
            'progress_pct': (days_active / 90) * 100,
            'total_signals': total_signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'flat_signals': flat_signals,
            'long_pct': (long_signals / total_signals * 100) if total_signals > 0 else 0,
            'short_pct': (short_signals / total_signals * 100) if total_signals > 0 else 0,
            'flat_pct': (flat_signals / total_signals * 100) if total_signals > 0 else 0,
            'avg_z_score': avg_z,
            'max_z_score': max_z,
            'min_z_score': min_z,
            'avg_confidence': avg_confidence,
            'high_confidence_pct': high_confidence_pct,
            'regime_dist': regime_dist,
            'signals_per_day': signals_per_day,
            'active_pct': active_pct * 100,
            'unique_dates': unique_dates
        }
        
        return metrics
    
    def calculate_pair_metrics(self, trades: pd.DataFrame) -> dict:
        pair_metrics = {}
        
        for pair in trades['pair'].unique():
            pair_trades = trades[trades['pair'] == pair]
            
            metrics = {
                'total_signals': len(pair_trades),
                'long': (pair_trades['signal'] == 1).sum(),
                'short': (pair_trades['signal'] == -1).sum(),
                'flat': (pair_trades['signal'] == 0).sum(),
                'active_pct': ((pair_trades['signal'] != 0).sum() / len(pair_trades) * 100),
                'avg_z': pair_trades['z_score'].abs().mean(),
                'max_z': pair_trades['z_score'].abs().max(),
                'avg_hedge_ratio': pair_trades['hedge_ratio'].mean() if 'hedge_ratio' in pair_trades.columns else None
            }
            
            if 'confidence' in pair_trades.columns:
                metrics['avg_confidence'] = pair_trades['confidence'].mean()
            
            pair_metrics[pair] = metrics
        
        return pair_metrics
    
    def generate_report(self):
        
        print("="*10)
        print(f" PAPER TRADING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*10)
        
        # Loading of  trades
        trades = self.load_trades()
        
        if trades is None:
            print("\n  No trading data available yet")
            print("Paper trading system needs to run for at least 1 day")
            return
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades)
        pair_metrics = self.calculate_pair_metrics(trades)
        
        print(f"\n Trading Period:")
        print(f"  Start: {metrics['start_date'].strftime('%Y-%m-%d')}")
        print(f"  End: {metrics['end_date'].strftime('%Y-%m-%d')}")
        print(f"  Days Active: {metrics['days_active']}")
        print(f"  Days Remaining: {metrics['days_remaining']}")
        print(f"  Progress: {metrics['progress_pct']:.1f}%")
        print(f"  Unique Trading Days: {metrics['unique_dates']}")
        
        # SIGNAL ACTIVITY
        print(f"\n📊 Signal Activity:")
        print(f"  Total Signals: {metrics['total_signals']}")
        print(f"  Signals/Day: {metrics['signals_per_day']:.2f}")
        print(f"  Long: {metrics['long_signals']} ({metrics['long_pct']:.1f}%)")
        print(f"  Short: {metrics['short_signals']} ({metrics['short_pct']:.1f}%)")
        print(f"  Flat: {metrics['flat_signals']} ({metrics['flat_pct']:.1f}%)")
        print(f"  Active Time: {metrics['active_pct']:.1f}%")
        
        # ENTRY QUALITY
        print(f"\n Entry Quality:")
        print(f"  Avg |Z-score|: {metrics['avg_z_score']:.2f}")
        print(f"  Max |Z-score|: {metrics['max_z_score']:.2f}")
        print(f"  Min |Z-score|: {metrics['min_z_score']:.2f}")
        
        if metrics['avg_confidence'] is not None:
            print(f"  Avg Confidence: {metrics['avg_confidence']:.2f}")
            print(f"  High Confidence Trades: {metrics['high_confidence_pct']*100:.1f}%")
        
        print(f"\n Performance by Pair:")
        
        for pair, pm in pair_metrics.items():
            print(f"\n  {pair}:")
            print(f"    Signals: {pm['total_signals']}")
            print(f"    Long/Short/Flat: {pm['long']}/{pm['short']}/{pm['flat']}")
            print(f"    Active: {pm['active_pct']:.1f}%")
            print(f"    Avg |Z|: {pm['avg_z']:.2f} (Max: {pm['max_z']:.2f})")
            
            if pm['avg_hedge_ratio']:
                print(f"    Avg Hedge Ratio: {pm['avg_hedge_ratio']:.3f}")
            
            if 'avg_confidence' in pm:
                print(f"    Avg Confidence: {pm['avg_confidence']:.2f}")
        
        # REGIME DISTRIBUTION
        print(f"\n  Market Regime Distribution:")
        
        for regime, count in metrics['regime_dist'].items():
            pct = (count / metrics['total_signals']) * 100
            print(f"  {regime}: {count} ({pct:.1f}%)")
        
        print(f"\n Status:")
        
        if metrics['days_active'] < 7:
            status = "JUST STARTED"
        elif metrics['days_active'] < 30:
            status = "EARLY STAGE"
        elif metrics['days_active'] < 60:
            status = "ON TRACK"
        else:
            status = "FINAL STRETCH"
        
        print(f"  {metrics['progress_pct']:.1f}% complete ({metrics['days_active']}/{90} days)")
        
        # ALERTS 

        print(f"\n  Alerts & Recommendations:")
        
        alerts = []
        
        # Check entry quality
        if metrics['avg_z_score'] < 1.5:
            alerts.append("  Low average Z-scores - entries may be too conservative")
        
        if metrics['avg_z_score'] > 3.5:
            alerts.append("  Very high Z-scores - may be overtrading")
        
        # Check activity
        if metrics['flat_pct'] > 90:
            alerts.append("  Too much time flat (>90%) - strategy not trading enough")
        
        if metrics['active_pct'] < 10:
            alerts.append("  Very low activity - check the generation of signals")
        
        # Check regime
        crisis_pct = 0
        if 'CRISIS' in metrics['regime_dist']:
            crisis_pct = (metrics['regime_dist']['CRISIS'] / metrics['total_signals']) * 100
            if crisis_pct > 50:
                alerts.append(f"ℹ  High crisis regime ({crisis_pct:.1f}%) - defensive behavior is normal")
        
        # Check data frequency
        if metrics['signals_per_day'] < 0.5:
            alerts.append("  Low signal frequency - check if daily updates are running")
        
        # Check balance
        if metrics['long_pct'] > 70 or metrics['short_pct'] > 70:
            alerts.append("  Imbalanced long/short ratio - check for market bias")
        
        if not alerts:
            print("   No issues detected - system operating normally")
        else:
            for alert in alerts:
                print(f"  {alert}")
        
        # COMPARISON FROM BACKTEST
        print(f"\n Expected vs Actual (Backtest Reference):")
        print(f"  Expected Sharpe: {self.backtest_sharpe:.2f}")
        print(f"  Expected Return: {self.backtest_return:.2f}%")
        print(f"  Note: Actual performance requires price data - see account summary")
        
        print("\n")
        print("\n")
        
        # Try to load account performance if available
        try:
            from alpaca.trading.client import TradingClient
            
            trading_client = TradingClient(
                config.ALPACA_API_KEY,
                config.ALPACA_SECRET_KEY,
                paper=config.PAPER_TRADING
            )
            
            account = trading_client.get_account()
            portfolio_value = float(account.portfolio_value)
            total_return = ((portfolio_value - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL) * 100
            
            print(f"\n Current Account Performance:")
            print(f"  Portfolio Value: ${portfolio_value:,.2f}")
            print(f"  Total Return: {total_return:+.2f}%")
            
            if metrics['days_active'] > 30:
                annualized = (total_return / metrics['days_active']) * 365
                print(f"  Annualized (est): {annualized:+.2f}%")
            
            print("\n")
            print("\n")
            print("\n")
            
        except Exception as e:
            pass  
    
    def plot_performance(self, save_path: str = 'paper_trading_monitor.png'):
        trades = self.load_trades()
        
        if trades is None or len(trades) < 2:
            print(" Insufficient data for plotting")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :2])
        
        for pair in trades['pair'].unique():
            pair_data = trades[trades['pair'] == pair]
            ax1.plot(pair_data['timestamp'], pair_data['z_score'], 
                    label=pair, alpha=0.7, linewidth=1.5)
        
        ax1.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Entry threshold')
        ax1.axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Exit threshold')
        ax1.axhline(y=-0.5, color='g', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        
        ax1.set_title('Z-Score Evolution Over Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Z-Score')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        ax2 = fig.add_subplot(gs[0, 2])
        
        signal_counts = trades['signal'].value_counts().sort_index()
        signal_labels = {-1: 'Short', 0: 'Flat', 1: 'Long'}
        colors = {-1: '#ff6b6b', 0: '#95a5a6', 1: '#51cf66'}
        
        bars = ax2.bar(
            [signal_labels.get(s, s) for s in signal_counts.index],
            signal_counts.values,
            color=[colors.get(s, '#3498db') for s in signal_counts.index]
        )
        
        ax2.set_title('Signal Distribution', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        ax3 = fig.add_subplot(gs[1, 0])
        
        regime_counts = trades['regime'].value_counts()
        colors_regime = {'CRISIS': '#e74c3c', 'VOLATILE': '#f39c12', 'NORMAL': '#27ae60'}
        
        ax3.pie(
            regime_counts.values,
            labels=regime_counts.index,
            autopct='%1.1f%%',
            colors=[colors_regime.get(r, '#3498db') for r in regime_counts.index],
            startangle=90
        )
        ax3.set_title('Regime Distribution', fontsize=12, fontweight='bold')
        
        ax4 = fig.add_subplot(gs[1, 1])
        
        ax4.hist(trades['z_score'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        ax4.axvline(x=2.0, color='r', linestyle='--', alpha=0.7, label='Entry')
        ax4.axvline(x=-2.0, color='r', linestyle='--', alpha=0.7)
        ax4.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='Exit')
        ax4.axvline(x=-0.5, color='g', linestyle='--', alpha=0.7)
        
        ax4.set_title('Z-Score Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Z-Score')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 2])
        
        if 'confidence' in trades.columns:
            ax5.scatter(trades['timestamp'], trades['confidence'], 
                       alpha=0.5, c=trades['signal'], cmap='RdYlGn', s=50)
            ax5.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5)
            ax5.set_title('Signal Confidence Over Time', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Confidence')
            ax5.grid(True, alpha=0.3)
            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        else:
            ax5.text(0.5, 0.5, 'Confidence data\nnot available', 
                    ha='center', va='center', fontsize=12)
            ax5.set_title('Signal Confidence', fontsize=12, fontweight='bold')
        
        ax6 = fig.add_subplot(gs[2, :])
        
        # Group by date and count active signals
        daily_activity = trades.groupby('date').agg({
            'signal': lambda x: (x != 0).sum(),
            'z_score': lambda x: x.abs().mean()
        }).reset_index()
        
        ax6_twin = ax6.twinx()
        
        ax6.bar(daily_activity['date'], daily_activity['signal'], 
               alpha=0.6, color='#3498db', label='Active Signals')
        ax6_twin.plot(daily_activity['date'], daily_activity['z_score'], 
                     color='#e74c3c', marker='o', linewidth=2, 
                     label='Avg |Z-Score|', markersize=4)
        
        ax6.set_title('Daily Activity Timeline', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Active Signals', color='#3498db')
        ax6_twin.set_ylabel('Avg |Z-Score|', color='#e74c3c')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='upper left')
        ax6_twin.legend(loc='upper right')
        
        plt.suptitle('Paper Trading Performance Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n Dashboard saved to: {save_path}")
        
        plt.show()


def main():
    monitor = TradingMonitor()
    
    monitor.generate_report()
    try:
        print("\n Generating performance charts")
        monitor.plot_performance()
    except Exception as e:
        print(f"\n  Could not generate charts: {e}")
        print("Charts require at least 2 days of trading data")


if __name__ == "__main__":
    main()