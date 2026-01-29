"""
Configuration for Alpaca Paper Trading System
Copy this to Paper_Trading/config.py and fill in your API keys
"""

from pathlib import Path
from datetime import datetime

# ============================================================================
# ALPACA API CREDENTIALS
# ============================================================================
# Get your API keys from: https://app.alpaca.markets/paper/dashboard/overview
# IMPORTANT: These are PAPER trading keys (not live trading)

ALPACA_API_KEY ="PKXXJHCPU45ZGSIF6DDPL6RXM4"  # Replace with your actual key
ALPACA_SECRET_KEY ="BYYim4tp8TFBiCmiJEN65rA1s8tiroA6RLZHMzA4offv"  # Replace with your actual secret

# Trading mode
PAPER_TRADING = True  # Set to False for live trading (BE CAREFUL!)

# ============================================================================
# VALIDATED PAIRS CONFIGURATION
# ============================================================================
# These are your validated pairs from the strategy analysis
# Format: 'pair_name': {config parameters}

VALIDATED_PAIRS = {
    'SO_SRE': {
        'ticker_y': 'SO',
        'ticker_x': 'SRE',
        'entry_z_normal': 1.5,
        'exit_z_normal': 0.5,
        'entry_z_volatile': 2.0,
        'exit_z_volatile': 1.0,
        'min_hold_days': 3,
        'z_score_window': 30,
        'position_size': 0.20  # Use 20% of portfolio for this pair
    }
}

# Benchmark for regime detection
BENCHMARK_SYMBOL = 'SPY'

# ============================================================================
# RISK MANAGEMENT
# ============================================================================
INITIAL_CAPITAL = 100000  # $100k starting capital
MAX_POSITION_SIZE = 0.25  # Maximum 25% per pair
MAX_DRAWDOWN = 0.15  # 15% maximum drawdown
DAILY_LOSS_LIMIT = 0.05  # 5% daily loss limit

# ============================================================================
# DATA SETTINGS
# ============================================================================
DATA_LOOKBACK_DAYS = 365  # Days of historical data to fetch
MIN_DATA_POINTS = 100  # Minimum data points required for strategy

# ============================================================================
# ORDER EXECUTION
# ============================================================================
ORDER_TYPE = 'market'  # 'market' or 'limit'
TIME_IN_FORCE = 'day'  # 'day', 'gtc', 'ioc', 'fok'
LIMIT_PRICE_OFFSET = 0.02  # 2% offset for limit orders

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

# File paths (relative to project root)
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / 'paper_trading.log'
TRADE_LOG_FILE = LOG_DIR / 'trades.csv'
PERFORMANCE_LOG_FILE = LOG_DIR / 'performance.csv'

# ============================================================================
# SCHEDULING
# ============================================================================
# Run after market close (4:00 PM ET = 9:00 PM UTC)
TRADING_SCHEDULE = {
    'hour': 21,  # 9 PM UTC = 4 PM ET (after market close)
    'minute': 0,
    'timezone': 'UTC'
}

# ============================================================================
# MONITORING
# ============================================================================
ENABLE_MONITORING = True
MONITORING_INTERVAL = 3600  # Check every hour (in seconds)

# Alert thresholds
ALERT_DRAWDOWN_THRESHOLD = 0.10  # Alert if drawdown > 10%
ALERT_DAILY_LOSS_THRESHOLD = 0.03  # Alert if daily loss > 3%

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config() -> bool:
    """
    Validate configuration settings
    
    Returns:
        True if configuration is valid
    """
    errors = []
    
    # Check API keys
    if ALPACA_API_KEY == "YOUR_ALPACA_API_KEY_HERE":
        errors.append("❌ ALPACA_API_KEY not set! Please update config.py")
    
    if ALPACA_SECRET_KEY == "YOUR_ALPACA_SECRET_KEY_HERE":
        errors.append("❌ ALPACA_SECRET_KEY not set! Please update config.py")
    
    # Check pairs
    if not VALIDATED_PAIRS:
        errors.append("❌ No validated pairs configured!")
    
    # Check position sizes
    total_position_size = sum(p['position_size'] for p in VALIDATED_PAIRS.values())
    if total_position_size > 1.0:
        errors.append(f"⚠️  Total position size ({total_position_size:.0%}) exceeds 100%")
    
    # Check risk parameters
    if MAX_DRAWDOWN <= 0 or MAX_DRAWDOWN >= 1:
        errors.append("❌ MAX_DRAWDOWN must be between 0 and 1")
    
    if DAILY_LOSS_LIMIT <= 0 or DAILY_LOSS_LIMIT >= 1:
        errors.append("❌ DAILY_LOSS_LIMIT must be between 0 and 1")
    
    # Print errors
    if errors:
        print("\n❌ Configuration Validation Failed:")
        for error in errors:
            print(f"  {error}")
        print("\nPlease fix the issues above before running the system.\n")
        return False
    
    print("✅ Configuration validation passed")
    return True


def print_config_summary():
    """Print configuration summary"""
    
    print("\n" + "="*80)
    print("⚙️  CONFIGURATION SUMMARY")
    print("="*80)
    
    print(f"\n🔑 API Configuration:")
    print(f"  Paper Trading: {'✅ ENABLED' if PAPER_TRADING else '⚠️  LIVE TRADING'}")
    print(f"  API Key: {'✅ Set' if ALPACA_API_KEY != 'YOUR_ALPACA_API_KEY_HERE' else '❌ Not Set'}")
    print(f"  Secret Key: {'✅ Set' if ALPACA_SECRET_KEY != 'YOUR_ALPACA_SECRET_KEY_HERE' else '❌ Not Set'}")
    
    print(f"\n📊 Trading Pairs ({len(VALIDATED_PAIRS)}):")
    for pair_name, config in VALIDATED_PAIRS.items():
        print(f"  {pair_name}:")
        print(f"    Tickers: {config['ticker_y']} / {config['ticker_x']}")
        print(f"    Position Size: {config['position_size']:.1%}")
        print(f"    Entry Z (Normal): ±{config['entry_z_normal']}")
        print(f"    Exit Z (Normal): ±{config['exit_z_normal']}")
    
    print(f"\n💰 Risk Management:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Max Position Size: {MAX_POSITION_SIZE:.0%}")
    print(f"  Max Drawdown: {MAX_DRAWDOWN:.0%}")
    print(f"  Daily Loss Limit: {DAILY_LOSS_LIMIT:.0%}")
    
    print(f"\n📈 Data Settings:")
    print(f"  Lookback Days: {DATA_LOOKBACK_DAYS}")
    print(f"  Min Data Points: {MIN_DATA_POINTS}")
    print(f"  Benchmark: {BENCHMARK_SYMBOL}")
    
    print(f"\n📝 Logging:")
    print(f"  Log Level: {LOG_LEVEL}")
    print(f"  Log File: {LOG_FILE}")
    print(f"  Trade Log: {TRADE_LOG_FILE}")
    
    print(f"\n⏰ Scheduling:")
    print(f"  Run Time: {TRADING_SCHEDULE['hour']:02d}:{TRADING_SCHEDULE['minute']:02d} {TRADING_SCHEDULE['timezone']}")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# RUNTIME CHECKS
# ============================================================================
if __name__ == "__main__":
    # If config.py is run directly, validate and print summary
    if validate_config():
        print_config_summary()