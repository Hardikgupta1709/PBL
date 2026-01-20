"""
Alpaca API Configuration
DO NOT COMMIT THIS FILE TO GIT!
Add to .gitignore
"""

# Alpaca API Credentials
ALPACA_API_KEY ="PKGHMR6IOMAHJFW2X5DXGDO2WF"  # Replace with your actual key
ALPACA_SECRET_KEY ="7dfuuE9i2d5KWvamhdJRcuYss8eUye6hqRwxivwNY7L2"  # Replace with your actual secret

# Alpaca Endpoints
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading

# Trading Configuration
PORTFOLIO_SIZE = 100000  # Starting with $100k paper money
POSITION_SIZE_PCT = 0.10  # Use 10% per pair

# Validated Pairs
VALIDATED_PAIRS = {
    'SO_SRE': {
        'ticker_y': 'SO',
        'ticker_x': 'SRE',
        'entry_z_normal': 2.0,
        'exit_z_normal': 0.5,
        'entry_z_volatile': 2.5,
        'exit_z_volatile': 0.3,
        'min_hold_days': 3,
        'z_score_window': 60,
        'position_size': POSITION_SIZE_PCT,
        'quality_score': 85,
        'backtest_sharpe': 0.30,
        'backtest_profit_factor': 1.51,
        'backtest_win_rate': 0.50
    }
}

# Trading Schedule
TRADING_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
SIGNAL_CHECK_TIME = '16:05'  # 4:05 PM ET (after market close)

# Risk Management
MAX_POSITION_SIZE = 0.20  # Maximum 20% per pair
MAX_TOTAL_EXPOSURE = 0.50  # Maximum 50% total
STOP_LOSS_PCT = 0.15  # Stop loss at 15% drawdown

# Monitoring
LOG_FILE = 'paper_trading_log.txt'
TRADE_LOG_FILE = 'paper_trades_log.csv'
PERFORMANCE_LOG = 'performance_summary.csv'