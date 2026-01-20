#!/bin/bash

# Navigate to project directory
cd /Users/hardik/PBL_Run/Paper_Trading

# Log start
echo "[$(date)] Starting paper trading update..." >> ../paper_trading_log.txt

# Run paper trading
python -c "
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('.'))))
sys.path.insert(0, '.')

from alpaca_paper_trader import AlpacaPaperTrader

trader = AlpacaPaperTrader()
trader.run_daily_update()
trader.get_performance_summary()
" >> ../paper_trading_log.txt 2>&1

# Check exit code
if [ \$? -eq 0 ]; then
    echo "[$(date)] ✅ Update completed successfully" >> ../paper_trading_log.txt
else
    echo "[$(date)] ❌ Update failed" >> ../paper_trading_log.txt
fi
