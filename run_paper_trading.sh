#!/bin/bash

cd /Users/hardik/PBL_Run

# Alpaca API credentials — keep this file out of version control!
export ALPACA_API_KEY="PKXXJHCPU45ZGSIF6DDPL6RXM4"
export ALPACA_SECRET_KEY="BYYim4tp8TFBiCmiJEN65rA1s8tiroA6RLZHMzA4offv"

# Log start
echo "[$(date)] Starting paper trading update..." >> paper_trading_log.txt

# Run paper trading using the Python runner
/Users/hardik/PBL_Run/venv/bin/python run_paper_trading.py --trade >> paper_trading_log.txt 2>&1

# Check exit code
if [ $? -eq 0 ]; then
    echo "[$(date)]  Update completed successfully" >> paper_trading_log.txt
else
    echo "[$(date)]  Update failed" >> paper_trading_log.txt
fi
