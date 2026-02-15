#!/bin/bash

cd /Users/hardik/PBL_Run

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
