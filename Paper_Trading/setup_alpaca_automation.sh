#!/bin/bash

# ============================================================================
# Setup Daily Paper Trading Automation
# Run this once to set up automatic daily trading
# ============================================================================

echo "🚀 Setting up Alpaca Paper Trading Automation"
echo "="

# Check if config.py has API keys
if grep -q "YOUR_API_KEY_HERE" config.py; then
    echo "❌ ERROR: Please add your Alpaca API keys to config.py first!"
    echo ""
    echo "Steps:"
    echo "1. Open config.py"
    echo "2. Replace YOUR_API_KEY_HERE with your actual key"
    echo "3. Replace YOUR_SECRET_KEY_HERE with your actual secret"
    echo "4. Run this script again"
    exit 1
fi

echo "✅ API keys found in config.py"
echo ""

# Create wrapper script
echo "📝 Creating run script..."
cat > run_paper_trading.sh << 'WRAPPER_EOF'
#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

# Log start
echo "["$(date)"] Starting paper trading update..." >> paper_trading_log.txt

# Run paper trading
python3 alpaca_paper_trader.py >> paper_trading_log.txt 2>&1

# Check exit code
if [ $? -eq 0 ]; then
    echo "["$(date)"] ✅ Update completed successfully" >> paper_trading_log.txt
else
    echo "["$(date)"] ❌ Update failed" >> paper_trading_log.txt
fi
WRAPPER_EOF

chmod +x run_paper_trading.sh

echo "✅ Run script created: run_paper_trading.sh"
echo ""

# Set up cron job
echo "⏰ Setting up daily automation..."
echo ""
echo "The script will run at 4:05 PM ET every weekday (after market close)"
echo ""

# Check current crontab
if crontab -l 2>/dev/null | grep -q "run_paper_trading.sh"; then
    echo "⚠️ Cron job already exists. Skipping..."
else
    # Get current directory
    CURRENT_DIR=$(pwd)
    
    # Create cron entry
    # 4:05 PM ET = 9:05 PM UTC (during EST) or 8:05 PM UTC (during EDT)
    # We'll use 9:05 PM UTC to be safe
    CRON_ENTRY="5 21 * * 1-5 cd $CURRENT_DIR && ./run_paper_trading.sh"
    
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    
    echo "✅ Cron job added successfully"
fi

echo ""
echo "="*80
echo "🎉 SETUP COMPLETE!"
echo "="*80
echo ""
echo "📋 Summary:"
echo "  • Script location: $(pwd)/run_paper_trading.sh"
echo "  • Log file: $(pwd)/paper_trading_log.txt"
echo "  • Trade log: $(pwd)/paper_trades_log.csv"
echo "  • Schedule: Weekdays at 4:05 PM ET"
echo ""
echo "✅ Next Steps:"
echo ""
echo "1. Test manually first:"
echo "   ./run_paper_trading.sh"
echo ""
echo "2. Check logs:"
echo "   tail -f paper_trading_log.txt"
echo ""
echo "3. View trades:"
echo "   cat paper_trades_log.csv"
echo ""
echo "4. Monitor cron jobs:"
echo "   crontab -l"
echo ""
echo "5. To disable automation:"
echo "   crontab -e"
echo "   (then delete the line with run_paper_trading.sh)"
echo ""