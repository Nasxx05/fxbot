#!/bin/bash
# Quick setup — install deps, create .env, run bot
# NOTE: MetaTrader 5 only runs on Windows. This script is for reference.

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating .env file..."
cat > .env << 'EOF'
MT5_LOGIN=your_demo_account_number
MT5_PASSWORD=your_demo_account_password
MT5_SERVER=Exness-MT5Trial
TELEGRAM_BOT_TOKEN=placeholder
TELEGRAM_CHAT_ID=placeholder
EOF

echo ""
echo "=== IMPORTANT ==="
echo "MetaTrader 5 requires a Windows machine with the MT5 terminal installed and logged in."
echo "Edit .env with your actual MT5 login, password, and server name."
echo "Server name can be found in MT5 terminal: File > Open Account"
echo ""
echo "To run the bot:  python main.py"
echo "To run backtest: python backtest/run_6month_backtest.py"
