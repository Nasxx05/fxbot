#!/bin/bash
# Quick setup for GitHub Codespaces — install deps, create .env, run backtest

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating .env file..."
cat > .env << 'EOF'
OANDA_API_KEY=e144ba41-62bc-4ecb-af19-24bdb142f6e1
OANDA_ACCOUNT_ID=101-001-1234567-001
TELEGRAM_BOT_TOKEN=placeholder
TELEGRAM_CHAT_ID=placeholder
EOF

echo "Running 6-month backtest (Jul-Dec 2025)..."
python backtest/run_6month_backtest.py
