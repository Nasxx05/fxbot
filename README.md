# Forex Trading Bot

Professional automated forex trading bot built with Python, connecting to OANDA's v20 API.

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python main.py`

## Project Structure

- `src/` — Core bot modules (data engine, feature engine, strategy, risk, execution)
- `tests/` — Unit tests
- `config/` — Configuration files
- `backtest/` — Backtesting engine
- `dashboard/` — Streamlit monitoring dashboard
- `data/` — Historical data storage
- `logs/` — Log files

## Docker

```bash
docker-compose up --build
```
