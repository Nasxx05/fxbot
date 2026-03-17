"""Run 6-month backtest on all 4 pairs using historical M15 data."""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from backtest.backtest_engine import BacktestEngine
from src.logger import BotLogger

INSTRUMENTS = ["EUR_USD", "GBP_USD", "GBP_JPY", "XAU_USD"]

# Most recent 6 months of available data
START_DATE = "2025-07-01"
END_DATE = "2025-12-31"

SPREAD_MAP = {
    "EUR_USD": 1.2,
    "GBP_USD": 1.5,
    "GBP_JPY": 2.5,
    "XAU_USD": 3.5,
}


def run_backtests():
    """Run backtests on all 4 pairs for the 6-month period."""
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger = BotLogger(config)
    engine = BacktestEngine(config, logger)

    all_results = {}

    for instrument in INSTRUMENTS:
        print(f"\n{'='*60}")
        print(f"BACKTESTING {instrument} ({START_DATE} to {END_DATE})")
        print(f"{'='*60}")

        spread = SPREAD_MAP.get(instrument, 1.5)
        metrics = engine.run_backtest(instrument, START_DATE, END_DATE,
                                       spread_pips=spread, slippage_pips=0.3)
        engine.print_report(metrics)
        engine.save_results(metrics, f"{instrument.lower()}_6month")
        all_results[instrument] = metrics

    # Print summary comparison
    print(f"\n{'='*70}")
    print("6-MONTH BACKTEST SUMMARY — ALL PAIRS")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"{'='*70}")
    print(f"{'Pair':<12} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Net PnL':>10} {'MaxDD%':>8} {'Exp(R)':>7} {'Sharpe':>7}")
    print("-" * 70)

    for inst in INSTRUMENTS:
        m = all_results[inst]
        print(f"{inst:<12} {m['total_trades']:>7} {m['win_rate']:>6.1f}% {m['profit_factor']:>7.2f} "
              f"${m['total_net_pnl_usd']:>9.2f} {m['max_drawdown_pct']:>7.2f}% "
              f"{m['expectancy_per_trade_r']:>7.2f} {m['sharpe_ratio']:>7.2f}")

    print("=" * 70)

    # Acceptance criteria check
    print(f"\n{'='*70}")
    print("ACCEPTANCE CRITERIA CHECK")
    print(f"{'='*70}")
    for inst in INSTRUMENTS:
        m = all_results[inst]
        checks = []
        checks.append(("Win Rate > 50%", m["win_rate"] > 50))
        checks.append(("Profit Factor > 1.5", m["profit_factor"] > 1.5))
        checks.append(("Max Drawdown < 20%", m["max_drawdown_pct"] < 20))
        checks.append(("Expectancy > 0.3R", m["expectancy_per_trade_r"] > 0.3))
        checks.append(("Min 20 trades", m["total_trades"] >= 20))

        passed = sum(1 for _, v in checks if v)
        status = "PASS" if passed >= 4 else "FAIL"
        print(f"\n{inst}: {passed}/5 criteria met — {status}")
        for name, val in checks:
            mark = "PASS" if val else "FAIL"
            print(f"  [{mark}] {name}")

    # Save combined results
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    combined_path = f"data/backtest_all_pairs_6month_{ts}.json"
    with open(combined_path, "w") as f:
        json.dump({
            "period": {"start": START_DATE, "end": END_DATE},
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("FOREX BOT — 6-MONTH BACKTEST (ALL PAIRS)")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Pairs: {', '.join(INSTRUMENTS)}")
    print(f"Timeframe: M15")
    print("=" * 60)

    run_backtests()
