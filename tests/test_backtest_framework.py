from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

# Permite rodar via comando direto:
#   python tests/test_backtest_framework.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest_framework.engine import BacktestEngine
from backtest_framework.metrics import compute_metrics
from backtest_framework.models import Side, Signal, Tick
from backtest_framework.strategy import Strategy


class FlipStrategy(Strategy):
    def __init__(self) -> None:
        self.count = 0

    def on_tick(self, tick: Tick) -> Signal | None:
        self.count += 1
        if self.count == 1:
            return Signal(timestamp=tick.timestamp, side=Side.BUY, quantity=1)
        if self.count == 2:
            return Signal(timestamp=tick.timestamp, side=Side.SELL, quantity=1)
        return None


class BacktestFrameworkTests(unittest.TestCase):
    def test_engine_executes_and_closes_trade(self) -> None:
        t0 = datetime(2025, 1, 2, 9, 0)
        ticks = [
            Tick(timestamp=t0, price=100.0),
            Tick(timestamp=t0 + timedelta(seconds=1), price=103.0),
            Tick(timestamp=t0 + timedelta(seconds=2), price=103.0),
        ]

        engine = BacktestEngine(strategy=FlipStrategy())
        result = engine.run(ticks)

        self.assertEqual(len(result.trades), 1)
        self.assertGreater(result.trades[0].gross_pnl, 0)
        self.assertAlmostEqual(result.net_profit, result.trades[0].net_pnl)

    def test_metrics_compute_expected_fields(self) -> None:
        t0 = datetime(2025, 1, 2, 9, 0)
        ticks = [
            Tick(timestamp=t0, price=100.0),
            Tick(timestamp=t0 + timedelta(seconds=1), price=101.0),
            Tick(timestamp=t0 + timedelta(seconds=2), price=102.0),
        ]
        engine = BacktestEngine(strategy=FlipStrategy())
        result = engine.run(ticks)

        metrics = compute_metrics(result)
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertGreaterEqual(metrics.max_drawdown, 0.0)
        self.assertAlmostEqual(metrics.net_profit, result.net_profit)


if __name__ == "__main__":
    unittest.main()
