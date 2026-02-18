from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtest_framework import (
    B3CostModel,
    B3Costs,
    BacktestConfig,
    BacktestEngine,
    SlippageConfig,
    SlippageModel,
    Tick,
    compute_metrics,
    format_report,
)
from backtest_framework.strategy import MomentumTicksStrategy


def generate_sample_ticks(total: int = 1_000) -> list[Tick]:
    ticks: list[Tick] = []
    t0 = datetime(2025, 1, 2, 9, 0)
    price = 130_000.0
    for i in range(total):
        # micro variação deterministicamente oscilante
        price += (1 if i % 7 < 4 else -1) * 5 + ((i % 13) - 6) * 0.2
        ticks.append(Tick(timestamp=t0 + timedelta(seconds=i), price=price, volume=1.0))
    return ticks


def main() -> None:
    strategy = MomentumTicksStrategy(lookback=25, quantity=1)

    slippage = SlippageModel(
        SlippageConfig(
            fixed_ticks=1.0,
            proportional_rate=0.00002,
            tick_size=5.0,
        )
    )

    costs = B3CostModel(
        B3Costs(
            brokerage_per_order=1.20,
            exchange_fee_rate=0.0002,
            registration_fee_rate=0.00005,
            emoluments_rate=0.00003,
            iss_rate=0.05,
        )
    )

    engine = BacktestEngine(
        strategy=strategy,
        slippage_model=slippage,
        cost_model=costs,
        config=BacktestConfig(initial_capital=100_000, risk_free_rate=0.1175, periods_per_year=252),
    )

    ticks = generate_sample_ticks()
    result = engine.run(ticks)
    metrics = compute_metrics(result, risk_free_rate=0.1175, periods_per_year=252)

    print(format_report(metrics))
    print(f"Trades executados: {len(result.trades)}")


if __name__ == "__main__":
    main()
