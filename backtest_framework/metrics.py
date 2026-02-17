from __future__ import annotations

import math
from dataclasses import dataclass

from backtest_framework.models import BacktestResult


@dataclass(slots=True)
class PerformanceMetrics:
    sharpe_ratio: float
    max_drawdown: float
    net_profit: float
    gross_profit: float
    total_costs: float
    win_rate: float
    profit_factor: float


def compute_metrics(result: BacktestResult, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> PerformanceMetrics:
    returns = _equity_returns(result.equity_curve)
    sharpe = _sharpe_ratio(returns, risk_free_rate, periods_per_year)
    max_dd = _max_drawdown(result.equity_curve)

    wins = [t.net_pnl for t in result.trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in result.trades if t.net_pnl < 0]
    win_rate = (len(wins) / len(result.trades)) if result.trades else 0.0

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else math.inf

    return PerformanceMetrics(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        net_profit=result.net_profit,
        gross_profit=result.gross_profit,
        total_costs=result.total_costs,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


def _equity_returns(curve: list[tuple]) -> list[float]:
    if len(curve) < 2:
        return []
    values = [v for _, v in curve]
    out: list[float] = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]
        if prev != 0:
            out.append((curr - prev) / prev)
    return out


def _sharpe_ratio(returns: list[float], risk_free_rate: float, periods_per_year: int) -> float:
    if not returns:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = [r - rf_per_period for r in returns]
    mean = sum(excess) / len(excess)
    variance = sum((r - mean) ** 2 for r in excess) / max(len(excess) - 1, 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def _max_drawdown(curve: list[tuple]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for _, equity in curve:
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
    return max_dd
