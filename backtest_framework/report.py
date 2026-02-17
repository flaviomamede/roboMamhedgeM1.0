from __future__ import annotations

from backtest_framework.metrics import PerformanceMetrics


def format_report(metrics: PerformanceMetrics) -> str:
    return "\n".join(
        [
            "=== Relatório de Performance ===",
            f"Sharpe Ratio      : {metrics.sharpe_ratio:.4f}",
            f"Max Drawdown      : {metrics.max_drawdown:.2%}",
            f"Lucro Bruto       : R$ {metrics.gross_profit:,.2f}",
            f"Custos Totais B3  : R$ {metrics.total_costs:,.2f}",
            f"Lucro Líquido     : R$ {metrics.net_profit:,.2f}",
            f"Win Rate          : {metrics.win_rate:.2%}",
            f"Profit Factor     : {metrics.profit_factor:.4f}",
        ]
    )
