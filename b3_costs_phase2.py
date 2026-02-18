"""
Custos realistas (Fase 2) usando o modelo do `backtest_framework`.

Notas:
- Para o Mini-Índice (WIN), 1 ponto ≈ R$ 0,20 por contrato.
- O `B3CostModel` calcula custos por ORDEM com:
  corretagem + ISS sobre corretagem + taxas variáveis (rate × valor bruto).

Este módulo existe para evitar que a Fase 2 dependa do `utils_fuso.py`
(que fica restrito à Fase 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from backtest_framework.costs import B3CostModel, B3Costs


WIN_POINT_VALUE_BRL = 0.20


@dataclass(frozen=True, slots=True)
class TradePoints:
    entry_price_points: float
    exit_price_points: float
    quantity: int = 1

    @property
    def gross_pnl_points(self) -> float:
        return (self.exit_price_points - self.entry_price_points) * self.quantity

    @property
    def gross_pnl_brl(self) -> float:
        return (self.exit_price_points - self.entry_price_points) * WIN_POINT_VALUE_BRL * self.quantity

    @property
    def entry_gross_value_brl(self) -> float:
        return self.entry_price_points * WIN_POINT_VALUE_BRL * self.quantity

    @property
    def exit_gross_value_brl(self) -> float:
        return self.exit_price_points * WIN_POINT_VALUE_BRL * self.quantity


def default_b3_cost_model() -> B3CostModel:
    """Config padrão para WIN day trade: custo fixo por contrato/ordem."""
    return B3CostModel(
        B3Costs(
            pricing_model="futures_per_contract",
            fixed_fee_per_contract=0.30,
            brokerage_per_order=0.0,
            iss_rate=0.05,
        )
    )


def trade_costs_brl(trade: TradePoints, cost_model: B3CostModel) -> float:
    """Custo total (entrada+saída) em R$ para um trade."""
    return cost_model.calculate(trade.entry_gross_value_brl, quantity=trade.quantity) + cost_model.calculate(
        trade.exit_gross_value_brl, quantity=trade.quantity
    )


def trade_net_pnl_brl(trade: TradePoints, cost_model: B3CostModel) -> float:
    """P&L líquido (já com custos) em R$."""
    return trade.gross_pnl_brl - trade_costs_brl(trade, cost_model)

