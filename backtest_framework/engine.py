from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from backtest_framework.costs import B3CostModel
from backtest_framework.models import BacktestConfig, BacktestResult, Fill, Position, Side, Tick, Trade
from backtest_framework.slippage import SlippageModel
from backtest_framework.strategy import Strategy


@dataclass(slots=True)
class _OpenLot:
    entry_fill: Fill


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        slippage_model: SlippageModel | None = None,
        cost_model: B3CostModel | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.slippage_model = slippage_model or SlippageModel()
        self.cost_model = cost_model or B3CostModel()
        self.config = config or BacktestConfig()

    def run(self, ticks: Iterable[Tick]) -> BacktestResult:
        realized_pnl = 0.0
        position = Position()
        open_lot: _OpenLot | None = None
        trades: list[Trade] = []
        total_costs = 0.0
        equity_curve: list[tuple[datetime, float]] = []
        last_tick: Tick | None = None

        for tick in ticks:
            last_tick = tick
            signal = self.strategy.on_tick(tick)

            if signal:
                if position.quantity == 0:
                    entry_fill = self._execute_order(tick, signal.side, signal.quantity)
                    total_costs += entry_fill.costs
                    open_lot = _OpenLot(entry_fill=entry_fill)
                    position = Position(quantity=signal.quantity, average_price=entry_fill.price, side=signal.side)
                elif position.side != signal.side:
                    assert open_lot is not None
                    exit_fill = self._execute_order(tick, signal.side, position.quantity)
                    total_costs += exit_fill.costs
                    trade = self._close_trade(open_lot.entry_fill, exit_fill)
                    trades.append(trade)
                    realized_pnl += trade.net_pnl

                    remaining_qty = signal.quantity - position.quantity
                    if remaining_qty > 0:
                        new_entry = self._execute_order(tick, signal.side, remaining_qty)
                        total_costs += new_entry.costs
                        open_lot = _OpenLot(entry_fill=new_entry)
                        position = Position(quantity=remaining_qty, average_price=new_entry.price, side=signal.side)
                    else:
                        open_lot = None
                        position = Position()

            equity = self.config.initial_capital + realized_pnl + self._mark_to_market(position, tick.price)
            equity_curve.append((tick.timestamp, equity))

        if last_tick is not None and open_lot is not None and position.side is not None:
            closing_side = Side.SELL if position.side == Side.BUY else Side.BUY
            exit_fill = self._execute_order(last_tick, closing_side, position.quantity)
            total_costs += exit_fill.costs
            trade = self._close_trade(open_lot.entry_fill, exit_fill)
            trades.append(trade)
            realized_pnl += trade.net_pnl
            equity_curve[-1] = (last_tick.timestamp, self.config.initial_capital + realized_pnl)

        gross_profit = sum(t.gross_pnl for t in trades)
        net_profit = sum(t.net_pnl for t in trades)
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            gross_profit=gross_profit,
            total_costs=total_costs,
            net_profit=net_profit,
        )

    def _execute_order(self, tick: Tick, side: Side, quantity: int) -> Fill:
        fill_price = self.slippage_model.apply(tick.price, side)
        gross_value = fill_price * quantity
        costs = self.cost_model.calculate(gross_value, quantity=quantity)
        return Fill(
            timestamp=tick.timestamp,
            side=side,
            quantity=quantity,
            price=fill_price,
            gross_value=gross_value,
            costs=costs,
        )

    @staticmethod
    def _close_trade(entry: Fill, exit_fill: Fill) -> Trade:
        side = entry.side
        direction = 1 if side == Side.BUY else -1
        gross_pnl = direction * (exit_fill.price - entry.price) * entry.quantity
        costs = entry.costs + exit_fill.costs
        net_pnl = gross_pnl - costs
        return Trade(
            entry_time=entry.timestamp,
            exit_time=exit_fill.timestamp,
            quantity=entry.quantity,
            entry_price=entry.price,
            exit_price=exit_fill.price,
            side=side,
            gross_pnl=gross_pnl,
            costs=costs,
            net_pnl=net_pnl,
        )

    @staticmethod
    def _mark_to_market(position: Position, last_price: float) -> float:
        if position.quantity <= 0 or position.side is None:
            return 0.0
        direction = 1 if position.side == Side.BUY else -1
        return direction * (last_price - position.average_price) * position.quantity
