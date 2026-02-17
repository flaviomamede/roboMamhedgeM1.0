from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from backtest_framework.models import Side, Signal, Tick


class Strategy(ABC):
    @abstractmethod
    def on_tick(self, tick: Tick) -> Signal | None:
        raise NotImplementedError


class MomentumTicksStrategy(Strategy):
    """Estratégia exemplo: compra quando preço rompe máxima recente e vende na mínima."""

    def __init__(self, lookback: int = 20, quantity: int = 1) -> None:
        self.lookback = lookback
        self.quantity = quantity
        self.window: deque[float] = deque(maxlen=lookback)

    def on_tick(self, tick: Tick) -> Signal | None:
        if len(self.window) < self.lookback:
            self.window.append(tick.price)
            return None

        max_price = max(self.window)
        min_price = min(self.window)

        signal = None
        if tick.price > max_price:
            signal = Signal(timestamp=tick.timestamp, side=Side.BUY, quantity=self.quantity)
        elif tick.price < min_price:
            signal = Signal(timestamp=tick.timestamp, side=Side.SELL, quantity=self.quantity)

        self.window.append(tick.price)
        return signal
