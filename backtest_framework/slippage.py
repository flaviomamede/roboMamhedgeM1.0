from __future__ import annotations

from dataclasses import dataclass

from backtest_framework.models import Side


@dataclass(slots=True)
class SlippageConfig:
    fixed_ticks: float = 0.0
    proportional_rate: float = 0.0
    tick_size: float = 0.5


class SlippageModel:
    def __init__(self, config: SlippageConfig | None = None) -> None:
        self.config = config or SlippageConfig()

    def apply(self, raw_price: float, side: Side) -> float:
        directional = self.config.fixed_ticks * self.config.tick_size
        proportional = raw_price * self.config.proportional_rate
        impact = directional + proportional
        if side == Side.BUY:
            return raw_price + impact
        return raw_price - impact
