from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(slots=True)
class Tick:
    timestamp: datetime
    price: float
    volume: float = 0.0


@dataclass(slots=True)
class Signal:
    timestamp: datetime
    side: Side
    quantity: int


@dataclass(slots=True)
class Fill:
    timestamp: datetime
    side: Side
    quantity: int
    price: float
    gross_value: float
    costs: float


@dataclass(slots=True)
class Trade:
    entry_time: datetime
    exit_time: datetime
    quantity: int
    entry_price: float
    exit_price: float
    side: Side
    gross_pnl: float
    costs: float
    net_pnl: float


@dataclass(slots=True)
class Position:
    quantity: int = 0
    average_price: float = 0.0
    side: Optional[Side] = None


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 50_000.0
    risk_free_rate: float = 0.0
    periods_per_year: int = 252


@dataclass(slots=True)
class BacktestResult:
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    gross_profit: float = 0.0
    total_costs: float = 0.0
    net_profit: float = 0.0
