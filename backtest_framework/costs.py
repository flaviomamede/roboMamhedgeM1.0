from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class B3Costs:
    brokerage_per_order: float = 0.0
    exchange_fee_rate: float = 0.00025
    registration_fee_rate: float = 0.00005
    emoluments_rate: float = 0.00003
    iss_rate: float = 0.05


class B3CostModel:
    """Modelo simplificado de custos para operações de DayTrade na B3."""

    def __init__(self, config: B3Costs | None = None) -> None:
        self.config = config or B3Costs()

    def calculate(self, gross_value: float) -> float:
        variable_fees = gross_value * (
            self.config.exchange_fee_rate
            + self.config.registration_fee_rate
            + self.config.emoluments_rate
        )
        iss = self.config.brokerage_per_order * self.config.iss_rate
        return variable_fees + self.config.brokerage_per_order + iss
