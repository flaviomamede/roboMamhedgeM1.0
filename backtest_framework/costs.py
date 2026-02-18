from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class B3Costs:
    # Modelo de custo:
    # - "equities_ad_valorem": percentual sobre notional (ações)
    # - "futures_per_contract": valor fixo por contrato/ordem (futuros)
    pricing_model: Literal["equities_ad_valorem", "futures_per_contract"] = "equities_ad_valorem"

    # Ações (ad valorem)
    brokerage_per_order: float = 0.0
    exchange_fee_rate: float = 0.00025
    registration_fee_rate: float = 0.00005
    emoluments_rate: float = 0.00003

    # Futuros (fixo por contrato/ordem)
    fixed_fee_per_contract: float = 0.30

    iss_rate: float = 0.05


class B3CostModel:
    """Modelo simplificado de custos para operações de DayTrade na B3."""

    def __init__(self, config: B3Costs | None = None) -> None:
        self.config = config or B3Costs()

    def calculate(
        self,
        gross_value: float,
        quantity: int = 1,
        pricing_model: Literal["equities_ad_valorem", "futures_per_contract"] | None = None,
    ) -> float:
        model = pricing_model or self.config.pricing_model
        if model == "futures_per_contract":
            return self.calculate_futures(quantity=quantity)
        return self.calculate_equities(gross_value=gross_value)

    def calculate_equities(self, gross_value: float) -> float:
        variable_fees = gross_value * (
            self.config.exchange_fee_rate
            + self.config.registration_fee_rate
            + self.config.emoluments_rate
        )
        iss = self.config.brokerage_per_order * self.config.iss_rate
        return variable_fees + self.config.brokerage_per_order + iss

    def calculate_futures(self, quantity: int = 1) -> float:
        # Custo por ORDEM em futuros: taxa fixa por contrato + corretagem + ISS da corretagem.
        qty = max(int(quantity), 1)
        b3_fees = self.config.fixed_fee_per_contract * qty
        brokerage = self.config.brokerage_per_order * qty
        iss = brokerage * self.config.iss_rate
        return b3_fees + brokerage + iss
