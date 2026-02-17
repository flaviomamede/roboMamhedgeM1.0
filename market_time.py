"""
Utilitários de tempo (Fase 2).

Separado de custos propositalmente:
- A Fase 1 usa `fase1_antigravity/utils_fuso.py` (inclui custos simplificados).
- A Fase 2 deve usar custos realistas via `backtest_framework.costs`.
"""

from __future__ import annotations

import pandas as pd


def converter_para_brt(df: pd.DataFrame) -> pd.DataFrame:
    """Converte o index para America/Sao_Paulo (BRT).

    Regras:
    - Se o index não tem fuso (tz-naive), assume que está em UTC.
    - Converte para BRT para aplicar filtros de sessão corretamente.
    """
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/Sao_Paulo")
    else:
        df.index = df.index.tz_convert("America/Sao_Paulo")
    return df


def dentro_horario_operacao(ts) -> bool:
    """Retorna True se o timestamp (em BRT) está no horário permitido.

    Janela atual:
    - 10:45 (inclusive) até 16:30 (exclusive)
    """
    h = ts.hour
    m = ts.minute
    if h < 10 or h >= 17:
        return False
    if h == 10 and m < 45:
        return False
    if h == 16 and m >= 30:
        return False
    return True

