"""
Utilitário: converte dados para BRT e aplica filtro de horário corretamente.
Evita confusão UTC vs BRT.

Escala: dados do Mini-Índice (WIN) em PONTOS.
  1 ponto = R$ 0,20 por contrato.

Modelo de custo:
  - P&L dos robôs é em PONTOS PUROS (sem custo).
  - Na exibição: P&L_reais = pnl_pontos × N_COTAS × MULT_PONTOS_REAIS - CUSTO_REAIS
  - CUSTO_REAIS é fixo por round-trip (R$ 2,50), não depende de posição.
"""
# 1 ponto WIN = R$ 0,20
MULT_PONTOS_REAIS = 0.20

# Quantidade de contratos por operação
N_COTAS = 1

# Custo fixo por round-trip em R$ (corretagem + emolumentos + slippage)
CUSTO_REAIS = 2.50


def pnl_reais(pnl_pontos):
    """Converte P&L de pontos para R$ (já com custo)."""
    return pnl_pontos * N_COTAS * MULT_PONTOS_REAIS - CUSTO_REAIS

import pandas as pd

def converter_para_brt(df):
    """Converte index para America/Sao_Paulo (BRT)."""
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/Sao_Paulo')
    else:
        df.index = df.index.tz_convert('America/Sao_Paulo')
    return df

def dentro_horario_operacao(ts):
    """
    Retorna True se o timestamp (em BRT) está no horário permitido.
    Janela: 10:45 às 16:30.
    """
    h = ts.hour
    m = ts.minute
    if h < 10 or h >= 17:
        return False
    if h == 10 and m < 45:
        return False
    if h == 11:
        return False
    if h == 16 and m >= 30:
        return False
    return True
