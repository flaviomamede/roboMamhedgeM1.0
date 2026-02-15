"""
Utilitário: converte dados para BRT e aplica filtro de horário corretamente.
Evita confusão UTC vs BRT.

Escala: dados em PONTOS (BOVA11 × 1000).
  1 ponto = R$ 0,001 por cota.

Modelo de custo:
  - P&L dos robôs é em PONTOS PUROS (sem custo).
  - Na exibição: P&L_reais = pnl_pontos × N_COTAS × MULT_PONTOS_REAIS - CUSTO_REAIS
  - CUSTO_REAIS é fixo por round-trip (R$ 2,50), não depende de posição.
"""
# 1 ponto (BOVA11 em escala Ibovespa) = R$ 0,001 por cota
MULT_PONTOS_REAIS = 0.001

# Quantidade de cotas por operação (define tamanho da posição)
N_COTAS = 100

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
    Exclui: 10h-10:45, 11h, 16:30 em diante.
    """
    h = ts.hour
    m = ts.minute
    if h < 10 or h >= 17:
        return False
    if h == 10 and m <= 45:
        return False
    if h == 11:
        return False
    if h == 16 and m >= 30:
        return False
    return True
