"""
Diagnóstico: escala dos dados, conversão para reais, e filtros.
Dados em PONTOS (BOVA11 × 1000). P&L em pontos × MULT_PONTOS_REAIS = R$
"""
import pandas as pd
import numpy as np
from roboMamhedgeR6 import run_backtest as r6
from roboMamhedgeR8 import run_backtest as r8
from utils_fuso import MULT_PONTOS_REAIS


def main():
    df = pd.read_csv("WIN_5min.csv", index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()

    print("=" * 60)
    print("DIAGNÓSTICO DE ESCALA E FILTROS")
    print("=" * 60)

    # 1. Dados (em pontos)
    n = len(df)
    dias = (df.index[-1] - df.index[0]).days
    print(f"\n1. DADOS: {n} candles, ~{dias} dias")
    print(f"   Close (pontos): {df['close'].min():.0f} a {df['close'].max():.0f}")
    atr = (df['high'] - df['low']).rolling(14).mean().mean()
    print(f"   ATR médio: {atr:.0f} pontos")

    # 2. EMA200 em 5min
    print(f"\n2. EMA200 em 5min = 200 barras = {200*5} min = {200*5/60:.1f}h")
    print("   → Com 60 dias de dados, EMA50 (4h) é mais adequado que EMA200.")

    # 3. Comparar R6 vs R8
    t6 = r6()
    t8 = r8()
    print(f"\n3. R6 (sem filtro tendência): {len(t6)} trades, Total=R$ {t6.sum() * MULT_PONTOS_REAIS:.2f}")
    print(f"   R8 (EMA50 + momentum): {len(t8)} trades, Total=R$ {t8.sum() * MULT_PONTOS_REAIS:.2f}")

    # 4. Conversão
    print(f"\n4. CONVERSÃO: 1 ponto = R$ {MULT_PONTOS_REAIS}")
    print(f"   R6 Total ≈ R$ {t6.sum() * MULT_PONTOS_REAIS:.2f} (1 mês)")
    print(f"   E[P&L] por trade ≈ R$ {t6.mean() * MULT_PONTOS_REAIS:.2f}")

    # 5. Custo
    from utils_fuso import CUSTO_POR_TRADE
    print(f"\n5. CUSTO no código: {CUSTO_POR_TRADE} pontos (= R$ {CUSTO_POR_TRADE * MULT_PONTOS_REAIS:.2f})")
    print(f"   Custo real B3+corretora: ~R$ 2-5 por round-trip")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
