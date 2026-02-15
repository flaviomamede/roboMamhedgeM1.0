"""
Investigação da escala dos dados em WIN_5min.csv.
Determina se os dados são WIN (mini-índice) ou BOVA11 (ETF) e como converter P&L.
"""
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("WIN_5min.csv", index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()

    close_min, close_max = df['close'].min(), df['close'].max()
    close_mean = df['close'].mean()
    atr = (df['high'] - df['low']).rolling(14).mean()
    atr_mean = atr.mean()
    diff_typical = df['close'].diff().abs().median()

    print("=" * 70)
    print("INVESTIGAÇÃO DE ESCALA - WIN_5min.csv")
    print("=" * 70)

    print("\n1. ESTATÍSTICAS DOS DADOS")
    print("-" * 50)
    print(f"   Close: {close_min:.2f} a {close_max:.2f} (média {close_mean:.2f})")
    print(f"   ATR(14) médio: {atr_mean:.4f}")
    print(f"   Variação típica (|Δclose| mediana): {diff_typical:.4f}")

    print("\n2. REFERÊNCIAS DE MERCADO (Jan–Fev 2026)")
    print("-" * 50)
    print("   WIN (mini-índice):  ~125.000 - 135.000 PONTOS")
    print("   Ibovespa (^BVSP):   ~183.000 - 190.000 pontos")
    print("   BOVA11 (ETF):       ~160 - 190 REAIS (R$/cota)")

    print("\n3. DIAGNÓSTICO")
    print("-" * 50)

    # Critério: WIN tem 5 dígitos, BOVA11 tem 2-3
    if close_mean > 100_000:
        ativo = "WIN (mini-índice)"
        escala = "pontos"
        mult_reais = 0.20
        print(f"   ✅ Escala WIN: preços em PONTOS (~{close_mean:.0f})")
        print(f"   → 1 ponto = R$ {mult_reais}")
    elif close_mean >= 1000:
        ativo = "BOVA11 em PONTOS (×1000)"
        escala = "pontos"
        mult_reais = 0.001
        print(f"   ✅ Escala BOVA11 em pontos (~{close_mean:.0f})")
        print(f"   → 1 ponto = R$ {mult_reais} (MULT_PONTOS_REAIS)")
    elif 50 < close_mean < 500:
        ativo = "BOVA11 (R$/cota) — converter com converter_csv_para_pontos.py"
        escala = "reais"
        mult_reais = 1.0
        print(f"   ⚠️ Escala BOVA11: preços em R$/cota (~{close_mean:.1f})")
        print(f"   → Execute: python converter_csv_para_pontos.py")
    else:
        ativo = "INDETERMINADO"
        escala = "?"
        mult_reais = None
        print(f"   ⚠️ Escala não reconhecida (média {close_mean:.1f})")

    print("\n4. CONCLUSÃO")
    print("-" * 50)
    if "BOVA11" in ativo and close_mean < 1000:
        print("   Dados em R$/cota. Converta para pontos: python converter_csv_para_pontos.py")
    elif "PONTOS" in ativo or ativo == "WIN (mini-índice)":
        print("   Dados em pontos (BOVA11×1000 ou WIN).")
        print("   • P&L em pontos × MULT_PONTOS_REAIS = R$")
        print("   • CUSTO_POR_TRADE em pontos (2500 = R$ 2,50)")

    print("\n5. VERIFICAÇÃO RÁPIDA (Yahoo Finance)")
    print("-" * 50)
    try:
        import yfinance as yf
        bova = yf.download("BOVA11.SA", period="5d", interval="1h")
        if not bova.empty:
            if isinstance(bova.columns, pd.MultiIndex):
                bova.columns = bova.columns.get_level_values(0)
            bova_close = bova['Close'].iloc[-1] if 'Close' in bova.columns else None
            if bova_close is not None:
                print(f"   BOVA11 hoje (1h): close ≈ {bova_close:.2f}")
                if abs(bova_close - close_mean) < 30:
                    print("   → Compatível com seus dados (mesma ordem de grandeza)")
    except Exception as e:
        print(f"   (yfinance não disponível ou erro: {e})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
