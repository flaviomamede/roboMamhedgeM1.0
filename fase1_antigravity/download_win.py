"""
Baixa dados de 5min do Yahoo Finance.
- WIN=F: mini-índice (se disponível)
- BOVA11.SA: fallback — convertido para PONTOS (×1000, inteiro) para alinhar com Ibovespa
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

TICKER_PRINCIPAL = "WIN=F"
TICKER_FALLBACK = "BOVA11.SA"
ARQUIVO_SAIDA = Path(__file__).resolve().parent / "WIN_5min.csv"
# BOVA11 em R$/cota → pontos (como Ibovespa): multiplicar por 1000
BOVA11_PARA_PONTOS = 1000


def bova11_para_pontos(df):
    """Converte BOVA11 (R$/cota) para pontos: ×1000, inteiro."""
    df = df.copy()
    for col in ["close", "high", "low"]:
        if col in df.columns:
            df[col] = (df[col] * BOVA11_PARA_PONTOS).round().astype(int)
    return df


def eh_bova11(df):
    """True se os dados parecem BOVA11 (preço < 1000)."""
    if df.empty or "close" not in df.columns:
        return False
    return df["close"].mean() < 1000


def main() -> None:
    print(f"Baixando dados de 5min para {TICKER_PRINCIPAL}...")
    df = yf.download(TICKER_PRINCIPAL, period="60d", interval="5m")

    if df.empty:
        print(f"Falha ao baixar {TICKER_PRINCIPAL}. Tentando {TICKER_FALLBACK}...")
        df = yf.download(TICKER_FALLBACK, period="60d", interval="5m")

    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={
            "Close": "close", "High": "high", "Low": "low", "Open": "open"
        })
        colunas = [c for c in ["close", "high", "low"] if c in df.columns]
        df = df[colunas].dropna()

        if eh_bova11(df):
            print("   Dados de BOVA11 — convertendo para pontos (×1000)...")
            df = bova11_para_pontos(df)

        df.to_csv(ARQUIVO_SAIDA)
        print(f"Sucesso! Arquivo {ARQUIVO_SAIDA} criado com {len(df)} candles.")
    else:
        print("Erro: não foi possível baixar dados de nenhum ticker.")


if __name__ == "__main__":
    main()
