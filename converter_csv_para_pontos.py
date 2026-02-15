"""
Converte WIN_5min.csv de BOVA11 (R$/cota) para PONTOS (×1000, inteiro).
Use para arquivos já baixados antes da atualização do download_win.py.
"""
import pandas as pd
import sys

ARQUIVO = "WIN_5min.csv"
BOVA11_PARA_PONTOS = 1000


def converter(arquivo=ARQUIVO):
    df = pd.read_csv(arquivo, index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()

    if df["close"].mean() >= 1000:
        print(f"Arquivo já parece estar em pontos (close médio {df['close'].mean():.0f}).")
        return

    for col in ["close", "high", "low"]:
        if col in df.columns:
            df[col] = (df[col] * BOVA11_PARA_PONTOS).round().astype(int)

    df.to_csv(arquivo)
    print(f"Convertido {arquivo} para pontos (×{BOVA11_PARA_PONTOS}).")


if __name__ == "__main__":
    arq = sys.argv[1] if len(sys.argv) > 1 else ARQUIVO
    converter(arq)
