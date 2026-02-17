"""
R6 Original (Flavio) — A ideia inicial.
EMA4 + RSI Peak Detection: antecipa reversão usando pico no IFR.

CONCEITO:
  1. "Janela bullish": se RSI > 40 nos últimos 5 candles, há força compradora
  2. EMA4 virando para cima: momentum de curtíssimo prazo confirmando
  3. Saída por PICO no RSI (peak detection): quando RSI faz topo local,
     o momentum exauriu — sai ANTES do preço cair (antecipação)
  4. Reversão de mão: se estava vendido e surge sinal de compra, vira direto

Sem stop loss, sem MACD, sem BB — puramente RSI peak + EMA4.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "WIN_5min.csv"


def run_backtest(csv_path=DEFAULT_CSV_PATH):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    # Indicadores de curto prazo (os favoritos do Flavio)
    df['ema4'] = df['close'].ewm(span=4, adjust=False).mean()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()

    # MACD e RSI (cálculo manual)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    df['macd_hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Peak Detection: detecta 'V' ou 'V invertido' no RSI
    df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & (df['rsi'].shift(1) > df['rsi'])
    df['rsi_peak_min'] = (df['rsi'].shift(1) < df['rsi'].shift(2)) & (df['rsi'].shift(1) < df['rsi'])

    # Memória de sinal: RSI > 40 nos últimos 5 candles
    df['rsi_bullish_window'] = (df['rsi'] > 40).rolling(window=5).max()

    position = 0
    entry_price = 0
    trades = []  # P&L em pontos puros

    for i in range(2, len(df)):
        if not dentro_horario_operacao(df.index[i]):
            continue

        # COMPRA (Antecipação): RSI bullish + EMA4 subindo
        if position <= 0:
            rsi_win = df['rsi_bullish_window'].iloc[i]
            if pd.isna(rsi_win):
                continue
            if rsi_win and df['ema4'].iloc[i] > df['ema4'].iloc[i - 1]:
                if position == -1:
                    trades.append(entry_price - df['close'].iloc[i])
                entry_price = df['close'].iloc[i]
                position = 1

        # SAÍDA ANTECIPADA: Pico de máximo no RSI
        elif position == 1:
            if df['rsi_peak_max'].iloc[i]:
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0

    return np.array(trades) if trades else np.array([])

if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R6 Original] Nenhum trade.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        wr = (trades_r > 0).sum() / len(trades_r)
        print(f"[R6 Original (Flavio)] Trades: {len(trades_r)} ({N_COTAS} cotas)")
        print(f"  Win: {wr*100:.1f}% | E[P&L]: R$ {trades_r.mean():.2f}/trade | Total: R$ {trades_r.sum():.2f}")
