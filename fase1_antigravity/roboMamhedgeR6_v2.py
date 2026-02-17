"""
R6 v2: EMA4 + RSI Peak — ajustes conservadores pelo Cursor.
Evolução do R6 Original do Flavio com:
  - MACD Hist > 0 como confirmação de entrada
  - Stop loss 1.5×ATR (proteção contra queda forte)
  - Take profit 2.5×ATR (captura se não houver pico RSI antes)
  - Filtro de horário BRT
  - Prioridade de saída: stop > peak RSI > take profit
"""
import pandas as pd
import numpy as np
from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS

def run_backtest(csv_path="WIN_5min.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    df['ema4'] = df['close'].ewm(span=4, adjust=False).mean()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    df['macd_hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & (df['rsi'].shift(1) > df['rsi'])
    df['rsi_bullish_window'] = (df['rsi'] > 40).rolling(window=5).max()

    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift()),
                   abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()

    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trades = []

    for i in range(2, len(df)):
        if not dentro_horario_operacao(df.index[i]):
            continue

        atr_val = df['atr'].iloc[i]
        if pd.isna(atr_val):
            continue

        # Fechar no fim do dia
        if df.index[i].hour >= 17 and position != 0:
            trades.append(df['close'].iloc[i] - entry_price)
            position = 0
            continue

        if position <= 0:
            rsi_win = df['rsi_bullish_window'].iloc[i]
            macd = df['macd_hist'].iloc[i]
            if pd.isna(rsi_win) or pd.isna(macd):
                continue
            if rsi_win and df['ema4'].iloc[i] > df['ema4'].iloc[i - 1] and macd > 0:
                if position == -1:
                    trades.append(entry_price - df['close'].iloc[i])
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - 1.5 * atr_val
                take_profit = entry_price + 2.5 * atr_val
                position = 1

        if position == 1:
            hit_stop = df['low'].iloc[i] <= stop_loss
            hit_tp = df['high'].iloc[i] >= take_profit
            peak = df['rsi_peak_max'].iloc[i]
            if hit_stop:
                trades.append(stop_loss - entry_price)
                position = 0
            elif peak:
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0
            elif hit_tp:
                trades.append(take_profit - entry_price)
                position = 0

    return np.array(trades) if trades else np.array([])

if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R6 v2] Nenhum trade executado.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        wr = (trades_r > 0).sum() / len(trades_r)
        print(f"[R6 v2 (Cursor)] Trades: {len(trades_r)} ({N_COTAS} cotas)")
        print(f"  Win: {wr*100:.1f}% | E[P&L]: R$ {trades_r.mean():.2f}/trade | Total: R$ {trades_r.sum():.2f}")
