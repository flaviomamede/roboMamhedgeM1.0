"""
R3: EMAs 20 e 50 (em vez de 9 e 21) — tendência mais suave.
"""
import pandas as pd
import numpy as np
from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS, MULT_PONTOS_REAIS

def run_backtest(csv_path="WIN_5min.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['Momentum'] = df['close'] - df['close'].shift(10)
    df['TR'] = np.maximum(df['high']-df['low'],
        np.maximum(abs(df['high']-df['close'].shift()),
                   abs(df['low']-df['close'].shift())))
    df['ATR'] = df['TR'].rolling(14).mean()

    position = 0
    trades = []

    for i in range(1, len(df)):
        if not dentro_horario_operacao(df.index[i]):
            continue

        if position == 0:
            atr_val = df['ATR'].iloc[i]
            ema200 = df['EMA200'].iloc[i]
            momentum = df['Momentum'].iloc[i]
            if pd.isna(atr_val) or pd.isna(ema200) or pd.isna(momentum):
                continue
            if (df['EMA20'].iloc[i] > df['EMA50'].iloc[i] and
                df['close'].iloc[i] > ema200 and momentum > 0):
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - (1.5 * atr_val)
                take_profit = entry_price + (2.0 * atr_val)
                position = 1
            elif (df['EMA20'].iloc[i] < df['EMA50'].iloc[i] and
                  df['close'].iloc[i] < ema200 and momentum < 0):
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price + (1.5 * atr_val)
                take_profit = entry_price - (2.0 * atr_val)
                position = -1

        elif position == 1:
            if df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit:
                exit_price = stop_loss if df['low'].iloc[i] <= stop_loss else take_profit
                trades.append(exit_price - entry_price)
                position = 0
        elif position == -1:
            if df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit:
                exit_price = stop_loss if df['high'].iloc[i] >= stop_loss else take_profit
                trades.append(entry_price - exit_price)
                position = 0

    return np.array(trades) if trades else np.array([])

if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R3] Nenhum trade executado.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        win_rate = (trades_r > 0).sum() / len(trades_r)
        print(f"[R3 - EMAs 20/50] Trades: {len(trades_r)} ({N_COTAS} cotas) | Win: {win_rate*100:.1f}% | "
              f"E[P&L]: R$ {trades_r.mean():.2f}/trade")
