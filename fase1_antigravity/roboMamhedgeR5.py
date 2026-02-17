"""
R5: MACD + RSI com Long e Short.
- Compra: MACD Hist > 0 e RSI cruzando acima de 40
- Venda: MACD Hist < 0 e RSI cruzando abaixo de 60
- Stop: 2×ATR | Target: 3×ATR
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

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    df['macd_hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift()),
                   abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(14).mean()

    position = 0
    entry_price = 0
    trades = []

    for i in range(1, len(df)):
        h = df.index[i].hour
        if h >= 17 and position != 0:
            pnl = (df['close'].iloc[i] - entry_price) if position == 1 else (entry_price - df['close'].iloc[i])
            trades.append(pnl)
            position = 0
            continue
        if not dentro_horario_operacao(df.index[i]):
            continue

        if position == 0:
            macd = df['macd_hist'].iloc[i]
            rsi = df['rsi'].iloc[i]
            rsi_prev = df['rsi'].iloc[i - 1]
            atr_val = df['atr'].iloc[i]
            if pd.isna(macd) or pd.isna(rsi) or pd.isna(atr_val):
                continue
            if macd > 0 and rsi > 40 and rsi_prev <= 40:
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - (2 * atr_val)
                take_profit = entry_price + (3 * atr_val)
                position = 1
            elif macd < 0 and rsi < 60 and rsi_prev >= 60:
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price + (2 * atr_val)
                take_profit = entry_price - (3 * atr_val)
                position = -1
        else:
            if position == 1:
                if df['low'].iloc[i] <= stop_loss:
                    trades.append(stop_loss - entry_price)
                    position = 0
                elif df['high'].iloc[i] >= take_profit:
                    trades.append(take_profit - entry_price)
                    position = 0
            elif position == -1:
                if df['high'].iloc[i] >= stop_loss:
                    trades.append(entry_price - stop_loss)
                    position = 0
                elif df['low'].iloc[i] <= take_profit:
                    trades.append(entry_price - take_profit)
                    position = 0

    return np.array(trades) if trades else np.array([])

if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R5] Nenhum trade executado.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        win_rate = (trades_r > 0).sum() / len(trades_r)
        print(f"[R5 - MACD+RSI Long/Short] Trades: {len(trades_r)} ({N_COTAS} cotas) | Win: {win_rate*100:.1f}% | "
              f"E[P&L]: R$ {trades_r.mean():.2f}/trade")
