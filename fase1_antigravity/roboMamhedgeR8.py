"""
R8: Híbrido R2 + R6 — filtros de tendência + saída por pico RSI.
- Entrada: EMA9 > EMA21, close > EMA_tendência, momentum > 0
- EMA50 (4h) em vez de EMA200 — adequado para 60 dias de dados
- Saída: Pico RSI ou stop 2×ATR (como R6)
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS, MULT_PONTOS_REAIS


def run_backtest(csv_path="WIN_5min.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & (df['rsi'].shift(1) > df['rsi'])

    position = 0
    entry_price = 0
    stop_loss = 0
    trades = []

    for i in range(50, len(df)):
        if not dentro_horario_operacao(df.index[i]):
            continue

        atr_val = df['atr'].iloc[i]
        ema50 = df['ema50'].iloc[i]
        momentum = df['momentum'].iloc[i]
        if pd.isna(atr_val) or pd.isna(ema50) or pd.isna(momentum):
            continue

        if position <= 0:
            if (df['ema9'].iloc[i] > df['ema21'].iloc[i] and
                df['close'].iloc[i] > ema50 and momentum > 0):
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - 2.0 * atr_val
                position = 1

        elif position == 1:
            hit_stop = df['low'].iloc[i] <= stop_loss
            peak = df['rsi_peak_max'].iloc[i]
            if hit_stop:
                trades.append(stop_loss - entry_price)
                position = 0
            elif peak:
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0

    return np.array(trades) if trades else np.array([])


if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R8] Nenhum trade.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        wr = (trades_r > 0).mean() * 100
        print(f"[R8 R2+R6 híbrido] Trades: {len(trades_r)} ({N_COTAS} cotas) | Win: {wr:.1f}% | "
              f"E[P&L]: R$ {trades_r.mean():.2f}/trade | Total: R$ {trades_r.sum():.2f}")
