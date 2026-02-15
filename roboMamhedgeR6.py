"""
R6: EMA4 + RSI Peak Detection + MACD + Bollinger Bands.
- Compra: RSI bullish window (40+) + EMA4 virando para cima + MACD Hist > 0
- Filtro BB: preço <= BB inferior (oversold) — inspirado em robo-daytrade-v1
- Saída: Pico de máximo no RSI ou stop loss 2×ATR
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

from config import (
    RSI_PERIOD, RSI_BULLISH, RSI_BULLISH_WINDOW,
    EMA_FAST, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, BB_USE, BB_ENTRY,
    ATR_PERIOD, STOP_ATR_MULT,
)
from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS, MULT_PONTOS_REAIS


def run_backtest(csv_path="WIN_5min.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    df['rsi'] = RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
    df['ema4'] = EMAIndicator(df['close'], window=EMA_FAST).ema_indicator()

    macd = MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df['macd_hist'] = macd.macd_diff()

    bb = BollingerBands(df['close'], window=BB_PERIOD, window_dev=BB_STD)
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()

    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=ATR_PERIOD).average_true_range()

    df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & (df['rsi'].shift(1) > df['rsi'])
    df['rsi_bullish_window'] = (df['rsi'] > RSI_BULLISH).rolling(window=RSI_BULLISH_WINDOW).max()

    position = 0
    entry_price = 0
    stop_loss = 0
    trades = []  # P&L em pontos puros

    for i in range(2, len(df)):
        if not dentro_horario_operacao(df.index[i]):
            continue
        if position <= 0:
            rsi_win = df['rsi_bullish_window'].iloc[i]
            if pd.isna(rsi_win):
                continue
            macd_val = df['macd_hist'].iloc[i]
            atr_val = df['atr'].iloc[i]
            close_val = df['close'].iloc[i]
            bb_low_val = df['bb_low'].iloc[i]
            bb_mid_val = df['bb_mid'].iloc[i]

            bb_ok = True
            if BB_USE:
                ref = bb_mid_val if BB_ENTRY == 'mid' else bb_low_val
                if not pd.isna(ref):
                    bb_ok = close_val <= ref

            ema4_up = df['ema4'].iloc[i] > df['ema4'].iloc[i - 1]
            macd_ok = pd.isna(macd_val) or macd_val > 0

            if rsi_win and ema4_up and macd_ok and bb_ok:
                if position == -1:
                    trades.append(entry_price - close_val)
                entry_price = close_val
                stop_loss = entry_price - STOP_ATR_MULT * atr_val if not pd.isna(atr_val) else entry_price - 1000
                position = 1

        elif position == 1:
            hit_stop = df['low'].iloc[i] <= stop_loss
            if hit_stop:
                trades.append(stop_loss - entry_price)
                position = 0
            elif df['rsi_peak_max'].iloc[i]:
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0

    return np.array(trades) if trades else np.array([])


if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R6 - EMA4+RSI Peak+BB] Nenhum trade executado.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        win_rate = (trades_r > 0).sum() / len(trades_r)
        print(f"[R6 - EMA4+RSI Peak+BB] Trades: {len(trades_r)} ({N_COTAS} cotas) | Win: {win_rate*100:.1f}% | "
              f"E[P&L]: R$ {trades_r.mean():.2f}/trade | Total: R$ {trades_r.sum():.2f}")
