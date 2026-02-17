"""
R6 (Professional): EMA4 + RSI Window + MACD (+ BB opcional) + Stop ATR.
ALINHADO 1:1 COM PADRÃO PROFISSIONAL (WIN 0.20).
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
from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS


def run_backtest(
    csv_path="WIN_5min.csv",
    stop_atr=STOP_ATR_MULT,
    min_atr=1e-9,
    use_macd_filter=True,
):
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
    entry_price = 0.0
    stop_loss = 0.0
    entry_day = None
    trades = []

    for i in range(2, len(df)):
        ts = df.index[i]

        if position == 1:
            # Fechamento por troca de dia ou fora da sessão
            if ts.date() != entry_day or not dentro_horario_operacao(ts):
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0
                continue

            # Check Stop Loss
            if df['low'].iloc[i] <= stop_loss:
                trades.append(stop_loss - entry_price)
                position = 0
                continue

            # RSI Peak Exit
            if bool(df['rsi_peak_max'].iloc[i]):
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0
                continue

        if position == 0:
            if not dentro_horario_operacao(ts):
                continue

            rsi_win = df['rsi_bullish_window'].iloc[i]
            atr_val = df['atr'].iloc[i]
            close_val = df['close'].iloc[i]
            bb_low_val = df['bb_low'].iloc[i]
            bb_mid_val = df['bb_mid'].iloc[i]
            macd_val = df['macd_hist'].iloc[i]

            if pd.isna(rsi_win) or pd.isna(atr_val) or atr_val <= min_atr:
                continue

            bb_ok = True
            if BB_USE:
                ref = bb_mid_val if BB_ENTRY == 'mid' else bb_low_val
                if not pd.isna(ref):
                    bb_ok = close_val <= ref

            ema4_up = df['ema4'].iloc[i] > df['ema4'].iloc[i - 1]
            macd_ok = (not use_macd_filter) or pd.isna(macd_val) or macd_val > 0

            if bool(rsi_win) and ema4_up and macd_ok and bb_ok:
                entry_price = close_val
                stop_loss = entry_price - stop_atr * atr_val
                entry_day = ts.date()
                position = 1

    if position == 1:
        trades.append(df['close'].iloc[-1] - entry_price)

    return np.array(trades) if trades else np.array([])


if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R6 Pro] Nenhum trade executado.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        win_rate = (trades_r > 0).sum() / len(trades_r)
        print(
            f"[R6 Pro] Trades: {len(trades_r)} ({N_COTAS} contratos) | "
            f"Win: {win_rate*100:.1f}% | E[P&L]: R$ {trades_r.mean():.2f}/trade | "
            f"Total: R$ {trades_r.sum():.2f}"
        )
