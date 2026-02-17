"""
Robô Contrário: inverte os sinais do R6 + stops assimétricos.
- Quando R6 compraria, nós vendemos (short)
- Stop pequeno (0.8×ATR): corta perdas rápido
- Target grande (3×ATR): deixa ganhos correrem
"""
import pandas as pd
import numpy as np
from pathlib import Path
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

from config import (
    RSI_PERIOD, RSI_BULLISH, RSI_BULLISH_WINDOW,
    EMA_FAST, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, BB_USE, BB_ENTRY,
    ATR_PERIOD,
)
from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS, MULT_PONTOS_REAIS

STOP_ATR = 0.8
TARGET_ATR = 3.0
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "WIN_5min.csv"


def run_backtest(csv_path=DEFAULT_CSV_PATH, stop_atr=None, target_atr=None):
    stop = stop_atr if stop_atr is not None else STOP_ATR
    target = target_atr if target_atr is not None else TARGET_ATR
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

    df['rsi_bullish_window'] = (df['rsi'] > RSI_BULLISH).rolling(window=RSI_BULLISH_WINDOW).max()

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

        if position >= 0:
            rsi_win = df['rsi_bullish_window'].iloc[i]
            if pd.isna(rsi_win):
                continue
            macd_val = df['macd_hist'].iloc[i]
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
                if position == 1:
                    trades.append(close_val - entry_price)
                entry_price = close_val
                stop_loss = entry_price + stop * atr_val
                take_profit = entry_price - target * atr_val
                position = -1

        elif position == -1:
            hit_stop = df['high'].iloc[i] >= stop_loss
            hit_tp = df['low'].iloc[i] <= take_profit
            if hit_stop:
                trades.append(entry_price - stop_loss)
                position = 0
            elif hit_tp:
                trades.append(entry_price - take_profit)
                position = 0

    return np.array(trades) if trades else np.array([])


if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[Contrário] Nenhum trade.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        wr = (trades_r > 0).mean() * 100
        avg_win = trades_r[trades_r > 0].mean() if (trades_r > 0).any() else 0
        avg_loss = trades_r[trades_r <= 0].mean() if (trades_r <= 0).any() else 0
        print(f"[Contrário] stop={STOP_ATR} target={TARGET_ATR} ({N_COTAS} cotas)")
        print(f"  Trades: {len(trades_r)} | Win: {wr:.1f}% | E[P&L]: R$ {trades_r.mean():.2f}/trade | Total: R$ {trades_r.sum():.2f}")
        print(f"  Ganho médio: R$ {avg_win:.2f} | Perda média: R$ {avg_loss:.2f}")
