"""
R9 Professional: RSI Window + EMA rápida + ADX/MACD opcionais + Stop ATR/Target ATR.
ALINHADO 1:1 COM PADRÃO PROFISSIONAL (WIN 0.20).
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange

from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS


def run_backtest(
    csv_path="WIN_5min.csv",
    ema_fast=4,
    rsi_period=14,
    rsi_thresh=40,
    rsi_window=5,
    stop_atr=2.0,
    target_atr=0,
    use_macd=True,
    use_adx=True,
    adx_min=20,
    min_atr=1e-9,
    max_bars_in_trade=48,
):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    df['rsi'] = RSIIndicator(df['close'], window=rsi_period).rsi()
    df['ema'] = EMAIndicator(df['close'], window=ema_fast).ema_indicator()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    if use_macd:
        macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_hist'] = macd.macd_diff()

    if use_adx:
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

    df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & (df['rsi'].shift(1) > df['rsi'])
    df['rsi_bullish_window'] = (df['rsi'] > rsi_thresh).rolling(window=rsi_window).max()

    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    bars_in_trade = 0
    entry_day = None
    trades = []

    for i in range(2, len(df)):
        ts = df.index[i]
        atr_val = df['atr'].iloc[i]

        if position == 1:
            bars_in_trade += 1

            # Session Exit
            if ts.date() != entry_day or not dentro_horario_operacao(ts):
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0
                continue

            # Stop first
            if df['low'].iloc[i] <= stop_loss:
                trades.append(stop_loss - entry_price)
                position = 0
                continue

            # Target second
            if take_profit > 0 and df['high'].iloc[i] >= take_profit:
                trades.append(take_profit - entry_price)
                position = 0
                continue

            # RSI peak or time stop
            if bool(df['rsi_peak_max'].iloc[i]) or bars_in_trade >= max_bars_in_trade:
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0
                continue

        if position == 0:
            if not dentro_horario_operacao(ts):
                continue

            if pd.isna(atr_val) or atr_val <= min_atr:
                continue

            if use_adx and (pd.isna(df['adx'].iloc[i]) or df['adx'].iloc[i] < adx_min):
                continue

            rsi_win = df['rsi_bullish_window'].iloc[i]
            if pd.isna(rsi_win) or not bool(rsi_win):
                continue

            ema_up = df['ema'].iloc[i] > df['ema'].iloc[i - 1]
            if not ema_up:
                continue

            if use_macd:
                macd_val = df['macd_hist'].iloc[i]
                if not (pd.isna(macd_val) or macd_val > 0):
                    continue

            # ENTRY
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price - stop_atr * atr_val
            take_profit = entry_price + target_atr * atr_val if target_atr > 0 else 0.0
            bars_in_trade = 0
            entry_day = ts.date()
            position = 1

    if position == 1:
        trades.append(df['close'].iloc[-1] - entry_price)

    return np.array(trades) if trades else np.array([])


if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) > 0:
        tr = np.array([pnl_reais(t) for t in trades])
        wr = (tr > 0).mean() * 100
        print(
            f"[R9 Pro] Trades: {len(tr)} ({N_COTAS} contratos) | "
            f"Win: {wr:.1f}% | E[P&L]: R$ {tr.mean():.2f}/trade | Total: R$ {tr.sum():.2f}"
        )
    else:
        print("[R9 Pro] Nenhum trade.")
