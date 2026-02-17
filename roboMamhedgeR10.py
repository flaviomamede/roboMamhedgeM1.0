"""
R10 (revisado): evolução do R9 com filtro de regime + trailing stop ATR + break-even.

Objetivo desta versão:
- Melhorar robustez prática ANTES de otimizar parâmetros.
- Evitar ambiguidade intrabar de stop/trailing (atualiza trailing após checar stop do candle).
- Controlar exposição com time-stop e fechamento por fim de sessão/troca de dia.
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.volatility import AverageTrueRange

from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS


def run_backtest(
    csv_path="WIN_5min.csv",
    quantity=1,
    ema_fast=6,
    ema_slow=21,
    rsi_period=14,
    rsi_thresh=40,
    rsi_window=4,
    stop_atr=1.7,
    trail_atr=2.4,
    breakeven_trigger_atr=1.5,
    use_adx=True,
    adx_min=20.0,
    use_macd=False,
    min_atr=1e-9,
    max_bars_in_trade=60,
):
    if quantity <= 0:
        raise ValueError("quantity deve ser > 0")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    # Indicadores
    df['rsi'] = RSIIndicator(df['close'], window=rsi_period).rsi()
    df['ema_fast'] = EMAIndicator(df['close'], window=ema_fast).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=ema_slow).ema_indicator()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    if use_adx:
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    if use_macd:
        df['macd_hist'] = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()

    df['rsi_bullish_window'] = (df['rsi'] > rsi_thresh).rolling(window=rsi_window).max()

    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    highest_since_entry = 0.0
    bars_in_trade = 0
    entry_day = None
    trades = []

    prev_ema_fast = None

    for i in range(2, len(df)):
        ts = df.index[i]
        row = df.iloc[i]

        current_ema_fast = row['ema_fast']
        if pd.isna(current_ema_fast):
            continue
        if prev_ema_fast is None:
            prev_ema_fast = current_ema_fast
            continue

        ema_up = current_ema_fast > prev_ema_fast
        regime_up = row['ema_fast'] > row['ema_slow'] if not pd.isna(row['ema_slow']) else False

        atr_val = row['atr']

        if position == 1:
            bars_in_trade += 1

            # saída operacional (sessão/dia)
            if ts.date() != entry_day or not dentro_horario_operacao(ts):
                trades.append((row['close'] - entry_price) * quantity)
                position = 0
                bars_in_trade = 0
                prev_ema_fast = current_ema_fast
                continue

            # 1) Checa stop com stop vigente (conservador; evita look-ahead do trailing)
            if row['low'] <= stop_loss:
                trades.append((stop_loss - entry_price) * quantity)
                position = 0
                bars_in_trade = 0
                prev_ema_fast = current_ema_fast
                continue

            # 2) Saída por quebra de regime / time stop
            if (not regime_up) or (bars_in_trade >= max_bars_in_trade):
                trades.append((row['close'] - entry_price) * quantity)
                position = 0
                bars_in_trade = 0
                prev_ema_fast = current_ema_fast
                continue

            # 3) Só agora atualiza trailing para vigorar nos próximos candles
            if pd.isna(atr_val) or atr_val <= min_atr:
                prev_ema_fast = current_ema_fast
                continue

            highest_since_entry = max(highest_since_entry, row['high'])
            trailing_stop = highest_since_entry - trail_atr * atr_val

            if (highest_since_entry - entry_price) >= breakeven_trigger_atr * atr_val:
                stop_loss = max(stop_loss, entry_price)

            stop_loss = max(stop_loss, trailing_stop)

            prev_ema_fast = current_ema_fast
            continue

        # Entrada
        if not dentro_horario_operacao(ts):
            prev_ema_fast = current_ema_fast
            continue

        if not regime_up or not ema_up:
            prev_ema_fast = current_ema_fast
            continue

        if pd.isna(atr_val) or atr_val <= min_atr:
            prev_ema_fast = current_ema_fast
            continue

        if use_adx:
            adx_val = row['adx']
            if pd.isna(adx_val) or adx_val < adx_min:
                prev_ema_fast = current_ema_fast
                continue

        if use_macd:
            macd_val = row['macd_hist']
            if not (pd.isna(macd_val) or macd_val > 0):
                prev_ema_fast = current_ema_fast
                continue

        rsi_ok = row['rsi_bullish_window']
        if pd.isna(rsi_ok) or not bool(rsi_ok):
            prev_ema_fast = current_ema_fast
            continue

        entry_price = float(row['close'])
        highest_since_entry = entry_price
        stop_loss = entry_price - stop_atr * atr_val
        bars_in_trade = 0
        entry_day = ts.date()
        position = 1

        prev_ema_fast = current_ema_fast

    if position == 1:
        trades.append((df['close'].iloc[-1] - entry_price) * quantity)

    return np.array(trades) if trades else np.array([])


if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R10 revisado] Nenhum trade.")
    else:
        tr = np.array([pnl_reais(t) for t in trades])
        wr = (tr > 0).mean() * 100
        print(
            f"[R10 revisado] Trades: {len(tr)} ({N_COTAS} cotas) | "
            f"Win: {wr:.1f}% | E[P&L]: R$ {tr.mean():.2f}/trade | Total: R$ {tr.sum():.2f}"
        )
