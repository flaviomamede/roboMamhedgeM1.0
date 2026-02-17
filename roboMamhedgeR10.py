"""
R10 Professional: evolução do R9 com filtro de regime + trailing stop ATR + break-even.
ALINHADO 1:1 COM PADRÃO PROFISSIONAL (WIN 0.20).
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.volatility import AverageTrueRange

from market_time import converter_para_brt, dentro_horario_operacao
from b3_costs_phase2 import TradePoints, default_b3_cost_model, trade_net_pnl_brl


DEFAULT_QUANTITY = 1


def run_backtest_trades(
    csv_path="fase1_antigravity/WIN_5min.csv",
    quantity: int = DEFAULT_QUANTITY,
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
    use_macd=True,
    min_atr=1e-9,
    max_bars_in_trade=20,
):
    """Retorna trades com entry/exit (pontos) para custos realistas."""
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
    trades: list[TradePoints] = []

    for i in range(2, len(df)):
        ts = df.index[i]
        row = df.iloc[i]

        ema_up = df['ema_fast'].iloc[i] > df['ema_fast'].iloc[i - 1]
        regime_up = row['ema_fast'] > row['ema_slow'] if not pd.isna(row['ema_slow']) else False
        atr_val = row['atr']

        if position == 1:
            bars_in_trade += 1

            # 0) Saída Operacional (Mandatória 17:00 ou fim de sessão)
            if ts.date() != entry_day or not dentro_horario_operacao(ts):
                exit_price = float(row['close'])
                trades.append(TradePoints(float(entry_price), exit_price, quantity))
                position = 0
                continue

            # 1) Checa STOP VIGENTE (Anti-bias: Low do candle vs Stop calculado no candle anterior)
            if row['low'] <= stop_loss:
                exit_price = float(stop_loss)
                trades.append(TradePoints(float(entry_price), exit_price, quantity))
                position = 0
                continue

            # 2) Saída por Regime ou Time-stop
            if (not regime_up) or (bars_in_trade >= max_bars_in_trade):
                exit_price = float(row['close'])
                trades.append(TradePoints(float(entry_price), exit_price, quantity))
                position = 0
                continue

            # 3) Só agora atualiza Watermark e Trailing para vigorar nos candles SEGUINTES
            highest_since_entry = max(highest_since_entry, row['high'])

            if (highest_since_entry - entry_price) >= breakeven_trigger_atr * atr_val:
                stop_loss = max(stop_loss, entry_price)

            trailing_stop = highest_since_entry - trail_atr * atr_val
            stop_loss = max(stop_loss, trailing_stop)
            continue

        # Entrada
        if not dentro_horario_operacao(ts):
            continue

        if not regime_up or not ema_up:
            continue

        if use_adx and (pd.isna(row.get('adx')) or row['adx'] < adx_min):
            continue

        if use_macd and not (pd.isna(row.get('macd_hist')) or row['macd_hist'] > 0):
            continue

        if pd.isna(atr_val) or atr_val <= min_atr:
            continue

        if not bool(row.get('rsi_bullish_window', False)):
            continue

        # BUY
        entry_price = float(row['close'])
        highest_since_entry = entry_price
        stop_loss = float(entry_price - stop_atr * atr_val)
        bars_in_trade = 0
        entry_day = ts.date()
        position = 1

    if position == 1:
        exit_price = float(df['close'].iloc[-1])
        trades.append(TradePoints(float(entry_price), exit_price, quantity))

    return trades


def run_backtest(
    csv_path="fase1_antigravity/WIN_5min.csv",
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
    use_macd=True,
    min_atr=1e-9,
    max_bars_in_trade=20,
):
    # Mantém compatibilidade: retorna P&L em pontos (sem custos).
    trades = run_backtest_trades(
        csv_path=csv_path,
        quantity=quantity,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        rsi_thresh=rsi_thresh,
        rsi_window=rsi_window,
        stop_atr=stop_atr,
        trail_atr=trail_atr,
        breakeven_trigger_atr=breakeven_trigger_atr,
        use_adx=use_adx,
        adx_min=adx_min,
        use_macd=use_macd,
        min_atr=min_atr,
        max_bars_in_trade=max_bars_in_trade,
    )
    pnl_pts = np.array([(t.exit_price_points - t.entry_price_points) * t.quantity for t in trades], dtype=float)
    return pnl_pts if len(pnl_pts) else np.array([])


if __name__ == "__main__":
    trades = run_backtest_trades(quantity=DEFAULT_QUANTITY)
    if len(trades) == 0:
        print("[R10 Pro] Nenhum trade.")
    else:
        cost_model = default_b3_cost_model()
        tr = np.array([trade_net_pnl_brl(t, cost_model) for t in trades], dtype=float)
        wr = float((tr > 0).mean() * 100)
        print(
            f"[R10 Pro] Trades: {len(tr)} ({DEFAULT_QUANTITY} contratos) | "
            f"Win: {wr:.1f}% | E[P&L]: R$ {tr.mean():.2f}/trade | Total: R$ {tr.sum():.2f}"
        )
