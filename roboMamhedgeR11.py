"""
R11 (Omni-Regime - The Adaptive Ghost v1.1):
- Governor de regime via Efficiency Ratio (Kaufman)
  · er_trend_min = 0.48  (zona morta reduzida vs v1.0)
  · er_meanrev_max = 0.38 (captura lateralidade mais cedo)
- Módulo de Tendência quando ER >= er_trend_min
  · ADX opcional (use_adx_trend=False por padrão — menos restritivo)
  · ER usado APENAS para entrada; saída gerida só por EMA, trailing-ATR e time-stop
  · Time-stop adaptativo por volatilidade relativa (_adaptive_max_bars, igual ao R10v2)
- Módulo de Reversão à Média quando ER <= er_meanrev_max
  · Bollinger Bands + RSI
  · Filtro de distância mínima do BB_mid (Risk/Reward): só entra se preço está
    a pelo menos meanrev_min_dist_atr × ATR da média das bandas
  · Saída por alvo na banda central, time-stop ou regime de tendência retomado
- Position sizing adaptativo por volatilidade relativa (ATR)
- Custos realistas via b3_costs_phase2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

from market_time import converter_para_brt, dentro_horario_operacao
from b3_costs_phase2 import TradePoints, default_b3_cost_model, trade_net_pnl_brl


DEFAULT_QUANTITY = 1


def _efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    er = direction / volatility.replace(0.0, np.nan)
    return er.clip(lower=0.0, upper=1.0).fillna(0.0)


def _intraday_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    cum_pv = (typical_price * volume).groupby(df.index.date).cumsum()
    cum_v = volume.groupby(df.index.date).cumsum().replace(0.0, np.nan)
    return (cum_pv / cum_v).fillna(df["close"])


def _adaptive_quantity(base_qty: int, vol_ratio: float, cutoff: float) -> int:
    qty = max(int(base_qty), 1)
    if pd.isna(vol_ratio) or vol_ratio <= cutoff:
        return qty
    scaled = qty * (cutoff / max(float(vol_ratio), 1e-9))
    return max(int(round(scaled)), 1)


def _adaptive_max_bars(base_bars: int, vol_ratio: float, high_cutoff: float = 1.5, low_cutoff: float = 0.75) -> int:
    """Alta volatilidade → sai mais rápido; baixa volatilidade → dá mais tempo."""
    b = float(max(base_bars, 3))
    if pd.isna(vol_ratio):
        return int(round(b))
    if vol_ratio > high_cutoff:
        b *= 0.70
    elif vol_ratio < low_cutoff:
        b *= 1.40
    return max(int(round(b)), 3)


def _trade_from_side(side: int, entry_price: float, exit_price: float, quantity: int) -> TradePoints:
    # TradePoints usa (exit-entry)*qty para P&L; para short invertimos entry/exit.
    if side >= 0:
        return TradePoints(float(entry_price), float(exit_price), int(quantity))
    return TradePoints(float(exit_price), float(entry_price), int(quantity))


def _append_trade(
    trades: list,
    trades_ts: list,
    side: int,
    mode: str,
    entry_price: float,
    exit_price: float,
    qty_open: int,
    entry_ts,
    exit_ts,
    with_timestamps: bool,
) -> TradePoints:
    t = _trade_from_side(side, entry_price, exit_price, qty_open)
    trades.append(t)
    if with_timestamps and entry_ts is not None:
        trades_ts.append(
            {
                "trade": t,
                "mode": mode,
                "side": side,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
            }
        )
    return t


def run_backtest_trades(
    csv_path: str = "fase1_antigravity/WIN_5min.csv",
    quantity: int = DEFAULT_QUANTITY,
    # Indicadores
    ema_fast: int = 10,
    ema_slow: int = 34,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    er_period: int = 20,
    # Governor: fronteiras reduzidas — menos zona morta
    er_trend_min: float = 0.48,
    er_meanrev_max: float = 0.38,
    # Módulo Tendência
    use_adx_trend: bool = False,   # ADX opcional — menos restritivo
    adx_min_trend: float = 18.0,
    use_macd_trend: bool = True,
    use_vwap_filter: bool = True,
    trend_stop_atr: float = 2.0,
    trend_trail_atr: float = 2.2,
    trend_breakeven_atr: float = 1.8,
    max_bars_trend: int = 18,      # base para _adaptive_max_bars
    # Módulo Reversão
    meanrev_stop_atr: float = 1.2,
    meanrev_target_atr: float = 1.2,
    meanrev_min_dist_atr: float = 0.5,  # R/R: distância mínima do BB_mid para entrar
    rsi_oversold: int = 35,
    rsi_overbought: int = 65,
    max_bars_meanrev: int = 10,
    # Position sizing
    atr_vol_window: int = 30,
    vol_explosion_cutoff: float = 1.6,
    min_atr: float = 1e-9,
    with_timestamps: bool = False,
):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)
    df = df.sort_index()

    df["rsi"] = RSIIndicator(df["close"], window=rsi_period).rsi()
    df["ema_fast"] = EMAIndicator(df["close"], window=ema_fast).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], window=ema_slow).ema_indicator()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=atr_period).average_true_range()
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["macd_hist"] = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff()
    bb = BollingerBands(df["close"], window=bb_period, window_dev=bb_std)
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_high"] = bb.bollinger_hband()
    df["er"] = _efficiency_ratio(df["close"], er_period)
    df["vwap"] = _intraday_vwap(df)
    df["atr_ma"] = df["atr"].rolling(atr_vol_window).mean()
    df["vol_ratio"] = df["atr"] / df["atr_ma"].replace(0.0, np.nan)

    side = 0        # 1 long, -1 short
    mode = ""       # "trend" ou "meanrev"
    entry_price = 0.0
    stop_loss = 0.0
    target_price = np.nan
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    qty_open = max(int(quantity), 1)
    max_bars_open = max_bars_trend
    entry_day = None
    entry_ts = None

    trades: list[TradePoints] = []
    trades_ts: list[dict] = []

    start_i = max(er_period + 2, atr_vol_window + 2, bb_period + 2)
    for i in range(start_i, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else np.nan

        if side != 0:
            bars_in_trade += 1
            current_close = float(row["close"])

            # Saída operacional obrigatória (fim de dia ou fora do horário)
            if ts.date() != entry_day or not dentro_horario_operacao(ts):
                _append_trade(trades, trades_ts, side, mode, entry_price, current_close,
                               qty_open, entry_ts, ts, with_timestamps)
                side = 0; mode = ""
                continue

            # Stop-loss (hard)
            if side == 1 and row["low"] <= stop_loss:
                _append_trade(trades, trades_ts, side, mode, entry_price, float(stop_loss),
                               qty_open, entry_ts, ts, with_timestamps)
                side = 0; mode = ""
                continue

            if side == -1 and row["high"] >= stop_loss:
                _append_trade(trades, trades_ts, side, mode, entry_price, float(stop_loss),
                               qty_open, entry_ts, ts, with_timestamps)
                side = 0; mode = ""
                continue

            if mode == "trend":
                # Saída por EMA cruzamento ou time-stop.
                # ER NÃO fecha mais a posição — trailing-ATR e EMA cuidam disso.
                ema_exit_long = side == 1 and row["ema_fast"] < row["ema_slow"]
                ema_exit_short = side == -1 and row["ema_fast"] > row["ema_slow"]
                time_exit = bars_in_trade >= max_bars_open

                if ema_exit_long or ema_exit_short or time_exit:
                    _append_trade(trades, trades_ts, side, mode, entry_price, current_close,
                                   qty_open, entry_ts, ts, with_timestamps)
                    side = 0; mode = ""
                    continue

                # Atualização do trailing-stop e break-even
                if side == 1:
                    highest_since_entry = max(highest_since_entry, float(row["high"]))
                    if not pd.isna(atr_val):
                        if (highest_since_entry - entry_price) >= trend_breakeven_atr * atr_val:
                            stop_loss = max(stop_loss, entry_price)
                        stop_loss = max(stop_loss, highest_since_entry - trend_trail_atr * atr_val)
                else:
                    lowest_since_entry = min(lowest_since_entry, float(row["low"]))
                    if not pd.isna(atr_val):
                        if (entry_price - lowest_since_entry) >= trend_breakeven_atr * atr_val:
                            stop_loss = min(stop_loss, entry_price)
                        stop_loss = min(stop_loss, lowest_since_entry + trend_trail_atr * atr_val)

            else:  # meanrev
                # Alvo na banda central
                if side == 1 and row["high"] >= target_price:
                    _append_trade(trades, trades_ts, side, mode, entry_price, float(target_price),
                                   qty_open, entry_ts, ts, with_timestamps)
                    side = 0; mode = ""
                    continue

                if side == -1 and row["low"] <= target_price:
                    _append_trade(trades, trades_ts, side, mode, entry_price, float(target_price),
                                   qty_open, entry_ts, ts, with_timestamps)
                    side = 0; mode = ""
                    continue

                # Time-stop ou retomada de tendência → fecha a posição de reversão
                if bars_in_trade >= max_bars_open or row["er"] > er_trend_min:
                    _append_trade(trades, trades_ts, side, mode, entry_price, current_close,
                                   qty_open, entry_ts, ts, with_timestamps)
                    side = 0; mode = ""
                    continue

            continue  # ainda em posição, segue para próxima barra

        # ── Sem posição: decidir entrada pelo regime ──────────────────────────
        if not dentro_horario_operacao(ts):
            continue
        if pd.isna(atr_val) or atr_val <= min_atr:
            continue
        if pd.isna(row["er"]):
            continue

        price = float(row["close"])
        vol_ratio = float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else np.nan
        qty_eff = _adaptive_quantity(int(quantity), vol_ratio, vol_explosion_cutoff)

        if row["er"] >= er_trend_min:
            # ── Módulo de Tendência ──────────────────────────────────────────
            if use_adx_trend and (pd.isna(row["adx"]) or row["adx"] < adx_min_trend):
                continue

            ema_up = row["ema_fast"] > row["ema_slow"] and row["ema_fast"] > df["ema_fast"].iloc[i - 1]
            ema_down = row["ema_fast"] < row["ema_slow"] and row["ema_fast"] < df["ema_fast"].iloc[i - 1]
            macd_long = (not use_macd_trend) or pd.isna(row["macd_hist"]) or row["macd_hist"] > 0
            macd_short = (not use_macd_trend) or pd.isna(row["macd_hist"]) or row["macd_hist"] < 0
            vwap_long = (not use_vwap_filter) or pd.isna(row["vwap"]) or price >= float(row["vwap"])
            vwap_short = (not use_vwap_filter) or pd.isna(row["vwap"]) or price <= float(row["vwap"])

            long_signal = ema_up and macd_long and vwap_long
            short_signal = ema_down and macd_short and vwap_short

            bars_eff = _adaptive_max_bars(max_bars_trend, vol_ratio)

            if long_signal:
                side = 1; mode = "trend"
                entry_price = price
                stop_loss = float(entry_price - trend_stop_atr * atr_val)
                target_price = np.nan
                highest_since_entry = entry_price; lowest_since_entry = entry_price
                bars_in_trade = 0; qty_open = qty_eff; max_bars_open = max(bars_eff, 3)
                entry_day = ts.date(); entry_ts = ts
                continue

            if short_signal:
                side = -1; mode = "trend"
                entry_price = price
                stop_loss = float(entry_price + trend_stop_atr * atr_val)
                target_price = np.nan
                highest_since_entry = entry_price; lowest_since_entry = entry_price
                bars_in_trade = 0; qty_open = qty_eff; max_bars_open = max(bars_eff, 3)
                entry_day = ts.date(); entry_ts = ts
                continue

        elif row["er"] <= er_meanrev_max:
            # ── Módulo de Reversão à Média ───────────────────────────────────
            if pd.isna(row["bb_low"]) or pd.isna(row["bb_mid"]) or pd.isna(row["bb_high"]) or pd.isna(row["rsi"]):
                continue

            bb_mid = float(row["bb_mid"])

            # Filtro de distância mínima (R/R): só entra se houver espaço suficiente
            # até a média — garante que o alvo compensa o risco do stop.
            if pd.isna(atr_val) or atr_val <= min_atr:
                continue
            dist_to_mid = abs(price - bb_mid)
            if dist_to_mid < meanrev_min_dist_atr * atr_val:
                continue

            vwap_long = (not use_vwap_filter) or pd.isna(row["vwap"]) or price <= float(row["vwap"])
            vwap_short = (not use_vwap_filter) or pd.isna(row["vwap"]) or price >= float(row["vwap"])

            long_signal = price <= float(row["bb_low"]) and row["rsi"] <= rsi_oversold and vwap_long
            short_signal = price >= float(row["bb_high"]) and row["rsi"] >= rsi_overbought and vwap_short

            if long_signal:
                side = 1; mode = "meanrev"
                entry_price = price
                stop_loss = float(entry_price - meanrev_stop_atr * atr_val)
                target_price = bb_mid
                highest_since_entry = entry_price; lowest_since_entry = entry_price
                bars_in_trade = 0; qty_open = qty_eff; max_bars_open = max(max_bars_meanrev, 2)
                entry_day = ts.date(); entry_ts = ts
                continue

            if short_signal:
                side = -1; mode = "meanrev"
                entry_price = price
                stop_loss = float(entry_price + meanrev_stop_atr * atr_val)
                target_price = bb_mid
                highest_since_entry = entry_price; lowest_since_entry = entry_price
                bars_in_trade = 0; qty_open = qty_eff; max_bars_open = max(max_bars_meanrev, 2)
                entry_day = ts.date(); entry_ts = ts
                continue

    if side != 0 and len(df):
        final_price = float(df["close"].iloc[-1])
        _append_trade(trades, trades_ts, side, mode, entry_price, final_price,
                       qty_open, entry_ts, df.index[-1], with_timestamps)

    return trades_ts if with_timestamps else trades


def run_backtest(
    csv_path: str = "fase1_antigravity/WIN_5min.csv",
    quantity: int = DEFAULT_QUANTITY,
    **kwargs,
):
    trades = run_backtest_trades(csv_path=csv_path, quantity=quantity, with_timestamps=False, **kwargs)
    pnl_pts = np.array([(t.exit_price_points - t.entry_price_points) * t.quantity for t in trades], dtype=float)
    return pnl_pts if len(pnl_pts) else np.array([])


if __name__ == "__main__":
    trades = run_backtest_trades(quantity=DEFAULT_QUANTITY, with_timestamps=False)
    if len(trades) == 0:
        print("[R11 Omni-Regime v1.1] Nenhum trade.")
    else:
        cost_model = default_b3_cost_model()
        tr = np.array([trade_net_pnl_brl(t, cost_model) for t in trades], dtype=float)
        wr = float((tr > 0).mean() * 100)
        print(
            f"[R11 Omni-Regime v1.1] Trades: {len(tr)} ({DEFAULT_QUANTITY} contratos) | "
            f"Win: {wr:.1f}% | E[P&L]: R$ {tr.mean():.2f}/trade | Total: R$ {tr.sum():.2f}"
        )
