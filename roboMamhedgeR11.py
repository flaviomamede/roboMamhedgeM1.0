"""
R11 (ex-R10v2 Professional):
- Trend-following com filtro de regime por Efficiency Ratio (Kaufman)
- Filtro intradiário de VWAP (opcional)
- Ajuste de quantity e tempo em trade por volatilidade relativa
- Stop ATR + trailing ATR + break-even
- Opera comprado e vendido (long + short)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

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


def _adaptive_quantity(base_qty: int, vol_ratio: float, vol_explosion_cutoff: float) -> int:
    qty = max(int(base_qty), 1)
    if pd.isna(vol_ratio) or vol_ratio <= vol_explosion_cutoff:
        return qty
    scaled = qty * (vol_explosion_cutoff / max(float(vol_ratio), 1e-9))
    return max(int(round(scaled)), 1)


def _adaptive_max_bars(base_bars: int, vol_ratio: float, high_cutoff: float = 1.5, low_cutoff: float = 0.75) -> int:
    b = float(max(base_bars, 3))
    if pd.isna(vol_ratio):
        return int(round(b))
    if vol_ratio > high_cutoff:
        b *= 0.70
    elif vol_ratio < low_cutoff:
        b *= 1.40
    return max(int(round(b)), 3)


def _trade_from_side(side: int, entry_price: float, exit_price: float, quantity: int) -> TradePoints:
    # TradePoints calcula P&L como (exit-entry)*qty.
    # Para short, invertendo entry/exit obtemos P&L correto sem quantity negativa.
    if side >= 0:
        return TradePoints(float(entry_price), float(exit_price), int(quantity))
    return TradePoints(float(exit_price), float(entry_price), int(quantity))


def run_backtest_trades(
    csv_path: str = "fase1_antigravity/WIN_5min.csv",
    quantity: int = DEFAULT_QUANTITY,
    ema_fast: int = 6,
    ema_slow: int = 34,
    rsi_period: int = 10,
    rsi_thresh: int = 55,
    rsi_window: int = 4,
    stop_atr: float = 1.4,
    trail_atr: float = 1.8,
    breakeven_trigger_atr: float = 2.2,
    use_adx: bool = True,
    adx_min: float = 20.0,
    use_macd: bool = False,
    use_vwap_filter: bool = False,
    er_period: int = 14,
    er_trend_min: float = 0.4,
    atr_vol_window: int = 30,
    vol_explosion_cutoff: float = 1.2,
    max_bars_in_trade: int = 12,
    enable_long: bool = True,
    enable_short: bool = True,
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
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    if use_adx:
        df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    if use_macd:
        df["macd_hist"] = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff()

    df["rsi_bullish_window"] = (df["rsi"] > rsi_thresh).rolling(window=rsi_window).max()
    df["rsi_bearish_window"] = (df["rsi"] < (100 - rsi_thresh)).rolling(window=rsi_window).max()
    df["er"] = _efficiency_ratio(df["close"], period=er_period)
    df["vwap"] = _intraday_vwap(df)
    df["atr_ma"] = df["atr"].rolling(atr_vol_window).mean()
    df["vol_ratio"] = df["atr"] / df["atr_ma"].replace(0.0, np.nan)

    side = 0  # 1 = long, -1 = short
    entry_price = 0.0
    stop_loss = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    entry_day = None
    entry_ts = None
    qty_open = max(int(quantity), 1)
    max_bars_open = max_bars_in_trade

    trades: list[TradePoints] = []
    trades_ts: list[dict] = []

    start_i = max(er_period + 2, atr_vol_window + 2)
    for i in range(start_i, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else np.nan

        if side != 0:
            bars_in_trade += 1

            if ts.date() != entry_day or not dentro_horario_operacao(ts):
                t = _trade_from_side(side, entry_price, float(row["close"]), qty_open)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append(
                        {
                            "trade": t,
                            "side": side,
                            "entry_time": entry_ts,
                            "exit_time": ts,
                            "entry_price": float(entry_price),
                            "exit_price": float(row["close"]),
                        }
                    )
                side = 0
                continue

            if side == 1 and row["low"] <= stop_loss:
                t = _trade_from_side(side, entry_price, float(stop_loss), qty_open)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append(
                        {
                            "trade": t,
                            "side": side,
                            "entry_time": entry_ts,
                            "exit_time": ts,
                            "entry_price": float(entry_price),
                            "exit_price": float(stop_loss),
                        }
                    )
                side = 0
                continue

            if side == -1 and row["high"] >= stop_loss:
                t = _trade_from_side(side, entry_price, float(stop_loss), qty_open)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append(
                        {
                            "trade": t,
                            "side": side,
                            "entry_time": entry_ts,
                            "exit_time": ts,
                            "entry_price": float(entry_price),
                            "exit_price": float(stop_loss),
                        }
                    )
                side = 0
                continue

            if side == 1:
                if row["ema_fast"] < row["ema_slow"] or bars_in_trade >= max_bars_open:
                    t = _trade_from_side(side, entry_price, float(row["close"]), qty_open)
                    trades.append(t)
                    if with_timestamps and entry_ts is not None:
                        trades_ts.append(
                            {
                                "trade": t,
                                "side": side,
                                "entry_time": entry_ts,
                                "exit_time": ts,
                                "entry_price": float(entry_price),
                                "exit_price": float(row["close"]),
                            }
                        )
                    side = 0
                    continue

                highest_since_entry = max(highest_since_entry, float(row["high"]))
                if not pd.isna(atr_val):
                    if (highest_since_entry - entry_price) >= breakeven_trigger_atr * atr_val:
                        stop_loss = max(stop_loss, entry_price)
                    stop_loss = max(stop_loss, highest_since_entry - trail_atr * atr_val)

            else:
                if row["ema_fast"] > row["ema_slow"] or bars_in_trade >= max_bars_open:
                    t = _trade_from_side(side, entry_price, float(row["close"]), qty_open)
                    trades.append(t)
                    if with_timestamps and entry_ts is not None:
                        trades_ts.append(
                            {
                                "trade": t,
                                "side": side,
                                "entry_time": entry_ts,
                                "exit_time": ts,
                                "entry_price": float(entry_price),
                                "exit_price": float(row["close"]),
                            }
                        )
                    side = 0
                    continue

                lowest_since_entry = min(lowest_since_entry, float(row["low"]))
                if not pd.isna(atr_val):
                    if (entry_price - lowest_since_entry) >= breakeven_trigger_atr * atr_val:
                        stop_loss = min(stop_loss, entry_price)
                    stop_loss = min(stop_loss, lowest_since_entry + trail_atr * atr_val)
            continue

        if not dentro_horario_operacao(ts):
            continue
        if pd.isna(atr_val) or atr_val <= min_atr:
            continue
        if pd.isna(row["er"]) or row["er"] < er_trend_min:
            continue
        if use_adx and (pd.isna(row.get("adx")) or row["adx"] < adx_min):
            continue

        price = float(row["close"])
        vwap_ok_long = (not use_vwap_filter) or pd.isna(row["vwap"]) or price >= float(row["vwap"])
        vwap_ok_short = (not use_vwap_filter) or pd.isna(row["vwap"]) or price <= float(row["vwap"])

        qty_eff = _adaptive_quantity(int(quantity), float(row["vol_ratio"]), vol_explosion_cutoff)
        bars_eff = _adaptive_max_bars(max_bars_in_trade, float(row["vol_ratio"]))

        ema_up = row["ema_fast"] > row["ema_slow"] and row["ema_fast"] > df["ema_fast"].iloc[i - 1]
        ema_down = row["ema_fast"] < row["ema_slow"] and row["ema_fast"] < df["ema_fast"].iloc[i - 1]
        rsi_bull = bool(row["rsi_bullish_window"]) if not pd.isna(row["rsi_bullish_window"]) else False
        rsi_bear = bool(row["rsi_bearish_window"]) if not pd.isna(row["rsi_bearish_window"]) else False

        macd_ok_long = True
        macd_ok_short = True
        if use_macd:
            macd_val = row.get("macd_hist")
            macd_ok_long = pd.isna(macd_val) or macd_val > 0
            macd_ok_short = pd.isna(macd_val) or macd_val < 0

        long_signal = enable_long and ema_up and rsi_bull and macd_ok_long and vwap_ok_long
        short_signal = enable_short and ema_down and rsi_bear and macd_ok_short and vwap_ok_short

        if long_signal:
            side = 1
            entry_price = price
            stop_loss = float(entry_price - stop_atr * atr_val)
            highest_since_entry = entry_price
            lowest_since_entry = entry_price
            bars_in_trade = 0
            entry_day = ts.date()
            entry_ts = ts
            qty_open = qty_eff
            max_bars_open = bars_eff
        elif short_signal:
            side = -1
            entry_price = price
            stop_loss = float(entry_price + stop_atr * atr_val)
            highest_since_entry = entry_price
            lowest_since_entry = entry_price
            bars_in_trade = 0
            entry_day = ts.date()
            entry_ts = ts
            qty_open = qty_eff
            max_bars_open = bars_eff

    if side != 0 and len(df):
        final_price = float(df["close"].iloc[-1])
        t = _trade_from_side(side, entry_price, final_price, qty_open)
        trades.append(t)
        if with_timestamps and entry_ts is not None:
            trades_ts.append(
                {
                    "trade": t,
                    "side": side,
                    "entry_time": entry_ts,
                    "exit_time": df.index[-1],
                    "entry_price": float(entry_price),
                    "exit_price": float(final_price),
                }
            )

    return trades_ts if with_timestamps else trades


def run_backtest(
    csv_path: str = "fase1_antigravity/WIN_5min.csv",
    quantity: int = DEFAULT_QUANTITY,
    ema_fast: int = 6,
    ema_slow: int = 34,
    rsi_period: int = 10,
    rsi_thresh: int = 55,
    rsi_window: int = 4,
    stop_atr: float = 1.4,
    trail_atr: float = 1.8,
    breakeven_trigger_atr: float = 2.2,
    use_adx: bool = True,
    adx_min: float = 20.0,
    use_macd: bool = False,
    use_vwap_filter: bool = False,
    er_period: int = 14,
    er_trend_min: float = 0.4,
    atr_vol_window: int = 30,
    vol_explosion_cutoff: float = 1.2,
    max_bars_in_trade: int = 12,
    enable_long: bool = True,
    enable_short: bool = True,
    min_atr: float = 1e-9,
):
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
        use_vwap_filter=use_vwap_filter,
        er_period=er_period,
        er_trend_min=er_trend_min,
        atr_vol_window=atr_vol_window,
        vol_explosion_cutoff=vol_explosion_cutoff,
        max_bars_in_trade=max_bars_in_trade,
        enable_long=enable_long,
        enable_short=enable_short,
        min_atr=min_atr,
        with_timestamps=False,
    )
    pnl_pts = np.array([(t.exit_price_points - t.entry_price_points) * t.quantity for t in trades], dtype=float)
    return pnl_pts if len(pnl_pts) else np.array([])


if __name__ == "__main__":
    trades = run_backtest_trades(quantity=DEFAULT_QUANTITY, with_timestamps=False)
    if len(trades) == 0:
        print("[R11] Nenhum trade.")
    else:
        cost_model = default_b3_cost_model()
        tr = np.array([trade_net_pnl_brl(t, cost_model) for t in trades], dtype=float)
        wr = float((tr > 0).mean() * 100)
        print(
            f"[R11] Trades: {len(tr)} ({DEFAULT_QUANTITY} contratos) | "
            f"Win: {wr:.1f}% | E[P&L]: R$ {tr.mean():.2f}/trade | Total: R$ {tr.sum():.2f}"
        )
