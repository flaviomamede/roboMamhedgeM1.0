"""
R11 (Omni-Regime - Mean-Variance Optimized):
- Governor de regime via Efficiency Ratio (Kaufman)
- Módulo de Tendência (trend-following) quando ER alto
- Módulo de Reversão à Média (Bollinger + RSI) quando ER baixo
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


def _trade_from_side(side: int, entry_price: float, exit_price: float, quantity: int) -> TradePoints:
    # TradePoints usa (exit-entry)*qty para P&L; para short invertimos entry/exit.
    if side >= 0:
        return TradePoints(float(entry_price), float(exit_price), int(quantity))
    return TradePoints(float(exit_price), float(entry_price), int(quantity))


def run_backtest_trades(
    csv_path: str = "fase1_antigravity/WIN_5min.csv",
    quantity: int = DEFAULT_QUANTITY,
    ema_fast: int = 10,
    ema_slow: int = 34,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    er_period: int = 20,
    er_trend_min: float = 0.60,
    er_meanrev_max: float = 0.30,
    adx_min_trend: float = 18.0,
    use_macd_trend: bool = True,
    use_vwap_filter: bool = True,
    trend_stop_atr: float = 2.0,
    trend_trail_atr: float = 2.2,
    trend_breakeven_atr: float = 1.8,
    meanrev_stop_atr: float = 1.2,
    meanrev_target_atr: float = 1.2,
    rsi_oversold: int = 35,
    rsi_overbought: int = 65,
    atr_vol_window: int = 30,
    vol_explosion_cutoff: float = 1.6,
    max_bars_trend: int = 18,
    max_bars_meanrev: int = 10,
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

    side = 0  # 1 long, -1 short
    mode = ""  # "trend" ou "meanrev"
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

            # Saída operacional obrigatória
            if ts.date() != entry_day or not dentro_horario_operacao(ts):
                t = _trade_from_side(side, entry_price, current_close, qty_open)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append(
                        {
                            "trade": t,
                            "mode": mode,
                            "side": side,
                            "entry_time": entry_ts,
                            "exit_time": ts,
                            "entry_price": float(entry_price),
                            "exit_price": current_close,
                        }
                    )
                side = 0
                mode = ""
                continue

            # Stop
            if side == 1 and row["low"] <= stop_loss:
                t = _trade_from_side(side, entry_price, float(stop_loss), qty_open)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append(
                        {
                            "trade": t,
                            "mode": mode,
                            "side": side,
                            "entry_time": entry_ts,
                            "exit_time": ts,
                            "entry_price": float(entry_price),
                            "exit_price": float(stop_loss),
                        }
                    )
                side = 0
                mode = ""
                continue

            if side == -1 and row["high"] >= stop_loss:
                t = _trade_from_side(side, entry_price, float(stop_loss), qty_open)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append(
                        {
                            "trade": t,
                            "mode": mode,
                            "side": side,
                            "entry_time": entry_ts,
                            "exit_time": ts,
                            "entry_price": float(entry_price),
                            "exit_price": float(stop_loss),
                        }
                    )
                side = 0
                mode = ""
                continue

            if mode == "trend":
                # Saídas de tendência
                if side == 1 and (row["ema_fast"] < row["ema_slow"] or bars_in_trade >= max_bars_open or row["er"] < 0.40):
                    t = _trade_from_side(side, entry_price, current_close, qty_open)
                    trades.append(t)
                    if with_timestamps and entry_ts is not None:
                        trades_ts.append(
                            {
                                "trade": t,
                                "mode": mode,
                                "side": side,
                                "entry_time": entry_ts,
                                "exit_time": ts,
                                "entry_price": float(entry_price),
                                "exit_price": current_close,
                            }
                        )
                    side = 0
                    mode = ""
                    continue

                if side == -1 and (row["ema_fast"] > row["ema_slow"] or bars_in_trade >= max_bars_open or row["er"] < 0.40):
                    t = _trade_from_side(side, entry_price, current_close, qty_open)
                    trades.append(t)
                    if with_timestamps and entry_ts is not None:
                        trades_ts.append(
                            {
                                "trade": t,
                                "mode": mode,
                                "side": side,
                                "entry_time": entry_ts,
                                "exit_time": ts,
                                "entry_price": float(entry_price),
                                "exit_price": current_close,
                            }
                        )
                    side = 0
                    mode = ""
                    continue

                # Atualização de trailing/breakeven
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

            else:
                # meanrev: saída por alvo no centro da banda, tempo, ou mudança de regime
                if side == 1 and row["high"] >= target_price:
                    t = _trade_from_side(side, entry_price, float(target_price), qty_open)
                    trades.append(t)
                    if with_timestamps and entry_ts is not None:
                        trades_ts.append(
                            {
                                "trade": t,
                                "mode": mode,
                                "side": side,
                                "entry_time": entry_ts,
                                "exit_time": ts,
                                "entry_price": float(entry_price),
                                "exit_price": float(target_price),
                            }
                        )
                    side = 0
                    mode = ""
                    continue

                if side == -1 and row["low"] <= target_price:
                    t = _trade_from_side(side, entry_price, float(target_price), qty_open)
                    trades.append(t)
                    if with_timestamps and entry_ts is not None:
                        trades_ts.append(
                            {
                                "trade": t,
                                "mode": mode,
                                "side": side,
                                "entry_time": entry_ts,
                                "exit_time": ts,
                                "entry_price": float(entry_price),
                                "exit_price": float(target_price),
                            }
                        )
                    side = 0
                    mode = ""
                    continue

                if bars_in_trade >= max_bars_open or row["er"] > er_trend_min:
                    t = _trade_from_side(side, entry_price, current_close, qty_open)
                    trades.append(t)
                    if with_timestamps and entry_ts is not None:
                        trades_ts.append(
                            {
                                "trade": t,
                                "mode": mode,
                                "side": side,
                                "entry_time": entry_ts,
                                "exit_time": ts,
                                "entry_price": float(entry_price),
                                "exit_price": current_close,
                            }
                        )
                    side = 0
                    mode = ""
                    continue
            continue

        # Sem posição: decidir módulo pelo regime
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
            # Módulo de tendência
            if pd.isna(row["adx"]) or row["adx"] < adx_min_trend:
                continue
            ema_up = row["ema_fast"] > row["ema_slow"] and row["ema_fast"] > df["ema_fast"].iloc[i - 1]
            ema_down = row["ema_fast"] < row["ema_slow"] and row["ema_fast"] < df["ema_fast"].iloc[i - 1]
            macd_long = (not use_macd_trend) or pd.isna(row["macd_hist"]) or row["macd_hist"] > 0
            macd_short = (not use_macd_trend) or pd.isna(row["macd_hist"]) or row["macd_hist"] < 0
            vwap_long = (not use_vwap_filter) or pd.isna(row["vwap"]) or price >= float(row["vwap"])
            vwap_short = (not use_vwap_filter) or pd.isna(row["vwap"]) or price <= float(row["vwap"])

            long_signal = ema_up and macd_long and vwap_long
            short_signal = ema_down and macd_short and vwap_short

            if long_signal:
                side = 1
                mode = "trend"
                entry_price = price
                stop_loss = float(entry_price - trend_stop_atr * atr_val)
                target_price = np.nan
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                bars_in_trade = 0
                qty_open = qty_eff
                max_bars_open = max(max_bars_trend, 3)
                entry_day = ts.date()
                entry_ts = ts
                continue

            if short_signal:
                side = -1
                mode = "trend"
                entry_price = price
                stop_loss = float(entry_price + trend_stop_atr * atr_val)
                target_price = np.nan
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                bars_in_trade = 0
                qty_open = qty_eff
                max_bars_open = max(max_bars_trend, 3)
                entry_day = ts.date()
                entry_ts = ts
                continue

        elif row["er"] <= er_meanrev_max:
            # Módulo de reversão à média
            if pd.isna(row["bb_low"]) or pd.isna(row["bb_mid"]) or pd.isna(row["bb_high"]) or pd.isna(row["rsi"]):
                continue
            vwap_long = (not use_vwap_filter) or pd.isna(row["vwap"]) or price <= float(row["vwap"])
            vwap_short = (not use_vwap_filter) or pd.isna(row["vwap"]) or price >= float(row["vwap"])

            long_signal = price <= float(row["bb_low"]) and row["rsi"] <= rsi_oversold and vwap_long
            short_signal = price >= float(row["bb_high"]) and row["rsi"] >= rsi_overbought and vwap_short

            if long_signal:
                side = 1
                mode = "meanrev"
                entry_price = price
                stop_loss = float(entry_price - meanrev_stop_atr * atr_val)
                target_price = float(row["bb_mid"]) if not pd.isna(row["bb_mid"]) else float(entry_price + meanrev_target_atr * atr_val)
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                bars_in_trade = 0
                qty_open = qty_eff
                max_bars_open = max(max_bars_meanrev, 2)
                entry_day = ts.date()
                entry_ts = ts
                continue

            if short_signal:
                side = -1
                mode = "meanrev"
                entry_price = price
                stop_loss = float(entry_price + meanrev_stop_atr * atr_val)
                target_price = float(row["bb_mid"]) if not pd.isna(row["bb_mid"]) else float(entry_price - meanrev_target_atr * atr_val)
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                bars_in_trade = 0
                qty_open = qty_eff
                max_bars_open = max(max_bars_meanrev, 2)
                entry_day = ts.date()
                entry_ts = ts
                continue

    if side != 0 and len(df):
        final_price = float(df["close"].iloc[-1])
        t = _trade_from_side(side, entry_price, final_price, qty_open)
        trades.append(t)
        if with_timestamps and entry_ts is not None:
            trades_ts.append(
                {
                    "trade": t,
                    "mode": mode,
                    "side": side,
                    "entry_time": entry_ts,
                    "exit_time": df.index[-1],
                    "entry_price": float(entry_price),
                    "exit_price": final_price,
                }
            )

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
        print("[R11 Omni-Regime] Nenhum trade.")
    else:
        cost_model = default_b3_cost_model()
        tr = np.array([trade_net_pnl_brl(t, cost_model) for t in trades], dtype=float)
        wr = float((tr > 0).mean() * 100)
        print(
            f"[R11 Omni-Regime] Trades: {len(tr)} ({DEFAULT_QUANTITY} contratos) | "
            f"Win: {wr:.1f}% | E[P&L]: R$ {tr.mean():.2f}/trade | Total: R$ {tr.sum():.2f}"
        )
