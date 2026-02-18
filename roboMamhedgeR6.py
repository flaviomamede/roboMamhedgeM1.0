"""
R6 (Phase 2 — Professional):
Lógica idêntica ao R6_v2 (Fase 1) com os parâmetros otimizados:
  - EMA4 subindo (momentum direcional)
  - RSI > rsi_thresh em pelo menos rsi_window barras recentes (momentum de força)
  - MACD Hist > 0 como confirmação de tendência (opcional)
  - Stop loss: stop_atr × ATR
  - Take profit: target_atr × ATR (saída de ganho rápido)
  - Saída adicional: pico de RSI (sinal de exaustão)
  - Saída intradiária: fim de dia ou fora de horário

Parâmetros padrão = valores otimizados pela Fase 1 (μ/σ² via Monte Carlo 1000 sims):
  stop_atr=2.0, target_atr=3.5, rsi_thresh=55, rsi_window=2, use_macd_filter=True
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator

from market_time import converter_para_brt, dentro_horario_operacao
from b3_costs_phase2 import TradePoints, default_b3_cost_model, trade_net_pnl_brl

DEFAULT_QUANTITY = 1


def _sma_macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram com ewm(adjust=False) — idêntico ao cálculo da Fase 1."""
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _sma_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR por SMA do True Range — idêntico ao cálculo da Fase 1."""
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ),
    )
    return tr.rolling(period).mean()


def _sma_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI calculado com SMA (rolling mean) — idêntico ao cálculo da Fase 1."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def run_backtest_trades(
    csv_path: str = "fase1_antigravity/WIN_5min.csv",
    stop_atr: float = 2.0,
    target_atr: float = 3.5,
    rsi_thresh: int = 55,
    rsi_window: int = 2,
    use_macd_filter: bool = True,
    quantity: int = DEFAULT_QUANTITY,
    min_atr: float = 1e-9,
    with_timestamps: bool = False,
):
    """Retorna lista de TradePoints (ou dicts com timestamps) para uso direto no comparativo."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)
    df.sort_index(inplace=True)

    df["rsi"] = _sma_rsi(df["close"], period=14)              # SMA — alinhado com Fase 1
    df["ema4"] = EMAIndicator(df["close"], window=4).ema_indicator()
    df["macd_hist"] = _sma_macd_hist(df["close"])             # ewm(adjust=False) — alinhado com Fase 1
    df["atr"] = _sma_atr(df, period=14)                       # SMA — alinhado com Fase 1

    df["rsi_peak_max"] = (df["rsi"].shift(1) > df["rsi"].shift(2)) & (df["rsi"].shift(1) > df["rsi"])
    df["rsi_bullish_window"] = (df["rsi"] > rsi_thresh).rolling(window=rsi_window).max()

    position = 0
    entry_price = 0.0
    entry_ts = None
    stop_loss = 0.0
    take_profit = 0.0
    entry_day = None
    trades: list[TradePoints] = []
    trades_ts: list[dict] = []

    for i in range(2, len(df)):
        ts = df.index[i]
        atr_val = float(df["atr"].iloc[i])
        if pd.isna(atr_val) or atr_val <= min_atr:
            continue

        if position == 1:
            # Saída por troca de dia ou fora de horário
            if ts.date() != entry_day or not dentro_horario_operacao(ts):
                exit_price = float(df["close"].iloc[i])
                t = TradePoints(float(entry_price), exit_price, quantity)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append({"trade": t, "entry_time": entry_ts, "exit_time": ts})
                position = 0
                continue

            # Saída por stop loss
            if df["low"].iloc[i] <= stop_loss:
                t = TradePoints(float(entry_price), float(stop_loss), quantity)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append({"trade": t, "entry_time": entry_ts, "exit_time": ts})
                position = 0
                continue

            # Saída por take profit
            if df["high"].iloc[i] >= take_profit:
                t = TradePoints(float(entry_price), float(take_profit), quantity)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append({"trade": t, "entry_time": entry_ts, "exit_time": ts})
                position = 0
                continue

            # Saída por pico de RSI (exaustão do momentum)
            if bool(df["rsi_peak_max"].iloc[i]):
                exit_price = float(df["close"].iloc[i])
                t = TradePoints(float(entry_price), exit_price, quantity)
                trades.append(t)
                if with_timestamps and entry_ts is not None:
                    trades_ts.append({"trade": t, "entry_time": entry_ts, "exit_time": ts})
                position = 0
                continue

        if position == 0:
            if not dentro_horario_operacao(ts):
                continue

            rsi_win = df["rsi_bullish_window"].iloc[i]
            macd_val = df["macd_hist"].iloc[i]
            if pd.isna(rsi_win) or pd.isna(macd_val):
                continue

            ema4_up = df["ema4"].iloc[i] > df["ema4"].iloc[i - 1]
            macd_ok = (not use_macd_filter) or macd_val > 0

            if bool(rsi_win) and ema4_up and macd_ok:
                entry_price = float(df["close"].iloc[i])
                entry_ts = ts
                stop_loss = entry_price - stop_atr * atr_val
                take_profit = entry_price + target_atr * atr_val
                entry_day = ts.date()
                position = 1

    if position == 1:
        exit_price = float(df["close"].iloc[-1])
        t = TradePoints(float(entry_price), exit_price, quantity)
        trades.append(t)
        if with_timestamps and entry_ts is not None:
            trades_ts.append({"trade": t, "entry_time": entry_ts, "exit_time": df.index[-1]})

    return trades_ts if with_timestamps else trades


def run_backtest(
    csv_path: str = "fase1_antigravity/WIN_5min.csv",
    stop_atr: float = 2.0,
    target_atr: float = 3.5,
    rsi_thresh: int = 55,
    rsi_window: int = 2,
    use_macd_filter: bool = True,
    min_atr: float = 1e-9,
):
    """Compatibilidade: retorna P&L bruto em pontos (sem custos)."""
    trades = run_backtest_trades(
        csv_path=csv_path,
        stop_atr=stop_atr,
        target_atr=target_atr,
        rsi_thresh=rsi_thresh,
        rsi_window=rsi_window,
        use_macd_filter=use_macd_filter,
        quantity=DEFAULT_QUANTITY,
        min_atr=min_atr,
    )
    pnl_pts = np.array([(t.exit_price_points - t.entry_price_points) * t.quantity for t in trades], dtype=float)
    return pnl_pts if len(pnl_pts) else np.array([])


if __name__ == "__main__":
    trades = run_backtest_trades(quantity=DEFAULT_QUANTITY)
    if len(trades) == 0:
        print("[R6 Pro] Nenhum trade executado.")
    else:
        cost_model = default_b3_cost_model()
        trades_r = np.array([trade_net_pnl_brl(t, cost_model) for t in trades], dtype=float)
        wr = float((trades_r > 0).mean())
        print(
            f"[R6 Pro] Trades: {len(trades_r)} ({DEFAULT_QUANTITY} contratos) | "
            f"Win: {wr*100:.1f}% | E[P&L]: R$ {trades_r.mean():.2f}/trade | "
            f"Total: R$ {trades_r.sum():.2f}"
        )
