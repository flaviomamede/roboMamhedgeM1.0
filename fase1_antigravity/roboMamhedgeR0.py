from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils_fuso import converter_para_brt, dentro_horario_operacao
from utils_metrics_pwb import metrics


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = BASE_DIR / "WIN_5min.csv"


def run_backtest(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    ema_fast: int = 9,
    ema_slow: int = 21,
    atr_period: int = 14,
    atr_mean_period: int = 20,
) -> np.ndarray:
    """R0: protótipo de sinal (EMA9/21 + filtro de ATR).

    - Retorna P&L em pontos puros (sem custo), para ser compatível com os executivos da Fase 1.
    - Não gera dados sintéticos: se o CSV não existir, levanta erro.
    - Execução: simula entrada/saída por reversão de sinal (long/short).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)
    df = df.sort_index()

    # Indicadores
    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift())),
    )
    df["atr"] = df["tr"].rolling(atr_period).mean()
    df["atr_mean"] = df["atr"].rolling(atr_mean_period).mean()

    # Sinais: 1 (long), -1 (short), 0 (sem operar)
    df["signal"] = 0
    df.loc[(df["ema_fast"] > df["ema_slow"]) & (df["atr"] > df["atr_mean"]), "signal"] = 1
    df.loc[(df["ema_fast"] < df["ema_slow"]) & (df["atr"] > df["atr_mean"]), "signal"] = -1

    position = 0  # 1 long, -1 short, 0 flat
    entry_price = 0.0
    entry_day = None
    trades: list[float] = []

    for i in range(1, len(df)):
        ts = df.index[i]
        if not dentro_horario_operacao(ts):
            continue

        sig = int(df["signal"].iloc[i])
        price = float(df["close"].iloc[i])

        # fecha no fim do dia / troca de data
        if position != 0 and entry_day is not None and ts.date() != entry_day:
            trades.append((price - entry_price) if position == 1 else (entry_price - price))
            position = 0

        if position == 0:
            if sig == 1:
                position = 1
                entry_price = price
                entry_day = ts.date()
            elif sig == -1:
                position = -1
                entry_price = price
                entry_day = ts.date()
            continue

        # reversão de sinal: fecha e vira
        if position == 1 and sig == -1:
            trades.append(price - entry_price)
            position = -1
            entry_price = price
            entry_day = ts.date()
        elif position == -1 and sig == 1:
            trades.append(entry_price - price)
            position = 1
            entry_price = price
            entry_day = ts.date()

    # fecha no fim
    if position != 0 and len(df):
        last_price = float(df["close"].iloc[-1])
        trades.append((last_price - entry_price) if position == 1 else (entry_price - last_price))

    return np.array(trades, dtype=float) if trades else np.array([])


def optimize_params(csv_path: str | Path = DEFAULT_CSV_PATH):
    """Grid search simples para o R0 (treino).

    Retorna:
      best: (ema_fast, ema_slow, atr_period, atr_mean_period, n_trades, win%, e_pl, total_pl)
    """
    best = None
    best_score = -1e18

    for ema_fast in [5, 7, 9, 12]:
        for ema_slow in [15, 21, 34]:
            if ema_fast >= ema_slow:
                continue
            for atr_period in [10, 14]:
                for atr_mean_period in [10, 20, 30]:
                    trades = run_backtest(
                        csv_path=csv_path,
                        ema_fast=ema_fast,
                        ema_slow=ema_slow,
                        atr_period=atr_period,
                        atr_mean_period=atr_mean_period,
                    )
                    if len(trades) < 10:
                        continue
                    m = metrics(trades)
                    # score: prioriza lucro total, depois sharpe
                    score = m["total_pl"] + (m["sharpe"] * 200.0)
                    if score > best_score:
                        best_score = score
                        best = (
                            ema_fast,
                            ema_slow,
                            atr_period,
                            atr_mean_period,
                            m["n"],
                            m["win_rate"] * 100,
                            m["e_pl"],
                            m["total_pl"],
                        )

    return best


if __name__ == "__main__":
    t = run_backtest()
    print(f"[R0] Trades: {len(t)} | Total (pontos): {t.sum():.2f}")