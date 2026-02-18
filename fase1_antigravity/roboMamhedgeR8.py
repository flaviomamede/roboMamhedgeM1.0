"""
R8: Híbrido R2 + R6 — filtros de tendência + saída por pico RSI.
- Entrada: EMA9 > EMA21, close > EMA_tendência, momentum > 0
- EMA50 (4h) em vez de EMA200 — adequado para 60 dias de dados
- Saída: Pico RSI ou stop 2×ATR (como R6)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS, MULT_PONTOS_REAIS
from utils_metrics_pwb import metrics


DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "WIN_5min.csv"


def run_backtest(
    csv_path=DEFAULT_CSV_PATH,
    ema_fast=9,
    ema_slow=21,
    ema_trend=100,
    momentum_lookback=30,
    stop_atr=1.5,
    rsi_period=14,
):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    df['ema_fast'] = EMAIndicator(df['close'], window=ema_fast).ema_indicator()
    df['ema_slow'] = EMAIndicator(df['close'], window=ema_slow).ema_indicator()
    df['ema_trend'] = EMAIndicator(df['close'], window=ema_trend).ema_indicator()
    df['momentum'] = df['close'] - df['close'].shift(momentum_lookback)
    df['rsi'] = RSIIndicator(df['close'], window=rsi_period).rsi()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & (df['rsi'].shift(1) > df['rsi'])

    position = 0
    entry_price = 0
    stop_loss = 0
    trades = []

    warmup = max(ema_trend, momentum_lookback, 50)
    for i in range(warmup, len(df)):
        if not dentro_horario_operacao(df.index[i]):
            continue

        atr_val = df['atr'].iloc[i]
        ema_t = df['ema_trend'].iloc[i]
        momentum = df['momentum'].iloc[i]
        if pd.isna(atr_val) or pd.isna(ema_t) or pd.isna(momentum):
            continue

        if position <= 0:
            if (df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
                df['close'].iloc[i] > ema_t and momentum > 0):
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - stop_atr * atr_val
                position = 1

        elif position == 1:
            hit_stop = df['low'].iloc[i] <= stop_loss
            peak = df['rsi_peak_max'].iloc[i]
            if hit_stop:
                trades.append(stop_loss - entry_price)
                position = 0
            elif peak:
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0

    return np.array(trades) if trades else np.array([])


def optimize_params(csv_path=DEFAULT_CSV_PATH):
    """Grid search simples para o R8 (treino)."""
    best = None
    best_score = -1e18

    for ema_fast in [5, 9, 12]:
        for ema_slow in [21, 34]:
            if ema_fast >= ema_slow:
                continue
            for ema_trend in [34, 50, 80]:
                for mom in [5, 10, 20]:
                    for stop in [1.5, 2.0, 2.5]:
                        trades = run_backtest(
                            csv_path=csv_path,
                            ema_fast=ema_fast,
                            ema_slow=ema_slow,
                            ema_trend=ema_trend,
                            momentum_lookback=mom,
                            stop_atr=stop,
                        )
                        if len(trades) < 10:
                            continue
                        m = metrics(trades)
                        score = m["total_pl"] + (m["sharpe"] * 200.0)
                        if score > best_score:
                            best_score = score
                            best = (ema_fast, ema_slow, ema_trend, mom, stop, m["n"], m["win_rate"] * 100, m["e_pl"], m["total_pl"])

    return best


if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R8] Nenhum trade.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        wr = (trades_r > 0).mean() * 100
        print(f"[R8 R2+R6 híbrido] Trades: {len(trades_r)} ({N_COTAS} cotas) | Win: {wr:.1f}% | "
              f"E[P&L]: R$ {trades_r.mean():.2f}/trade | Total: R$ {trades_r.sum():.2f}")
