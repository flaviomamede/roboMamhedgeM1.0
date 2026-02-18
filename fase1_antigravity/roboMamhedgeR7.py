"""
R7: R6 + Take Profit — otimizado para meta >55% win, 11k em 60 dias.
- Take profit 2.5×ATR (captura ganhos)
- Stop 1.5×ATR (proteção)
- R:R ~1.67 — com 55% win: E = 0.55*2.5 - 0.45*1.5 = 1.0 ATR
- Aceita parâmetros via kwargs para otimização
"""
import pandas as pd
import numpy as np
from pathlib import Path
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

from config import (
    RSI_PERIOD, RSI_BULLISH, RSI_BULLISH_WINDOW,
    EMA_FAST, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_PERIOD,
)
from utils_fuso import converter_para_brt, dentro_horario_operacao, pnl_reais, N_COTAS, MULT_PONTOS_REAIS

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "WIN_5min.csv"


def run_backtest(
    csv_path=DEFAULT_CSV_PATH,
    stop_atr=2.3,
    target_atr=4.0,
    rsi_bullish=45,
    use_macd_filter=False,
):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    df['rsi'] = RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
    df['ema4'] = EMAIndicator(df['close'], window=EMA_FAST).ema_indicator()
    macd = MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df['macd_hist'] = macd.macd_diff()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=ATR_PERIOD).average_true_range()

    df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & (df['rsi'].shift(1) > df['rsi'])
    df['rsi_bullish_window'] = (df['rsi'] > rsi_bullish).rolling(window=RSI_BULLISH_WINDOW).max()

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

        # Fechar no fim do dia se em posição
        if df.index[i].hour >= 17 and position != 0:
            pnl = df['close'].iloc[i] - entry_price
            trades.append(pnl)
            position = 0
            continue

        if position <= 0:
            rsi_win = df['rsi_bullish_window'].iloc[i]
            macd_val = df['macd_hist'].iloc[i]
            if pd.isna(rsi_win):
                continue
            macd_ok = (not use_macd_filter) or pd.isna(macd_val) or macd_val > 0
            ema4_up = df['ema4'].iloc[i] > df['ema4'].iloc[i - 1]

            if rsi_win and ema4_up and macd_ok:
                if position == -1:
                    trades.append(entry_price - df['close'].iloc[i])
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - stop_atr * atr_val
                take_profit = entry_price + target_atr * atr_val
                position = 1

        if position == 1:
            hit_stop = df['low'].iloc[i] <= stop_loss
            hit_tp = df['high'].iloc[i] >= take_profit
            peak = df['rsi_peak_max'].iloc[i]
            if hit_stop:
                trades.append(stop_loss - entry_price)
                position = 0
            elif peak:
                trades.append(df['close'].iloc[i] - entry_price)
                position = 0
            elif hit_tp:
                trades.append(take_profit - entry_price)
                position = 0

    return np.array(trades) if trades else np.array([])


def optimize_params(csv_path=DEFAULT_CSV_PATH):
    """Grid search para encontrar melhores parâmetros."""
    best = None
    best_score = -1e9
    results = []

    for stop in [1.5, 1.8, 2.0]:
        for target in [2.0, 2.5, 3.0]:
            for rsi in [40, 45, 50, 55]:
                for use_macd in [True, False]:
                    trades = run_backtest(
                        csv_path, stop_atr=stop, target_atr=target,
                        rsi_bullish=rsi, use_macd_filter=use_macd
                    )
                    if len(trades) < 10:
                        continue
                    trades_r = np.array([pnl_reais(t) for t in trades])
                    wr = (trades_r > 0).mean()
                    e_pl = trades_r.mean()
                    total = trades_r.sum()
                    score = wr * 80 + e_pl * 20 + total * 0.1
                    if wr >= 0.55:
                        score += 100
                    results.append((stop, target, rsi, use_macd, len(trades), wr * 100, e_pl, total, score))
                    if score > best_score:
                        best_score = score
                        best = (stop, target, rsi, use_macd, len(trades), wr * 100, e_pl, total)

    return best, sorted(results, key=lambda x: -x[8])


if __name__ == "__main__":
    trades = run_backtest()
    if len(trades) == 0:
        print("[R7] Nenhum trade.")
    else:
        trades_r = np.array([pnl_reais(t) for t in trades])
        wr = (trades_r > 0).mean() * 100
        total = trades_r.sum()
        print(f"[R7] Trades: {len(trades_r)} ({N_COTAS} cotas) | Win: {wr:.1f}% | E[P&L]: R$ {trades_r.mean():.2f}/trade | Total: R$ {total:.2f}")

    print("\n--- Otimização (stop, target, rsi, macd) ---")
    best, all_res = optimize_params()
    if best:
        print(f"Melhor: stop={best[0]}, target={best[1]}, rsi={best[2]}, macd={best[3]}")
        print(f"  → {best[4]} trades, {best[5]:.1f}% win, E[P&L]=R$ {best[6]:.2f}, Total R$ {best[7]:.2f}")
        print("\nTop 5:")
        for r in all_res[:5]:
            print(f"  s={r[0]} t={r[1]} rsi={r[2]} m={r[3]}: {r[4]} trades, {r[5]:.1f}% win, E=R${r[6]:.2f}, Tot=R${r[7]:.2f}")
