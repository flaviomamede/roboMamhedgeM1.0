"""
R9: Versão otimizada do R6 (a ideia do Flavio, refinada).

Mantém o conceito central: RSI Peak Detection + EMA rápida.
Otimizações sobre o R6 original:
  1. Grid search nos parâmetros (RSI threshold, janela, EMA, stop, target)
  2. Stop ATR obrigatório (o original não tinha stop)
  3. Filtro ADX > 20 (só opera em mercado com tendência)
  4. Fechamento forçado no fim do dia
  5. Sem Bollinger Bands (simplificação — menos overfitting)

Parâmetros otimizáveis via run_backtest() kwargs.
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
    target_atr=0,       # 0 = sem target (sai só por peak ou stop)
    use_macd=False,
    use_adx=True,
    adx_min=20,
):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)

    # Indicadores
    df['rsi'] = RSIIndicator(df['close'], window=rsi_period).rsi()
    df['ema'] = EMAIndicator(df['close'], window=ema_fast).ema_indicator()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    if use_macd:
        macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_hist'] = macd.macd_diff()

    if use_adx:
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

    # Peak detection e janela bullish (a ideia central do Flavio)
    df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & (df['rsi'].shift(1) > df['rsi'])
    df['rsi_bullish_window'] = (df['rsi'] > rsi_thresh).rolling(window=rsi_window).max()

    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trades = []

    for i in range(2, len(df)):
        ts = df.index[i]

        # Fechar no fim do dia
        if ts.hour >= 17 and position != 0:
            if position == 1:
                trades.append(df['close'].iloc[i] - entry_price)
            elif position == -1:
                trades.append(entry_price - df['close'].iloc[i])
            position = 0
            continue

        if not dentro_horario_operacao(ts):
            continue

        atr_val = df['atr'].iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            continue

        # Filtro ADX
        if use_adx:
            adx_val = df['adx'].iloc[i]
            if pd.isna(adx_val) or adx_val < adx_min:
                continue

        # --- ENTRADA ---
        if position <= 0:
            rsi_win = df['rsi_bullish_window'].iloc[i]
            if pd.isna(rsi_win) or not rsi_win:
                continue

            ema_up = df['ema'].iloc[i] > df['ema'].iloc[i - 1]
            if not ema_up:
                continue

            macd_ok = True
            if use_macd:
                macd_val = df['macd_hist'].iloc[i]
                macd_ok = pd.isna(macd_val) or macd_val > 0

            if macd_ok:
                if position == -1:
                    trades.append(entry_price - df['close'].iloc[i])
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price - stop_atr * atr_val
                take_profit = entry_price + target_atr * atr_val if target_atr > 0 else 0
                position = 1

        # --- SAÍDA ---
        elif position == 1:
            hit_stop = df['low'].iloc[i] <= stop_loss
            hit_tp = take_profit > 0 and df['high'].iloc[i] >= take_profit
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


def optimize(csv_path="WIN_5min.csv"):
    """Grid search nos parâmetros chave do R6."""
    best = None
    best_score = -1e9
    results = []

    for ema in [3, 4, 5, 6]:
        for rsi_t in [35, 40, 45, 50]:
            for rsi_w in [3, 5, 7]:
                for stop in [1.5, 2.0, 2.5, 3.0]:
                    for target in [0, 2.5, 3.5]:
                        for macd in [False, True]:
                            for adx in [False, True]:
                                trades = run_backtest(
                                    csv_path,
                                    ema_fast=ema, rsi_thresh=rsi_t, rsi_window=rsi_w,
                                    stop_atr=stop, target_atr=target,
                                    use_macd=macd, use_adx=adx,
                                )
                                if len(trades) < 15:
                                    continue
                                tr = np.array([pnl_reais(t) for t in trades])
                                wr = (tr > 0).mean()
                                epl = tr.mean()
                                total = tr.sum()
                                # Score: E[P&L] × sqrt(n_trades) — balanceia retorno e consistência
                                score = epl * np.sqrt(len(tr))
                                if wr >= 0.45:
                                    score *= 1.2
                                results.append({
                                    'ema': ema, 'rsi_t': rsi_t, 'rsi_w': rsi_w,
                                    'stop': stop, 'target': target, 'macd': macd, 'adx': adx,
                                    'n': len(tr), 'wr': wr * 100, 'epl': epl, 'total': total,
                                    'score': score,
                                })
                                if score > best_score:
                                    best_score = score
                                    best = results[-1]

    results.sort(key=lambda x: -x['score'])
    return best, results


if __name__ == "__main__":
    # Primeiro roda com defaults
    trades = run_backtest()
    if len(trades) > 0:
        tr = np.array([pnl_reais(t) for t in trades])
        wr = (tr > 0).mean() * 100
        print(f"[R9 default] Trades: {len(tr)} ({N_COTAS} cotas) | Win: {wr:.1f}% | "
              f"E[P&L]: R$ {tr.mean():.2f}/trade | Total: R$ {tr.sum():.2f}")
    else:
        print("[R9 default] Nenhum trade.")

    # Otimização
    print("\n--- Otimização R9 (grid search) ---")
    best, all_res = optimize()
    if best:
        print(f"\nMelhor configuração:")
        print(f"  EMA={best['ema']}, RSI>{best['rsi_t']} (janela {best['rsi_w']}), "
              f"Stop={best['stop']}×ATR, Target={best['target']}×ATR, "
              f"MACD={'Sim' if best['macd'] else 'Não'}, ADX={'Sim' if best['adx'] else 'Não'}")
        print(f"  → {best['n']} trades, {best['wr']:.1f}% win, E[P&L]=R$ {best['epl']:.2f}, Total R$ {best['total']:.2f}")

        print(f"\nTop 10:")
        for r in all_res[:10]:
            print(f"  E{r['ema']} R{r['rsi_t']}w{r['rsi_w']} s{r['stop']} t{r['target']} "
                  f"m{'Y' if r['macd'] else 'N'} a{'Y' if r['adx'] else 'N'}: "
                  f"{r['n']}t {r['wr']:.0f}%w R${r['epl']:.2f}/t R${r['total']:.0f}")

        # Roda o melhor para confirmar
        print(f"\n--- Rodando melhor config ---")
        trades_best = run_backtest(
            ema_fast=best['ema'], rsi_thresh=best['rsi_t'], rsi_window=best['rsi_w'],
            stop_atr=best['stop'], target_atr=best['target'],
            use_macd=best['macd'], use_adx=best['adx'],
        )
        tr_best = np.array([pnl_reais(t) for t in trades_best])
        print(f"  Confirmado: {len(tr_best)} trades, {(tr_best > 0).mean()*100:.1f}% win, "
              f"E[P&L]=R$ {tr_best.mean():.2f}, Total=R$ {tr_best.sum():.2f}")
