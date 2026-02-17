"""
Análise de desempenho por horário (em BRT).
"""
import pandas as pd
import numpy as np
from utils_fuso import converter_para_brt, dentro_horario_operacao, CUSTO_POR_TRADE

# Carregar e calcular (R2: EMAs 9/21)
df = pd.read_csv("WIN_5min.csv", index_col=0, parse_dates=True)
df.columns = df.columns.str.lower()
df = converter_para_brt(df)

df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
df['Momentum'] = df['close'] - df['close'].shift(10)
df['TR'] = np.maximum(df['high']-df['low'],
    np.maximum(abs(df['high']-df['close'].shift()),
               abs(df['low']-df['close'].shift())))
df['ATR'] = df['TR'].rolling(14).mean()

# Backtest registrando HORA de entrada de cada trade
position = 0
trades_with_hour = []  # (pnl, hour_brt)

for i in range(1, len(df)):
    if not dentro_horario_operacao(df.index[i]):
        continue
    atr_val = df['ATR'].iloc[i]
    ema200 = df['EMA200'].iloc[i]
    momentum = df['Momentum'].iloc[i]
    if pd.isna(atr_val) or pd.isna(ema200) or pd.isna(momentum):
        continue

    if position == 0:
        if (df['EMA9'].iloc[i] > df['EMA21'].iloc[i] and
            df['close'].iloc[i] > ema200 and momentum > 0):
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price - (1.5 * atr_val)
            take_profit = entry_price + (2.0 * atr_val)
            position = 1
            entry_hour = df.index[i].hour
        elif (df['EMA9'].iloc[i] < df['EMA21'].iloc[i] and
              df['close'].iloc[i] < ema200 and momentum < 0):
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price + (1.5 * atr_val)
            take_profit = entry_price - (2.0 * atr_val)
            position = -1
            entry_hour = df.index[i].hour

    elif position == 1:
        if df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit:
            exit_price = stop_loss if df['low'].iloc[i] <= stop_loss else take_profit
            pnl = exit_price - entry_price - CUSTO_POR_TRADE
            trades_with_hour.append((pnl, entry_hour))
            position = 0
    elif position == -1:
        if df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit:
            exit_price = stop_loss if df['high'].iloc[i] >= stop_loss else take_profit
            pnl = entry_price - exit_price - CUSTO_POR_TRADE
            trades_with_hour.append((pnl, entry_hour))
            position = 0

# Agregar por hora
by_hour = {}
for pnl, h in trades_with_hour:
    if h not in by_hour:
        by_hour[h] = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl_total': 0}
    by_hour[h]['trades'] += 1
    by_hour[h]['pnl_total'] += pnl
    if pnl > 0:
        by_hour[h]['wins'] += 1
    else:
        by_hour[h]['losses'] += 1

# Ordenar por hora
hours_sorted = sorted(by_hour.keys())

print("=" * 70)
print("ANÁLISE POR HORÁRIO (R2 - EMAs 9/21)")
print("Horário em BRT (convertido de UTC)")
print("=" * 70)
print(f"{'Hora':^6} | {'Trades':^6} | {'Wins':^5} | {'Losses':^6} | {'Win%':^6} | {'P&L Total':^12}")
print("-" * 70)

for h in hours_sorted:
    d = by_hour[h]
    wr = (d['wins'] / d['trades'] * 100) if d['trades'] > 0 else 0
    flag = " ⚠ 10-11h" if h in (10, 11) else ""
    print(f"  {h:02d}h  | {d['trades']:^6} | {d['wins']:^5} | {d['losses']:^6} | {wr:^5.1f}% | {d['pnl_total']:>+10.2f}  {flag}")

print("-" * 70)
total_trades = sum(by_hour[h]['trades'] for h in hours_sorted)
total_pnl = sum(by_hour[h]['pnl_total'] for h in hours_sorted)
print(f"TOTAL | {total_trades:^6} |      |        |       | {total_pnl:>+10.2f}")

# Destaque 10-11h BRT (13-14 UTC)
h10_11 = [h for h in hours_sorted if h in (10, 11)]
if h10_11:
    t_10_11 = sum(by_hour[h]['trades'] for h in h10_11)
    pnl_10_11 = sum(by_hour[h]['pnl_total'] for h in h10_11)
    print("\n--- Zona 10h-11h BRT ---")
    print(f"Trades: {t_10_11} | P&L: R$ {pnl_10_11:+.2f}")
    pct = abs(pnl_10_11) / abs(total_pnl) * 100 if (pnl_10_11 < 0 and total_pnl < 0) else 0
    if pnl_10_11 < 0 and total_pnl < 0:
        print(f"→ {pct:.0f}% do prejuízo total vem dessa faixa.")
    print("\n--- CONFIRMAÇÃO: 10h-11h BRT continua com muitas perdas? ---")
    if pnl_10_11 < -10 and pct > 40:
        print("SIM. A faixa 10h-11h BRT concentra >40% do prejuízo.")
    else:
        print("Parcialmente. Ver tabela acima para detalhes.")
