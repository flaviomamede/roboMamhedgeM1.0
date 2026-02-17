"""
Diagnóstico: por que até o ALEATÓRIO tem ~5% de acerto?
"""
import numpy as np
import pandas as pd
from utils_fuso import converter_para_brt, dentro_horario_operacao

df = pd.read_csv("WIN_5min.csv", index_col=0, parse_dates=True)
df.columns = df.columns.str.lower()
df = converter_para_brt(df)

df['TR'] = np.maximum(df['high']-df['low'],
    np.maximum(abs(df['high']-df['close'].shift()),
               abs(df['low']-df['close'].shift())))
df['ATR'] = df['TR'].rolling(14).mean()

# Contar quantos trades saem por STOP vs TARGET
np.random.seed(42)
exits_stop = 0
exits_target = 0
position = 0
entry_price = 0
stop_loss = 0
take_profit = 0

for i in range(1, len(df)):
    if not dentro_horario_operacao(df.index[i]):
        continue
    atr = df['ATR'].iloc[i]
    if np.isnan(atr):
        continue
    if position == 0:
        if np.random.rand() < 0.5:
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price - 1.5 * atr
            take_profit = entry_price + 2.0 * atr
            position = 1
        else:
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price + 1.5 * atr
            take_profit = entry_price - 2.0 * atr
            position = -1
    else:
        hit_stop = (position == 1 and df['low'].iloc[i] <= stop_loss) or \
                   (position == -1 and df['high'].iloc[i] >= stop_loss)
        hit_target = (position == 1 and df['high'].iloc[i] >= take_profit) or \
                     (position == -1 and df['low'].iloc[i] <= take_profit)
        if hit_stop and hit_target:
            # Ambos no mesmo candle: qual aconteceu primeiro? Não sabemos.
            # Estamos assumindo STOP primeiro (ordem do if)
            exits_stop += 1
        elif hit_stop:
            exits_stop += 1
        elif hit_target:
            exits_target += 1
        if hit_stop or hit_target:
            position = 0

print("=" * 70)
print("DIAGNÓSTICO: STOP vs TARGET")
print("=" * 70)
total = exits_stop + exits_target
print(f"Saídas por STOP:   {exits_stop} ({100*exits_stop/total:.1f}%)")
print(f"Saídas por TARGET: {exits_target} ({100*exits_target/total:.1f}%)")
print(f"\n→ Se ~95% saem por STOP, o stop está MUITO apertado ou o target longe.")

# Verificar escala: ATR típico vs custo
atr_medio = df['ATR'].dropna().mean()
gain_medio = 2.0 * atr_medio
loss_medio = 1.5 * atr_medio
print(f"\n--- ESCALA ---")
print(f"ATR médio: {atr_medio:.4f}")
print(f"Ganho típico (2×ATR): {gain_medio:.4f}")
print(f"Perda típica (1.5×ATR): {loss_medio:.4f}")
print(f"Custo por trade: 1.0")
print(f"→ Ganho líquido se acerta: {gain_medio - 1.0:.4f}")
print(f"→ Perda líquida se erra: {-loss_medio - 1.0:.4f}")
if gain_medio < 1.0:
    print(f"\n⚠ CUSTO COME O GANHO! Mesmo acertando, ganhamos {gain_medio:.2f} - 1.0 = {gain_medio-1:.2f} (negativo!)")

# Teste SEM CUSTO
print("\n" + "=" * 70)
print("TESTE: Baseline ALEATÓRIO SEM CUSTO (custo=0)")
print("=" * 70)
np.random.seed(42)
trades_sem_custo = []
position = 0
for i in range(1, len(df)):
    if not dentro_horario_operacao(df.index[i]):
        continue
    atr = df['ATR'].iloc[i]
    if np.isnan(atr):
        continue
    if position == 0:
        if np.random.rand() < 0.5:
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price - 1.5 * atr
            take_profit = entry_price + 2.0 * atr
            position = 1
        else:
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price + 1.5 * atr
            take_profit = entry_price - 2.0 * atr
            position = -1
    else:
        if position == 1:
            if df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit:
                exit_p = stop_loss if df['low'].iloc[i] <= stop_loss else take_profit
                trades_sem_custo.append(exit_p - entry_price)  # SEM -1.0
                position = 0
        else:
            if df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit:
                exit_p = stop_loss if df['high'].iloc[i] >= stop_loss else take_profit
                trades_sem_custo.append(entry_price - exit_p)  # SEM -1.0
                position = 0

trades_sem_custo = np.array(trades_sem_custo)
wr_sc = (trades_sem_custo > 0).mean() * 100
print(f"Win Rate SEM custo: {wr_sc:.1f}%")
print(f"E[P&L] SEM custo: R$ {trades_sem_custo.mean():.2f}")
