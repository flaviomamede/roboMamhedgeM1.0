"""
Script de inspeção para validar cálculos do robô linha a linha.
Executa o mesmo pipeline do R2 e exibe um relatório detalhado para um instante específico.
"""
import pandas as pd
import numpy as np

# --- Carregar e calcular (igual ao R2) ---
df = pd.read_csv("WIN_5min.csv", index_col=0, parse_dates=True)
df.columns = df.columns.str.lower()

df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
df['Momentum'] = df['close'] - df['close'].shift(10)

df['TR'] = np.maximum(df['high'] - df['low'],
    np.maximum(abs(df['high'] - df['close'].shift()),
               abs(df['low'] - df['close'].shift())))
df['ATR'] = df['TR'].rolling(14).mean()

# --- Encontrar o índice da PRIMEIRA entrada (compra ou venda) ---
first_entry_idx = None
position = 0
for i in range(1, len(df)):
    if df.index[i].hour < 10 or df.index[i].hour >= 17:
        continue
    atr_val = df['ATR'].iloc[i]
    ema200 = df['EMA200'].iloc[i]
    momentum = df['Momentum'].iloc[i]
    if pd.isna(atr_val) or pd.isna(ema200) or pd.isna(momentum):
        continue
    if position == 0:
        if (df['EMA9'].iloc[i] > df['EMA21'].iloc[i] and
            df['close'].iloc[i] > ema200 and momentum > 0):
            first_entry_idx = i
            break
        elif (df['EMA9'].iloc[i] < df['EMA21'].iloc[i] and
              df['close'].iloc[i] < ema200 and momentum < 0):
            first_entry_idx = i
            break

if first_entry_idx is None:
    print("Nenhuma entrada encontrada. Tentando índice 300 (após warmup)...")
    first_entry_idx = 300

idx = first_entry_idx
row = df.iloc[idx]

print("=" * 70)
print(f"INSPEÇÃO MANUAL — Linha {idx} | Data/Hora: {df.index[idx]}")
print("=" * 70)

# --- 1. DADOS BRUTOS ---
print("\n--- 1. DADOS BRUTOS (OHLC) ---")
print(f"  open:  {row.get('open', 'N/A'):.4f}" if 'open' in df.columns else "")
print(f"  high:  {row['high']:.4f}")
print(f"  low:   {row['low']:.4f}")
print(f"  close: {row['close']:.4f}")

# --- 2. MOMENTUM (manual) ---
close_at = row['close']
close_10_ago = df['close'].iloc[idx - 10] if idx >= 10 else np.nan
momentum_calc = close_at - close_10_ago
print("\n--- 2. MOMENTUM (close - close[10 atrás]) ---")
print(f"  close atual:     {close_at:.4f}")
print(f"  close 10 atrás:  {close_10_ago:.4f}")
print(f"  Momentum = {close_at:.4f} - {close_10_ago:.4f} = {momentum_calc:.4f}")
print(f"  Pandas df['Momentum']: {row['Momentum']:.4f}")
print(f"  ✓ OK" if abs(momentum_calc - row['Momentum']) < 1e-6 else "  ✗ DIFERENTE!")

# --- 3. TR (True Range) manual ---
high = row['high']
low = row['low']
prev_close = df['close'].iloc[idx - 1] if idx >= 1 else np.nan
tr1 = high - low
tr2 = abs(high - prev_close)
tr3 = abs(low - prev_close)
tr_manual = max(tr1, tr2, tr3)
print("\n--- 3. TRUE RANGE ---")
print(f"  TR1 = high - low = {high:.4f} - {low:.4f} = {tr1:.4f}")
print(f"  TR2 = |high - prev_close| = |{high:.4f} - {prev_close:.4f}| = {tr2:.4f}")
print(f"  TR3 = |low - prev_close| = |{low:.4f} - {prev_close:.4f}| = {tr3:.4f}")
print(f"  TR = max(TR1, TR2, TR3) = {tr_manual:.4f}")
print(f"  Pandas df['TR']: {row['TR']:.4f}")
print(f"  ✓ OK" if abs(tr_manual - row['TR']) < 1e-6 else "  ✗ DIFERENTE!")

# --- 4. ATR (média dos últimos 14 TR) ---
tr_window = df['TR'].iloc[idx-13:idx+1]
atr_manual = tr_window.mean()
print("\n--- 4. ATR (média dos últimos 14 TR) ---")
print(f"  TRs usados: índices {idx-13} a {idx}")
print(f"  ATR manual = mean(TR) = {atr_manual:.4f}")
print(f"  Pandas df['ATR']: {row['ATR']:.4f}")
print(f"  ✓ OK" if abs(atr_manual - row['ATR']) < 1e-6 else "  ✗ DIFERENTE!")

# --- 5. EMA (fórmula: α=2/(span+1), EMA = α*close + (1-α)*EMA_prev) ---
def ema_manual(series, span, idx):
    alpha = 2 / (span + 1)
    ema = series.iloc[0]
    for j in range(1, idx + 1):
        ema = alpha * series.iloc[j] + (1 - alpha) * ema
    return ema

ema9_manual = ema_manual(df['close'], 9, idx)
ema21_manual = ema_manual(df['close'], 21, idx)
ema200_manual = ema_manual(df['close'], 200, idx)

print("\n--- 5. EMAs (fórmula: α=2/(span+1), EMA = α*close + (1-α)*EMA_prev) ---")
print(f"  EMA9  manual: {ema9_manual:.4f}  |  Pandas: {row['EMA9']:.4f}")
print(f"  EMA21 manual: {ema21_manual:.4f}  |  Pandas: {row['EMA21']:.4f}")
print(f"  EMA200 manual: {ema200_manual:.4f}  |  Pandas: {row['EMA200']:.4f}")

# --- 6. CONDIÇÕES DE ENTRADA ---
print("\n--- 6. CONDIÇÕES DE ENTRADA ---")
c1 = row['EMA9'] > row['EMA21']
c2 = row['close'] > row['EMA200']
c3 = row['Momentum'] > 0
c4 = row['EMA9'] < row['EMA21']
c5 = row['close'] < row['EMA200']
c6 = row['Momentum'] < 0

print("  COMPRA (todas devem ser True):")
print(f"    EMA9 > EMA21?     {row['EMA9']:.4f} > {row['EMA21']:.4f}  →  {c1}")
print(f"    close > EMA200?   {row['close']:.4f} > {row['EMA200']:.4f}  →  {c2}")
print(f"    Momentum > 0?    {row['Momentum']:.4f} > 0  →  {c3}")
print(f"    → Compra válida?  {c1 and c2 and c3}")

print("  VENDA (todas devem ser True):")
print(f"    EMA9 < EMA21?     {row['EMA9']:.4f} < {row['EMA21']:.4f}  →  {c4}")
print(f"    close < EMA200?   {row['close']:.4f} < {row['EMA200']:.4f}  →  {c5}")
print(f"    Momentum < 0?     {row['Momentum']:.4f} < 0  →  {c6}")
print(f"    → Venda válida?   {c4 and c5 and c6}")

# --- 7. STOP LOSS e TAKE PROFIT (se entrou) ---
if c1 and c2 and c3:
    entry = row['close']
    sl = entry - 1.5 * row['ATR']
    tp = entry + 2.0 * row['ATR']
    print("\n--- 7. SE COMPROU NESTE CANDLE ---")
    print(f"  Entry:      {entry:.4f}")
    print(f"  Stop Loss:  {entry:.4f} - 1.5×{row['ATR']:.4f} = {sl:.4f}")
    print(f"  Take Profit: {entry:.4f} + 2.0×{row['ATR']:.4f} = {tp:.4f}")
    print(f"  Low do candle: {row['low']:.4f}  (atingiu SL? {row['low'] <= sl})")
    print(f"  High do candle: {row['high']:.4f}  (atingiu TP? {row['high'] >= tp})")
elif c4 and c5 and c6:
    entry = row['close']
    sl = entry + 1.5 * row['ATR']
    tp = entry - 2.0 * row['ATR']
    print("\n--- 7. SE VENDEU NESTE CANDLE ---")
    print(f"  Entry:      {entry:.4f}")
    print(f"  Stop Loss:  {entry:.4f} + 1.5×{row['ATR']:.4f} = {sl:.4f}")
    print(f"  Take Profit: {entry:.4f} - 2.0×{row['ATR']:.4f} = {tp:.4f}")
    print(f"  High do candle: {row['high']:.4f}  (atingiu SL? {row['high'] >= sl})")
    print(f"  Low do candle: {row['low']:.4f}  (atingiu TP? {row['low'] <= tp})")

# --- 8. SIMULAR PRIMEIRO TRADE COMPLETO (entrada → saída) ---
print("\n--- 8. PRIMEIRO TRADE COMPLETO (entrada até saída) ---")
position = 0
entry_price = None
stop_loss = None
take_profit = None
trade_direction = None
for i in range(1, min(len(df), 500)):
    if df.index[i].hour < 10 or df.index[i].hour >= 17:
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
            trade_direction = "COMPRA"
            entry_idx = i
        elif (df['EMA9'].iloc[i] < df['EMA21'].iloc[i] and
              df['close'].iloc[i] < ema200 and momentum < 0):
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price + (1.5 * atr_val)
            take_profit = entry_price - (2.0 * atr_val)
            position = -1
            trade_direction = "VENDA"
            entry_idx = i
    elif position == 1:
        if df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit:
            exit_price = stop_loss if df['low'].iloc[i] <= stop_loss else take_profit
            pnl = exit_price - entry_price - 1.0
            print(f"  Entrada idx {entry_idx}: {df.index[entry_idx]} @ {entry_price:.4f}")
            print(f"  Saída idx {i}: {df.index[i]} @ {exit_price:.4f}")
            print(f"  SL={stop_loss:.4f} TP={take_profit:.4f}")
            print(f"  P&L = {exit_price:.4f} - {entry_price:.4f} - 1.0 = R$ {pnl:.4f}")
            break
    elif position == -1:
        if df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit:
            exit_price = stop_loss if df['high'].iloc[i] >= stop_loss else take_profit
            pnl = entry_price - exit_price - 1.0
            print(f"  Entrada idx {entry_idx}: {df.index[entry_idx]} @ {entry_price:.4f}")
            print(f"  Saída idx {i}: {df.index[i]} @ {exit_price:.4f}")
            print(f"  SL={stop_loss:.4f} TP={take_profit:.4f}")
            print(f"  P&L (venda) = {entry_price:.4f} - {exit_price:.4f} - 1.0 = R$ {pnl:.4f}")
            break

print("\n" + "=" * 70)
