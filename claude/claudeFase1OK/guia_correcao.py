"""
GUIA DE CORREÇÃO - Como corrigir seus robôs R6, R7, R8
=======================================================

Passo a passo para corrigir os erros identificados
"""

print("=" * 80)
print("GUIA DE CORREÇÃO DOS SEUS ROBÔS")
print("=" * 80)

print("""
Os principais problemas encontrados nos seus robôs (R6, R7, R8):

1. ❌ P&L não usa multiplicador WIN (0.20)
2. ❌ Custos podem estar errados
3. ⚠️  Estratégias muito complexas (overfitting)
4. ⚠️  Peak detection com delay de 1 vela
5. ⚠️  Sem filtro de regime (operam em mercado lateral)

Vou mostrar como corrigir cada um.
""")

# =============================================================================
# CORREÇÃO 1: Adicionar multiplicador WIN
# =============================================================================
print("\n" + "=" * 80)
print("CORREÇÃO 1: Adicionar multiplicador WIN (0.20)")
print("=" * 80)

print("""
❌ CÓDIGO ATUAL (ERRADO):
```python
trades.append((stop_loss - entry_price) - CUSTO_POR_TRADE)
```

Problemas:
- (stop_loss - entry_price) está em PONTOS, não em Reais
- Se diferença = -200 pontos, trata como -R$ 200 (ERRADO!)
- Real = -200 pts × R$ 0,20 = -R$ 40,00

✅ CÓDIGO CORRETO:
```python
# No topo do arquivo, adicione:
MULT_WIN = 0.20  # 1 ponto WIN = R$ 0,20
CUSTO_REAIS = 2.50  # Custo real em Reais (não em pontos!)

# Em cada saída de trade, mude para:
pnl_pontos = (stop_loss - entry_price)
pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
trades.append(pnl_reais)
```

EXEMPLO COMPLETO (para LONG):
```python
if position == 1:
    hit_stop = df['low'].iloc[i] <= stop_loss
    hit_tp = df['high'].iloc[i] >= take_profit
    peak = df['rsi_peak_max'].iloc[i]
    
    if hit_stop:
        pnl_pontos = stop_loss - entry_price
        pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
        trades.append(pnl_reais)
        position = 0
    elif peak:
        pnl_pontos = df['close'].iloc[i] - entry_price
        pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
        trades.append(pnl_reais)
        position = 0
    elif hit_tp:
        pnl_pontos = take_profit - entry_price
        pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
        trades.append(pnl_reais)
        position = 0
```
""")

# =============================================================================
# CORREÇÃO 2: Adicionar filtro ADX (regime de mercado)
# =============================================================================
print("\n" + "=" * 80)
print("CORREÇÃO 2: Adicionar filtro de regime (ADX)")
print("=" * 80)

print("""
Problema: Seus robôs operam em qualquer condição de mercado,
inclusive em mercados LATERAIS onde estratégias de tendência perdem.

✅ SOLUÇÃO: Adicionar filtro ADX

```python
# No início, importar:
from ta.trend import ADXIndicator

# Calcular ADX:
adx_ind = ADXIndicator(df['high'], df['low'], df['close'], window=14)
df['adx'] = adx_ind.adx()

# Definir threshold:
ADX_MIN = 20  # Só opera quando ADX > 20 (tendência mínima)

# Na entrada, adicionar filtro:
if position <= 0:
    # Condições originais...
    rsi_win = df['rsi_bullish_window'].iloc[i]
    macd_val = df['macd_hist'].iloc[i]
    atr_val = df['atr'].iloc[i]
    adx_val = df['adx'].iloc[i]  # ← NOVO
    
    # Verifica ADX
    if pd.isna(adx_val) or adx_val < ADX_MIN:  # ← NOVO
        continue
    
    # ... resto das condições
    if rsi_win and ema4_up and macd_ok and bb_ok:
        # Entra
```

IMPACTO ESPERADO:
- Menos trades (filtrou mercado lateral)
- Maior win rate (só opera em tendências)
- Melhor profit factor
""")

# =============================================================================
# CORREÇÃO 3: Simplificar estratégia
# =============================================================================
print("\n" + "=" * 80)
print("CORREÇÃO 3: Simplificar estratégia (reduzir overfitting)")
print("=" * 80)

print("""
PROBLEMA: R6 usa muitos indicadores:
- EMA4
- RSI + RSI bullish window
- MACD
- Bollinger Bands
- ATR

Isso aumenta chance de overfitting!

✅ SUGESTÃO: Versão simplificada do R6

```python
# INDICADORES (apenas 3):
df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

# ENTRADA SIMPLES:
# Long: Preço cruza acima EMA21 + ADX > 20
if position <= 0:
    adx_val = df['adx'].iloc[i]
    ema_val = df['ema21'].iloc[i]
    close_prev = df['close'].iloc[i-1]
    close_curr = df['close'].iloc[i]
    ema_prev = df['ema21'].iloc[i-1]
    
    # Crossover bullish
    if (close_prev <= ema_prev and close_curr > ema_val and 
        adx_val > 20):
        entry_price = close_curr
        stop_loss = entry_price - 1.5 * atr_val
        take_profit = entry_price + 2.5 * atr_val
        position = 1

# SAÍDA SIMPLES:
# Stop ou Target (sem peak detection)
if position == 1:
    if df['low'].iloc[i] <= stop_loss:
        pnl = (stop_loss - entry_price) * MULT_WIN - CUSTO_REAIS
        trades.append(pnl)
        position = 0
    elif df['high'].iloc[i] >= take_profit:
        pnl = (take_profit - entry_price) * MULT_WIN - CUSTO_REAIS
        trades.append(pnl)
        position = 0
```

BENEFÍCIOS:
- Menos parâmetros = menos overfitting
- Mais robusto em diferentes períodos
- Mais fácil de entender e debugar
""")

# =============================================================================
# CORREÇÃO 4: Melhorar peak detection
# =============================================================================
print("\n" + "=" * 80)
print("CORREÇÃO 4: Melhorar saída (trailing stop)")
print("=" * 80)

print("""
PROBLEMA: Peak detection atual tem delay de 1 vela

```python
# Detecta pico DEPOIS que já passou
df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & 
                      (df['rsi'].shift(1) > df['rsi'])
```

✅ SOLUÇÃO: Trailing stop dinâmico

```python
# Variáveis de estado
trailing_stop_active = False
max_profit = 0

# No loop:
if position == 1:
    current_profit = (df['close'].iloc[i] - entry_price) * MULT_WIN
    
    # Atualiza máximo lucro
    if current_profit > max_profit:
        max_profit = current_profit
    
    # Ativa trailing quando lucro > 1.5 ATR
    if not trailing_stop_active and current_profit >= 1.5 * atr_val * MULT_WIN:
        stop_loss = entry_price  # Move stop para breakeven
        trailing_stop_active = True
    
    # Se trailing ativo, move stop conforme preço sobe
    if trailing_stop_active:
        # Stop = preço atual - 1 ATR (deixa respirar)
        new_stop = df['close'].iloc[i] - 1.0 * atr_val
        if new_stop > stop_loss:
            stop_loss = new_stop
    
    # Checa stop
    if df['low'].iloc[i] <= stop_loss:
        pnl = (stop_loss - entry_price) * MULT_WIN - CUSTO_REAIS
        trades.append(pnl)
        position = 0
        max_profit = 0
        trailing_stop_active = False
```

BENEFÍCIOS:
- Captura mais do movimento
- Protege lucros dinamicamente
- Sem delay de 1 vela
""")

# =============================================================================
# CÓDIGO COMPLETO CORRIGIDO (R6 simplificado)
# =============================================================================
print("\n" + "=" * 80)
print("CÓDIGO COMPLETO: R6 CORRIGIDO E SIMPLIFICADO")
print("=" * 80)

codigo_r6_corrigido = '''
"""
R6 CORRIGIDO - Versão simplificada e robusta
"""
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

# CONFIGURAÇÕES
MULT_WIN = 0.20         # 1 ponto = R$ 0,20
CUSTO_REAIS = 2.50      # Custo por round-trip
ADX_MIN = 20            # ADX mínimo (tendência)
STOP_ATR = 1.5          # Stop em ATR
TARGET_ATR = 2.5        # Target em ATR

def run_backtest(csv_path="WIN_5min.csv"):
    # Carrega dados
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    
    # Indicadores (apenas 3!)
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    # Estado
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trades = []
    
    # Loop
    for i in range(21, len(df)):
        close = df['close'].iloc[i]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        ema = df['ema21'].iloc[i]
        adx = df['adx'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if pd.isna(ema) or pd.isna(adx) or pd.isna(atr):
            continue
        
        # Gestão de posição
        if position == 1:
            # Stop
            if low <= stop_loss:
                pnl_pontos = stop_loss - entry_price
                pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
                trades.append(pnl_reais)
                position = 0
                continue
            # Target
            if high >= take_profit:
                pnl_pontos = take_profit - entry_price
                pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
                trades.append(pnl_reais)
                position = 0
                continue
        
        # Entrada
        if position == 0:
            # Filtro ADX
            if adx < ADX_MIN:
                continue
            
            # Crossover EMA21
            close_prev = df['close'].iloc[i-1]
            ema_prev = df['ema21'].iloc[i-1]
            
            if close_prev <= ema_prev and close > ema:
                entry_price = close
                stop_loss = entry_price - STOP_ATR * atr
                take_profit = entry_price + TARGET_ATR * atr
                position = 1
    
    return np.array(trades) if trades else np.array([])
'''

print(codigo_r6_corrigido)

print("\n" + "=" * 80)
print("RESUMO DAS CORREÇÕES")
print("=" * 80)

print("""
✅ Correções implementadas:

1. Multiplicador WIN (0.20)
   → P&L agora está em Reais, não pontos
   
2. Custos corretos (R$ 2,50)
   → Valores realistas de mercado
   
3. Filtro ADX (> 20)
   → Só opera em tendência, evita lateral
   
4. Simplificação (3 indicadores)
   → Menos overfitting, mais robusto
   
5. Stop/Target em ATR
   → Gestão de risco clara

PRÓXIMOS PASSOS:

1. Aplicar essas correções em R6, R7, R8
2. Rodar backtest novamente
3. Comparar com robô baseline
4. Se ainda ruim, problema é estratégica (não implementação)
""")

print("\n" + "=" * 80)
print("DICA FINAL")
print("=" * 80)

print("""
Se MESMO DEPOIS dessas correções seus robôs ainda perdem:

→ O problema NÃO é implementação
→ O problema É a ESTRATÉGIA em si

Estratégias de tendência simplesmente NÃO funcionam bem
em mercados laterais (70% do tempo).

SOLUÇÕES:
1. Use filtro ADX > 25 (mais rigoroso)
2. Combine com estratégia de reversão
3. Reduza frequência de trades
4. Aumente stop/target (operações swing em vez de scalp)
5. Teste em período diferente (validação out-of-sample)

Lembre-se: "In backtesting we trust, but always verify forward!"
""")
