"""
Configurações centralizadas do roboMT5.
Parâmetros de indicadores e estratégia para facilitar ajustes e otimização.
"""

# ==================== R6 — EMA4 + RSI Peak + MACD + BB ====================
# Indicadores
RSI_PERIOD = 14
RSI_BULLISH = 40
RSI_BULLISH_WINDOW = 5

EMA_FAST = 4
EMA_SLOW = 12

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD = 2
BB_USE = False  # True = exige preço <= BB (oversold)
BB_ENTRY = 'mid'  # 'low' = bb_low (estrito), 'mid' = bb_mid (menos restritivo)

ATR_PERIOD = 14

# Gestão de risco
STOP_ATR_MULT = 2.0
TARGET_ATR_MULT = 3.0  # Para robôs que usam take profit
