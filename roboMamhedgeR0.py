import pandas as pd
import numpy as np
import os

# Tenta carregar, se não existir, cria dados sintéticos para teste
file_name = "WIN_5min.csv"
if os.path.exists(file_name):
    df = pd.read_csv(file_name)
else:
    print("Aviso: CSV não encontrado. Gerando dados de teste...")
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    df = pd.DataFrame({'close': np.random.randn(100).cumsum() + 100000, 
                       'high': 100100, 'low': 99900}, index=dates)

# Cálculo das Médias (EMA) conforme seu código original
df['EMA9'] = df['close'].ewm(span=9).mean()
df['EMA21'] = df['close'].ewm(span=21).mean()

# TR e ATR
df['TR'] = np.maximum(df['high']-df['low'],
             np.maximum(abs(df['high']-df['close'].shift()),
                        abs(df['low']-df['close'].shift())))
df['ATR'] = df['TR'].rolling(14).mean()
df['ATR_mean'] = df['ATR'].rolling(20).mean()

# Lógica de Sinais
df['signal'] = 0
df.loc[(df['EMA9'] > df['EMA21']) & (df['ATR'] > df['ATR_mean']), 'signal'] = 1
df.loc[(df['EMA9'] < df['EMA21']) & (df['ATR'] > df['ATR_mean']), 'signal'] = -1

print(df[['close', 'EMA9', 'EMA21', 'signal']].tail())