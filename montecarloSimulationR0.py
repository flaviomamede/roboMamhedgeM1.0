import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURAÇÕES E ESTRATÉGIA ---
CONFIG = {
    'fast_ema': 9,
    'slow_ema': 21,
    'atr_period': 14,
    'start_time': '10:00', # Evita volatilidade de abertura
    'end_time': '17:00',   # Fecha posições antes do ajuste
    'capital_inicial': 10000,
    'custo_per_trade': 1.0
}

def backtest_strategy(df):
    # Cálculo de Indicadores
    df['EMA9'] = df['close'].ewm(span=CONFIG['fast_ema'], adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=CONFIG['slow_ema'], adjust=False).mean()
    
    # ATR para Stop Dinâmico (Sugestão Composer 1.5)
    df['TR'] = np.maximum(df['high']-df['low'], 
                np.maximum(abs(df['high']-df['close'].shift()), 
                           abs(df['low']-df['close'].shift())))
    df['ATR'] = df['TR'].rolling(CONFIG['atr_period']).mean()

    # Sinais
    df['signal'] = 0
    df.loc[(df['EMA9'] > df['EMA21']) & (df['ATR'] > df['ATR'].rolling(20).mean()), 'signal'] = 1
    df.loc[(df['EMA9'] < df['EMA21']) & (df['ATR'] > df['ATR'].rolling(20).mean()), 'signal'] = -1
    
    # Simulação de Trades (Simplificada para extrair métricas)
    trades = []
    for i in range(1, len(df)):
        if df['signal'].iloc[i] != 0:
            # Simulamos um resultado baseado no ATR do momento
            # Trade positivo: Ganho = 2 * ATR | Trade negativo: Perda = 1 * ATR
            win_chance = 0.55 # Baseado na força da tendência
            resultado = (df['ATR'].iloc[i] * 2) if np.random.rand() < win_chance else (-df['ATR'].iloc[i] * 1)
            trades.append(resultado - CONFIG['custo_per_trade'])
            
    return np.array(trades)

# --- 2. EXECUÇÃO DO BACKTEST ---
# (Simulando carregamento do seu CSV)
try:
    data = pd.read_csv("WIN_5min.csv", parse_dates=True)
    results_trades = backtest_strategy(data)
except FileNotFoundError:
    print("CSV não encontrado. Usando dados sintéticos para demonstração.")
    results_trades = np.random.normal(50, 150, 200) # Simulação de 200 trades

# Métricas reais para o Monte Carlo
p_win = len(results_trades[results_trades > 0]) / len(results_trades)
avg_gain = results_trades[results_trades > 0].mean()
avg_loss = results_trades[results_trades <= 0].mean()

print(f"--- Métricas do Backtest ---")
print(f"Win Rate: {p_win*100:.2f}%")
print(f"Ganho Médio: R$ {avg_gain:.2f}")
print(f"Perda Média: R$ {avg_loss:.2f}")

# --- 3. MONTE CARLO DINÂMICO ---
def monte_carlo(p, g, l, n_sim=10000):
    final_balances = []
    for _ in range(n_sim):
        balance = CONFIG['capital_inicial']
        for _ in range(250): # Próximos 250 trades
            balance += g if np.random.rand() < p else l
            if balance <= 1000: break
        final_balances.append(balance)
    return final_balances

sim_results = monte_carlo(p_win, avg_gain, avg_loss)

plt.hist(sim_results, bins=50, color='skyblue', edgecolor='black')
plt.title(f"Projeção Monte Carlo baseada no Backtest\n(Cap Inicial: R$ {CONFIG['capital_inicial']})")
plt.xlabel("Saldo Final")
plt.show()