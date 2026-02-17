#!/usr/bin/env python3
"""
Gerador de Dados de Exemplo para Teste do BCP
Simula série do IBOVESPA com mudanças de regime
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_ibovespa(n_days=30, points_per_day=78, 
                               initial_price=125000, 
                               add_regime_changes=True):
    """
    Gera dados sintéticos simulando IBOVESPA com:
    - Períodos de tendência de alta
    - Períodos de tendência de baixa  
    - Mudanças de regime (pontos de inflexão)
    - Volatilidade realística
    
    Parâmetros:
    -----------
    n_days : int
        Número de dias a simular
    points_per_day : int
        Pontos de dados por dia (78 = mercado de 6.5h em candles de 5min)
    initial_price : float
        Preço inicial
    add_regime_changes : bool
        Se True, adiciona mudanças claras de regime
    """
    
    n_total = n_days * points_per_day
    
    # Gera timestamps (mercado 10:00-16:30, segunda a sexta)
    timestamps = []
    current_date = datetime(2026, 1, 15, 10, 0, 0)  # começa em jan/2026
    
    for i in range(n_total):
        timestamps.append(current_date)
        current_date += timedelta(minutes=5)
        
        # Pula para próximo dia útil
        if current_date.hour >= 16 and current_date.minute >= 30:
            current_date = current_date.replace(hour=10, minute=0)
            current_date += timedelta(days=1)
            
            # Pula fins de semana
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
    
    # Gera série de preços com mudanças de regime
    np.random.seed(42)
    
    if add_regime_changes:
        # Define regimes com mudanças claras
        regime_lengths = [
            int(n_total * 0.25),  # Regime 1: 25% - Alta
            int(n_total * 0.20),  # Regime 2: 20% - Lateral
            int(n_total * 0.30),  # Regime 3: 30% - Baixa (REVERSÃO!)
            int(n_total * 0.25),  # Regime 4: 25% - Recuperação
        ]
        
        # Ajusta para somar exatamente n_total
        regime_lengths[-1] = n_total - sum(regime_lengths[:-1])
        
        regime_params = [
            {'drift': 0.0003, 'vol': 0.0025},  # Alta moderada
            {'drift': 0.0000, 'vol': 0.0015},  # Lateral (baixa vol)
            {'drift': -0.0004, 'vol': 0.0035}, # Baixa forte (REVERSÃO!)
            {'drift': 0.0002, 'vol': 0.0020},  # Recuperação
        ]
        
        returns = []
        for length, params in zip(regime_lengths, regime_params):
            regime_returns = np.random.normal(
                params['drift'], 
                params['vol'], 
                length
            )
            returns.extend(regime_returns)
        
        returns = np.array(returns)
        
    else:
        # Série simples sem mudanças claras
        drift = 0.0001
        vol = 0.002
        returns = np.random.normal(drift, vol, n_total)
    
    # Converte retornos em preços
    log_prices = np.log(initial_price) + np.cumsum(returns)
    prices = np.exp(log_prices)
    
    # Adiciona ruído de microestrutura (bid-ask bounce)
    prices += np.random.normal(0, initial_price * 0.0001, n_total)
    
    # Cria DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.0005, n_total)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_total))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_total))),
        'volume': np.random.lognormal(15, 1, n_total).astype(int)
    })
    
    # Garante High >= Close >= Low
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)
    df['high'] = df[['high', 'open']].max(axis=1)
    df['low'] = df[['low', 'open']].min(axis=1)
    
    return df


def generate_example_files():
    """
    Gera arquivos de exemplo para teste.
    """
    print("Gerando arquivos de exemplo...")
    print()
    
    # Arquivo 1: Com mudanças de regime claras
    print("1. ibovespa_5min_exemplo_com_reversao.csv")
    df1 = generate_synthetic_ibovespa(
        n_days=30, 
        add_regime_changes=True
    )
    df1.to_csv('ibovespa_5min_exemplo_com_reversao.csv', index=False)
    print(f"   ✓ {len(df1)} registros gerados")
    print(f"   - Período: {df1['timestamp'].iloc[0]} até {df1['timestamp'].iloc[-1]}")
    print(f"   - Preço inicial: {df1['close'].iloc[0]:.2f}")
    print(f"   - Preço final: {df1['close'].iloc[-1]:.2f}")
    print(f"   - Retorno total: {(df1['close'].iloc[-1]/df1['close'].iloc[0] - 1)*100:.2f}%")
    print()
    
    # Arquivo 2: Sem mudanças claras (controle)
    print("2. ibovespa_5min_exemplo_estavel.csv")
    df2 = generate_synthetic_ibovespa(
        n_days=30, 
        add_regime_changes=False
    )
    df2.to_csv('ibovespa_5min_exemplo_estavel.csv', index=False)
    print(f"   ✓ {len(df2)} registros gerados")
    print(f"   - Período: {df2['timestamp'].iloc[0]} até {df2['timestamp'].iloc[-1]}")
    print(f"   - Preço inicial: {df2['close'].iloc[0]:.2f}")
    print(f"   - Preço final: {df2['close'].iloc[-1]:.2f}")
    print(f"   - Retorno total: {(df2['close'].iloc[-1]/df2['close'].iloc[0] - 1)*100:.2f}%")
    print()
    
    print("Arquivos gerados com sucesso!")
    print()
    print("Para testar o detector, execute:")
    print()
    print("  python ibovespa_bcp_reversal_detector.py ibovespa_5min_exemplo_com_reversao.csv")
    print()
    print("Ou:")
    print()
    print("  python ibovespa_bcp_reversal_detector.py ibovespa_5min_exemplo_estavel.csv")
    print()
    
    return df1, df2


if __name__ == "__main__":
    # Gera arquivos de exemplo
    df_com_reversao, df_estavel = generate_example_files()
    
    # Mostra estatísticas
    print("="*70)
    print("ESTATÍSTICAS DOS DADOS GERADOS")
    print("="*70)
    print()
    
    print("Arquivo COM reversão:")
    returns1 = np.log(df_com_reversao['close'] / df_com_reversao['close'].shift(1)).dropna()
    print(f"  - Retorno médio: {returns1.mean():.6f}")
    print(f"  - Volatilidade: {returns1.std():.6f}")
    print(f"  - Skewness: {returns1.skew():.4f}")
    print(f"  - Kurtosis: {returns1.kurtosis():.4f}")
    print()
    
    print("Arquivo ESTÁVEL:")
    returns2 = np.log(df_estavel['close'] / df_estavel['close'].shift(1)).dropna()
    print(f"  - Retorno médio: {returns2.mean():.6f}")
    print(f"  - Volatilidade: {returns2.std():.6f}")
    print(f"  - Skewness: {returns2.skew():.4f}")
    print(f"  - Kurtosis: {returns2.kurtosis():.4f}")
