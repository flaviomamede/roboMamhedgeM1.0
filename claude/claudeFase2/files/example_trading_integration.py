#!/usr/bin/env python3
"""
EXEMPLO DE INTEGRAÇÃO - Trading Automatizado com BCP

Este script mostra como integrar o detector BCP em um sistema de trading.
"""

import pandas as pd
import numpy as np
from fast_bcp_detector import FastBCPDetector


class BCPTradingSignal:
    """
    Integra BCP em sistema de trading para gerar sinais.
    """
    
    def __init__(self, window=200, p0=0.18, threshold_high=75, threshold_low=40):
        """
        Parâmetros:
        -----------
        window : int
            Janela para análise BCP
        p0 : float
            Prior de probabilidade de mudança
        threshold_high : float
            Percentil acima do qual emitimos sinal de ALERTA
        threshold_low : float
            Percentil abaixo do qual consideramos mercado ESTÁVEL
        """
        self.detector = FastBCPDetector(window=window, p0=p0)
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        
    def update(self, price_history):
        """
        Atualiza detector com histórico de preços.
        
        Parâmetros:
        -----------
        price_history : array-like
            Série histórica de preços (pelo menos 'window' observações)
        
        Retorna:
        --------
        dict : Sinal de trading atual
        """
        # Calcula retornos
        prices = np.array(price_history)
        returns = np.diff(np.log(prices))
        
        if len(returns) < self.detector.window:
            return {
                'signal': 'WAIT',
                'reason': 'Dados insuficientes',
                'probability': 0.0,
                'percentile': 0.0
            }
        
        # Ajusta detector
        self.detector.fit(returns)
        
        # Status atual
        status = self.detector.get_current_status()
        
        # Gera sinal
        prob = status['prob_mudanca_atual']
        percentile = status['percentil']
        
        if percentile >= self.threshold_high:
            signal = 'REDUCE'  # Reduzir exposição
            reason = f'Alta prob. reversão ({prob:.1%}, P{percentile:.0f})'
        elif percentile >= 60:
            signal = 'CAUTION'  # Cautela
            reason = f'Prob. moderada ({prob:.1%}, P{percentile:.0f})'
        elif percentile <= self.threshold_low:
            signal = 'HOLD'  # Manter posição
            reason = f'Mercado estável ({prob:.1%}, P{percentile:.0f})'
        else:
            signal = 'MONITOR'  # Apenas monitorar
            reason = f'Normal ({prob:.1%}, P{percentile:.0f})'
        
        return {
            'signal': signal,
            'reason': reason,
            'probability': prob,
            'percentile': percentile,
            'volatility': status['volatilidade_atual'],
            'trend': status['tendencia'],
            'force': status['forca_sinal']
        }


def example_live_trading_loop():
    """
    Exemplo de loop de trading ao vivo com BCP.
    """
    # Inicializa detector
    bcp_signal = BCPTradingSignal(
        window=200,
        p0=0.18,
        threshold_high=75,
        threshold_low=40
    )
    
    # Simula histórico de preços (em produção, viria de API)
    df = pd.read_csv('ibovespa_5min_exemplo_com_reversao.csv')
    prices = df['close'].values
    
    # Configuração de trading
    position_size = 1.0  # Tamanho inicial da posição (100%)
    max_position = 1.0
    min_position = 0.2
    
    print("="*70)
    print("SIMULAÇÃO DE TRADING COM BCP")
    print("="*70)
    print()
    
    # Loop principal (simula pontos no tempo)
    checkpoints = [500, 1000, 1500, 2000, len(prices)-1]
    
    for i in checkpoints:
        # Pega histórico até este momento
        price_history = prices[:i+1]
        
        # Atualiza BCP
        result = bcp_signal.update(price_history)
        
        # Decisão de trading baseada no sinal
        if result['signal'] == 'REDUCE':
            # Reduz posição para 40% quando há alerta alto
            position_size = max(min_position, position_size * 0.4)
            action = f"⚠️  REDUZIR posição para {position_size:.0%}"
            
        elif result['signal'] == 'CAUTION':
            # Reduz posição para 70% em cautela
            position_size = max(min_position, position_size * 0.7)
            action = f"⚡ CAUTELA - posição em {position_size:.0%}"
            
        elif result['signal'] == 'HOLD':
            # Pode aumentar posição gradualmente
            position_size = min(max_position, position_size * 1.1)
            action = f"✅ HOLD - posição em {position_size:.0%}"
            
        else:  # MONITOR
            # Mantém posição atual
            action = f"👁️  MONITOR - posição em {position_size:.0%}"
        
        # Log da decisão
        print(f"t={i:4d} | Preço: {price_history[-1]:,.2f} | "
              f"P={result['probability']:.1%} (P{result['percentile']:.0f}) | "
              f"{action}")
        print(f"         Razão: {result['reason']}")
        print(f"         Tendência: {result['trend']}, Vol: {result['volatility']:.3%}")
        print()
    
    print("="*70)
    print(f"Posição final: {position_size:.0%}")
    print("="*70)


def example_risk_adjusted_sizing():
    """
    Exemplo de ajuste dinâmico de posição baseado em BCP.
    """
    print("\n" + "="*70)
    print("DIMENSIONAMENTO DINÂMICO DE POSIÇÃO COM BCP")
    print("="*70)
    print()
    
    # Carrega dados
    df = pd.read_csv('ibovespa_5min_exemplo_com_reversao.csv')
    prices = df['close'].values
    
    # Calcula retornos
    returns = np.diff(np.log(prices))
    
    # Detector BCP
    detector = FastBCPDetector(window=200, p0=0.18)
    detector.fit(returns)
    
    # Status atual
    status = detector.get_current_status()
    
    # Regras de dimensionamento
    base_position = 1.0  # Posição base: 100% do capital
    
    # Ajusta baseado na probabilidade de mudança
    prob_factor = 1.0 - (status['prob_mudanca_atual'] ** 2)  # Quanto maior prob, menor posição
    
    # Ajusta baseado na volatilidade
    vol_factor = 1.0 / (1.0 + status['volatilidade_atual'] * 100)  # Maior vol, menor posição
    
    # Ajusta baseado no percentil
    percentile_factor = 1.0 - (status['percentil'] / 100) * 0.5  # Percentil alto = reduz
    
    # Posição final
    adjusted_position = base_position * prob_factor * vol_factor * percentile_factor
    adjusted_position = np.clip(adjusted_position, 0.2, 1.0)  # Entre 20% e 100%
    
    print(f"Análise de Dimensionamento:")
    print(f"  • Posição Base: {base_position:.0%}")
    print(f"  • Fator Probabilidade: {prob_factor:.2f} (prob={status['prob_mudanca_atual']:.1%})")
    print(f"  • Fator Volatilidade: {vol_factor:.2f} (vol={status['volatilidade_atual']:.3%})")
    print(f"  • Fator Percentil: {percentile_factor:.2f} (P{status['percentil']:.0f})")
    print(f"  • Posição Ajustada: {adjusted_position:.0%}")
    print()
    
    # Interpretação
    if adjusted_position < 0.4:
        print("⚠️  POSIÇÃO DEFENSIVA - Alto risco de reversão")
    elif adjusted_position < 0.7:
        print("⚡ POSIÇÃO CAUTELOSA - Risco moderado")
    else:
        print("✅ POSIÇÃO NORMAL - Baixo risco de mudança")


def example_stop_loss_adjustment():
    """
    Exemplo de ajuste dinâmico de stop loss com BCP.
    """
    print("\n" + "="*70)
    print("AJUSTE DINÂMICO DE STOP LOSS COM BCP")
    print("="*70)
    print()
    
    # Carrega dados
    df = pd.read_csv('ibovespa_5min_exemplo_com_reversao.csv')
    prices = df['close'].values
    current_price = prices[-1]
    
    # Calcula retornos
    returns = np.diff(np.log(prices))
    
    # Detector BCP
    detector = FastBCPDetector(window=200, p0=0.18)
    detector.fit(returns)
    status = detector.get_current_status()
    
    # Stop loss base (ex: 5% abaixo)
    base_stop_distance = 0.05
    
    # Ajusta stop baseado em BCP
    # Quanto maior a prob de reversão, stop mais apertado
    prob_adjustment = 1.0 - (status['prob_mudanca_atual'] * 0.5)
    
    # Ajusta baseado em volatilidade
    # Maior volatilidade = stop mais largo para não ser pego por ruído
    vol_adjustment = 1.0 + (status['volatilidade_atual'] * 50)
    
    # Stop final
    adjusted_stop_distance = base_stop_distance * prob_adjustment * vol_adjustment
    adjusted_stop_distance = np.clip(adjusted_stop_distance, 0.02, 0.10)  # Entre 2% e 10%
    
    stop_loss_price = current_price * (1 - adjusted_stop_distance)
    
    print(f"Configuração de Stop Loss:")
    print(f"  • Preço Atual: {current_price:,.2f}")
    print(f"  • Stop Base: {base_stop_distance:.1%} → {current_price * (1-base_stop_distance):,.2f}")
    print(f"  • Ajuste por Prob: {prob_adjustment:.2f}x")
    print(f"  • Ajuste por Vol: {vol_adjustment:.2f}x")
    print(f"  • Stop Ajustado: {adjusted_stop_distance:.1%} → {stop_loss_price:,.2f}")
    print()
    
    if adjusted_stop_distance < 0.03:
        print("⚠️  STOP APERTADO - Alta probabilidade de reversão")
        print("    Protege capital mas pode ser ativado por ruído.")
    elif adjusted_stop_distance > 0.07:
        print("📊 STOP LARGO - Acomoda volatilidade atual")
        print("    Menos proteção mas evita stops falsos.")
    else:
        print("✅ STOP BALANCEADO - Configuração equilibrada")


if __name__ == "__main__":
    print("="*70)
    print("EXEMPLOS DE INTEGRAÇÃO BCP EM TRADING")
    print("="*70)
    print()
    
    # Exemplo 1: Loop de trading ao vivo
    example_live_trading_loop()
    
    # Exemplo 2: Dimensionamento de posição
    example_risk_adjusted_sizing()
    
    # Exemplo 3: Ajuste de stop loss
    example_stop_loss_adjustment()
    
    print("\n" + "="*70)
    print("CONCLUSÃO")
    print("="*70)
    print()
    print("O BCP pode ser integrado de várias formas:")
    print()
    print("1. 📊 Sinais de Trading")
    print("   - REDUCE/CAUTION/HOLD baseado em percentil")
    print()
    print("2. 💰 Dimensionamento de Posição")
    print("   - Ajusta % do capital baseado em prob. mudança")
    print()
    print("3. 🛡️  Stop Loss Dinâmico")
    print("   - Aperta/alarga stop conforme contexto")
    print()
    print("4. ⚡ Gestão de Risco")
    print("   - Reduz exposição quando instabilidade aumenta")
    print()
    print("Use estes exemplos como ponto de partida para")
    print("integrar o BCP no seu sistema de trading!")
    print()
