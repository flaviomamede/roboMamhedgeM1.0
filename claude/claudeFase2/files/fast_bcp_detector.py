#!/usr/bin/env python3
"""
Detector Rápido de Mudança de Regime via Janela Móvel

NOTA IMPORTANTE: Este detector usa uma heurística baseada em razão de
verossimilhança com janela móvel. NÃO é uma implementação do algoritmo
Bayesian Change Point Detection (BCP) de Barry & Hartigan (1993).
Para o BCP verdadeiro com forward-backward, use ibovespa_bcp_reversal_detector.py.

Vantagens desta abordagem heurística:
  - Rápida: O(n · window) vs O(n · max_block) do BCP
  - Simples de entender
  - Boa para monitoramento em tempo real

Limitações:
  - Não marginaliza sobre todas as partições possíveis
  - Probabilidades não são bem calibradas
  - Só testa 4 pontos de quebra por janela
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class FastBCPDetector:
    """
    Versão rápida do BCP usando janela móvel e aproximações.
    
    Ideal para análise em tempo real de ativos de alta frequência.
    """
    
    def __init__(self, window=200, p0=0.2):
        """
        Parâmetros:
        -----------
        window : int
            Tamanho da janela móvel (padrão: 200 períodos)
        p0 : float
            Prior para probabilidade de mudança
        """
        self.window = window
        self.p0 = p0
        
        self.change_probability = None
        self.regime_mean = None
        self.regime_vol = None
        
    def _detect_change_in_window(self, returns_window):
        """
        Detecta probabilidade de mudança em uma janela usando
        abordagem simplificada de razão de verossimilhança.
        """
        n = len(returns_window)
        if n < 20:
            return 0.0, np.mean(returns_window), np.std(returns_window)
        
        # Testa múltiplos pontos de quebra possíveis
        log_likelihoods = []
        split_points = []
        
        # Testa splits em 20%, 40%, 60%, 80% da janela
        for split_pct in [0.2, 0.4, 0.6, 0.8]:
            split = int(n * split_pct)
            if split < 10 or (n - split) < 10:
                continue
                
            # Segmento 1
            seg1 = returns_window[:split]
            mu1, sigma1 = np.mean(seg1), np.std(seg1)
            
            # Segmento 2  
            seg2 = returns_window[split:]
            mu2, sigma2 = np.mean(seg2), np.std(seg2)
            
            # Log-verossimilhança de haver mudança
            if sigma1 > 0 and sigma2 > 0:
                ll_split = (
                    -len(seg1) * np.log(sigma1 + 1e-10) -
                    np.sum((seg1 - mu1)**2) / (2 * sigma1**2 + 1e-10) -
                    len(seg2) * np.log(sigma2 + 1e-10) -
                    np.sum((seg2 - mu2)**2) / (2 * sigma2**2 + 1e-10)
                )
                log_likelihoods.append(ll_split)
                split_points.append(split)
        
        # Log-verossimilhança de NÃO haver mudança (modelo nulo)
        mu_all = np.mean(returns_window)
        sigma_all = np.std(returns_window)
        
        if sigma_all > 0:
            ll_no_split = (
                -n * np.log(sigma_all + 1e-10) -
                np.sum((returns_window - mu_all)**2) / (2 * sigma_all**2 + 1e-10)
            )
        else:
            ll_no_split = -np.inf
        
        # Razão de verossimilhança (Bayes Factor aproximado)
        if len(log_likelihoods) > 0:
            ll_max_split = np.max(log_likelihoods)
            log_bf = ll_max_split - ll_no_split
            
            # Converte para probabilidade posterior (aproximação)
            # P(mudança|dados) ∝ P(dados|mudança) * P(mudança)
            prior_odds = self.p0 / (1 - self.p0)
            posterior_odds = np.exp(log_bf) * prior_odds
            prob_change = posterior_odds / (1 + posterior_odds)
            
            # Limita entre 0 e 1
            prob_change = np.clip(prob_change, 0, 1)
        else:
            prob_change = 0.0
        
        return prob_change, mu_all, sigma_all
    
    def fit(self, returns):
        """
        Detecta mudanças estruturais usando janela móvel.
        """
        X = np.array(returns)
        n = len(X)
        
        print(f"Executando Fast BCP Detection (janela móvel)...")
        print(f"  - Tamanho da amostra: {n}")
        print(f"  - Tamanho da janela: {self.window}")
        print(f"  - Prior p0: {self.p0:.3f}")
        
        self.change_probability = np.zeros(n)
        self.regime_mean = np.zeros(n)
        self.regime_vol = np.zeros(n)
        
        # Processa janela móvel
        for i in range(n):
            start = max(0, i - self.window + 1)
            end = i + 1
            
            window_data = X[start:end]
            
            prob, mu, vol = self._detect_change_in_window(window_data)
            
            self.change_probability[i] = prob
            self.regime_mean[i] = mu
            self.regime_vol[i] = vol
        
        print(f"  ✓ Detecção concluída!")
        print(f"  - Probabilidade média: {np.mean(self.change_probability):.4f}")
        print(f"  - Máxima probabilidade: {np.max(self.change_probability):.4f}")
        
        return self
    
    def get_current_status(self, lookback=100):
        """
        Avalia status atual (última observação).
        """
        if self.change_probability is None:
            raise ValueError("Modelo não ajustado. Execute .fit() primeiro.")
        
        n = len(self.change_probability)
        
        # Métricas atuais
        current_prob = self.change_probability[-1]
        current_vol = self.regime_vol[-1]
        
        # Comparação com histórico
        recent_window = min(lookback, n)
        recent_probs = self.change_probability[-recent_window:]
        
        percentile = stats.percentileofscore(recent_probs, current_prob)
        
        # Tendência recente (últimos 5-10 períodos)
        lookback_trend = min(10, n)
        if n >= lookback_trend:
            trend_direction = np.mean(np.diff(self.regime_mean[-lookback_trend:]))
            trend_text = "ALTA" if trend_direction > 0 else "BAIXA"
        else:
            trend_text = "INDEFINIDA"
        
        # Status
        if percentile > 90:
            status = "🔴 ALERTA ALTO - Reversão iminente"
        elif percentile > 75:
            status = "🟡 ALERTA MODERADO - Probabilidade elevada"
        elif percentile > 60:
            status = "🟠 ATENÇÃO - Probabilidade moderada"
        else:
            status = "🟢 ESTÁVEL - Baixa probabilidade"
        
        # Força do sinal
        signal_strength = current_prob * current_vol * 1000  # escala para visualização
        
        return {
            'status': status,
            'prob_mudanca_atual': current_prob,
            'volatilidade_atual': current_vol,
            'percentil': percentile,
            'tendencia': trend_text,
            'forca_sinal': signal_strength,
            'media_regime_atual': self.regime_mean[-1],
            'n_observacoes': n
        }
    
    def plot_analysis(self, dates=None, price=None, figsize=(15, 10)):
        """
        Plota análise visual.
        """
        if self.change_probability is None:
            raise ValueError("Modelo não ajustado.")
        
        n = len(self.change_probability)
        if dates is None:
            dates = np.arange(n)
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Preço (se fornecido)
        if price is not None:
            axes[0].plot(dates, price, 'k-', linewidth=1, alpha=0.7, label='Preço')
            axes[0].set_ylabel('Preço', fontsize=11, fontweight='bold')
            axes[0].legend(loc='upper left')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('Fast BCP Analysis - Detecção de Reversão de Tendência', 
                            fontsize=14, fontweight='bold')
        
        # Plot 2: Média do regime + banda de volatilidade
        axes[1].plot(dates, self.regime_mean, 'b-', linewidth=1.5, 
                    label='Média do Regime', alpha=0.8)
        axes[1].fill_between(dates,
                            self.regime_mean - self.regime_vol,
                            self.regime_mean + self.regime_vol,
                            alpha=0.2, color='blue', label='± 1 Vol')
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[1].set_ylabel('Retorno Médio', fontsize=11, fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Probabilidade de mudança
        axes[2].plot(dates, self.change_probability, 'r-', linewidth=2,
                    label='P(Change Point)', alpha=0.8)
        axes[2].fill_between(dates, 0, self.change_probability,
                            alpha=0.3, color='red')
        
        # Marca pontos de alta probabilidade
        high_prob = self.change_probability > 0.6
        if np.any(high_prob):
            axes[2].scatter(dates[high_prob], self.change_probability[high_prob],
                          color='darkred', s=50, zorder=5, 
                          label='Alta Probabilidade (>0.6)')
        
        # Linhas de referência
        axes[2].axhline(0.5, color='orange', linestyle='--', linewidth=1, 
                       alpha=0.7, label='Limiar 0.5')
        axes[2].axhline(0.75, color='red', linestyle=':', linewidth=1, 
                       alpha=0.7, label='Limiar 0.75')
        
        axes[2].set_ylabel('Probabilidade', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Tempo', fontsize=11, fontweight='bold')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        return fig


def quick_analysis(csv_file, date_col='timestamp', price_col='close', 
                  window=200, p0=0.2):
    """
    Análise rápida de reversão de tendência.
    """
    print("="*70)
    print("ANÁLISE RÁPIDA - DETECÇÃO DE REVERSÃO DE TENDÊNCIA")
    print("Fast BCP - Bayesian Change Point Detection (Versão Otimizada)")
    print("="*70)
    print()
    
    # Carrega dados
    print(f"📂 Carregando: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"   ✓ {len(df)} registros")
    except Exception as e:
        print(f"   ✗ Erro: {e}")
        return None
    
    # Valida colunas
    if date_col not in df.columns or price_col not in df.columns:
        print(f"   ✗ Colunas '{date_col}' ou '{price_col}' não encontradas")
        return None
    
    # Processa datas
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    except:
        pass
    
    print(f"   - Período: {df[date_col].iloc[0]} a {df[date_col].iloc[-1]}")
    print(f"   - Preço: {df[price_col].iloc[0]:.2f} → {df[price_col].iloc[-1]:.2f}")
    print()
    
    # Calcula retornos
    returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
    print(f"📊 Estatísticas:")
    print(f"   - Retorno médio: {returns.mean():.6f}")
    print(f"   - Volatilidade: {returns.std():.6f}")
    print(f"   - Sharpe (anualizado): {returns.mean() / returns.std() * np.sqrt(252 * 78):.2f}")
    print()
    
    # Executa detecção
    detector = FastBCPDetector(window=window, p0=p0)
    detector.fit(returns.values)
    print()
    
    # Status atual
    print("="*70)
    print("📍 STATUS ATUAL (Última Observação)")
    print("="*70)
    status = detector.get_current_status()
    
    print(f"\n{status['status']}\n")
    print(f"Métricas:")
    print(f"  • Probabilidade de Reversão: {status['prob_mudanca_atual']:.2%} "
          f"(percentil {status['percentil']:.0f})")
    print(f"  • Volatilidade Atual: {status['volatilidade_atual']:.4%}")
    print(f"  • Força do Sinal: {status['forca_sinal']:.2f}")
    print(f"  • Tendência Recente: {status['tendencia']}")
    print(f"  • Retorno Médio Atual: {status['media_regime_atual']:.4%}")
    print()
    
    # Interpretação
    print("="*70)
    print("💡 INTERPRETAÇÃO")
    print("="*70)
    
    if status['percentil'] > 90:
        print("⚠️  SINAL FORTE DE REVERSÃO IMINENTE!")
        print("    O modelo indica alta probabilidade de mudança estrutural.")
        print("    ➜ Considere: Aguardar confirmação antes de novas posições")
        print("    ➜ Ação: Reduzir exposição ou ajustar stop loss")
    elif status['percentil'] > 75:
        print("⚡ SINAIS DE INSTABILIDADE CRESCENTE")
        print("    Probabilidade de reversão acima da média histórica.")
        print("    ➜ Considere: Aumentar cautela e monitoramento")
        print("    ➜ Ação: Revisar estratégia e exposição ao risco")
    elif status['percentil'] > 60:
        print("👁️  ATENÇÃO - MONITORAR DE PERTO")
        print("    Sinais moderados de possível mudança de regime.")
        print("    ➜ Considere: Manter vigilância, mas sem alarme")
        print("    ➜ Ação: Preparar plano de contingência")
    else:
        print("✅ MERCADO CONSISTENTE")
        print("    Baixa probabilidade de reversão no curto prazo.")
        print("    ➜ Considere: Regime atual parece estável")
        print("    ➜ Ação: Manter estratégia atual")
    
    print()
    print(f"📈 Baseado em análise de {status['n_observacoes']} observações")
    print(f"   (janela móvel de {window} períodos)")
    print()
    
    # Plota
    print("📊 Gerando gráficos...")
    fig = detector.plot_analysis(
        dates=df[date_col].iloc[1:],
        price=df[price_col].iloc[1:]
    )
    
    return {
        'detector': detector,
        'status': status,
        'df': df,
        'returns': returns,
        'figure': fig
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "ibovespa_5min_exemplo_com_reversao.csv"
        print(f"Uso: python {sys.argv[0]} <arquivo.csv> [window] [p0]")
        print(f"Usando arquivo padrão: {csv_file}\n")
    
    # Parâmetros
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    p0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    
    # Análise
    results = quick_analysis(
        csv_file,
        window=window,
        p0=p0
    )
    
    if results is not None:
        # Salva
        output = f'bcp_analysis_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
        results['figure'].savefig(output, dpi=150, bbox_inches='tight')
        print(f"💾 Gráfico salvo: {output}")
        
        plt.show()
        
        print("\n" + "="*70)
        print("✅ Análise concluída com sucesso!")
        print("="*70)
