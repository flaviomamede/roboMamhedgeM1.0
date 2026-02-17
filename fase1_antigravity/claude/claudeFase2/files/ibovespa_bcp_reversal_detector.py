#!/usr/bin/env python3
"""
Bayesian Change Point Detection para Detecção de Reversão de Tendência
Baseado na metodologia de Tobias Setz & Diethelm Würtz (ETH Zurich)

Este script implementa o algoritmo BCP para detectar quando o IBOVESPA
está próximo ou em cima de um ponto de reversão de tendência.

Referências:
  - Barry & Hartigan (1993) "A Bayesian Analysis for Change Point Problems"
  - Setz (2017) "Stable Portfolio Design Using Bayesian Change Point Models"

Correções aplicadas (2026-02-15):
  - Backward pass reimplementado corretamente com variáveis backward separadas
  - Cohesions corrigidas para o último bloco
  - Posterior mean otimizado de O(n³) para O(n·W)
  - Janela máxima de bloco para manter complexidade tratável
  - Removido parâmetro w0 (não utilizado)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class BayesianChangePointDetector:
    """
    Implementação do Bayesian Change Point Detection baseado em:
    Barry & Hartigan (1993) "A Bayesian Analysis for Change Point Problems"

    Detecta mudanças estruturais na média e variância de uma série temporal.
    Usa o Product Partition Model com cohesions geométricas e programação
    dinâmica (Forward-Backward) para computar probabilidades posteriores
    de change point em cada observação.
    """

    def __init__(self, p0=0.2, max_block=200):
        """
        Parâmetros:
        -----------
        p0 : float (0, 1]
            Prior para probabilidade de mudança em cada observação.
            Menor = menos mudanças esperadas (mais conservador).
            Recomendado: 0.1-0.3 para dados financeiros de alta frequência.

        max_block : int
            Tamanho máximo de bloco permitido. Limita a complexidade
            a O(n · max_block) ao invés de O(n²).
            Recomendado: 100-300.
        """
        self.p0 = p0
        self.max_block = max_block

        # Resultados
        self.posterior_mean = None
        self.posterior_variance = None
        self.posterior_probability = None
        self.change_point_intensity = None

    @staticmethod
    def _log_data_factor(X, start, end):
        """
        Calcula log data factor para o bloco X[start:end].

        O data factor é a verossimilhança marginal do bloco,
        integrando sobre os parâmetros desconhecidos (μ, σ²)
        usando priors impróprios (Jeffreys).

        Fórmula completa (Student-t marginal likelihood):
            log f = Γ((n-1)/2) - ½·log(n) - (n-1)/2·log(π) - (n-1)/2·log(W + ε)

        onde W = Σ(Xᵢ - X̄)² é a soma dos quadrados dos desvios.

        Os termos Γ e log(n) normalizam corretamente para o tamanho do bloco,
        evitando que blocos pequenos sejam artificialmente favorecidos.
        """
        from scipy.special import gammaln

        block = X[start:end]
        n_block = len(block)

        if n_block < 2:
            return 0.0  # bloco de tamanho 1: contribuição neutra

        X_bar = np.mean(block)
        W = np.sum((block - X_bar) ** 2)

        df = n_block - 1
        log_f = (gammaln(df / 2.0)
                 - 0.5 * np.log(n_block)
                 - df / 2.0 * np.log(np.pi)
                 - df / 2.0 * np.log(W + 1e-5))

        return log_f

    def _log_cohesion(self, block_len, is_last_block):
        """
        Calcula log cohesion para um bloco de tamanho block_len.

        Cohesion geométrica de Barry & Hartigan:
            c(i,j) = p · (1-p)^(j-i-1)     se j < n (não é último bloco)
            c(i,n) = (1-p)^(n-i-1)          se j = n (último bloco)

        O último bloco não tem fator p porque não precisa que haja
        um change point à frente dele.
        """
        if is_last_block:
            return (block_len - 1) * np.log(1 - self.p0)
        else:
            return np.log(self.p0) + (block_len - 1) * np.log(1 - self.p0)

    def _forward_backward(self, X):
        """
        Algoritmo Forward-Backward para calcular probabilidades posteriores
        de change point em cada observação.

        Usa programação dinâmica O(n · max_block) ao invés de enumeração O(2^n).

        Forward pass:
            log_forward[j] = log Σ_partições P(X[0:j], partição termina em j)

        Backward pass:
            log_backward[j] = log Σ_partições P(X[j:n] | bloco começa em j)

        Probabilidade de change point em t:
            P(cp em t | X) = exp(log_forward[t] + log_backward[t] - log_forward[n])
        """
        n = len(X)

        # Pré-computa data factors para todos os blocos possíveis
        log_factors = {}
        for i in range(n):
            for j in range(i + 1, min(i + self.max_block + 1, n + 1)):
                log_factors[(i, j)] = self._log_data_factor(X, i, j)

        # === FORWARD PASS ===
        # log_forward[j] = log P(X[0:j]) marginalizado sobre todas as partições de [0, j)
        log_forward = np.full(n + 1, -np.inf)
        log_forward[0] = 0.0  # P(dados vazios) = 1

        for j in range(1, n + 1):
            log_probs = []
            for i in range(max(0, j - self.max_block), j):
                block_len = j - i
                is_last = (j == n)
                log_coh = self._log_cohesion(block_len, is_last)

                if (i, j) in log_factors:
                    log_prob = log_forward[i] + log_coh + log_factors[(i, j)]
                    log_probs.append(log_prob)

            if log_probs:
                log_forward[j] = logsumexp(log_probs)

        # === BACKWARD PASS ===
        # log_backward[j] = log P(X[j:n]) marginalizado sobre todas as partições de [j, n)
        log_backward = np.full(n + 1, -np.inf)
        log_backward[n] = 0.0  # P(dados vazios) = 1

        for j in range(n - 1, -1, -1):
            log_probs = []
            for k in range(j + 1, min(j + self.max_block + 1, n + 1)):
                block_len = k - j
                is_last = (k == n)
                log_coh = self._log_cohesion(block_len, is_last)

                if (j, k) in log_factors:
                    log_prob = log_coh + log_factors[(j, k)] + log_backward[k]
                    log_probs.append(log_prob)

            if log_probs:
                log_backward[j] = logsumexp(log_probs)

        # === PROBABILIDADE DE CHANGE POINT ===
        # P(change point em t | X) = P(alguma partição tem fronteira em t | X)
        # = exp(log_forward[t] + log_backward[t] - log_forward[n])
        posterior_prob = np.zeros(n)
        for t in range(1, n):
            log_post = log_forward[t] + log_backward[t] - log_forward[n]
            posterior_prob[t] = np.clip(np.exp(log_post), 0.0, 1.0)

        return posterior_prob, log_factors, log_forward, log_backward

    def _calculate_posterior_mean(self, X, log_factors, log_forward, log_backward):
        """
        Calcula média posterior E(μ_t | X) para cada observação.

        É a média ponderada sobre todos blocos que contêm t,
        ponderada pela relevância posterior de cada bloco.

        Otimizado para O(n · max_block) usando forward/backward.
        """
        n = len(X)
        posterior_mean = np.zeros(n)

        for k in range(n):
            weighted_sum = 0.0
            weight_sum = 0.0

            # Para cada bloco [i, j) que contém k
            for i in range(max(0, k - self.max_block + 1), k + 1):
                for j in range(k + 1, min(i + self.max_block + 1, n + 1)):
                    if (i, j) not in log_factors:
                        continue

                    block_len = j - i
                    is_last = (j == n)
                    log_coh = self._log_cohesion(block_len, is_last)

                    # Relevância do bloco = forward[i] * cohesion * data_factor * backward[j] / Z
                    log_relevance = (log_forward[i] + log_coh +
                                     log_factors[(i, j)] + log_backward[j] -
                                     log_forward[n])

                    relevance = np.exp(np.clip(log_relevance, -700, 0))

                    if relevance > 1e-15:
                        block_mean = np.mean(X[i:j])
                        weighted_sum += block_mean * relevance
                        weight_sum += relevance

            if weight_sum > 1e-15:
                posterior_mean[k] = weighted_sum / weight_sum
            else:
                posterior_mean[k] = X[k]

        return posterior_mean

    def _calculate_posterior_variance(self, X, posterior_mean):
        """
        Calcula variância posterior para cada observação
        usando janela móvel local.
        """
        n = len(X)
        posterior_var = np.zeros(n)
        window = 20

        for i in range(n):
            start = max(0, i - window)
            end = min(n, i + window + 1)

            residuals = X[start:end] - posterior_mean[start:end]
            posterior_var[i] = np.var(residuals) if len(residuals) > 1 else np.var(X)

        return posterior_var

    def fit(self, returns):
        """
        Ajusta o modelo BCP aos retornos.

        Parâmetros:
        -----------
        returns : array-like
            Série de retornos logarítmicos

        Retorna:
        --------
        self : retorna a instância para chaining
        """
        X = np.array(returns, dtype=np.float64)
        n = len(X)

        print(f"Executando Bayesian Change Point Detection...")
        print(f"  - Tamanho da amostra: {n}")
        print(f"  - Prior p0 (prob. mudança): {self.p0:.3f}")
        print(f"  - Bloco máximo: {self.max_block}")

        # Algoritmo Forward-Backward (corrigido)
        self.posterior_probability, log_factors, log_fwd, log_bwd = self._forward_backward(X)

        # Calcula estatísticas posteriores
        print(f"  ⏳ Calculando média posterior...")
        self.posterior_mean = self._calculate_posterior_mean(X, log_factors, log_fwd, log_bwd)
        self.posterior_variance = self._calculate_posterior_variance(X, self.posterior_mean)

        # Métrica de intensidade de mudança
        self.change_point_intensity = self.posterior_probability * np.sqrt(self.posterior_variance)

        print(f"  ✓ Detecção concluída!")
        print(f"  - Probabilidade média de mudança: {np.mean(self.posterior_probability):.4f}")
        print(f"  - Máxima probabilidade: {np.max(self.posterior_probability):.4f}"
              f" em t={np.argmax(self.posterior_probability)}")

        return self

    def get_current_status(self, window=50):
        """
        Avalia o status atual (última observação) quanto à probabilidade de reversão.
        """
        if self.posterior_probability is None:
            raise ValueError("Modelo não foi ajustado. Execute .fit() primeiro.")

        n = len(self.posterior_probability)

        current_prob = self.posterior_probability[-1]
        current_intensity = self.change_point_intensity[-1]

        recent_window = min(window, n)
        recent_probs = self.posterior_probability[-recent_window:]
        recent_intensities = self.change_point_intensity[-recent_window:]

        percentile_prob = stats.percentileofscore(recent_probs, current_prob)
        percentile_intensity = stats.percentileofscore(recent_intensities, current_intensity)

        if percentile_prob > 90:
            status = "🔴 ALERTA ALTO - Alta probabilidade de reversão"
        elif percentile_prob > 75:
            status = "🟡 ALERTA MODERADO - Probabilidade elevada de reversão"
        elif percentile_prob > 60:
            status = "🟠 ATENÇÃO - Probabilidade moderada de reversão"
        else:
            status = "🟢 ESTÁVEL - Baixa probabilidade de reversão"

        if n >= 5:
            trend_direction = np.mean(np.diff(self.posterior_mean[-5:]))
            trend_text = "ALTA" if trend_direction > 0 else "BAIXA"
        else:
            trend_text = "INDEFINIDA"

        return {
            'status': status,
            'prob_mudanca_atual': current_prob,
            'intensidade_atual': current_intensity,
            'percentil_prob': percentile_prob,
            'percentil_intensidade': percentile_intensity,
            'tendencia_recente': trend_text,
            'media_posterior_atual': self.posterior_mean[-1],
            'variancia_posterior_atual': self.posterior_variance[-1],
            'n_observacoes': n
        }

    def plot_analysis(self, dates=None, price=None, figsize=(15, 12)):
        """
        Plota análise completa do BCP.
        """
        if self.posterior_probability is None:
            raise ValueError("Modelo não foi ajustado. Execute .fit() primeiro.")

        n = len(self.posterior_probability)
        if dates is None:
            dates = np.arange(n)

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # Plot 1: Preço
        if price is not None:
            axes[0].plot(dates, price, 'k-', linewidth=1, alpha=0.7, label='Preço')
            axes[0].set_ylabel('Preço', fontsize=11)
            axes[0].legend(loc='upper left')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('Análise Bayesian Change Point - IBOVESPA',
                              fontsize=14, fontweight='bold')

        # Plot 2: Média Posterior
        axes[1].plot(dates, self.posterior_mean, 'b-', linewidth=1.5,
                     label='Média Posterior E(μ|X)')
        axes[1].fill_between(dates,
                             self.posterior_mean - np.sqrt(self.posterior_variance),
                             self.posterior_mean + np.sqrt(self.posterior_variance),
                             alpha=0.2, color='blue', label='± 1 Desvio Padrão')
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[1].set_ylabel('Média Posterior', fontsize=11)
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Variância Posterior
        axes[2].plot(dates, self.posterior_variance, 'g-', linewidth=1.5,
                     label='Variância Posterior Var(μ|X)')
        axes[2].fill_between(dates, 0, self.posterior_variance, alpha=0.2, color='green')
        axes[2].set_ylabel('Variância Posterior', fontsize=11)
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')

        # Plot 4: Probabilidade de Change Point
        axes[3].plot(dates, self.posterior_probability, 'r-', linewidth=1.5,
                     label='P(Change Point | X)')
        axes[3].fill_between(dates, 0, self.posterior_probability,
                             alpha=0.2, color='red')

        high_prob_mask = self.posterior_probability > 0.5
        if np.any(high_prob_mask):
            axes[3].scatter(dates[high_prob_mask],
                            self.posterior_probability[high_prob_mask],
                            color='darkred', s=50, zorder=5,
                            label='Alta Probabilidade (>0.5)')

        axes[3].axhline(0.5, color='orange', linestyle='--', linewidth=1,
                        alpha=0.7, label='Limiar 0.5')
        axes[3].set_ylabel('Probabilidade', fontsize=11)
        axes[3].set_xlabel('Tempo', fontsize=11)
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, 1)

        plt.tight_layout()

        return fig


def analyze_ibovespa_reversal(csv_file, date_column='timestamp', price_column='close',
                              p0=0.2, max_block=200):
    """
    Função principal para análise de reversão do IBOVESPA.

    Parâmetros:
    -----------
    csv_file : str
        Caminho para arquivo CSV com dados de 5 minutos
    date_column : str
        Nome da coluna com timestamp
    price_column : str
        Nome da coluna com preço de fechamento
    p0 : float
        Prior para probabilidade de mudança (padrão 0.2)
    max_block : int
        Tamanho máximo de bloco (padrão 200)
    """
    print("=" * 70)
    print("ANÁLISE DE REVERSÃO DE TENDÊNCIA - IBOVESPA")
    print("Metodologia: Bayesian Change Point Detection (Setz & Würtz, ETH)")
    print("=" * 70)
    print()

    # Carrega dados
    print(f"Carregando dados de: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"  ✓ {len(df)} registros carregados")
    except Exception as e:
        print(f"  ✗ Erro ao carregar arquivo: {e}")
        return None

    if date_column not in df.columns or price_column not in df.columns:
        print(f"  ✗ Erro: Colunas '{date_column}' ou '{price_column}' não encontradas")
        print(f"     Colunas disponíveis: {', '.join(df.columns)}")
        return None

    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
    except Exception:
        print("  ! Aviso: Não foi possível converter coluna de data")

    print(f"  - Período: {df[date_column].iloc[0]} até {df[date_column].iloc[-1]}")
    print(f"  - Preço inicial: {df[price_column].iloc[0]:.2f}")
    print(f"  - Preço final: {df[price_column].iloc[-1]:.2f}")
    print()

    # Calcula retornos logarítmicos
    returns = np.log(df[price_column] / df[price_column].shift(1)).dropna()
    print(f"Calculando retornos logarítmicos...")
    print(f"  - Retorno médio: {returns.mean():.6f}")
    print(f"  - Volatilidade: {returns.std():.6f}")
    print()

    # Executa BCP
    detector = BayesianChangePointDetector(p0=p0, max_block=max_block)
    detector.fit(returns.values)
    print()

    # Status atual
    print("=" * 70)
    print("STATUS ATUAL (Última Observação)")
    print("=" * 70)
    status = detector.get_current_status(window=100)

    print(f"\n{status['status']}\n")
    print(f"Métricas:")
    print(f"  • Probabilidade de Mudança: {status['prob_mudanca_atual']:.4f} "
          f"(percentil {status['percentil_prob']:.1f}%)")
    print(f"  • Intensidade de Mudança: {status['intensidade_atual']:.6f} "
          f"(percentil {status['percentil_intensidade']:.1f}%)")
    print(f"  • Tendência Recente: {status['tendencia_recente']}")
    print(f"  • Média Posterior Atual: {status['media_posterior_atual']:.6f}")
    print(f"  • Variância Posterior: {status['variancia_posterior_atual']:.8f}")
    print()

    # Interpretação
    print("=" * 70)
    print("INTERPRETAÇÃO")
    print("=" * 70)

    if status['percentil_prob'] > 90:
        print("⚠️  ALTA PROBABILIDADE DE REVERSÃO IMINENTE")
        print("   O modelo detecta forte evidência de mudança estrutural.")
        print("   Considere aguardar confirmação antes de tomar posições.")
    elif status['percentil_prob'] > 75:
        print("⚡ PROBABILIDADE ELEVADA DE REVERSÃO")
        print("   Há sinais de instabilidade crescente no mercado.")
        print("   Monitore de perto e considere reduzir exposição.")
    elif status['percentil_prob'] > 60:
        print("👀 ATENÇÃO - SINAIS MODERADOS DE MUDANÇA")
        print("   A probabilidade de reversão está acima da média.")
        print("   Mantenha cautela e acompanhe próximos períodos.")
    else:
        print("✅ MERCADO RELATIVAMENTE ESTÁVEL")
        print("   A probabilidade de reversão está baixa.")
        print("   Estrutura atual parece consistente.")

    print()
    print(f"Baseado em {status['n_observacoes']} observações de 5 minutos.")
    print()

    # Plota análise
    print("Gerando gráficos...")
    fig = detector.plot_analysis(
        dates=df[date_column].iloc[1:],
        price=df[price_column].iloc[1:]
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
        csv_file = "ibovespa_5min.csv"
        print(f"Uso: python {sys.argv[0]} <arquivo_csv> [p0] [max_block]")
        print(f"Usando arquivo padrão: {csv_file}\n")

    p0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    max_block = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    results = analyze_ibovespa_reversal(
        csv_file,
        date_column='timestamp',
        price_column='close',
        p0=p0,
        max_block=max_block
    )

    if results is not None:
        output_file = 'ibovespa_bcp_analysis.png'
        results['figure'].savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_file}")

        plt.show()

        print("\n" + "=" * 70)
        print("Análise concluída!")
        print("=" * 70)
