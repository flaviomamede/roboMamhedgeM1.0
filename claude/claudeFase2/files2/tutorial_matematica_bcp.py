#!/usr/bin/env python3
"""
TUTORIAL INTERATIVO - Matemática do BCP Passo a Passo

Este script demonstra cada componente matemático do BCP com exemplos práticos.
Execute seção por seção para entender a matemática em ação.

Correções aplicadas (2026-02-15):
  - Backward pass reimplementado corretamente com variáveis backward separadas
  - Cohesions corrigidas para o último bloco
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy import stats

print("=" * 70)
print("TUTORIAL MATEMÁTICO - BAYESIAN CHANGE POINT DETECTION")
print("Demonstração Prática de Cada Componente")
print("=" * 70)
print()

# ============================================================================
# SEÇÃO 1: DADOS SINTÉTICOS COM MUDANÇA CONHECIDA
# ============================================================================

print("\n" + "=" * 70)
print("SEÇÃO 1: GERANDO DADOS COM MUDANÇA CONHECIDA")
print("=" * 70)

np.random.seed(42)

# Regime 1: média 0.001, vol 0.01 (50 pontos)
regime1 = np.random.normal(0.001, 0.01, 50)

# Regime 2: média -0.002, vol 0.015 (50 pontos)
regime2 = np.random.normal(-0.002, 0.015, 50)

# Combina
X = np.concatenate([regime1, regime2])
true_changepoint = 50

print(f"Dados gerados:")
print(f"  • N = {len(X)} observações")
print(f"  • Regime 1 (t=1-50): μ={np.mean(regime1):.6f}, σ={np.std(regime1):.6f}")
print(f"  • Regime 2 (t=51-100): μ={np.mean(regime2):.6f}, σ={np.std(regime2):.6f}")
print(f"  • Change point verdadeiro: t={true_changepoint}")
print()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(X, 'k-', alpha=0.7, linewidth=1)
axes[0].axvline(true_changepoint, color='red', linestyle='--',
                linewidth=2, label='Change Point Real (t=50)')
axes[0].axhline(np.mean(regime1), color='blue', linestyle=':',
                xmax=0.5, label=f'Regime 1: μ={np.mean(regime1):.3f}')
axes[0].axhline(np.mean(regime2), color='green', linestyle=':',
                xmin=0.5, label=f'Regime 2: μ={np.mean(regime2):.3f}')
axes[0].set_ylabel('Retorno')
axes[0].set_title('Dados Sintéticos com Change Point em t=50')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(regime1, bins=20, alpha=0.5, label='Regime 1', color='blue')
axes[1].hist(regime2, bins=20, alpha=0.5, label='Regime 2', color='green')
axes[1].set_xlabel('Retorno')
axes[1].set_ylabel('Frequência')
axes[1].set_title('Distribuições dos Regimes')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tutorial_01_dados.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: tutorial_01_dados.png")
print()

# ============================================================================
# SEÇÃO 2: COHESIONS (PRIORS SOBRE PARTIÇÕES)
# ============================================================================

print("\n" + "=" * 70)
print("SEÇÃO 2: COHESIONS - PRIOR SOBRE PARTIÇÕES")
print("=" * 70)

p = 0.2  # probabilidade de mudança por observação
n = len(X)

print(f"Parâmetro p = {p:.2f}")
print(f"Interpretação: A cada ponto, há {p:.0%} de chance de mudança")
print()

# Calcula cohesions para alguns blocos
print("Cohesions de exemplo:")
print(f"  c[0,10] = p·(1-p)^9 = {p * (1 - p) ** 9:.6f}  (bloco curto)")
print(f"  c[0,50] = p·(1-p)^49 = {p * (1 - p) ** 49:.6f}  (bloco médio)")
print(f"  c[0,100] = (1-p)^99 = {(1 - p) ** 99:.6f}  (último bloco, sem fator p)")
print()

# Visualiza decaimento exponencial
block_sizes = np.arange(1, 101)
cohesions = [p * (1 - p) ** (size - 1) for size in block_sizes]

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(block_sizes, cohesions, 'b-', linewidth=2)
ax.set_xlabel('Tamanho do Bloco')
ax.set_ylabel('Cohesion c_ij (log scale)')
ax.set_title(f'Decaimento Exponencial das Cohesions (p={p})')
ax.grid(True, alpha=0.3)
ax.axhline(p * (1 - p) ** 49, color='red', linestyle='--',
           label=f'Bloco de tamanho 50: {p * (1 - p) ** 49:.6f}')
ax.legend()

plt.tight_layout()
plt.savefig('tutorial_02_cohesions.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: tutorial_02_cohesions.png")
print()

print("💡 Intuição:")
print("  • Cohesions pequenas = baixa probabilidade a priori")
print("  • Blocos longos sem mudança têm cohesion muito baixa")
print("  • Isso favorece múltiplos blocos curtos a médios")
print()

# ============================================================================
# SEÇÃO 3: DATA FACTORS (LIKELIHOOD)
# ============================================================================

print("\n" + "=" * 70)
print("SEÇÃO 3: DATA FACTORS - LIKELIHOOD MARGINAL")
print("=" * 70)


def calculate_log_data_factor(X_block):
    """
    Calcula log data factor (Student-t marginal likelihood):

    log f = Γ((n-1)/2) - ½·log(n) - (n-1)/2·log(π) - (n-1)/2·log(W)

    onde W = Σ(X_l - X̄)² é a soma dos quadrados dos desvios.

    Os termos Γ(·) e log(n) normalizam corretamente para o tamanho
    do bloco, evitando viés a favor de blocos pequenos.
    """
    from scipy.special import gammaln

    n_block = len(X_block)
    if n_block < 2:
        return 0.0

    X_mean = np.mean(X_block)
    W = np.sum((X_block - X_mean) ** 2)

    df = n_block - 1
    log_factor = (gammaln(df / 2.0)
                  - 0.5 * np.log(n_block)
                  - df / 2.0 * np.log(np.pi)
                  - df / 2.0 * np.log(W + 1e-5))

    return log_factor


# Testa em 3 blocos diferentes
print("Data factors para diferentes blocos:")
print()

# Bloco 1: Só regime 1 (homogêneo)
block1 = X[0:50]
log_f1 = calculate_log_data_factor(block1)
print(f"Bloco [0,50] (só regime 1):")
print(f"  X̄ = {np.mean(block1):.6f}")
print(f"  W = {np.sum((block1 - np.mean(block1)) ** 2):.6f}")
print(f"  log f = {log_f1:.2f}")
print()

# Bloco 2: Só regime 2 (homogêneo)
block2 = X[50:100]
log_f2 = calculate_log_data_factor(block2)
print(f"Bloco [50,100] (só regime 2):")
print(f"  X̄ = {np.mean(block2):.6f}")
print(f"  W = {np.sum((block2 - np.mean(block2)) ** 2):.6f}")
print(f"  log f = {log_f2:.2f}")
print()

# Bloco 3: Ambos regimes (heterogêneo!)
block3 = X[0:100]
log_f3 = calculate_log_data_factor(block3)
print(f"Bloco [0,100] (AMBOS regimes - heterogêneo!):")
print(f"  X̄ = {np.mean(block3):.6f}")
print(f"  W = {np.sum((block3 - np.mean(block3)) ** 2):.6f}")
print(f"  log f = {log_f3:.2f}")
print()

print("🔍 Observação CRÍTICA:")
print(f"  • Blocos homogêneos: log f ≈ {(log_f1 + log_f2) / 2:.2f}")
print(f"  • Bloco heterogêneo: log f = {log_f3:.2f}")
print(f"  • Diferença: {log_f3 - (log_f1 + log_f2) / 2:.2f} (!)")
print()
print("  O bloco heterogêneo tem likelihood MUITO MENOR!")
print("  Isso é o que detecta a mudança.")
print()

# ============================================================================
# SEÇÃO 4: BAYES FACTOR
# ============================================================================

print("\n" + "=" * 70)
print("SEÇÃO 4: BAYES FACTOR - EVIDÊNCIA DE MUDANÇA")
print("=" * 70)

# Modelo A: Mudança em t=50
log_cohesion_model_A = np.log(p) + 49 * np.log(1 - p) + 49 * np.log(1 - p)
log_likelihood_model_A = log_f1 + log_f2
log_posterior_A = log_cohesion_model_A + log_likelihood_model_A

print(f"Modelo A: Mudança em t=50")
print(f"  Prior (cohesions): {np.exp(log_cohesion_model_A):.2e}")
print(f"  Likelihood: exp({log_likelihood_model_A:.2f})")
print(f"  Posterior: exp({log_posterior_A:.2f})")
print()

# Modelo B: Sem mudança (último bloco — sem fator p)
log_cohesion_model_B = 99 * np.log(1 - p)
log_likelihood_model_B = log_f3
log_posterior_B = log_cohesion_model_B + log_likelihood_model_B

print(f"Modelo B: SEM mudança")
print(f"  Prior (cohesions): {np.exp(log_cohesion_model_B):.2e}")
print(f"  Likelihood: exp({log_likelihood_model_B:.2f})")
print(f"  Posterior: exp({log_posterior_B:.2f})")
print()

# Bayes Factor
log_BF = log_posterior_A - log_posterior_B
BF = np.exp(min(log_BF, 700))  # Evita overflow

print(f"Bayes Factor (A vs B):")
print(f"  log BF = {log_BF:.2f}")
print(f"  BF = {BF:.2e}")
print()

if BF > 100:
    print("  ✅ Evidência DECISIVA a favor de mudança!")
elif BF > 10:
    print("  ✅ Evidência FORTE a favor de mudança")
elif BF > 3:
    print("  ⚡ Evidência MODERADA a favor de mudança")
else:
    print("  ⚠️  Evidência fraca ou inconclusiva")
print()

# Converte para probabilidade
prior_odds = p / (1 - p)
posterior_odds = BF * prior_odds
posterior_prob_bayes = posterior_odds / (1 + posterior_odds)

print(f"Probabilidade Posterior:")
print(f"  P(mudança em t=50 | dados) = {posterior_prob_bayes:.6f} ({posterior_prob_bayes:.2%})")
print()

# ============================================================================
# SEÇÃO 5: FORWARD-BACKWARD COMPLETO (CORRIGIDO)
# ============================================================================

print("\n" + "=" * 70)
print("SEÇÃO 5: ALGORITMO FORWARD-BACKWARD COMPLETO")
print("=" * 70)

print("Executando algoritmo completo para detectar change point...")
print()

max_block = 60  # janela máxima para eficiência

# Cache de data factors
log_factors = {}
for i in range(n):
    for j in range(i + 1, min(i + max_block + 1, n + 1)):
        log_factors[(i, j)] = calculate_log_data_factor(X[i:j])


def log_cohesion_fn(block_len, is_last_block):
    """Cohesion geométrica de Barry & Hartigan."""
    if is_last_block:
        return (block_len - 1) * np.log(1 - p)
    else:
        return np.log(p) + (block_len - 1) * np.log(1 - p)


# === FORWARD PASS ===
log_forward = np.full(n + 1, -np.inf)
log_forward[0] = 0.0

for j in range(1, n + 1):
    log_probs = []
    for i in range(max(0, j - max_block), j):
        block_len = j - i
        is_last = (j == n)
        log_coh = log_cohesion_fn(block_len, is_last)

        if (i, j) in log_factors:
            log_prob = log_forward[i] + log_coh + log_factors[(i, j)]
            log_probs.append(log_prob)

    if log_probs:
        log_forward[j] = logsumexp(log_probs)

# === BACKWARD PASS (CORRETO) ===
log_backward = np.full(n + 1, -np.inf)
log_backward[n] = 0.0

for j in range(n - 1, -1, -1):
    log_probs = []
    for k in range(j + 1, min(j + max_block + 1, n + 1)):
        block_len = k - j
        is_last = (k == n)
        log_coh = log_cohesion_fn(block_len, is_last)

        if (j, k) in log_factors:
            log_prob = log_coh + log_factors[(j, k)] + log_backward[k]
            log_probs.append(log_prob)

    if log_probs:
        log_backward[j] = logsumexp(log_probs)

# === PROBABILIDADE DE CHANGE POINT ===
posterior_prob = np.zeros(n)
for t in range(1, n):
    log_post = log_forward[t] + log_backward[t] - log_forward[n]
    posterior_prob[t] = np.clip(np.exp(log_post), 0, 1)

print(f"Resultado:")
print(f"  • Change point verdadeiro: t={true_changepoint}")
print(f"  • Probabilidade em t=50: {posterior_prob[50]:.4f}")
print(f"  • Máxima probabilidade: {np.max(posterior_prob):.4f} em t={np.argmax(posterior_prob)}")
print()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Dados originais
axes[0].plot(X, 'k-', alpha=0.7, linewidth=1)
axes[0].axvline(true_changepoint, color='red', linestyle='--',
                linewidth=2, label='Change Point Real')
axes[0].set_ylabel('Retorno')
axes[0].set_title('Série Temporal Original')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Probabilidade posterior
axes[1].plot(posterior_prob, 'b-', linewidth=2, label='P(change point | dados)')
axes[1].axvline(true_changepoint, color='red', linestyle='--',
                linewidth=2, label='Change Point Real')
axes[1].axhline(0.5, color='orange', linestyle=':', alpha=0.7, label='Limiar 0.5')
axes[1].fill_between(range(n), 0, posterior_prob, alpha=0.3)
axes[1].set_xlabel('Tempo')
axes[1].set_ylabel('Probabilidade')
axes[1].set_title('Probabilidade Posterior de Change Point (Forward-Backward Correto)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('tutorial_05_forward_backward.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: tutorial_05_forward_backward.png")
print()

# ============================================================================
# SEÇÃO 6: ANÁLISE DE SENSIBILIDADE
# ============================================================================

print("\n" + "=" * 70)
print("SEÇÃO 6: ANÁLISE DE SENSIBILIDADE AO PARÂMETRO p")
print("=" * 70)

print("Testando diferentes valores de p...")
print()

p_values = [0.05, 0.10, 0.15, 0.20, 0.30]
results = {}

for p_test in p_values:
    # Forward pass
    log_forward_test = np.full(n + 1, -np.inf)
    log_forward_test[0] = 0.0

    for j in range(1, n + 1):
        log_probs = []
        for i in range(max(0, j - max_block), j):
            block_len = j - i
            is_last = (j == n)
            if is_last:
                log_coh = (block_len - 1) * np.log(1 - p_test)
            else:
                log_coh = np.log(p_test) + (block_len - 1) * np.log(1 - p_test)

            if (i, j) in log_factors:
                log_prob = log_forward_test[i] + log_coh + log_factors[(i, j)]
                log_probs.append(log_prob)

        if log_probs:
            log_forward_test[j] = logsumexp(log_probs)

    # Backward pass (correto)
    log_backward_test = np.full(n + 1, -np.inf)
    log_backward_test[n] = 0.0

    for j in range(n - 1, -1, -1):
        log_probs = []
        for k in range(j + 1, min(j + max_block + 1, n + 1)):
            block_len = k - j
            is_last = (k == n)
            if is_last:
                log_coh = (block_len - 1) * np.log(1 - p_test)
            else:
                log_coh = np.log(p_test) + (block_len - 1) * np.log(1 - p_test)

            if (j, k) in log_factors:
                log_prob = log_coh + log_factors[(j, k)] + log_backward_test[k]
                log_probs.append(log_prob)

        if log_probs:
            log_backward_test[j] = logsumexp(log_probs)

    # Probabilidade de change point
    prob_test = np.zeros(n)
    for t in range(1, n):
        log_post = log_forward_test[t] + log_backward_test[t] - log_forward_test[n]
        prob_test[t] = np.clip(np.exp(log_post), 0, 1)

    results[p_test] = prob_test

    print(f"p = {p_test:.2f}:")
    print(f"  • P(change em t=50): {prob_test[50]:.4f}")
    print(f"  • Máxima probabilidade: {np.max(prob_test):.4f} em t={np.argmax(prob_test)}")
    print()

# Plot comparação
fig, ax = plt.subplots(figsize=(12, 6))

for p_test, prob_test in results.items():
    ax.plot(prob_test, label=f'p={p_test:.2f}', linewidth=2, alpha=0.8)

ax.axvline(true_changepoint, color='red', linestyle='--',
           linewidth=2, label='Change Point Real')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Tempo')
ax.set_ylabel('P(Change Point)')
ax.set_title('Sensibilidade ao Parâmetro p')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('tutorial_06_sensitivity.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: tutorial_06_sensitivity.png")
print()

print("💡 Conclusão:")
print("  • p PEQUENO (0.05): Conservador, só detecta mudanças muito fortes")
print("  • p MÉDIO (0.15-0.20): Balanceado, detecta mudanças claras")
print("  • p GRANDE (0.30): Agressivo, pode dar falsos alarmes")
print()

# ============================================================================
# SEÇÃO 7: COMPARAÇÃO COM TESTE CLÁSSICO
# ============================================================================

print("\n" + "=" * 70)
print("SEÇÃO 7: COMPARAÇÃO COM TESTE t DE STUDENT")
print("=" * 70)

print("Comparando BCP com teste t tradicional...")
print()

# Teste t em vários pontos
t_statistics = []
p_values_classical = []

for t in range(10, n - 10):
    sample1 = X[:t]
    sample2 = X[t:]

    if len(sample1) > 1 and len(sample2) > 1:
        t_stat, p_val = stats.ttest_ind(sample1, sample2)
        t_statistics.append(abs(t_stat))
        p_values_classical.append(p_val)
    else:
        t_statistics.append(0)
        p_values_classical.append(1)

# Converte p-value clássico para "probabilidade de mudança"
prob_classical = [1 - pv for pv in p_values_classical]

# Plot comparação
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Dados
axes[0].plot(X, 'k-', alpha=0.7, linewidth=1)
axes[0].axvline(true_changepoint, color='red', linestyle='--', linewidth=2)
axes[0].set_ylabel('Retorno')
axes[0].set_title('Dados Originais')
axes[0].grid(True, alpha=0.3)

# BCP
axes[1].plot(posterior_prob, 'b-', linewidth=2, label='BCP')
axes[1].axvline(true_changepoint, color='red', linestyle='--', linewidth=2)
axes[1].axhline(0.5, color='orange', linestyle=':', alpha=0.7)
axes[1].set_ylabel('Probabilidade BCP')
axes[1].set_title('Bayesian Change Point Detection (Corrigido)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1)

# Teste t clássico
axes[2].plot(range(10, n - 10), prob_classical, 'g-', linewidth=2,
             label='1 - p-value (teste t)')
axes[2].axvline(true_changepoint, color='red', linestyle='--', linewidth=2)
axes[2].axhline(0.5, color='orange', linestyle=':', alpha=0.7)
axes[2].set_xlabel('Tempo')
axes[2].set_ylabel('1 - p-value')
axes[2].set_title('Teste t Clássico (Duas Amostras)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('tutorial_07_comparison.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico salvo: tutorial_07_comparison.png")
print()

print("🔍 Diferenças Chave:")
print()
print("BCP:")
print("  ✅ Suave (considera contexto temporal)")
print("  ✅ Incorpora prior bayesiano")
print("  ✅ Probabilidade bem calibrada")
print("  ✅ Detecta múltiplas mudanças automaticamente")
print()
print("Teste t:")
print("  ⚡ Simples e rápido")
print("  ⚠️  Ruidoso (testa cada ponto independentemente)")
print("  ⚠️  p-value ≠ probabilidade de mudança")
print("  ⚠️  Requer correção de múltiplas comparações")
print()

# ============================================================================
# CONCLUSÃO
# ============================================================================

print("\n" + "=" * 70)
print("CONCLUSÃO DO TUTORIAL")
print("=" * 70)
print()
print("Você aprendeu:")
print()
print("1. ✅ Como gerar dados com mudanças conhecidas")
print("2. ✅ O que são cohesions e como funcionam")
print("3. ✅ Como calcular data factors (likelihood)")
print("4. ✅ O que é Bayes Factor e como interpretar")
print("5. ✅ Algoritmo Forward-Backward completo (CORRIGIDO)")
print("6. ✅ Sensibilidade aos hiperparâmetros")
print("7. ✅ Comparação com métodos clássicos")
print()
print("📊 Gráficos salvos para referência!")
print()
print("Próximos passos:")
print("  • Aplique aos seus dados reais do IBOVESPA")
print("  • Experimente diferentes valores de p")
print("  • Combine com outros indicadores")
print()
print("=" * 70)
