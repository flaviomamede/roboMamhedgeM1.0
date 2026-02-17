# 📐 MATEMÁTICA COMPLETA DO BAYESIAN CHANGE POINT DETECTION

## Explicação Detalhada da Metodologia de Setz & Würtz (ETH Zurich)

Este documento apresenta **toda a matemática** por trás do algoritmo BCP implementado, com derivações completas, intuição e exemplos numéricos.

---

## 📚 ÍNDICE

1. [Fundamentos Conceituais](#1-fundamentos-conceituais)
2. [Product Partition Model (PPM)](#2-product-partition-model-ppm)
3. [Priors Bayesianos](#3-priors-bayesianos)
4. [Likelihood e Data Factors](#4-likelihood-e-data-factors)
5. [Posterior Distributions](#5-posterior-distributions)
6. [Algoritmo Forward-Backward](#6-algoritmo-forward-backward)
7. [Cálculo de Probabilidades](#7-cálculo-de-probabilidades)
8. [Implementação Prática](#8-implementação-prática)
9. [Exemplo Numérico Completo](#9-exemplo-numérico-completo)

---

## 1. FUNDAMENTOS CONCEITUAIS

### 1.1 O Problema de Change Point

**Objetivo:** Dado uma série temporal X = {X₁, X₂, ..., Xₙ}, queremos identificar pontos onde a **estrutura geradora** dos dados muda.

**Definição Formal:**

Existe uma **partição** ρ = (i₀, i₁, ..., iᵦ) tal que:

```
0 = i₀ < i₁ < i₂ < ... < iᵦ = n
```

E dentro de cada **bloco** [iₖ₋₁ + 1, iₖ], os dados são i.i.d. (independentes e identicamente distribuídos) com parâmetros θᵢₖ.

**Exemplo Visual:**

```
X: [2.1, 2.3, 2.0, 2.2 | 5.1, 5.3, 5.0, 5.2 | 1.9, 2.1, 2.0]
          Bloco 1        |     Bloco 2        |     Bloco 3
       μ₁ ≈ 2.15         |   μ₂ ≈ 5.15        |   μ₃ ≈ 2.0
   Change point em i₁=4  |  Change point em i₂=8
```

### 1.2 Abordagem Bayesiana

Na abordagem Bayesiana, **tudo** é uma distribuição de probabilidade:

1. **Prior:** P(ρ) - nossa crença a priori sobre partições
2. **Likelihood:** P(X|ρ, θ) - probabilidade dos dados dada uma partição
3. **Posterior:** P(ρ|X) - nossa crença a posteriori sobre partições

**Teorema de Bayes:**

```
P(ρ|X) = P(X|ρ) · P(ρ) / P(X)
```

Onde:
- P(X|ρ) = ∫ P(X|ρ, θ) · P(θ|ρ) dθ (marginalizando sobre parâmetros)
- P(X) = Σᵨ P(X|ρ) · P(ρ) (evidência total)

---

## 2. PRODUCT PARTITION MODEL (PPM)

### 2.1 Definição do PPM

O **Product Partition Model** assume que:

**1. Independência entre blocos:**
```
P(X|ρ, θ) = ∏ P(X_{iₖ₋₁+1:iₖ} | θᵢₖ)
           k=1..b
```

**2. Prior fatorável:**
```
P(θ|ρ) = ∏ P(θᵢₖ)
        k=1..b
```

**3. Cohesions (coesões):**

A probabilidade prior de uma partição é proporcional ao produto de **cohesions** cᵢⱼ:

```
P(ρ) ∝ ∏ c_{iₖ₋₁,iₖ}
       k=1..b
```

Onde cᵢⱼ mede a "coesão" (probabilidade a priori) do bloco [i+1, j].

### 2.2 Data Factors

O **data factor** fᵢⱼ é a verossimilhança marginal do bloco [i+1, j]:

```
fᵢⱼ(Xᵢⱼ) = ∫ P(Xᵢⱼ|θ) · P(θ) dθ
```

Esta integral **elimina** θ, deixando apenas os dados observados.

**Propriedade Fundamental:**

```
P(X|ρ) = ∏ fᵢⱼ(Xᵢⱼ)
        ij∈ρ
```

### 2.3 Relevâncias (Relevances)

A **relevância** rᵢⱼ é a probabilidade posterior de que o bloco [i+1, j] aparece em alguma partição:

```
rᵢⱼ(X) = P(bloco [i+1, j] está em ρ | X)
```

**Cálculo via λ-recursão:**

Defina λᵢⱼ como a soma de produtos sobre todas partições de [i+1, j]:

```
λᵢⱼ = Σ  ∏ cᵢₖ₋₁,ᵢₖ
     ρ k=1..b
```

Então:

```
rᵢⱼ(X) = (λ₀ᵢ · c̃ᵢⱼ · λⱼₙ) / λ₀ₙ
```

Onde c̃ᵢⱼ = cᵢⱼ · fᵢⱼ(Xᵢⱼ) é a **posterior cohesion**.

---

## 3. PRIORS BAYESIANOS

### 3.1 Prior sobre Partições (Cohesions)

**Modelo Geométrico de Barry & Hartigan:**

```
cᵢⱼ = p · (1-p)^(j-i-1)    para j < n
cᵢₙ = (1-p)^(n-i-1)         para j = n
```

Onde p ∈ (0,1] é a **probabilidade de mudança** por observação.

**Intuição:**
- p alto → muitas mudanças esperadas
- p baixo → poucas mudanças esperadas

**Interpretação Probabilística:**

A cada passo, há:
- Probabilidade p de ter um change point
- Probabilidade (1-p) de continuar no mesmo regime

### 3.2 Prior sobre Parâmetros (Modelo Normal)

O BCP assume que dentro de cada bloco:

```
Xᵢ ~ N(μ, σ²)    (observações)
μ  ~ N(μ₀, σ₀²/(j-i))   (prior conjugado)
```

**Motivação do Prior:**

- Blocos **longos** → pequeno desvio de μ₀ esperado → prior tight
- Blocos **curtos** → grande desvio de μ₀ possível → prior wide

Isto faz sentido porque é difícil detectar pequenas mudanças em blocos curtos!

### 3.3 Hiperpriors (Full Bayesian Approach)

No modelo completo de Setz, usamos hiperpriors:

```
P(μ₀) = 1                     (improper, -∞ < μ₀ < ∞)
P(σ²) = 1/σ²                  (Jeffreys prior, σ² > 0)
P(p) = 1/p₀                   (uniforme em [0, p₀])
P(w) = 1/w₀                   (uniforme em [0, w₀])
```

Onde w = σ²/(σ₀² + σ²) é a **razão de variâncias**.

**Invariâncias:**
- Invariante a translações (μ₀)
- Invariante a escala (σ²)

---

## 4. LIKELIHOOD E DATA FACTORS

### 4.1 Likelihood Condicional

Dado um bloco [i+1, j] com parâmetros (μ, σ²):

```
P(Xᵢⱼ | μ, σ²) = ∏ (1/√(2πσ²)) · exp(-(Xₗ - μ)²/(2σ²))
                l=i+1..j

               = (2πσ²)^(-(j-i)/2) · exp(-Σ(Xₗ - μ)²/(2σ²))
```

### 4.2 Data Factor (Integrando μ)

**Prior:** μ ~ N(μ₀, σ₀²/(j-i))

**Posterior (conjugado):**

```
μ | Xᵢⱼ, σ² ~ N(μ̂ᵢⱼ, σ̂²ᵢⱼ)
```

Onde:

```
σ̂²ᵢⱼ = 1 / (1/σ₀² + (j-i)/σ²)
μ̂ᵢⱼ = σ̂²ᵢⱼ · (μ₀/σ₀² + (j-i)·X̄ᵢⱼ/σ²)
```

**Data Factor (integrando μ):**

```
fᵢⱼ(Xᵢⱼ | σ²) = (2πσ²)^(-(j-i)/2) · (σ²/(σ₀² + σ²))^(1/2) · exp(Vᵢⱼ)
```

Onde:

```
Vᵢⱼ = -Σ(Xₗ - X̄ᵢⱼ)²/(2σ²) - (j-i)(X̄ᵢⱼ - μ₀)²/(2(σ₀² + σ²))
```

**Componentes de Vᵢⱼ:**

1. **W_ij = Σ(Xₗ - X̄ᵢⱼ)²:** Variação **dentro** do bloco (within-block variance)
2. **B_ij = (j-i)(X̄ᵢⱼ - μ₀)²:** Variação **entre** blocos (between-block variance)

### 4.3 Data Factor (Integrando μ e σ²)

**Prior:** σ² ~ 1/σ² (improper Jeffreys)

Integrando sobre σ²:

```
fᵢⱼ(Xᵢⱼ) = ∫₀^∞ fᵢⱼ(Xᵢⱼ | σ²) · (1/σ²) dσ²
```

**Resultado (usando improper priors):**

```
fᵢⱼ(Xᵢⱼ) ∝ ∫₀^w₀ w^((b-1)/2) / (W + Bw)^((n-1)/2) dw
```

Esta integral é uma **incomplete beta function** que pode ser calculada numericamente.

### 4.4 Aproximação Simplificada

Na implementação prática (versão rápida), usamos:

```
log fᵢⱼ ≈ -(j-i-1)/2 · log(W_ij + ε)
```

Onde:
- W_ij = Σ(Xₗ - X̄ᵢⱼ)² é a variância dentro do bloco
- ε > 0 é um termo de regularização pequeno

**Intuição:** Blocos com **baixa variância interna** têm **alta likelihood**.

---

## 5. POSTERIOR DISTRIBUTIONS

### 5.1 Posterior sobre Partições

**Teorema de Bayes:**

```
P(ρ|X) = P(X|ρ) · P(ρ) / P(X)
       = [∏ fᵢⱼ(Xᵢⱼ)] · [∏ cᵢⱼ] / P(X)
       = [∏ c̃ᵢⱼ] / P(X)
```

Onde c̃ᵢⱼ = cᵢⱼ · fᵢⱼ é a **posterior cohesion**.

### 5.2 Posterior sobre Parâmetros

Para um bloco [i+1, j], o parâmetro posterior é:

```
μᵢⱼ | Xᵢⱼ ~ N(μ̂ᵢⱼ, σ̂²ᵢⱼ)
```

Com:

```
μ̂ᵢⱼ = (1-w)·X̄ᵢⱼ + w·μ₀
```

Onde w = σ²/(σ₀² + σ²) é o **peso do prior**.

**Intuição:**
- w → 0: Prior fraco, μ̂ᵢⱼ ≈ X̄ᵢⱼ (acredita nos dados)
- w → 1: Prior forte, μ̂ᵢⱼ ≈ μ₀ (acredita no prior)

### 5.3 Marginalização sobre Partições

A **média posterior final** em cada ponto k é:

```
E[μₖ | X] = Σ  E[μₖ | Xᵢⱼ, ρ] · P(ρ | X)
           ρ
```

Esta é uma **média ponderada** sobre TODAS as partições possíveis!

**Usando relevâncias:**

```
E[μₖ | X] = Σ  E[μₖ | Xᵢⱼ] · rᵢⱼ(X)
          i<k≤j
```

Onde a soma é sobre todos blocos que **contêm** k.

---

## 6. ALGORITMO FORWARD-BACKWARD

### 6.1 Problema Computacional

Enumeração exaustiva:
- Número de partições de n elementos = **Número de Bell** Bₙ
- B₁₀ = 115,975
- B₂₀ = 51,724,158,235,372
- B₅₀ ≈ 10^47 (intratável!)

**Solução:** Programação dinâmica usando **Product Partition Model**.

### 6.2 Lambda-Recursão (Forward)

**Definição:**

λᵢⱼ = soma sobre todas partições de [i+1, j]

**Recursão:**

```
λᵢⱼ = Σ  cᵢₖ · λₖⱼ
     k=i..j-1
```

**Caso Base:**
```
λᵢᵢ = 1    (partição vazia)
```

**Complexidade:** O(n²)

### 6.3 Backward Pass (Relevâncias)

A relevância rᵢⱼ pode ser calculada como:

```
rᵢⱼ(X) = (λ₀ᵢ · c̃ᵢⱼ · λⱼₙ) / λ₀ₙ
```

**Algoritmo:**

```
1. Forward: Calcular λ₀ⱼ para j = 1..n (probabilidade de chegar em j)
2. Backward: Calcular λⱼₙ para j = 0..n-1 (probabilidade de j até n)
3. Relevance: rᵢⱼ = (λ₀ᵢ · c̃ᵢⱼ · λⱼₙ) / λ₀ₙ
```

### 6.4 Probabilidade de Change Point

A probabilidade de haver mudança no ponto t:

```
P(change em t | X) = Σ P(ρ | X)
                    ρ: t é change point em ρ
```

**Cálculo via Forward-Backward:**

```
P(change em t | X) = (λ₀,ₜ₋₁ · p · λₜₙ) / λ₀ₙ
```

Onde:
- λ₀,ₜ₋₁: probabilidade forward até t-1
- p: prior de mudança
- λₜₙ: probabilidade backward de t até n
- λ₀ₙ: normalização

---

## 7. CÁLCULO DE PROBABILIDADES

### 7.1 Log-Space Arithmetic

Para evitar **underflow numérico**, trabalhamos em log-space:

```
log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))
           = logsumexp([log(a), log(b)])
```

**Python:**
```python
from scipy.special import logsumexp
log_sum = logsumexp([log_a, log_b])
```

### 7.2 Fórmula Prática do Forward

```
log λ₀ⱼ = logsumexp([
    log λ₀ᵢ + log cᵢⱼ + log fᵢⱼ
    for i in range(j)
])
```

### 7.3 Conversão para Probabilidades

```
P(change em t) = exp(log_λ₀,ₜ₋₁ + log(p) + log_λₜₙ - log_λ₀ₙ)
```

**Clipping:** Limitar entre [0, 1] para evitar erros numéricos.

---

## 8. IMPLEMENTAÇÃO PRÁTICA

### 8.1 Versão Simplificada (Janela Móvel)

Para séries longas (n > 1000), usar janela móvel:

**Algoritmo:**

```
Para cada t = 1..n:
    1. Pegar janela [t - window, t]
    2. Detectar mudanças dentro da janela
    3. Atribuir probabilidade ao ponto t
```

**Vantagem:** Complexidade O(n · window²) ao invés de O(n³).

### 8.2 Teste de Múltiplos Split Points

Dentro da janela, testar splits em posições estratégicas:

```
split_positions = [0.2, 0.4, 0.6, 0.8] · window_size
```

Para cada split:

1. Calcular likelihood do modelo com split
2. Calcular likelihood do modelo sem split
3. Bayes Factor = likelihood_with / likelihood_without
4. Converter para probabilidade via prior

### 8.3 Bayes Factor

```
BF = P(dados | H₁: há mudança) / P(dados | H₀: não há mudança)
```

**Interpretação:**
- BF > 10: Evidência forte para mudança
- BF > 3: Evidência moderada
- BF < 1/3: Evidência contra mudança

**Conversão para Probabilidade:**

```
P(mudança | dados) = BF · P(mudança) / (BF · P(mudança) + 1 - P(mudança))
```

---

## 9. EXEMPLO NUMÉRICO COMPLETO

### 9.1 Setup

Dados simulados:
```
X = [2.0, 2.1, 2.0, 2.2, 5.0, 5.1, 5.0, 5.2]
      ←─── Regime 1 ────→  ←─── Regime 2 ───→
```

Parâmetros:
- p = 0.2 (prior de mudança)
- μ₀ = 0.0 (prior da média)
- σ² = 1.0 (variância conhecida)
- σ₀² = 10.0 (variância do prior)

### 9.2 Cálculo dos Data Factors

**Bloco [1,4]:** X = [2.0, 2.1, 2.0, 2.2]
```
X̄₁₄ = 2.075
W₁₄ = Σ(Xᵢ - 2.075)² = 0.0075 + 0.000625 + 0.005625 + 0.015625 = 0.0294

log f₁₄ ≈ -3/2 · log(0.0294) = 5.24
```

**Bloco [5,8]:** X = [5.0, 5.1, 5.0, 5.2]
```
X̄₅₈ = 5.075
W₅₈ = 0.0294   (mesma variância interna!)

log f₅₈ ≈ 5.24
```

**Bloco [1,8]:** X = [2.0, 2.1, 2.0, 2.2, 5.0, 5.1, 5.0, 5.2]
```
X̄₁₈ = 3.575
W₁₈ = Σ(Xᵢ - 3.575)² = 4·(1.575)² + 4·(1.475)² = 18.41

log f₁₈ ≈ -7/2 · log(18.41) = -10.13
```

### 9.3 Cohesions

```
c₁₄ = 0.2 · (0.8)³ = 0.1024
c₅₈ = 0.2 · (0.8)³ = 0.1024
c₁₈ = 0.2 · (0.8)⁷ = 0.0419
```

### 9.4 Posterior Cohesions

```
c̃₁₄ = c₁₄ · f₁₄ = 0.1024 · exp(5.24) = 19.2
c̃₅₈ = c₅₈ · f₅₈ = 0.1024 · exp(5.24) = 19.2
c̃₁₈ = c₁₈ · f₁₈ = 0.0419 · exp(-10.13) = 0.0000015
```

### 9.5 Comparação de Modelos

**Modelo A:** Mudança em t=4
```
P(X | modelo A) ∝ c̃₁₄ · c̃₅₈ = 19.2 · 19.2 = 368.64
```

**Modelo B:** Sem mudança
```
P(X | modelo B) ∝ c̃₁₈ = 0.0000015
```

**Bayes Factor:**
```
BF = 368.64 / 0.0000015 ≈ 2.5 × 10⁸  (!!!)
```

**Conclusão:** Evidência **extremamente forte** de mudança em t=4.

### 9.6 Probabilidade Posterior

```
prior_odds = p/(1-p) = 0.2/0.8 = 0.25
posterior_odds = BF · prior_odds = 2.5×10⁸ · 0.25 = 6.25×10⁷

P(mudança em t=4 | X) = posterior_odds / (1 + posterior_odds)
                       ≈ 0.9999999984  ≈ 100%
```

---

## 10. INTUIÇÃO E INSIGHTS

### 10.1 Por Que o BCP Funciona?

**1. Parsimônia Bayesiana:**
- Prior favorece poucas mudanças (via cohesions geométricas)
- Dados superam prior apenas quando mudança é **clara**

**2. Automatic Relevance Determination:**
- Blocos curtos têm prior largo → tolera grandes desvios
- Blocos longos têm prior apertado → exige consistência

**3. Marginalização:**
- Considera TODAS partições possíveis
- Robusto a incerteza sobre localização exata

### 10.2 Interpretação dos Hiperparâmetros

**p (probabilidade de mudança):**
- p pequeno (0.05-0.15): Conservador, detecta apenas mudanças drásticas
- p grande (0.25-0.40): Agressivo, detecta mudanças sutis

**w (razão de variâncias):**
- w pequeno (0.05-0.15): Prior fraco, acredita mais nos dados
- w grande (0.25-0.40): Prior forte, requer mudanças maiores

### 10.3 Limitações

**1. Assumem normalidade:**
- Se dados são heavy-tailed, pode dar falsos positivos
- Solução: Usar transformações (log, rank)

**2. Independência:**
- Não modela autocorrelação
- Em séries com forte dependência temporal, pode ser subótimo

**3. Retrospectivo:**
- Resultado muda conforme chegam novos dados
- A probabilidade em t depende de observações futuras!

### 10.4 Extensões Avançadas

**1. Mudança em múltiplos parâmetros:**
- Detectar mudanças em média **E** variância simultaneamente
- Modelo N-NGIG de Setz (Normal - Normal Generalized Inverse Gaussian)

**2. Markov Dependency:**
- Parâmetros de blocos adjacentes correlacionados
- Mais complexo, mas mais realista

**3. Online Detection:**
- Usar apenas dados até t para estimar P(change em t)
- Perde informação, mas permite uso em tempo real

---

## 11. COMPARAÇÃO COM OUTROS MÉTODOS

### 11.1 vs. CUSUM

**CUSUM (Cumulative Sum):**
```
Sₜ = max(0, Sₜ₋₁ + (Xₜ - μ₀) - k)
```

**Vantagens BCP:**
- ✅ Quantifica incerteza (probabilidade)
- ✅ Detecta múltiplas mudanças
- ✅ Não requer threshold manual

**Vantagens CUSUM:**
- ✅ Computacionalmente mais rápido
- ✅ Controle direto de false alarm rate

### 11.2 vs. Structural Break Tests

**Chow Test, Bai-Perron:**
- Baseados em F-statistics
- Requerem especificação do número de breaks

**Vantagens BCP:**
- ✅ Inferência automática de número de mudanças
- ✅ Prior bayesiano incorpora conhecimento a priori
- ✅ Posterior distribution completa

### 11.3 vs. Hidden Markov Models

**HMM:**
- Assume estados latentes discretos
- Transições entre estados via matriz

**Vantagens BCP:**
- ✅ Não requer especificar número de estados
- ✅ Mais interpretável (mudanças pontuais)
- ✅ Menos parâmetros a estimar

**Vantagens HMM:**
- ✅ Modela dependência temporal
- ✅ Permite transições reversíveis

---

## 12. CHECKLIST DE IMPLEMENTAÇÃO

### ✅ Pré-processamento
- [ ] Remover outliers extremos (> 5σ)
- [ ] Verificar stationaridade (se muito não-estacionário, diferenciar)
- [ ] Normalizar se escalas muito diferentes

### ✅ Escolha de Hiperparâmetros
- [ ] Começar com p = 0.2, w = 0.2
- [ ] Se muitos falsos positivos: reduzir p
- [ ] Se não detecta mudanças óbvias: aumentar p

### ✅ Validação
- [ ] Testar em dados sintéticos com mudanças conhecidas
- [ ] Verificar calibração: P(change) vs taxa real
- [ ] Cross-validation em séries similares

### ✅ Interpretação
- [ ] Percentil > 90: Alerta máximo
- [ ] Percentil 75-90: Cautela
- [ ] Percentil < 60: Normal
- [ ] Combinar com outros indicadores

---

## 13. CÓDIGO MÍNIMO COMENTADO

```python
import numpy as np
from scipy.special import logsumexp

def bcp_detect(returns, p0=0.2, max_block=200):
    """
    BCP via Forward-Backward (implementação correta).
    
    Args:
        returns: np.array de retornos
        p0: prior de probabilidade de mudança
        max_block: tamanho máximo de bloco (limita complexidade)
    
    Returns:
        posterior_prob: P(change em t | dados) para cada t
    """
    n = len(returns)
    
    # Log-cohesions (prior geométrico de Barry & Hartigan)
    log_p = np.log(p0)
    log_1mp = np.log(1 - p0)
    
    def log_cohesion(block_len, is_last):
        """Cohesion: p·(1-p)^(len-1) ou (1-p)^(len-1) para último bloco."""
        if is_last:
            return (block_len - 1) * log_1mp
        return log_p + (block_len - 1) * log_1mp
    
    # Data factors (log-likelihood marginal de cada bloco)
    log_factors = {}
    for i in range(n):
        for j in range(i+1, min(i + max_block + 1, n+1)):
            block = returns[i:j]
            if len(block) < 2:
                log_factors[(i,j)] = 0.0
            else:
                mu = np.mean(block)
                W = np.sum((block - mu)**2)
                log_factors[(i,j)] = -(len(block)-1)/2 * np.log(W + 1e-10)
    
    # Forward pass
    log_forward = np.full(n+1, -np.inf)
    log_forward[0] = 0.0
    
    for j in range(1, n+1):
        log_probs = []
        for i in range(max(0, j - max_block), j):
            log_coh = log_cohesion(j - i, j == n)
            if (i, j) in log_factors:
                log_probs.append(log_forward[i] + log_coh + log_factors[(i,j)])
        if log_probs:
            log_forward[j] = logsumexp(log_probs)
    
    # Backward pass (ESSENCIAL: computar separadamente!)
    log_backward = np.full(n+1, -np.inf)
    log_backward[n] = 0.0
    
    for j in range(n-1, -1, -1):
        log_probs = []
        for k in range(j+1, min(j + max_block + 1, n+1)):
            log_coh = log_cohesion(k - j, k == n)
            if (j, k) in log_factors:
                log_probs.append(log_coh + log_factors[(j,k)] + log_backward[k])
        if log_probs:
            log_backward[j] = logsumexp(log_probs)
    
    # Probabilidade de change point em t
    posterior_prob = np.zeros(n)
    for t in range(1, n):
        log_post = log_forward[t] + log_backward[t] - log_forward[n]
        posterior_prob[t] = np.clip(np.exp(log_post), 0, 1)
    
    return posterior_prob

# Uso:
# prob = bcp_detect(retornos_ibovespa, p0=0.18)
# print(f"Probabilidade atual: {prob[-1]:.2%}")
```

---

## 14. REFERÊNCIAS MATEMÁTICAS

### Papers Fundamentais

1. **Barry, D. & Hartigan, J.A. (1992)**
   "Product Partition Models for Change Point Problems"
   *Annals of Statistics*, 20(1), 260-279
   
2. **Barry, D. & Hartigan, J.A. (1993)**
   "A Bayesian Analysis for Change Point Problems"
   *Journal of the American Statistical Association*, 35, 309-319

3. **Setz, T. (2017)**
   "Stable Portfolio Design Using Bayesian Change Point Models and Geometric Shape Factors"
   *ETH Zurich PhD Thesis*
   DOI: 10.3929/ethz-b-000244960

### Livros Recomendados

4. **Gelman et al. (2013)**
   "Bayesian Data Analysis" (3rd ed.)
   Chapman & Hall/CRC

5. **Bishop, C.M. (2006)**
   "Pattern Recognition and Machine Learning"
   Springer (Capítulo sobre HMMs tem conexões)

---

## 📝 RESUMO EXECUTIVO

### O Que o BCP Faz?

Dado uma série X₁, ..., Xₙ, calcula para cada ponto t:

```
P(mudança estrutural em t | todos os dados observados)
```

### Como Funciona?

1. **Prior:** Assume poucas mudanças (via cohesions geométricas)
2. **Likelihood:** Calcula probabilidade dos dados sob cada partição possível
3. **Posterior:** Combina via Bayes para obter probabilidades
4. **Eficiência:** Usa programação dinâmica (Forward-Backward) O(n²)

### Hiperparâmetros Principais

- **p ∈ (0.1, 0.3):** Probabilidade a priori de mudança por ponto
  - Menor = mais conservador
  - Maior = mais sensível

### Output Típico

```
Tempo   Preço    P(change)   Status
t=100   125000   0.15        🟢 Estável
t=200   127000   0.42        🟡 Atenção
t=300   122000   0.87        🔴 Alerta!
```

### Interpretação

- **P > 0.75:** Alta probabilidade de reversão
- **P ∈ [0.5, 0.75]:** Probabilidade moderada
- **P < 0.5:** Baixa probabilidade

---

**FIM DO DOCUMENTO MATEMÁTICO COMPLETO**

*Para dúvidas sobre implementação específica, consulte o código comentado na seção 13 ou os scripts Python fornecidos.*
