# Detector de Reversão de Tendência - IBOVESPA
## Metodologia Bayesian Change Point Detection (Setz & Würtz, ETH Zurich)

Este projeto implementa o algoritmo de Bayesian Change Point Detection baseado na tese de doutorado de **Tobias Setz** (ETH Zurich, 2017) para detectar pontos de reversão de tendência em ativos financeiros.

## 📚 Fundamentação Teórica

O método se baseia em:
- **Barry & Hartigan (1993)** - "A Bayesian Analysis for Change Point Problems"
- **Setz (2017)** - "Stable Portfolio Design Using Bayesian Change Point Models and Geometric Shape Factors"

### Como Funciona

O algoritmo:
1. **Modela** a série de retornos como sequência de blocos com parâmetros (média/variância) constantes
2. **Detecta** mudanças estruturais calculando probabilidade posterior de change point em cada momento
3. **Quantifica** a probabilidade de estar em um ponto de inflexão

**Vantagens para ativos de alta volatilidade (cripto, IBOVESPA):**
- ✅ Detecção online (tempo real)
- ✅ Adapta-se a mudanças abruptas de regime
- ✅ Quantifica incerteza probabilisticamente
- ✅ Não assume distribuição estacionária

## 🚀 Uso Rápido

### Requisitos

```bash
pip install pandas numpy scipy matplotlib
```

### Preparar Dados

Seu arquivo CSV deve ter (no mínimo):
- Coluna de **timestamp/data** 
- Coluna de **preço de fechamento**

Exemplo de formato:
```csv
timestamp,close
2026-02-15 09:00:00,125000.50
2026-02-15 09:05:00,125100.75
2026-02-15 09:10:00,124950.25
...
```

### Executar Análise

```bash
# Uso básico
python ibovespa_bcp_reversal_detector.py seu_arquivo.csv

# Com parâmetros personalizados
python ibovespa_bcp_reversal_detector.py seu_arquivo.csv 0.15 0.15
#                                                      arquivo  p0   w0
```

## ⚙️ Parâmetros

### `p0` - Prior de Probabilidade de Mudança (default: 0.2)
- **Menor valor** (0.05-0.15): Detecta apenas mudanças muito significativas
- **Valor médio** (0.15-0.25): Balanço entre sensibilidade e ruído
- **Maior valor** (0.25-0.40): Mais sensível, detecta mudanças sutis

**Recomendação para IBOVESPA 5min**: 0.15 - 0.20

### `w0` - Prior de Magnitude de Mudança (default: 0.2)
- **Menor valor** (0.05-0.15): Exige mudanças de grande magnitude
- **Valor médio** (0.15-0.25): Moderado
- **Maior valor** (0.25-0.40): Detecta mudanças menores

**Recomendação para IBOVESPA 5min**: 0.15 - 0.20

## 📊 Interpretação dos Resultados

### Status da Análise

O script fornece um status claro:

- 🔴 **ALERTA ALTO** (>90º percentil): Alta probabilidade de reversão **iminente**
- 🟡 **ALERTA MODERADO** (75-90º percentil): Probabilidade **elevada** de reversão
- 🟠 **ATENÇÃO** (60-75º percentil): Probabilidade **moderada** de reversão  
- 🟢 **ESTÁVEL** (<60º percentil): Baixa probabilidade de reversão

### Métricas Principais

1. **Probabilidade de Mudança** (0-1)
   - Probabilidade posterior de haver change point na última observação
   - >0.7: Muito alta
   - 0.5-0.7: Alta
   - 0.3-0.5: Moderada
   - <0.3: Baixa

2. **Intensidade de Mudança**
   - Combina probabilidade com magnitude (volatilidade)
   - Indica não apenas SE haverá mudança, mas QUÃO DRÁSTICA será

3. **Percentil** 
   - Posição da probabilidade atual relativa aos últimos 100 períodos
   - Percentil alto = situação incomum = maior atenção

## 📈 Gráficos Gerados

O script gera 4 painéis:

1. **Preço**: Série temporal do ativo
2. **Média Posterior**: E(μ|X) - média estimada considerando estrutura de mudanças
3. **Variância Posterior**: Var(μ|X) - volatilidade estrutural
4. **Probabilidade de Change Point**: P(mudança|X) - métrica chave

## 🎯 Exemplo Prático

### Caso 1: IBOVESPA com Feeling de Reversão

Você mencionou que o futuro indica +4.6% mas seu "feeling" diz que estamos no ponto de inflexão.

**Com o BCP você pode:**

```python
results = analyze_ibovespa_reversal(
    'ibovespa_5min.csv',
    p0=0.18,  # sensibilidade moderada-alta
    w0=0.18
)

status = results['status']
prob = status['prob_mudanca_atual']
percentil = status['percentil_prob']

if percentil > 85:
    print("✅ Seu 'feeling' está correto!")
    print(f"   Probabilidade de reversão: {prob:.2%}")
    print(f"   Nível: {percentil:.0f}º percentil (muito alto)")
elif percentil > 70:
    print("⚡ Há evidências de mudança estrutural")
    print(f"   Mas não é conclusivo ainda ({percentil:.0f}º percentil)")
else:
    print("❌ Baixa probabilidade de reversão")
    print(f"   Mercado parece seguir tendência atual")
```

### Caso 2: Trading Intraday

Para trading de curto prazo com dados de 5 minutos:

```python
# Use parâmetros mais sensíveis
results = analyze_ibovespa_reversal(
    'ibovespa_5min_hoje.csv',
    p0=0.25,  # mais sensível a mudanças
    w0=0.20
)

# Monitore em tempo real
for i in range(0, len(df), 12):  # a cada hora (12 x 5min)
    window_data = df.iloc[:i+12]
    # ... reexecute análise
```

## 🔬 Implementação Técnica

### Algoritmo Forward-Backward

O script usa programação dinâmica **O(n²)** ao invés de enumeração exaustiva O(2^n):

```
1. Forward Pass: Calcula P(X[1:t], último change point em t)
2. Backward Pass: Calcula P(change point em t | todos os dados)
3. Posterior: Combina evidências para estimar parâmetros
```

### Product Partition Model (PPM)

- **Prior**: Define probabilidade de partições via cohesions
- **Likelihood**: Modelo Normal com variância desconhecida  
- **Posterior**: Bayesian update via integração analítica

## 🎓 Referências

1. **Setz, T.** (2017). "Stable Portfolio Design Using Bayesian Change Point Models and Geometric Shape Factors". ETH Zurich PhD Thesis.

2. **Barry, D. & Hartigan, J.A.** (1993). "A Bayesian Analysis for Change Point Problems". *Journal of the American Statistical Association*, 35, 309–319.

3. **Würtz, D., Chalabi, Y. & Setz, T.** (2011). "Stability Analytics of Vulnerabilities in Financial Time Series". ETH Econophysics Working Paper.

## 💡 Dicas

### Para Melhorar Detecção

1. **Mais dados = melhor**: Mínimo 200 observações, ideal >500
2. **Dados limpos**: Remova outliers extremos antes
3. **Consistência**: Use sempre a mesma frequência (5min)
4. **Contexto**: Combine com análise técnica tradicional

### Limitações

- ⚠️ Não faz previsão do **timing exato** da reversão
- ⚠️ Apenas indica **probabilidade** de mudança estrutural
- ⚠️ Resultados são retrospectivos (olhando para trás)
- ⚠️ Requer dados suficientes para calibração

### Quando NÃO usar

- ❌ Mercados sem liquidez
- ❌ Ativos sem histórico suficiente (<100 observações)
- ❌ Durante anúncios de dados macroeconômicos (jumps exógenos)

## 📞 Suporte

Para dúvidas sobre a metodologia, consulte:
- Tese original: https://doi.org/10.3929/ethz-b-000244960
- Paper BCP: Barry & Hartigan (1993)

---

**Desenvolvido com base na pesquisa de Tobias Setz & Diethelm Würtz (ETH Zurich)**  
*"Monitoring investments effectively and identifying risks early on"*
