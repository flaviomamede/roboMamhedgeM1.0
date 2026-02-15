# Benchmarking e Robôs de Referência

Pesquisa sobre: (1) robôs com benchmarking tipo MEF, (2) robôs que "contam vantagem", (3) ecossistema Papers With Backtest para estudo.

---

## 1. Benchmarking em trading — o equivalente ao MEF

Em simulação por MEF, você compara com **solução analítica** ou **outra simulação de referência**. Em trading, o equivalente seria:

### O que existe hoje

| Abordagem | Descrição | Limitação |
|-----------|-----------|-----------|
| **Backtest vs out-of-sample** | Treina em 70%, valida em 30% | Estudo com 888 algoritmos: **Sharpe do backtest tem R² < 0,025** para prever desempenho real |
| **Walk-forward** | Rebalanceia janelas no tempo | Fraco contra overfitting |
| **CPCV** (Combinatorial Purged Cross-Validation) | Validação cruzada com purga | Melhor que K-Fold, mas ainda não é "solução analítica" |
| **Live evaluation** | Dados em tempo real, sem vazamento | Custo alto, demora |

### Conclusão

**Não existe "solução analítica" em trading.** O mercado não tem equação fechada. O mais próximo de benchmark rigoroso é:

1. **Datasets padronizados** — mesmo dado para todos
2. **Walk-forward / out-of-sample** — nunca validar no mesmo período que treinou
3. **LiveTradeBench, AI-Trader** — avaliação em tempo real (evita look-ahead bias)

---

## 2. Benchmarks acadêmicos e industriais

### [Papers With Backtest (pwb-toolbox)](https://github.com/paperswithbacktest/pwb-toolbox)

- **~1 TB** de dados limpos (Stocks, Bonds, Crypto, Forex, ETFs)
- **140+ papers** codificados em estratégias executáveis
- `load_dataset("Stocks-Daily-Price")` — mesmo dataset para todos
- Integração com **Backtrader**
- Requer Python 3.10+, API key ou Hugging Face

### [LiveTradeBench](https://github.com/ulab-uiuc/live-trade-bench)

- Avaliação **em tempo real** (evita overfitting de backtest)
- Ações US + Polymarket
- Suporta GPT, Claude, Gemini
- Paper: [LiveTradeBench: Seeking Real-World Alpha with LLMs](https://arxiv.org/abs/2511.03628)

### [QuantBench](https://arxiv.org/html/2504.18600v1)

- Benchmark industrial para investimento quantitativo
- Padronização, flexibilidade para IA

### [FinTSB](https://arxiv.org/html/2502.18834v1)

- Benchmark para forecasting de séries financeiras
- Métricas padronizadas, custos de transação

---

## 3. Robôs que "contam vantagem" no Brasil

| Robô / Plataforma | Alegação | Contexto |
|--------------------|----------|----------|
| **Trade2go** | Top 10 OnTick (XP), 150% rentabilidade bruta (Maui #3) | Fintech com IA, opera WIN, dólar, Bitcoin |
| **N2 Trading** (Nexus/Empiricus) | 81% acerto, até R$ 5.600/mês, 21× Ibovespa | 1–2 ops/dia, gestão de risco |
| **AlphaBit 01** (SmarttStore) | 214% retorno em 1 ano, 61% trades lucrativos | BITFUT, tendência+momentum+congestão |
| **Sistemas patrocinados** | "R$ 5.600/mês", "81% acerto" | Conteúdo de marca — **cuidado** |

### Avisos

- Resultados **não consideram impostos e custos** (AlphaBit)
- Conteúdo patrocinado tende a exagerar
- Estudo com 888 algoritmos: **backtest não prediz desempenho real**
- "Solução analítica" não existe — não há como provar que um robô é "bom" de forma definitiva

---

## 4. Ecossistema Papers With Backtest — para estudo

O que sugeri antes (pwb-toolbox, AlphaEvolve) forma um ecossistema coerente:

```
pwb-toolbox          → Datasets + Backtrader + run_backtest()
pwb-alphaevolve      → LLM evolui estratégias automaticamente
pwb-backtrader       → Engine de backtest
```

### [pwb-alphaevolve](https://github.com/paperswithbacktest/pwb-alphaevolve)

- **LLM** (OpenAI o3 ou local) evolui estratégias
- Usa datasets do pwb-toolbox
- Avaliador: Backtrader walk-forward, KPIs (Sharpe, CAGR, Calmar, DD)
- GUI Streamlit para acompanhar evolução
- Requer: `OPENAI_API_KEY`, `HF_ACCESS_TOKEN` (Hugging Face)

### Plano de estudo sugerido

| Etapa | Ação |
|-------|------|
| 1 | Instalar `pwb-toolbox`, carregar `Stocks-Daily-Price` ou similar |
| 2 | Rodar estratégia de exemplo (ex.: Golden Cross) no mesmo dataset |
| 3 | Comparar roboMT5 R6 com baseline do pwb (mesmo período, mesmo ativo se houver) |
| 4 | (Opcional) Instalar AlphaEvolve, evoluir seed strategy e comparar |

### Compatibilidade com roboMT5

- **pwb-toolbox** usa dados **diários** (daily); roboMT5 usa **5 min**
- Para benchmark direto: converter WIN 5min → daily ou buscar dataset WIN no pwb
- Alternativa: usar pwb como **referência de metodologia** (walk-forward, métricas) e aplicar no roboMT5

---

## 6. Implementação no roboMT5

- **benchmark_pwb.py** — Walk-forward (70% train / 30% test), métricas padronizadas
- **roboMamhedgeR7.py** — R6 + Take Profit + otimização de parâmetros
- **roboMamhedgeR8.py** — Híbrido R2 (EMA200, momentum) + R6 (saída por pico)
- **run_benchmark_completo.py** — Executa benchmark + otimização + projeção 60 dias

---

## 5. Resumo

| Pergunta | Resposta |
|---------|----------|
| **Existe benchmark tipo MEF?** | Não há "solução analítica". O mais próximo: datasets padronizados + out-of-sample + live evaluation |
| **Robôs que contam vantagem?** | Sim (Trade2go, N2, AlphaBit), mas resultados são de backtest/simulação e nem sempre consideram custos |
| **Vale estudar pwb?** | Sim — para datasets, metodologia de validação e (opcional) evolução automática com AlphaEvolve |

---

## Links

- [pwb-toolbox](https://github.com/paperswithbacktest/pwb-toolbox)
- [pwb-alphaevolve](https://github.com/paperswithbacktest/pwb-alphaevolve)
- [LiveTradeBench](https://github.com/ulab-uiuc/live-trade-bench)
- [Paper: Backtest vs Out-of-Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2745220)
- [Papers With Backtest (site)](https://paperswithbacktest.com/)
