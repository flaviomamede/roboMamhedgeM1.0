# Histórico de Versões — roboMT5

Documentação das melhorias e alterações implementadas em cada versão do projeto.

---

## roboMamhedge

### R0 — Versão inicial

**Arquivo:** `roboMamhedgeR0.py`

- Sistema de sinais baseado em EMA9 e EMA21
- Filtro de volatilidade: ATR > ATR_mean (20 períodos)
- Sinais: 1 (compra), -1 (venda), 0 (neutro)
- Fallback para dados sintéticos se CSV não existir
- Saída: últimas 5 linhas com close, EMAs e sinal

---

### R1 — Sugestões do Gemini

**Arquivo:** `roboMamhedgeR1.py`

| Melhoria | Descrição |
|----------|-----------|
| **Backtest** | Simulação de trades reais com capital inicial R$ 10.000 |
| **Stop loss** | 1,5 × ATR do preço de entrada |
| **Take profit** | 2,0 × ATR do preço de entrada |
| **Filtro de horário** | Opera apenas entre 10h e 17h |
| **Remoção ATR_mean** | Entrada apenas por cruzamento de EMAs |
| **Métricas** | Win rate, ganho/perda médios, expectativa por trade |
| **EMA** | `adjust=False` nas médias exponenciais |
| **Custo** | R$ 1,00 por trade |

---

### R2 — Filtros de tendência macro e momentum

**Arquivo:** `roboMamhedgeR2.py`

| Melhoria | Descrição |
|----------|-----------|
| **EMA200** | Média de 200 para tendência macro |
| **Momentum** | close - close.shift(10) — força de tendência |
| **Compra** | EMA9 > EMA21 **e** close > EMA200 **e** Momentum > 0 |
| **Venda** | EMA9 < EMA21 **e** close < EMA200 **e** Momentum < 0 |

Objetivo: operar apenas no sentido da tendência de maior prazo.

---

### R6 — EMA4 + RSI Peak Detection

**Arquivo:** `roboMamhedgeR6.py`

| Elemento | Descrição |
|----------|-----------|
| **Entrada** | RSI bullish window (40+ nos últimos 5) + EMA4 virando para cima + MACD Hist > 0 |
| **Saída** | Pico de máximo no RSI (antecipação) ou stop loss 2×ATR |
| **Stop loss** | 2×ATR abaixo do preço de entrada (proteção contra drawdown) |
| **Filtro BRT** | Exclui 10h–10:45, 11h e 16:30+ (via `utils_fuso`) |

**Resultados (backtest):** ~134 trades, 33,6% win rate, E[P&L] R$ 0,09/trade. Melhor desempenho entre R1–R6.

**Refatoração (config + ta + BB):**
- `config.py` — parâmetros centralizados (RSI, EMA, MACD, BB, ATR)
- Lib `ta` — indicadores padronizados (RSIIndicator, EMAIndicator, MACD, BollingerBands, AverageTrueRange)
- Bandas de Bollinger — filtro opcional (`BB_USE=True`, `BB_ENTRY='low'|'mid'`)

---

### R5 — MACD + RSI (long e short)

**Arquivo:** `roboMamhedgeR5.py`

| Elemento | Descrição |
|----------|-----------|
| **Compra** | MACD Hist > 0 e RSI cruzando acima de 40 |
| **Venda** | MACD Hist < 0 e RSI cruzando abaixo de 60 |
| **Stop/Target** | 2×ATR / 3×ATR |

---

### R4 — MACD + RSI (long-only)

**Arquivo:** `roboMamhedgeR4.py`

| Elemento | Descrição |
|----------|-----------|
| **MACD** | Histograma (12, 26, 9) > 0 |
| **RSI** | Cruzamento acima de 40 (saída de oversold) |
| **Entrada** | MACD Hist > 0 **e** RSI cruza de ≤40 para >40 |
| **Stop/Target** | 2×ATR / 3×ATR |
| **Direção** | Apenas compra (long-only) |

---

### R3 — EMAs 20 e 50

**Arquivo:** `roboMamhedgeR3.py`

| Alteração | Descrição |
|-----------|-----------|
| **EMA20/50** | Substitui EMA9/21 — tendência mais suave, menos sinais falsos |

---

### analise_horario.py

Script para analisar desempenho por horário. Revelou que **60% do prejuízo** vem de 10h-11h BRT (13h-14h UTC).

---

### R7 — R6 + Take Profit (otimizado)

**Arquivo:** `roboMamhedgeR7.py`

- R6 + take profit 2.5×ATR, stop 1.5–2×ATR
- Saída: stop > peak > TP (prioridade)
- Grid search: stop, target, RSI, MACD
- Melhor: stop=2, target=1.5–2.5, rsi=40, macd=False

### R8 — Híbrido R2 + R6

**Arquivo:** `roboMamhedgeR8.py`

- Entrada: EMA9>EMA21, close>EMA200, momentum>0 (R2)
- Saída: pico RSI ou stop 2×ATR (R6)

### utils_metrics_pwb.py / run_benchmark_completo.py

- **Walk-forward** 70/30 (train/test) — metodologia Papers With Backtest
- Métricas: win rate, E[P&L], Sharpe, max drawdown
- Projeção 60 dias e escala para meta 11k

### run_all_montecarlo.py

Executa R1–R8 e o Monte Carlo com as métricas de cada robô. Gera `montecarlo_comparativo.png`.

---

## montecarloSimulation

### R0 — Versão atual

**Arquivo:** `montecarloSimulationR0.py`

- Parâmetros fixos (p, gain, loss)
- 20.000 simulações × 500 trades
- Capitais: R$ 5.000, R$ 10.000, R$ 20.000
- Visualização: histogramas, curvas de capital, probabilidade de ruína

---

## Outros módulos

### download_win.py

- Download via Yahoo Finance (yfinance)
- Tickers: WIN=F (principal), BOVA11.SA (fallback)
- Formato: close, high, low — compatível com o robô

---

## Legenda

- **R0, R1, …** — Revisões do robô principal
- Sugestões futuras podem ser registradas aqui antes da implementação
