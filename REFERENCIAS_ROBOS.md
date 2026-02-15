# Referências de Robôs DayTrade — O que copiar ou parametrizar

> "Na engenharia nada se cria, tudo se copia."

Pesquisa de robôs open source para inspiração e reaproveitamento no roboMT5.

---

## 1. Robôs analisados (seus links)

### [robo-daytrade-v1](https://github.com/Rony-s-barbosa/robo-daytrade-v1) — OCR + RSI/EMA/BB

| Aspecto | Detalhe |
|---------|---------|
| **Estratégia** | RSI + EMA 9/21 + Bandas de Bollinger |
| **Compra** | RSI < 30 **e** EMA9 > EMA21 **e** preço ≤ BB inferior |
| **Venda** | RSI > 70 **e** EMA9 < EMA21 **e** preço ≥ BB superior |
| **Biblioteca** | `ta` (Technical Analysis) |
| **Entrada de dados** | OCR na tela (Tesseract) — não usa CSV/API |

**O que copiar:**
- **Bandas de Bollinger** como filtro de sobrecompra/sobrevenda (complementa RSI)
- Uso da lib `ta` para indicadores padronizados
- Lógica de compra em oversold + tendência de alta

**Parametrização sugerida:** Adicionar BB ao R6 como filtro extra (preço ≤ BB_low na entrada).

---

### [Robot-Trader-IA-actor-critic-2](https://github.com/MilianoJunior/Robot-Trader-IA-actor-critic-2) — WIN + RL + MetaTrader

| Aspecto | Detalhe |
|---------|---------|
| **Ativo** | WIN (mesmo que o seu) |
| **Abordagem** | Aprendizado por reforço (Actor-Critic, tf-agents) |
| **Dados** | OHLC 1 min, 6 anos |
| **Integração** | Socket com MetaTrader |
| **Trade.py** | Classe com stop/gain, compra/venda, recompensas |

**O que copiar:**
- **Estrutura de Trade** com stop/gain em pontos
- **Verificação de stop** no candle (high/low vs stop)
- **Cálculo de duração** do trade (penaliza trades longos)
- Ideia de **recompensa** para RL futuro

**Parametrização sugerida:** O `Trade.py` tem lógica de stop/gain que pode inspirar um módulo `Trade` no roboMT5.

---

### [robo-daytrade-b3](https://github.com/dogdirobo/robo-daytrade-b3)

Template Streamlit (GDP dashboard). **Não é robô de trading** — irrelevante.

---

### [sample.daytrader7](https://github.com/WASdev/sample.daytrader7) / [sample.daytrader8](https://github.com/OpenLiberty/sample.daytrader8)

Aplicações Java EE de **sistema de corretagem** (login, portfolio, compra/venda de ações). **Não são robôs algorítmicos** — irrelevantes para estratégia.

---

### [daytrader-example-webrepo](https://github.com/davegree-n/daytrader-example-webrepo)

Microserviços Spring Boot para DayTrader. **Infraestrutura web**, não estratégia de trading.

---

## 2. Outros projetos relevantes (busca ampliada)

### [Freqtrade](https://github.com/freqtrade/freqtrade) — 46k+ stars

| Aspecto | Detalhe |
|---------|---------|
| **Foco** | Cripto (Binance, Kraken, OKX) |
| **Backtest** | Nativo, com otimização |
| **Estratégias** | Python, classe com `populate_indicators`, `populate_entry_trend`, `populate_exit_trend` |
| **UI** | Web + Telegram |

**O que copiar:**
- **Padrão de estratégia** (indicadores → entrada → saída)
- **Otimização de parâmetros** (hiperparâmetros)
- **Métricas** (ROI, drawdown, Sharpe)

**Limitação:** Focado em cripto; WIN/B3 exigiria adaptação ou uso de dados locais.

---

### [AlphaEvolve](https://github.com/paperswithbacktest/pwb-alphaevolve)

Usa LLM para **evoluir estratégias** automaticamente. Integra com Backtrader. Ideia avançada para futura exploração.

---

### [Grid Trading Bot](https://github.com/jordantete/grid_trading_bot)

Grid trading com CCXT. Estratégia diferente (range), mas estrutura de backtest e métricas reaproveitáveis.

---

## 3. O que parametrizar no roboMT5

### 3.1 Indicadores (do robo-daytrade-v1)

```python
# Bandas de Bollinger — adicionar ao R6 ou R7
from ta.volatility import BollingerBands
bb = BollingerBands(close, window=20, window_dev=2)
df['bb_high'] = bb.bollinger_hband()
df['bb_low'] = bb.bollinger_lband()
# Entrada long: preço <= bb_low (oversold) + seus filtros atuais
```

### 3.2 Config centralizada (do robo-daytrade-v1)

```python
# config.py
RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
BB_PERIOD = 20
BB_STD = 2
STOP_ATR_MULT = 2.0
TARGET_ATR_MULT = 3.0
```

### 3.3 Classe Trade (do Robot-Trader-IA)

- Stop/gain em pontos
- Verificação de high/low do candle
- Registro de entrada/saída/duração

### 3.4 Biblioteca `ta`

Substituir cálculos manuais por `ta` para padronizar e reduzir bugs:

```python
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
```

---

## 4. Próximos passos sugeridos

| Prioridade | Ação |
|------------|------|
| ~~1~~ | ~~Bandas de Bollinger~~ — Implementado em R6 (`config.BB_USE`, `BB_ENTRY`) |
| ~~2~~ | ~~config.py~~ — Implementado |
| ~~3~~ | ~~Lib ta~~ — Implementado em R6 |
| 4 | Estudar **Freqtrade** para padrão de estratégia e otimização |
| 5 | (Futuro) Explorar **RL** ou **AlphaEvolve** para otimização automática |

---

## 5. Links úteis

- [robo-daytrade-v1](https://github.com/Rony-s-barbosa/robo-daytrade-v1) — RSI/EMA/BB, lib `ta`
- [Robot-Trader-IA-actor-critic-2](https://github.com/MilianoJunior/Robot-Trader-IA-actor-critic-2) — WIN, RL, MetaTrader
- [Freqtrade](https://github.com/freqtrade/freqtrade) — Framework completo
- [ta - Technical Analysis](https://github.com/bukosabino/ta) — Indicadores Python
