# Robô de DayTrade - roboMT5

## O que é o projeto

Projeto de **robô de investimentos para DayTrade** que combina:
1. **Sistema de sinais** baseado em indicadores técnicos
2. **Simulação Monte Carlo** para avaliar risco de ruína e distribuição de resultados

O objetivo é testar estratégias de forma quantitativa antes de operar com dinheiro real.

---

## Escala e modelo de custo

- **Dados:** BOVA11 × 1000 (pontos, alinhado ao Ibovespa)
- **1 ponto = R$ 0,001 por cota** (`MULT_PONTOS_REAIS = 0.001`)
- **Posição:** 100 cotas por operação (`N_COTAS = 100`)
- **Custo:** R$ 2,50 fixo por round-trip (`CUSTO_REAIS = 2.50`)
- **P&L:** robôs retornam pontos puros; `pnl_reais(pts) = pts × 100 × 0.001 - 2.50`

---

## Resultados atuais (100 cotas, custo R$ 2,50)

| Robô | Trades | Win% | E[P&L]/trade | Total R$ | MC Saldo médio |
|------|--------|------|-------------|----------|----------------|
| **R9** | 81 | 35.8% | **R$ 13.97** | R$ 1.132 | **R$ 13.504** |
| **R6** | 132 | 31.8% | R$ 7.79 | R$ 1.028 | R$ 11.940 |
| **R8** | 135 | 35.6% | R$ 5.39 | R$ 728 | R$ 11.348 |
| R6orig | 204 | 36.3% | R$ 3.87 | R$ 789 | R$ 10.969 |
| Contrário | 83 | 24.1% | R$ 1.91 | R$ 159 | R$ 10.473 |
| R7 | 230 | 30.9% | R$ 0.77 | R$ 177 | R$ 10.193 |
| R2 | 76 | 50.0% | R$ 0.68 | R$ 52 | R$ 10.165 |
| R1 | 116 | 46.6% | R$ 0.53 | R$ 61 | R$ 10.128 |
| R5 | 30 | 43.3% | R$ -2.50 | R$ -75 | R$ 9.390 |
| R3 | 77 | 46.8% | R$ -1.91 | R$ -147 | R$ 9.532 |
| R6v2 | 152 | 31.6% | R$ -3.01 | R$ -458 | R$ 9.249 |
| R4 | 13 | 30.8% | R$ -10.19 | R$ -132 | R$ 7.450 |

---

## Evolução da família R6 (a ideia central)

### A ideia do Flavio: RSI Peak Detection

> "Quando o RSI faz um pico local (topo), o momentum exauriu. Sair ANTES que o preço caia."

Essa é a tese central: usar **peak detection no RSI** para antecipar reversões, em vez de esperar cruzamentos de média ou outros sinais atrasados.

### R6 Original (`roboMamhedgeR6 copy.py`) — A ideia pura

- **Entrada:** RSI > 40 nos últimos 5 candles ("janela bullish") + EMA4 subindo
- **Saída:** Pico de máximo no RSI (peak detection)
- **Sem stop loss, sem target, sem MACD, sem BB**
- Reversão de mão: se vendido e surge compra, vira direto
- 204 trades, 36.3% win, **R$ 3.87/trade**

### R6 v2 (`roboMamhedgeR6_v2.py`) — Ajustes conservadores (Cursor)

- Adicionou **MACD > 0** como confirmação de entrada
- Stop loss **1.5×ATR** e take profit **2.5×ATR**
- Prioridade: stop > peak RSI > target
- Mais filtros = menos trades e **piorou** (R$ -3.01/trade)
- Conclusão: o MACD como filtro obrigatório cortou bons trades

### R6 (`roboMamhedgeR6.py`) — Versão com config centralizada

- Usa `config.py` para parâmetros (RSI, EMA, MACD, BB, ATR)
- Bollinger Bands opcional (`BB_USE = False` por default)
- Stop loss **2×ATR** (sem target — sai por peak ou stop)
- MACD como filtro soft (aceita NaN)
- 132 trades, 31.8% win, **R$ 7.79/trade** — o melhor da família R6

### R9 (`roboMamhedgeR9.py`) — Versão otimizada (IA)

- Grid search em 2000+ combinações de parâmetros
- **Melhor config:** EMA6, RSI>40, janela 3, Stop 1.5×ATR, sem target, MACD+ADX
- Filtro **ADX > 20** (só opera em mercado com tendência)
- 81 trades, 35.8% win, **R$ 13.97/trade** — melhor E[P&L] do projeto
- Score otimizado por `E[P&L] × √(n_trades)` (balanceia retorno e consistência)

### Lição da evolução

| Versão | Filtros | Resultado |
|--------|---------|-----------|
| R6 Original | Poucos (RSI+EMA4) | Bom, muitos trades |
| R6 v2 | Muitos (MACD obrigatório) | Piorou — overfiltering |
| R6 config | Equilibrado (BB off, MACD soft) | Melhorou |
| R9 | Otimizado (ADX sim, MACD sim, target off) | Melhor de todos |

**Padrão:** a saída por peak RSI funciona melhor que take profit fixo. O ADX filtra mercado lateral. O MACD ajuda quando não é obrigatório demais.

---

## Estrutura do projeto

```
roboMT5/
├── download_win.py            # Download BOVA11 → pontos (×1000)
├── converter_csv_para_pontos.py # Converte CSVs antigos
├── utils_fuso.py              # BRT, horários, pnl_reais(), N_COTAS, CUSTO
├── config.py                  # Parâmetros centralizados R6
│
├── roboMamhedgeR0.py          # Sistema de sinais (EMA + ATR)
├── roboMamhedgeR1.py          # Backtest com stop/target
├── roboMamhedgeR2.py          # R1 + EMA200 + Momentum
├── roboMamhedgeR3.py          # EMAs 20/50
├── roboMamhedgeR4.py          # MACD + RSI, long-only
├── roboMamhedgeR5.py          # MACD + RSI, long/short
├── roboMamhedgeR6 copy.py     # R6 Original (Flavio) — a ideia pura
├── roboMamhedgeR6_v2.py       # R6 v2 (Cursor) — +MACD +Stop +TP
├── roboMamhedgeR6.py          # R6 com config centralizada
├── roboMamhedgeR7.py          # R6 + Take Profit
├── roboMamhedgeR8.py          # Híbrido R2+R6 (EMA50+Peak)
├── roboMamhedgeR9.py          # R9 — R6 otimizado (melhor do projeto)
├── roboContrario.py           # Inverso do R6
│
├── run_all_montecarlo.py      # Executa todos + Monte Carlo
├── benchmark_pwb.py           # Walk-forward 70/30
├── diagnostico_escala.py      # Verifica escala dos dados
├── investigar_escala.py       # Diagnóstico BOVA11 vs WIN
│
├── WIN_5min.csv               # Dados (BOVA11 em pontos)
├── montecarlo_comparativo.png # Gráfico Monte Carlo
├── RESUMO.md                  # Este arquivo
├── CHANGELOG.md               # Histórico de versões
├── FUNDAMENTOS_TRADING.md     # WIN, custos, expectativa
└── claude/                    # Diagnósticos e baseline do Claude
```

---

## Como usar

```bash
# 1. Baixar dados
python download_win.py

# 2. Rodar um robô específico
python roboMamhedgeR9.py   # R9 (melhor) — inclui otimização
python roboMamhedgeR6.py   # R6, R1, R2...

# 3. Rodar todos + Monte Carlo
python run_all_montecarlo.py

# 4. Benchmark walk-forward
python benchmark_pwb.py
```

---

## Próximos passos

1. **Mais dados:** conseguir 6+ meses de histórico para validar out-of-sample
2. **Walk-forward no R9:** confirmar que os parâmetros otimizados não são overfitting
3. **Position sizing:** ajustar N_COTAS dinamicamente (Kelly Criterion)
4. **Integração MT5:** operar em tempo real com a estratégia validada
5. **Custos reais:** calibrar CUSTO_REAIS com dados da corretora
