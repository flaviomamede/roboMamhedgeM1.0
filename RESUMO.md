# Robô de DayTrade - roboMT5

## O que é o projeto

Projeto de **robô de investimentos para DayTrade** que combina:
1. **Sistema de sinais** baseado em indicadores técnicos
2. **Simulação Monte Carlo** para avaliar risco de ruína e distribuição de resultados

O objetivo é testar estratégias de forma quantitativa antes de operar com dinheiro real.

---

## Escala e modelo de custo

- **Dados:** Mini-Índice (WIN) em pontos
- **1 ponto = R$ 0,20 por contrato** (`MULT_PONTOS_REAIS = 0.20`)
- **Posição:** 1 contrato por operação (`N_COTAS = 1`)
- **Custo:** R$ 2,50 fixo por round-trip (`CUSTO_REAIS = 2.50`)
- **P&L:** robôs retornam pontos puros; `pnl_reais(pts) = pts × 1 × 0.20 - 2.50`

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
roboMamhedgeM1.0/
├── backtest_framework/         # Fase 2: framework profissional (Codex)
├── tests/                      # Fase 2: testes automatizados
├── run_backtest_framework.py   # Exemplo de uso do framework
├── utils_fuso.py               # BRT, horários, pnl_reais(), N_COTAS, CUSTO (WIN)
├── roboMamhedgeR6.py           # Robô revisado (fase 2)
├── roboMamhedgeR9.py           # Robô revisado (fase 2)
├── roboMamhedgeR10.py          # Robô revisado (fase 2)
├── comparativo_r6_r9_r10.py    # Comparativo R6/R9/R10 (fase 2)
│
├── fase1_antigravity/          # Fase 1 (legado)
│   ├── WIN_5min.csv            # Dados de entrada (legado)
│   ├── WIN_train.csv           # Split treino (gerado no benchmark)
│   ├── WIN_test.csv            # Split teste (gerado no benchmark)
│   ├── montecarlo_comparativo.png  # Saída (legado)
│   ├── flyer_r9.png            # Saída (legado)
│   ├── *.py                    # Robôs e utilitários da fase 1
│   └── claude/                 # Materiais/diagnósticos do Claude (legado)
│
├── RESUMO.md                   # Este arquivo
├── CHANGELOG.md                # Histórico de versões
└── FUNDAMENTOS_TRADING.md      # Notas de trading/custos
```

---

## Como usar

```bash
# 1. Baixar dados
python fase1_antigravity/download_win.py

# 2. Rodar um robô específico
python roboMamhedgeR9.py   # R9 (melhor) — inclui otimização
python roboMamhedgeR6.py   # R6, R1, R2...

# 3. Rodar todos + Monte Carlo
python run_all_montecarlo.py

# 4. Benchmark walk-forward
python fase1_antigravity/benchmark_pwb.py
```

---

## Próximos passos

1. **Mais dados:** conseguir 6+ meses de histórico para validar out-of-sample
2. **Walk-forward no R9:** confirmar que os parâmetros otimizados não são overfitting
3. **Position sizing:** ajustar N_COTAS dinamicamente (Kelly Criterion)
4. **Integração MT5:** operar em tempo real com a estratégia validada
5. **Custos reais:** calibrar CUSTO_REAIS com dados da corretora
