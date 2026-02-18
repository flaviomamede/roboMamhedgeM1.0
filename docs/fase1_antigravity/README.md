# Fase 1 (Antigravity) — Execução

Nesta fase, **os únicos scripts “executivos”** (para você rodar no terminal) são:

1) **Rodar 1 robô e ver desempenho**

- `python fase1_antigravity/exec_run_robot.py --robot R7`

2) **Rodar vários robôs e comparar**

- `python fase1_antigravity/exec_compare_robots.py`

3) **Otimizar parâmetros no treino (train) e avaliar no teste (test)**

- `python fase1_antigravity/exec_optimize_robot.py --robot R7`

### Critério unificado de otimização

O otimizador usa, para todos os robôs (R0, R6, R7, R8, R9, R10), a função-objetivo:

- `score = mu / variancia`

onde `mu` e `variancia` são calculados por bootstrap Monte Carlo dos trades no TRAIN.
Por padrão: **1000 simulações por conjunto de parâmetros**.

Para otimizar todos em sequência:

- `python fase1_antigravity/exec_optimize_robot.py --robot ALL --mc_sims 1000`

A busca usa heurística para reduzir risco de mínimo local:
- âncora atual + varredura por parâmetro
- exploração global aleatória
- refinamento coordenado nos melhores candidatos

No comparativo (`exec_compare_robots.py`) a tabela também mostra a coluna
`Kelly μ/σ²` (o mesmo critério da função-objetivo, via Monte Carlo).

Relatório final:

- `fase1_antigravity/otimize.md`

## Dados

- CSV principal: `fase1_antigravity/WIN_5min.csv`
- Split temporal 70/30:
  - `fase1_antigravity/WIN_train.csv`
  - `fase1_antigravity/WIN_test.csv`

## Observação

Os demais arquivos da pasta (diagnósticos, inspeções, conversores) foram movidos para `fase1_antigravity/dev_tools/`
e/ou deixados como “biblioteca” (não são para execução no fluxo normal).

