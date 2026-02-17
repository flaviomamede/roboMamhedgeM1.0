# Fase 1 (Antigravity) — Execução

Nesta fase, **os únicos scripts “executivos”** (para você rodar no terminal) são:

1) **Rodar 1 robô e ver desempenho**

- `python fase1_antigravity/exec_run_robot.py --robot R7`

2) **Rodar vários robôs e comparar**

- `python fase1_antigravity/exec_compare_robots.py`

3) **Otimizar parâmetros no treino (train) e avaliar no teste (test)**

- `python fase1_antigravity/exec_optimize_robot.py --robot R7`

## Dados

- CSV principal: `fase1_antigravity/WIN_5min.csv`
- Split temporal 70/30:
  - `fase1_antigravity/WIN_train.csv`
  - `fase1_antigravity/WIN_test.csv`

## Observação

Os demais arquivos da pasta (diagnósticos, inspeções, conversores) foram movidos para `fase1_antigravity/dev_tools/`
e/ou deixados como “biblioteca” (não são para execução no fluxo normal).

