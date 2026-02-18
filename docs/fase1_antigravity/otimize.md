# Otimizacao Heuristica Unificada - Fase 1 + Fase 2

- Data/hora: 2026-02-17T21:59:50
- CSV base: `/home/flavio/Documentos/PESSOAL/PROJETO_IA_FINANCAS/roboMamhedgeM1.0/fase1_antigravity/WIN_5min.csv`
- Split temporal: 70% train / 30% test
- Simulacoes Monte Carlo por conjunto: 1000
- Funcao-objetivo: `score = mu / variancia`
  - `mu`: media dos totais simulados (R$)
  - `variancia`: variancia dos totais simulados (R$^2)
- Busca heuristica:
  - ancora atual + varredura 1D
  - amostragem aleatoria global (diversificacao)
  - refinamento coordenado multi-start (elites)

| Robo | Combos testados | Combos validos | Score train | Score test | mu train (R$) | var train | Params otimos |
|---|---:|---:|---:|---:|---:|---:|---|
| R0 | 243 | 243 | 0.00151535 | -0.00060722 | 4754.60 | 3137630.66 | `ema_fast=11 ema_slow=18 atr_period=10 atr_mean_period=7` |
| R6 | 361 | 361 | -0.00136151 | -0.00025514 | -427.63 | 314089.22 | `stop_atr=2.0 target_atr=3.5 rsi_thresh=55 rsi_window=2 use_macd_filter=True` |
| R7 | 197 | 197 | -0.00149992 | 0.00054325 | -896.69 | 597825.12 | `stop_atr=2.3 target_atr=4.0 rsi_bullish=45 use_macd_filter=False` |
| R8 | 467 | 467 | 0.00199265 | 0.00046536 | 1504.89 | 755216.91 | `ema_fast=9 ema_slow=21 ema_trend=100 momentum_lookback=30 stop_atr=1.5 rsi_period=14` |
| R9 | 614 | 614 | 0.00178317 | 0.00080605 | 243.15 | 136356.88 | `ema_fast=4 rsi_period=7 rsi_thresh=45 rsi_window=9 stop_atr=2.4 target_atr=2.0 use_macd=True use_adx=True adx_min=30 max_bars_in_trade=48` |
| R10 | 873 | 806 | 0.00342741 | 0.00228405 | 575.27 | 167843.41 | `ema_fast=10 ema_slow=34 rsi_period=14 rsi_thresh=50 rsi_window=3 stop_atr=2.0 trail_atr=2.2 breakeven_trigger_atr=2.2 use_adx=True adx_min=20 use_macd=True max_bars_in_trade=12` |

## Resultado detalhado

### R0
- Params otimos: `ema_fast=11 ema_slow=18 atr_period=10 atr_mean_period=7`
- Busca usada: global_samples=220 elite_size=6 refine_rounds=3
- Objective train: 0.00151535 | mu=4754.60 | var=3137630.66
- Objective test: -0.00060722 | mu=-2609.53 | var=4297515.11
- TRAIN metrics: trades=109 win=48.6% epl=R$ 42.33 total=R$ 4613.50 payoff=2.15 riskF=0.13 ROI/m=21.97%
- TEST metrics: trades=48 win=33.3% epl=R$ -56.50 total=R$ -2712.00 payoff=1.20 riskF=1.49 ROI/m=-32.54%

### R6
- Params otimos: `stop_atr=2.0 target_atr=3.5 rsi_thresh=55 rsi_window=2 use_macd_filter=True`
- Busca usada: global_samples=220 elite_size=6 refine_rounds=3
- Objective train: -0.00136151 | mu=-427.63 | var=314089.22
- Objective test: -0.00025514 | mu=-147.32 | var=577404.07
- TRAIN metrics: trades=300 win=35.0% epl=R$ -1.47 total=R$ -441.71 payoff=1.62 riskF=1.55 ROI/m=-2.10%
- TEST metrics: trades=124 win=39.5% epl=R$ -1.49 total=R$ -184.14 payoff=1.44 riskF=7.57 ROI/m=-2.21%

### R7
- Params otimos: `stop_atr=2.3 target_atr=4.0 rsi_bullish=45 use_macd_filter=False`
- Busca usada: global_samples=220 elite_size=6 refine_rounds=3
- Objective train: -0.00149992 | mu=-896.69 | var=597825.12
- Objective test: 0.00054325 | mu=648.31 | var=1193391.37
- TRAIN metrics: trades=531 win=30.3% epl=R$ -1.68 total=R$ -890.04 payoff=1.96 riskF=1.79 ROI/m=-4.24%
- TEST metrics: trades=211 win=33.6% epl=R$ 2.82 total=R$ 594.19 payoff=2.22 riskF=2.15 ROI/m=7.13%

### R8
- Params otimos: `ema_fast=9 ema_slow=21 ema_trend=100 momentum_lookback=30 stop_atr=1.5 rsi_period=14`
- Busca usada: global_samples=220 elite_size=6 refine_rounds=3
- Objective train: 0.00199265 | mu=1504.89 | var=755216.91
- Objective test: 0.00046536 | mu=516.97 | var=1110893.39
- TRAIN metrics: trades=344 win=36.0% epl=R$ 4.35 total=R$ 1496.65 payoff=2.47 riskF=0.35 ROI/m=7.13%
- TEST metrics: trades=143 win=40.6% epl=R$ 3.68 total=R$ 526.61 payoff=1.68 riskF=2.10 ROI/m=6.32%

### R9
- Params otimos: `ema_fast=4 rsi_period=7 rsi_thresh=45 rsi_window=9 stop_atr=2.4 target_atr=2.0 use_macd=True use_adx=True adx_min=30 max_bars_in_trade=48`
- Busca usada: global_samples=220 elite_size=6 refine_rounds=3
- Objective train: 0.00178317 | mu=243.15 | var=136356.88
- Objective test: 0.00080605 | mu=360.98 | var=447845.32
- TRAIN metrics: trades=133 win=33.1% epl=R$ 1.59 total=R$ 211.45 payoff=2.31 riskF=1.88 ROI/m=1.01%
- TEST metrics: trades=68 win=41.2% epl=R$ 5.08 total=R$ 345.38 payoff=1.71 riskF=1.68 ROI/m=4.14%

### R10
- Params otimos: `ema_fast=10 ema_slow=34 rsi_period=14 rsi_thresh=50 rsi_window=3 stop_atr=2.0 trail_atr=2.2 breakeven_trigger_atr=2.2 use_adx=True adx_min=20 use_macd=True max_bars_in_trade=12`
- Busca usada: global_samples=220 elite_size=6 refine_rounds=3
- Objective train: 0.00342741 | mu=575.27 | var=167843.41
- Objective test: 0.00228405 | mu=1678.17 | var=734734.95
- TRAIN metrics: trades=71 win=57.7% epl=R$ 7.78 total=R$ 552.51 payoff=1.09 riskF=0.56 ROI/m=2.63%
- TEST metrics: trades=34 win=47.1% epl=R$ 49.29 total=R$ 1675.73 payoff=2.81 riskF=0.20 ROI/m=20.11%

