# Fase 2 - Guia Rapido

## Visao geral

A Fase 2 introduz um paradigma de backtest orientado a componentes, com separacao entre:
- dados de mercado (`Tick`)
- geracao de sinal (`Strategy`)
- execucao (`BacktestEngine`)
- friccoes de mercado (`SlippageModel` + `B3CostModel`)
- avaliacao (`compute_metrics`)

Objetivo: sair do backtest "monolitico por robo" para um pipeline reutilizavel, testavel e com custos mais realistas por trade.

## Classificacao dos arquivos Python

### Nucleo do framework (`backtest_framework/`)
- `models.py`: entidades centrais (`Tick`, `Signal`, `Fill`, `Trade`, `BacktestResult`).
- `strategy.py`: interface de estrategia e implementacao exemplo.
- `engine.py`: simulacao de ordens, fechamento de trades e curva de equity.
- `slippage.py`: impacto de execucao no preco de fill.
- `costs.py`: modelo de custos B3 (taxas + emolumentos + corretagem/ISS).
- `metrics.py`: Sharpe, drawdown, win rate, profit factor e consolidacao de performance.
- `report.py`: formatacao de relatorio textual.
- `data.py`: carregamento de ticks por CSV.

### Integracao da Fase 2 na raiz do projeto
- `roboMamhedgeR6.py`, `roboMamhedgeR9.py`, `roboMamhedgeR10.py`: estrategias com pipeline de custos mais realista.
- `market_time.py`: utilitarios de fuso BRT e janela operacional.
- `b3_costs_phase2.py`: ponte para calcular P&L liquido por trade com custo B3.

### Scripts organizados por fase (`scripts/`)
- `scripts/phase2/run_backtest_framework.py`: exemplo executavel fim-a-fim do framework.
- `scripts/phase2/comparativo_r6_r9_r10.py`: comparativo entre robos da Fase 2.
- `scripts/phase2/backtest_r9_flyer.py`: backtest detalhado com flyer do R9.
- `scripts/phase1/run_all_montecarlo.py`: Monte Carlo consolidado (fase 1 e comparativos).
- `scripts/phase1/run_benchmark_completo.py`: benchmark legado da Fase 1.

Arquivos na raiz com os mesmos nomes foram mantidos como wrappers para compatibilidade.

### Testes
- `tests/test_backtest_framework.py`: cobre execucao basica do engine e calculo de metricas.

## Validacao rapida

Comandos executados:
- `python -m unittest discover -s tests -p "test_*.py"` -> OK (2 testes)
- `python run_backtest_framework.py` -> OK (gera relatorio de performance)

## Diferenciais da Fase 2

- Modularidade: troca de estrategia sem reescrever engine/custos.
- Testabilidade: componentes pequenos e cobertura unit.
- Realismo operacional: slippage e custos por ordem/trade.
- Escalabilidade: base pronta para walk-forward, otimizacao robusta e multiplos ativos.
