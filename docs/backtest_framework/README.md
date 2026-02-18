# Backtest Framework (DayTrade)

Framework modular para backtest profissional em Python com:

- Replay de ticks (`Tick` + `BacktestEngine`)
- Slippage configurável (fixo em ticks + proporcional)
- Custos de operação B3 configuráveis
- Métricas de performance (`Sharpe`, `Max Drawdown`, `Lucro Líquido`)

## Estrutura

- `models.py`: modelos de domínio (Tick, Signal, Fill, Trade, Result)
- `strategy.py`: interface de estratégia e estratégia exemplo
- `engine.py`: loop de replay e execução de ordens
- `slippage.py`: modelo de slippage
- `costs.py`: modelo de custos B3
- `metrics.py`: cálculo de métricas
- `report.py`: formatação do relatório
- `data.py`: loader CSV para ticks

## Execução rápida

```bash
python run_backtest_framework.py
```

## Formato CSV de ticks

```csv
timestamp,price,volume
2025-01-02T09:00:00,130000.0,1
2025-01-02T09:00:01,130005.0,2
```

Use `TickCSVLoader("arquivo.csv").stream()` para alimentar o `BacktestEngine`.

## Publicar no GitHub como versão paralela (sem afetar a principal)

1. Crie uma branch de validação:

```bash
git checkout -b feat/backtest-framework-validation
```

2. Suba a branch:

```bash
git push -u origin feat/backtest-framework-validation
```

3. Abra um Pull Request no GitHub com:
- **base**: `main` (ou branch principal)
- **compare**: `feat/backtest-framework-validation`

4. Marque como **Draft PR** até terminar a validação. Assim o código fica paralelo e não altera a principal até o merge.
