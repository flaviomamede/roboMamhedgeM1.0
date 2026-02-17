# Fases do projeto

Este repositório foi organizado em duas fases de evolução do projeto:

## Fase 1 — Antigravity

Código e utilitários desenvolvidos na etapa inicial (legado).

- Pasta: `fase1_antigravity/`
- Dados de entrada (legado): `fase1_antigravity/WIN_*.csv`
- Saídas/artefatos (legado): `fase1_antigravity/montecarlo_comparativo.png`, `fase1_antigravity/flyer_r9.png`
- Materiais do Claude (legado): `fase1_antigravity/claude/`

## Fase 2 — Codex (backtest framework profissional)

Implementação profissional do framework de backtest e testes automatizados, além das revisões dos robôs.

- Framework: `backtest_framework/`
- Testes: `tests/`
- Robôs revisados: `roboMamhedgeR6.py`, `roboMamhedgeR9.py`, `roboMamhedgeR10.py`
- Utilitário (escala WIN e horário): `utils_fuso.py`

### Execução rápida

- Rodar exemplo do framework:
  - `python run_backtest_framework.py`
- Rodar testes:
  - `python -m unittest discover -s tests -p "test_*.py" -q`

