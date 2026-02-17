from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from utils_metrics_pwb import metrics_from_csv


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
# Quando executamos `python fase1_antigravity/exec_*.py`, o sys.path[0] vira a pasta
# `fase1_antigravity/`. Precisamos incluir a raiz do repositório para achar `config.py`.
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CSV = BASE_DIR / "WIN_5min.csv"


ROBOT_MODULES = {
    "R0": "roboMamhedgeR0",
    "R1": "roboMamhedgeR1",
    "R2": "roboMamhedgeR2",
    "R3": "roboMamhedgeR3",
    "R4": "roboMamhedgeR4",
    "R5": "roboMamhedgeR5",
    "R6ORIG": "roboMamhedgeR6 copy",  # arquivo com espaço
    "R6V2": "roboMamhedgeR6_v2",
    "R7": "roboMamhedgeR7",
    "R8": "roboMamhedgeR8",
    "CONTRARIO": "roboContrario",
}


def _load_run_backtest(robot_key: str):
    import importlib.util

    key = robot_key.upper()
    if key not in ROBOT_MODULES:
        raise SystemExit(f"Robô inválido: {robot_key}. Opções: {', '.join(sorted(ROBOT_MODULES))}")

    module_name = ROBOT_MODULES[key]
    if module_name == "roboMamhedgeR6 copy":
        # carregar via path (nome de arquivo tem espaço)
        path = BASE_DIR / "roboMamhedgeR6 copy.py"
        spec = importlib.util.spec_from_file_location("roboMamhedgeR6_copy", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Falha ao carregar {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.run_backtest

    mod = __import__(module_name, fromlist=["run_backtest"])
    return getattr(mod, "run_backtest")


def main() -> None:
    p = argparse.ArgumentParser(description="Fase 1: rodar 1 robô e imprimir métricas")
    p.add_argument("--robot", required=True, help="Ex.: R7, R6orig, R6v2, R1, Contrario")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="Caminho do CSV (default: WIN_5min.csv da fase1)")
    args = p.parse_args()

    run_backtest = _load_run_backtest(args.robot)
    trades_pts = run_backtest(args.csv)
    trades_pts = np.array(trades_pts) if trades_pts is not None else np.array([])

    m = metrics_from_csv(trades_pts, args.csv)
    print("=" * 70)
    print(f"FASE 1 — ROBÔ {args.robot.upper()} — {Path(args.csv).name}")
    print("=" * 70)
    print(f"Trades: {m['n']} | Win: {m['win_rate']*100:.1f}% | E[P&L]: R$ {m['e_pl']:.2f}/trade | Total: R$ {m['total_pl']:.2f}")
    print(
        f"Sharpe: {m['sharpe']:.2f} | Max DD: R$ {m['max_dd']:.2f} | "
        f"Payoff: {m['payoff']:.2f} | RiskFactor: {m['risk_factor']:.2f} | ROI mensal: {m['roi_mensal_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()

