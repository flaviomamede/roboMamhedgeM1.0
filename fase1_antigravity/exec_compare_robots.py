from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from utils_metrics_pwb import metrics_from_csv


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CSV = BASE_DIR / "WIN_5min.csv"


def _run_all(csv_path: str) -> list[tuple[str, dict]]:
    # Imports locais para evitar custo ao importar o módulo
    from roboMamhedgeR1 import run_backtest as run_r1
    from roboMamhedgeR2 import run_backtest as run_r2
    from roboMamhedgeR3 import run_backtest as run_r3
    from roboMamhedgeR4 import run_backtest as run_r4
    from roboMamhedgeR5 import run_backtest as run_r5
    from roboMamhedgeR6_v2 import run_backtest as run_r6v2
    from roboMamhedgeR7 import run_backtest as run_r7
    from roboMamhedgeR8 import run_backtest as run_r8
    from roboContrario import run_backtest as run_contra

    import importlib.util

    # R6 Original tem espaço no nome
    r6_copy_path = BASE_DIR / "roboMamhedgeR6 copy.py"
    spec = importlib.util.spec_from_file_location("roboMamhedgeR6_copy", r6_copy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Falha ao carregar {r6_copy_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_r6orig = mod.run_backtest

    robots = [
        ("R1", run_r1),
        ("R2", run_r2),
        ("R3", run_r3),
        ("R4", run_r4),
        ("R5", run_r5),
        ("R6orig", run_r6orig),
        ("R6v2", run_r6v2),
        ("R7", run_r7),
        ("R8", run_r8),
        ("Contrario", run_contra),
    ]

    out: list[tuple[str, dict]] = []
    for name, fn in robots:
        trades = fn(csv_path)
        trades = np.array(trades) if trades is not None else np.array([])
        out.append((name, metrics_from_csv(trades, csv_path)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fase 1: comparar vários robôs (tabela)")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="Caminho do CSV (default: WIN_5min.csv da fase1)")
    args = p.parse_args()

    rows = _run_all(args.csv)
    # ordenar por total_pl desc, depois e_pl desc
    rows.sort(key=lambda x: (x[1]["total_pl"], x[1]["e_pl"]), reverse=True)

    print("=" * 100)
    print(f"FASE 1 — COMPARATIVO — {Path(args.csv).name}")
    print("=" * 100)
    print(
        f"{'Robô':<10} | {'Trades':>6} | {'Win%':>6} | {'E[P&L] R$':>10} | {'Total R$':>10} | "
        f"{'Payoff':>6} | {'RiskF':>6} | {'ROI/m':>7} | {'Sharpe':>7} | {'MaxDD R$':>9}"
    )
    print("-" * 100)
    for name, m in rows:
        print(
            f"{name:<10} | {m['n']:>6} | {m['win_rate']*100:>5.1f}% | {m['e_pl']:>10.2f} | {m['total_pl']:>10.2f} | "
            f"{m['payoff']:>6.2f} | {m['risk_factor']:>6.2f} | {m['roi_mensal_pct']:>6.2f}% | {m['sharpe']:>7.2f} | {m['max_dd']:>9.2f}"
        )


if __name__ == "__main__":
    main()

