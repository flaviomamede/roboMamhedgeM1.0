from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from utils_fuso import pnl_reais
from utils_metrics_pwb import metrics_from_csv


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CSV = BASE_DIR / "WIN_5min.csv"


def _trades_to_reais(trades_pts) -> np.ndarray:
    if trades_pts is None:
        return np.array([])
    arr = np.asarray(trades_pts).tolist()
    return np.array([pnl_reais(t) for t in arr], dtype=float)


def _mc_mu_var(trades_r: np.ndarray, n_sims: int = 1000, seed: int = 42) -> tuple[float, float]:
    if len(trades_r) == 0:
        return 0.0, float("inf")
    if len(trades_r) == 1:
        return float(trades_r[0]), 0.0
    rng = np.random.default_rng(seed)
    n = len(trades_r)
    idx = rng.integers(0, n, size=(n_sims, n))
    totals = trades_r[idx].sum(axis=1)
    return float(totals.mean()), float(totals.var(ddof=1))


def _objective(mu: float, var: float, eps: float = 1e-9) -> float:
    return float(mu / (var + eps))


def _run_all(csv_path: str, mc_sims: int, seed: int) -> list[tuple[str, dict]]:
    # Imports locais para evitar custo ao importar o módulo
    from roboMamhedgeR0 import run_backtest as run_r0
    from roboMamhedgeR1 import run_backtest as run_r1
    from roboMamhedgeR2 import run_backtest as run_r2
    from roboMamhedgeR3 import run_backtest as run_r3
    from roboMamhedgeR4 import run_backtest as run_r4
    from roboMamhedgeR5 import run_backtest as run_r5
    from roboMamhedgeR6_v2 import run_backtest as run_r6v2
    from roboMamhedgeR7 import run_backtest as run_r7
    from roboMamhedgeR8 import run_backtest as run_r8
    from roboMamhedgeR9 import run_backtest as run_r9
    from roboMamhedgeR10 import run_backtest as run_r10
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
        ("R0", run_r0),
        ("R1", run_r1),
        ("R2", run_r2),
        ("R3", run_r3),
        ("R4", run_r4),
        ("R5", run_r5),
        ("R6orig", run_r6orig),
        ("R6v2", run_r6v2),
        ("R7", run_r7),
        ("R8", run_r8),
        ("R9", run_r9),
        ("R10", run_r10),
        ("Contrario", run_contra),
    ]

    out: list[tuple[str, dict]] = []
    for i, (name, fn) in enumerate(robots, start=1):
        trades = fn(csv_path)
        trades = np.array(trades) if trades is not None else np.array([])
        m = metrics_from_csv(trades, csv_path)
        trades_r = _trades_to_reais(trades)
        mu_mc, var_mc = _mc_mu_var(trades_r, n_sims=mc_sims, seed=seed + i)
        m["kelly_obj"] = _objective(mu_mc, var_mc)
        out.append((name, m))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fase 1: comparar vários robôs (tabela)")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="Caminho do CSV (default: WIN_5min.csv da fase1)")
    p.add_argument(
        "--robots",
        default="R0,R1,R2,R3,R4,R5,R6orig,R6v2,R7,R8,R9,R10,Contrario",
        help="Lista separada por vírgula. Ex.: R8,R1 ou R6orig,R7,R8",
    )
    p.add_argument(
        "--mc_sims",
        type=int,
        default=1000,
        help="Simulações Monte Carlo por robô para calcular Kelly μ/σ²",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed base do Monte Carlo")
    args = p.parse_args()

    selected = [r.strip() for r in str(args.robots).split(",") if r.strip()]
    rows_all = _run_all(args.csv, mc_sims=int(args.mc_sims), seed=int(args.seed))
    rows = [row for row in rows_all if row[0].lower() in {s.lower() for s in selected}]
    if not rows:
        raise SystemExit("Nenhum robô selecionado. Use --robots com nomes válidos (ex.: R8,R1).")
    # ordenar por total_pl desc, depois e_pl desc
    rows.sort(key=lambda x: (x[1]["total_pl"], x[1]["e_pl"]), reverse=True)

    print("=" * 128)
    print(f"FASE 1 — COMPARATIVO — {Path(args.csv).name}")
    print("=" * 128)
    print(
        f"{'Robô':<10} | {'Trades':>6} | {'Win%':>6} | {'E[P&L] R$':>10} | {'Total R$':>10} | "
        f"{'Payoff':>6} | {'RiskF':>6} | {'ROI/m':>7} | {'Sharpe':>7} | {'Kelly μ/σ²':>12} | {'MaxDD R$':>9}"
    )
    print("-" * 128)
    for name, m in rows:
        print(
            f"{name:<10} | {m['n']:>6} | {m['win_rate']*100:>5.1f}% | {m['e_pl']:>10.2f} | {m['total_pl']:>10.2f} | "
            f"{m['payoff']:>6.2f} | {m['risk_factor']:>6.2f} | {m['roi_mensal_pct']:>6.2f}% | {m['sharpe']:>7.2f} | "
            f"{m['kelly_obj']:>12.6e} | {m['max_dd']:>9.2f}"
        )


if __name__ == "__main__":
    main()

