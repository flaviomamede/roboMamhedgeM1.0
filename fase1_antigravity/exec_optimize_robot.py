from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CSV = BASE_DIR / "WIN_5min.csv"
TRAIN_CSV = BASE_DIR / "WIN_train.csv"
TEST_CSV = BASE_DIR / "WIN_test.csv"


# --------------------------------------------------------------------
# Conteúdo “embutido” do benchmark_pwb.py (Fase 1)
# --------------------------------------------------------------------
from utils_fuso import pnl_reais


def metrics_phase1(trades_pts):
    """Métricas padronizadas. trades_pts = P&L em pontos puros."""
    if trades_pts is None or len(trades_pts) == 0:
        return {"n": 0, "win_rate": 0, "e_pl": 0, "total_pl": 0, "sharpe": 0, "max_dd": 0}
    trades_r = np.array([pnl_reais(t) for t in trades_pts], dtype=float)
    wins = trades_r[trades_r > 0]
    n = len(trades_r)
    wr = len(wins) / n if n > 0 else 0
    e_pl = float(trades_r.mean())
    total = float(trades_r.sum())
    std = float(trades_r.std(ddof=1)) if n > 1 else 0.0
    sharpe = (e_pl / std * np.sqrt(252)) if std > 0 else 0.0
    cum = np.cumsum(trades_r)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    return {"n": int(n), "win_rate": float(wr), "e_pl": e_pl, "total_pl": total, "sharpe": float(sharpe), "max_dd": max_dd}


def _ensure_train_test(csv_path: Path, train_pct: float = 0.7) -> tuple[Path, Path]:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()
    cut = int(len(df) * train_pct)
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    train.to_csv(TRAIN_CSV)
    test.to_csv(TEST_CSV)
    return TRAIN_CSV, TEST_CSV


def _optimize_r7(train_csv: str):
    from roboMamhedgeR7 import optimize_params

    best, results = optimize_params(train_csv)
    if best is None:
        raise SystemExit("Otimização não encontrou resultados (poucos trades?).")
    # best tuple: (stop, target, rsi, use_macd, n, wr%, e_pl, total)
    return best


def _evaluate_r7(csv_path: str, best) -> dict:
    from roboMamhedgeR7 import run_backtest

    stop, target, rsi, use_macd = best[0], best[1], best[2], best[3]
    trades_pts = run_backtest(csv_path, stop_atr=stop, target_atr=target, rsi_bullish=rsi, use_macd_filter=use_macd)
    trades_pts = np.array(trades_pts) if trades_pts is not None else np.array([])
    return metrics_phase1(trades_pts)


def main() -> None:
    p = argparse.ArgumentParser(description="Fase 1: otimizar no TRAIN e avaliar no TEST (temporal 70/30)")
    p.add_argument("--robot", default="R7", help="Por enquanto só R7 (tem optimize_params na fase1)")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV base para split (default: WIN_5min.csv da fase1)")
    p.add_argument("--train_pct", type=float, default=0.7, help="Percentual para treino (default 0.7)")
    args = p.parse_args()

    robot = args.robot.upper()
    if robot != "R7":
        raise SystemExit("Nesta limpeza inicial, a otimização executiva foi padronizada para o R7.")

    base_csv = Path(args.csv)
    train_csv, test_csv = _ensure_train_test(base_csv, train_pct=float(args.train_pct))
    best = _optimize_r7(str(train_csv))

    train_m = _evaluate_r7(str(train_csv), best)
    test_m = _evaluate_r7(str(test_csv), best)

    print("=" * 90)
    print("FASE 1 — OTIMIZAÇÃO (TRAIN) + AVALIAÇÃO (TEST) — R7")
    print("=" * 90)
    print(f"Base:  {base_csv.name}")
    print(f"Train: {train_csv.name} | Test: {test_csv.name}")
    print("-" * 90)
    print(f"Melhor params: stop={best[0]} target={best[1]} rsi={best[2]} macd={best[3]}")
    print("-" * 90)
    print(f"TRAIN -> Trades {train_m['n']} | Win {train_m['win_rate']*100:.1f}% | E[P&L] R$ {train_m['e_pl']:.2f} | Total R$ {train_m['total_pl']:.2f}")
    print(f"TEST  -> Trades {test_m['n']} | Win {test_m['win_rate']*100:.1f}% | E[P&L] R$ {test_m['e_pl']:.2f} | Total R$ {test_m['total_pl']:.2f}")


if __name__ == "__main__":
    main()

