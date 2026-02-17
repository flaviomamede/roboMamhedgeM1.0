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


def metrics_phase1_from_csv(trades_pts, csv_path: str | Path, capital_inicial: float = 10_000.0) -> dict:
    """Métricas padronizadas (em R$) + ROI mensal baseado no período do CSV."""
    if trades_pts is None:
        iterable = []
    else:
        iterable = np.asarray(trades_pts).tolist()
    trades_r = np.array([pnl_reais(t) for t in iterable], dtype=float)
    if len(trades_r) == 0:
        return {
            "n": 0,
            "win_rate": 0.0,
            "e_pl": 0.0,
            "total_pl": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "payoff": 0.0,
            "risk_factor": float("inf"),
            "roi_total_pct": 0.0,
            "roi_mensal_pct": 0.0,
        }

    wins = trades_r[trades_r > 0]
    losses = trades_r[trades_r < 0]
    n = len(trades_r)
    win_rate = float((trades_r > 0).mean())
    e_pl = float(trades_r.mean())
    total = float(trades_r.sum())

    std = float(trades_r.std(ddof=1)) if n > 1 else 0.0
    sharpe = float((e_pl / std * np.sqrt(252)) if std > 0 else 0.0)

    cum = np.cumsum(trades_r)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff = float(avg_win / abs(avg_loss)) if avg_win > 0 and avg_loss < 0 else 0.0

    risk_factor = float(max_dd / abs(total)) if abs(total) > 1e-12 else float("inf")

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        idx = pd.to_datetime(df.index)
        days = (idx.max() - idx.min()).days if len(idx) else 0
    except Exception:
        days = 0

    roi_total = (total / capital_inicial) if capital_inicial else 0.0
    roi_total_pct = roi_total * 100.0
    roi_mensal_pct = (roi_total * (30.0 / days) * 100.0) if days and days > 0 else 0.0

    return {
        "n": int(n),
        "win_rate": win_rate,
        "e_pl": e_pl,
        "total_pl": total,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "payoff": payoff,
        "risk_factor": risk_factor,
        "roi_total_pct": float(roi_total_pct),
        "roi_mensal_pct": float(roi_mensal_pct),
    }


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
    return metrics_phase1_from_csv(trades_pts, csv_path)


def _optimize_r0(train_csv: str):
    from roboMamhedgeR0 import optimize_params

    best = optimize_params(train_csv)
    if best is None:
        raise SystemExit("Otimização R0 não encontrou resultados (poucos trades?).")
    return best


def _evaluate_r0(csv_path: str, best) -> dict:
    from roboMamhedgeR0 import run_backtest

    ema_fast, ema_slow, atr_period, atr_mean_period = best[0], best[1], best[2], best[3]
    trades_pts = run_backtest(
        csv_path=csv_path,
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        atr_period=int(atr_period),
        atr_mean_period=int(atr_mean_period),
    )
    return metrics_phase1_from_csv(trades_pts, csv_path)


def _optimize_r6(train_csv: str):
    # Fase 1: R6 = R6_v2
    from roboMamhedgeR6_v2 import optimize_params

    best = optimize_params(train_csv)
    if best is None:
        raise SystemExit("Otimização R6 (R6_v2) não encontrou resultados (poucos trades?).")
    return best


def _evaluate_r6(csv_path: str, best) -> dict:
    from roboMamhedgeR6_v2 import run_backtest

    stop, target, rsi, win, use_macd = best[0], best[1], best[2], best[3], best[4]
    trades_pts = run_backtest(
        csv_path=csv_path,
        stop_atr=float(stop),
        target_atr=float(target),
        rsi_thresh=float(rsi),
        rsi_window=int(win),
        use_macd_filter=bool(use_macd),
    )
    return metrics_phase1_from_csv(trades_pts, csv_path)


def _optimize_r8(train_csv: str):
    from roboMamhedgeR8 import optimize_params

    best = optimize_params(train_csv)
    if best is None:
        raise SystemExit("Otimização R8 não encontrou resultados (poucos trades?).")
    return best


def _evaluate_r8(csv_path: str, best) -> dict:
    from roboMamhedgeR8 import run_backtest

    ema_fast, ema_slow, ema_trend, mom, stop = best[0], best[1], best[2], best[3], best[4]
    trades_pts = run_backtest(
        csv_path=csv_path,
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        ema_trend=int(ema_trend),
        momentum_lookback=int(mom),
        stop_atr=float(stop),
    )
    return metrics_phase1_from_csv(trades_pts, csv_path)

def main() -> None:
    p = argparse.ArgumentParser(description="Fase 1: otimizar no TRAIN e avaliar no TEST (temporal 70/30)")
    p.add_argument("--robot", default="R7", help="R0, R6, R7 ou R8")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV base para split (default: WIN_5min.csv da fase1)")
    p.add_argument("--train_pct", type=float, default=0.7, help="Percentual para treino (default 0.7)")
    args = p.parse_args()

    robot = args.robot.upper()

    base_csv = Path(args.csv)
    train_csv, test_csv = _ensure_train_test(base_csv, train_pct=float(args.train_pct))
    if robot == "R0":
        best = _optimize_r0(str(train_csv))
        train_m = _evaluate_r0(str(train_csv), best)
        test_m = _evaluate_r0(str(test_csv), best)
        defaults_hint = f"ema_fast={best[0]} ema_slow={best[1]} atr_period={best[2]} atr_mean_period={best[3]}"
    elif robot == "R6":
        best = _optimize_r6(str(train_csv))
        train_m = _evaluate_r6(str(train_csv), best)
        test_m = _evaluate_r6(str(test_csv), best)
        defaults_hint = f"stop_atr={best[0]} target_atr={best[1]} rsi_thresh={best[2]} rsi_window={best[3]} use_macd_filter={best[4]}"
    elif robot == "R7":
        best = _optimize_r7(str(train_csv))
        train_m = _evaluate_r7(str(train_csv), best)
        test_m = _evaluate_r7(str(test_csv), best)
        defaults_hint = f"stop_atr={best[0]} target_atr={best[1]} rsi_bullish={best[2]} use_macd_filter={best[3]}"
    elif robot == "R8":
        best = _optimize_r8(str(train_csv))
        train_m = _evaluate_r8(str(train_csv), best)
        test_m = _evaluate_r8(str(test_csv), best)
        defaults_hint = f"ema_fast={best[0]} ema_slow={best[1]} ema_trend={best[2]} momentum_lookback={best[3]} stop_atr={best[4]}"
    else:
        raise SystemExit("Robô inválido. Use R0, R6, R7 ou R8.")

    print("=" * 90)
    print(f"FASE 1 — OTIMIZAÇÃO (TRAIN) + AVALIAÇÃO (TEST) — {robot}")
    print("=" * 90)
    print(f"Base:  {base_csv.name}")
    print(f"Train: {train_csv.name} | Test: {test_csv.name}")
    print("-" * 90)
    print(f"Melhor params: {defaults_hint}")
    print("-" * 90)
    print(f"TRAIN -> Trades {train_m['n']} | Win {train_m['win_rate']*100:.1f}% | E[P&L] R$ {train_m['e_pl']:.2f} | Total R$ {train_m['total_pl']:.2f}")
    print(f"TEST  -> Trades {test_m['n']} | Win {test_m['win_rate']*100:.1f}% | E[P&L] R$ {test_m['e_pl']:.2f} | Total R$ {test_m['total_pl']:.2f}")


if __name__ == "__main__":
    main()

