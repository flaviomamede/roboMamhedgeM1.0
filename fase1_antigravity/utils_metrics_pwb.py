"""
Benchmark estilo Papers With Backtest (pwb).
P&L dos robôs em pontos puros; conversão para R$ via pnl_reais().
- Walk-forward: treino 70% / teste 30% (evita overfitting)
- Métricas padronizadas: win rate, E[P&L], Sharpe, drawdown, payoff, fator de risco, ROI mensal
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from utils_fuso import pnl_reais, N_COTAS, CUSTO_REAIS

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = BASE_DIR / "WIN_5min.csv"


def walk_forward_split(df, train_pct=0.7):
    n = len(df)
    cut = int(n * train_pct)
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    return train, test


def run_benchmark(run_backtest_fn, csv_path=DEFAULT_CSV_PATH, train_pct=0.7):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()

    train_df, test_df = walk_forward_split(df, train_pct)
    train_start = train_df.index[0]
    train_end = train_df.index[-1]
    test_start = test_df.index[0]
    test_end = test_df.index[-1]

    train_path = BASE_DIR / "WIN_train.csv"
    test_path = BASE_DIR / "WIN_test.csv"
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)

    trades_train = run_backtest_fn(train_path)
    trades_test = run_backtest_fn(test_path)

    return {
        "train": {"trades": trades_train, "start": train_start, "end": train_end},
        "test": {"trades": trades_test, "start": test_start, "end": test_end},
    }


def metrics(trades_pts):
    """Métricas padronizadas. trades_pts = P&L em pontos puros.

    Observações:
    - Payoff = ganho médio / |perda média|
    - Fator de risco (risk_factor) = MaxDD / |Lucro|  (quanto menor, melhor)
    - ROI mensal usa o período do CSV via `metrics_from_csv(...)` (recomendado).
    """
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
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # negativo
    payoff = float(avg_win / abs(avg_loss)) if avg_win > 0 and avg_loss < 0 else 0.0

    risk_factor = float(max_dd / abs(total)) if abs(total) > 1e-12 else float("inf")

    # ROI (total/mensal) só faz sentido quando calculado com capital + janela temporal.
    return {
        "n": int(n),
        "win_rate": win_rate,
        "e_pl": e_pl,
        "total_pl": total,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "payoff": payoff,
        "risk_factor": risk_factor,
        "roi_total_pct": 0.0,
        "roi_mensal_pct": 0.0,
    }


def metrics_from_csv(trades_pts, csv_path: str | Path, capital_inicial: float = 10_000.0) -> dict:
    """Métricas + ROI mensal baseado no período do CSV."""
    m = metrics(trades_pts)
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        idx = pd.to_datetime(df.index)
        days = (idx.max() - idx.min()).days if len(idx) else 0
    except Exception:
        days = 0

    total = float(m["total_pl"])
    roi_total = (total / capital_inicial) if capital_inicial else 0.0
    roi_total_pct = roi_total * 100.0
    roi_mensal_pct = (roi_total * (30.0 / days) * 100.0) if days and days > 0 else 0.0

    m["roi_total_pct"] = float(roi_total_pct)
    m["roi_mensal_pct"] = float(roi_mensal_pct)
    return m


def print_benchmark(result, name="Robô"):
    print(f"\n{'='*60}")
    print(f"BENCHMARK {name} ({N_COTAS} cotas, custo R$ {CUSTO_REAIS:.2f}/trade)")
    print("=" * 60)
    for phase in ["train", "test"]:
        d = result[phase]
        m = metrics(d["trades"])
        print(f"\n--- {phase.upper()} ({d['start'].date()} a {d['end'].date()}) ---")
        print(f"  Trades: {m['n']} | Win: {m['win_rate']*100:.1f}% | "
              f"E[P&L]: R$ {m['e_pl']:.2f}/trade | Total: R$ {m['total_pl']:.2f}")
        print(f"  Sharpe: {m['sharpe']:.2f} | Max DD: R$ {m['max_dd']:.2f}")
    print()


if __name__ == "__main__":
    raise SystemExit(
        "Este arquivo é biblioteca da Fase 1.\n"
        "Use os scripts executivos:\n"
        "- python fase1_antigravity/exec_run_robot.py --robot R7\n"
        "- python fase1_antigravity/exec_compare_robots.py\n"
        "- python fase1_antigravity/exec_optimize_robot.py --robot R7\n"
    )
