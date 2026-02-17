"""
Benchmark estilo Papers With Backtest (pwb).
P&L dos robôs em pontos puros; conversão para R$ via pnl_reais().
- Walk-forward: treino 70% / teste 30% (evita overfitting)
- Métricas padronizadas: win rate, E[P&L], Sharpe, drawdown
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
    """Métricas padronizadas. trades_pts = P&L em pontos puros."""
    if trades_pts is None or len(trades_pts) == 0:
        return {"n": 0, "win_rate": 0, "e_pl": 0, "total_pl": 0, "sharpe": 0, "max_dd": 0}
    # Converter para R$
    trades_r = np.array([pnl_reais(t) for t in trades_pts])
    wins = trades_r[trades_r > 0]
    n = len(trades_r)
    wr = len(wins) / n if n > 0 else 0
    e_pl = trades_r.mean()
    total = trades_r.sum()
    sharpe = (trades_r.mean() / trades_r.std() * np.sqrt(252)) if trades_r.std() > 0 else 0
    cum = np.cumsum(trades_r)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) > 0 else 0
    return {
        "n": n,
        "win_rate": wr,
        "e_pl": e_pl,
        "total_pl": total,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


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
    from roboMamhedgeR6 import run_backtest

    result = run_benchmark(run_backtest)
    print_benchmark(result, "R6")
