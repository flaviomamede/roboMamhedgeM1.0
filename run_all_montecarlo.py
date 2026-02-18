"""
Executa cada robô (R1-R10 + Contrário) e Monte Carlo com métricas.
P&L dos robôs é em pontos puros; conversão para R$ via pnl_reais().
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
FASE1_DIR = BASE_DIR / "fase1_antigravity"
OUTPUT_PNG = FASE1_DIR / "montecarlo_comparativo.png"

# A Fase 1 foi movida para `fase1_antigravity/`. Colocamos este diretório no PYTHONPATH
# para que imports como `roboMamhedgeR1` continuem funcionando.
sys.path.insert(0, str(FASE1_DIR))

from roboMamhedgeR1 import run_backtest as run_r1
from roboMamhedgeR2 import run_backtest as run_r2
from roboMamhedgeR3 import run_backtest as run_r3
from roboMamhedgeR4 import run_backtest as run_r4
from roboMamhedgeR5 import run_backtest as run_r5
from roboMamhedgeR6 import run_backtest as run_r6
from roboMamhedgeR7 import run_backtest as run_r7
from roboMamhedgeR8 import run_backtest as run_r8
from roboContrario import run_backtest as run_contrario
from utils_fuso import pnl_reais, N_COTAS, CUSTO_REAIS

# Importa as 3 versões do R6 e o R9
from roboMamhedgeR6_v2 import run_backtest as run_r6v2
from roboMamhedgeR9 import run_backtest as run_r9
from roboMamhedgeR10 import run_backtest as run_r10

# `roboMamhedgeR6 copy.py` tem espaço no nome; carregamos via path.
_R6_COPY_PATH = FASE1_DIR / "roboMamhedgeR6 copy.py"
_spec = importlib.util.spec_from_file_location("roboMamhedgeR6_copy", _R6_COPY_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Falha ao carregar {_R6_COPY_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
r6_orig = _mod

ROBOTS = [
    ("R1", run_r1, "EMAs 9/21"),
    ("R2", run_r2, "EMAs 9/21 + EMA200 + Momentum"),
    ("R3", run_r3, "EMAs 20/50 + EMA200 + Momentum"),
    ("R4", run_r4, "MACD+RSI long-only"),
    ("R5", run_r5, "MACD+RSI long/short"),
    ("R6orig", r6_orig.run_backtest, "R6 Original (Flavio): Peak RSI puro"),
    ("R6v2", run_r6v2, "R6 v2 (Cursor): +MACD +Stop +TP"),
    ("R6", run_r6, "R6 (config): +BB +Stop 2×ATR"),
    ("R7", run_r7, "R7 defaults (otimizados)"),
    ("R8", run_r8, "R2+R6 (EMA50+Momentum+Peak)"),
    ("R9", run_r9, "R9 defaults (otimizados)"),
    ("R10", run_r10, "R10 defaults (otimizados)"),
    ("Contrário", run_contrario, "Inverso R6, stop 0.8 / target 3"),
]

CAPITAL = 10000
N_SIM = 10000
N_TRADES_SIM = 250
MARGEM = 1000

def monte_carlo(p_win, avg_gain_r, avg_loss_r):
    """Monte Carlo com P&L já em R$."""
    if np.isnan(avg_gain_r) and np.isnan(avg_loss_r):
        return None, None
    if np.isnan(avg_gain_r):
        avg_gain_r = 0
    if np.isnan(avg_loss_r):
        avg_loss_r = 0
    balances = []
    ruined = 0
    for _ in range(N_SIM):
        bal = CAPITAL
        for _ in range(N_TRADES_SIM):
            bal += avg_gain_r if np.random.rand() < p_win else avg_loss_r
            if bal <= MARGEM:
                ruined += 1
                break
        balances.append(bal)
    return np.array(balances), ruined / N_SIM

def main():
    print("=" * 70)
    print(f"ROBÔS R1-R10 + MONTE CARLO ({N_COTAS} contratos, custo R$ {CUSTO_REAIS:.2f}/trade)")
    print("=" * 70)

    results = []
    for name, run_fn, desc in ROBOTS:
        trades_pts = run_fn()
        if len(trades_pts) == 0:
            print(f"\n--- {name} ({desc}) ---")
            print("  Nenhum trade. Monte Carlo não aplicável.")
            results.append((name, desc, 0, np.nan, np.nan, np.nan, None, None))
            continue

        # Converter para R$
        trades_r = np.array([pnl_reais(t) for t in trades_pts])
        p_win = (trades_r > 0).sum() / len(trades_r)
        avg_gain_r = trades_r[trades_r > 0].mean() if (trades_r > 0).any() else 0
        avg_loss_r = trades_r[trades_r <= 0].mean() if (trades_r <= 0).any() else 0

        print(f"\n--- {name} ({desc}) ---")
        print(f"  Trades: {len(trades_r)} | Win: {p_win*100:.1f}% | E[P&L]: R$ {trades_r.mean():.2f}/trade")

        balances, prob_ruin = monte_carlo(p_win, avg_gain_r, avg_loss_r)
        if balances is not None:
            print(f"  Monte Carlo: Prob. Ruína {prob_ruin*100:.1f}% | Saldo médio: R$ {balances.mean():.2f}")
            results.append((name, desc, len(trades_r), p_win, avg_gain_r, avg_loss_r, balances, prob_ruin))
        else:
            results.append((name, desc, len(trades_r), p_win, avg_gain_r, avg_loss_r, None, None))

    # Gráfico comparativo
    n_robots = len(results)
    n_cols = 3
    n_rows = (n_robots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.atleast_2d(axes)
    axes = axes.flatten()
    for idx, (name, desc, n, p, g, l, balances, prob) in enumerate(results):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break
        if balances is not None and len(balances) > 0:
            ax.hist(balances, bins=50, color='skyblue', edgecolor='black')
            ax.axvline(CAPITAL, color='green', linestyle='--', label='Inicial')
            ax.axvline(MARGEM, color='red', linestyle='--', label='Margem')
        ax.set_title(f"{name}: {desc[:20]}...")
        ax.set_xlabel("Saldo Final (R$)")
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=100)
    plt.close()
    print(f"\n--- Gráfico salvo em {OUTPUT_PNG} ---")

if __name__ == "__main__":
    main()
