"""
Executa benchmark completo estilo pwb + otimização.
Meta: >55% win rate, 11k em 60 dias (10k inicial + 1k lucro).
"""
import numpy as np
import sys
from pathlib import Path

# utilitários da Fase 1
FASE1_DIR = Path(__file__).resolve().parent / "fase1_antigravity"
sys.path.insert(0, str(FASE1_DIR))

from utils_metrics_pwb import run_benchmark, metrics, print_benchmark  # noqa: E402
from roboMamhedgeR6 import run_backtest as run_r6  # noqa: E402
from roboMamhedgeR7 import run_backtest as run_r7, optimize_params  # noqa: E402

CAPITAL = 10000
META_LUCRO = 1000
DIAS = 60

def main():
    print("=" * 70)
    print("BENCHMARK COMPLETO — Metodologia Papers With Backtest")
    print("=" * 70)

    # 1. Benchmark R6 (walk-forward)
    result_r6 = run_benchmark(run_r6)
    print_benchmark(result_r6, "R6")

    # 2. Otimização R7
    print("--- Otimização R7 ---")
    best, all_res = optimize_params()
    if best:
        print(f"Melhor: stop={best[0]}, target={best[1]}, rsi={best[2]}, macd={best[3]}")
        print(f"  → {best[4]} trades, {best[5]:.1f}% win, E[P&L]={best[6]:.2f}, Total={best[7]:.2f}")

    # 3. R7 com melhores params
    if best:
        trades_r7 = run_r7(
            stop_atr=best[0], target_atr=best[1],
            rsi_bullish=best[2], use_macd_filter=best[3]
        )
        m = metrics(trades_r7)
        print(f"\nR7 (otimizado): {m['n']} trades, {m['win_rate']*100:.1f}% win, Total R$ {m['total_pl']:.2f}")

    # 4. Projeção 60 dias
    print("\n" + "=" * 70)
    print("PROJEÇÃO 60 DIAS")
    print("=" * 70)
    for name, run_fn in [("R6", run_r6), ("R7 (best)", lambda: run_r7(stop_atr=2.0, target_atr=1.5, rsi_bullish=40, use_macd_filter=False))]:
        trades = run_fn()
        if len(trades) < 5:
            continue
        n = len(trades)
        total = trades.sum()
        # Dados atuais ~1 mês. 60 dias ≈ 2x
        proj_60d = total * (60 / 30) if n > 0 else 0
        wr = (trades > 0).mean() * 100
        print(f"{name}: {n} trades/mês | Win {wr:.1f}% | Total R$ {total:.2f} | Proj. 60d R$ {proj_60d:.2f}")
        if proj_60d >= META_LUCRO:
            print(f"  ✓ Meta 11k atingível!")
        else:
            mult = META_LUCRO / proj_60d if proj_60d > 0 else 0
            print(f"  → Para 11k: escalar posição ~{mult:.0f}x (ou mais dados/otimização)")

    print("\n--- Nota: P&L em pontos. WIN: 1 pt ≈ R$ 0,20/contrato. ---")

if __name__ == "__main__":
    main()
