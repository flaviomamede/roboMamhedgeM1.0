"""
Flyer de desempenho do R11 (Omni-Regime: Trend-Following + Mean Reversion via ER).
Consome roboMamhedgeR11.run_backtest_trades() diretamente com with_timestamps=True.
Saída: reports/phase2/flyer_r11.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from b3_costs_phase2 import default_b3_cost_model, trade_costs_brl, trade_net_pnl_brl
from market_time import converter_para_brt
from roboMamhedgeR11 import run_backtest_trades

INITIAL_CAPITAL = 10_000.0
CDI_ANNUAL = 0.12
DEFAULT_QUANTITY = 1
_COST_MODEL = default_b3_cost_model()

DEFAULT_CSV_PATH = os.path.join(REPO_ROOT, "fase1_antigravity", "WIN_5min.csv")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "reports", "phase2")
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "flyer_r11.png")


def run_detailed_backtest(csv_path=DEFAULT_CSV_PATH):
    print(f"Carregando dados de {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)
    df.sort_index(inplace=True)

    print("Simulando trades com roboMamhedgeR11.run_backtest_trades()...")
    trades_with_ts = run_backtest_trades(
        csv_path=csv_path, quantity=DEFAULT_QUANTITY, with_timestamps=True
    )

    trades_rows = []
    for idx, item in enumerate(trades_with_ts, start=1):
        t = item["trade"]
        trades_rows.append(
            {
                "trade_id": idx,
                "entry_time": item["entry_time"],
                "exit_time": item["exit_time"],
                "entry_price": float(t.entry_price_points),
                "exit_price": float(t.exit_price_points),
                "points": float(t.exit_price_points - t.entry_price_points),
                "pnl_brl": float(trade_net_pnl_brl(t, _COST_MODEL)),
                "costs_brl": float(trade_costs_brl(t, _COST_MODEL)),
            }
        )

    equity = [INITIAL_CAPITAL]
    for row in trades_rows:
        equity.append(equity[-1] + row["pnl_brl"])
    return df, trades_rows, equity


def calculate_metrics(trades, equity_curve):
    if not trades:
        return {}
    df_trades = pd.DataFrame(trades)
    total_net_profit = df_trades["pnl_brl"].sum()
    final_equity = equity_curve[-1]
    roi_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wins = df_trades[df_trades["pnl_brl"] > 0]
    losses = df_trades[df_trades["pnl_brl"] <= 0]
    gross_profit = wins["pnl_brl"].sum()
    gross_loss = abs(losses["pnl_brl"].sum())
    eq_series = pd.Series(equity_curve)
    drawdown = (eq_series - eq_series.cummax()) / eq_series.cummax() * 100
    return {
        "Net Profit": total_net_profit,
        "ROI (%)": roi_pct,
        "Trades": len(df_trades),
        "Win Rate (%)": (len(wins) / len(df_trades)) * 100,
        "Profit Factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "Max Drawdown (%)": drawdown.min(),
    }


def generate_flyer(df, trades, equity, metrics, filename=DEFAULT_OUTPUT_FILE):
    print("Gerando gráfico flyer R11...")
    df_trades = pd.DataFrame(trades)
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["close"], color="gray", alpha=0.5, linewidth=1, label="WIN Index")
    if not df_trades.empty:
        wins = df_trades[df_trades["pnl_brl"] > 0]
        losses = df_trades[df_trades["pnl_brl"] <= 0]
        ax1.scatter(
            wins["exit_time"], wins["exit_price"],
            color="green", marker="o", s=60, edgecolors="black", zorder=5, label="Win",
        )
        ax1.scatter(
            losses["exit_time"], losses["exit_price"],
            color="red", marker="o", s=60, edgecolors="black", zorder=5, label="Loss",
        )
        equity_times = [df.index[0]] + list(df_trades["exit_time"])
    else:
        equity_times = [df.index[0]]

    ax2.step(equity_times, equity, where="post", color="purple", linewidth=2.2, label="Equity (R$)")
    ax1.set_title(
        f"Robot R11 — Omni-Regime (Trend + Mean Reversion) | {df.index[0].date()} → {df.index[-1].date()}",
        fontsize=16, fontweight="bold",
    )
    ax1.grid(alpha=0.3)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    ax3 = fig.add_subplot(gs[1], sharex=ax1)
    eq_series = pd.Series(equity)
    drawdown = (eq_series - eq_series.cummax()) / eq_series.cummax() * 100
    ax3.fill_between(equity_times, drawdown, 0, color="red", alpha=0.3, label="Drawdown %")
    ax3.plot(equity_times, drawdown, color="red", linewidth=1)
    ax3.legend(loc="lower left")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[2])
    ax4.axis("off")
    days = (df.index[-1] - df.index[0]).days
    cdi_return = (1 + CDI_ANNUAL) ** (days / 365) - 1
    stats_text = (
        f"Net Profit: R$ {metrics.get('Net Profit', 0):,.2f} | "
        f"ROI: {metrics.get('ROI (%)', 0):.2f}% | "
        f"Drawdown: {metrics.get('Max Drawdown (%)', 0):.2f}% | "
        f"Trades: {metrics.get('Trades', 0)} | "
        f"Win Rate: {metrics.get('Win Rate (%)', 0):.1f}% | "
        f"Profit Factor: {metrics.get('Profit Factor', 0):.2f}"
        f"\nBenchmark CDI ({days} dias): {cdi_return * 100:.2f}%"
    )
    ax4.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=13, fontweight="bold")

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em {filename}")


if __name__ == "__main__":
    if not os.path.exists(DEFAULT_CSV_PATH):
        raise SystemExit(f"CSV não encontrado: {DEFAULT_CSV_PATH}")
    df, trades, equity = run_detailed_backtest(csv_path=DEFAULT_CSV_PATH)
    metrics = calculate_metrics(trades, equity)
    print("\n--- Relatório de Performance R11 ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    generate_flyer(df, trades, equity, metrics, filename=DEFAULT_OUTPUT_FILE)
