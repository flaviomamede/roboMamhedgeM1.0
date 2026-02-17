from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from roboMamhedgeR6 import run_backtest as run_r6
from roboMamhedgeR9 import run_backtest as run_r9
from roboMamhedgeR10 import run_backtest as run_r10
from utils_fuso import pnl_reais

CAPITAL_INICIAL = 10_000.0
CDI_ANUAL = 0.1175  # ajuste conforme cenário desejado
MC_SIMS = 100
SEED = 42


def _period_business_days(csv_path: str = "WIN_5min.csv") -> int:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.to_datetime(df.index)
    if len(idx) == 0:
        return 1
    start = idx.min().date()
    end = idx.max().date()
    bdays = pd.bdate_range(start=start, end=end)
    return max(len(bdays), 1)


def _metrics(trades_pts: np.ndarray, capital: float, cdi_anual: float, bdays: int) -> dict:
    trades_r = np.array([pnl_reais(t) for t in trades_pts], dtype=float)
    if len(trades_r) == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "media_trade_r$": 0.0,
            "desvio_trade_r$": 0.0,
            "variancia_trade_r$": 0.0,
            "sharpe_trade": 0.0,
            "retorno_total_%": 0.0,
            "cdi_periodo_%": ((1 + cdi_anual) ** (bdays / 252) - 1) * 100,
            "alpha_vs_cdi_pp": -((1 + cdi_anual) ** (bdays / 252) - 1) * 100,
            "trades_r": trades_r,
        }

    win_rate = float((trades_r > 0).mean())
    media = float(trades_r.mean())
    desvio = float(trades_r.std(ddof=1)) if len(trades_r) > 1 else 0.0
    variancia = float(trades_r.var(ddof=1)) if len(trades_r) > 1 else 0.0

    returns_pct = trades_r / capital
    mean_ret = float(returns_pct.mean())
    std_ret = float(returns_pct.std(ddof=1)) if len(returns_pct) > 1 else 0.0
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    retorno_total = float(trades_r.sum() / capital)
    cdi_periodo = float((1 + cdi_anual) ** (bdays / 252) - 1)
    alpha_vs_cdi = retorno_total - cdi_periodo

    return {
        "n_trades": int(len(trades_r)),
        "win_rate": win_rate,
        "media_trade_r$": media,
        "desvio_trade_r$": desvio,
        "variancia_trade_r$": variancia,
        "sharpe_trade": float(sharpe),
        "retorno_total_%": retorno_total * 100,
        "cdi_periodo_%": cdi_periodo * 100,
        "alpha_vs_cdi_pp": alpha_vs_cdi * 100,
        "trades_r": trades_r,
    }


def _monte_carlo_returns_pct(trades_r: np.ndarray, capital: float, sims: int, seed: int) -> np.ndarray:
    if len(trades_r) == 0:
        return np.zeros(sims)
    rng = np.random.default_rng(seed)
    n = len(trades_r)
    out = []
    for _ in range(sims):
        sample = rng.choice(trades_r, size=n, replace=True)
        out.append(sample.sum() / capital * 100)
    return np.array(out)


def _plot_mc_distributions(mc_data: dict[str, np.ndarray], output_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, values) in zip(axes, mc_data.items()):
        ax.hist(values, bins=18, alpha=0.85, color="#3b82f6", edgecolor="black")
        mean = values.mean() if len(values) else 0.0
        ax.axvline(mean, color="red", linestyle="--", linewidth=1.8, label=f"Média {mean:.2f}%")
        ax.set_title(f"{name} - {len(values)} simulações")
        ax.set_xlabel("Retorno acumulado (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Frequência")
    fig.suptitle("Monte Carlo (100 simulações) - Distribuição de retornos em torno da média", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def run_comparison() -> pd.DataFrame:
    bdays = _period_business_days("WIN_5min.csv")

    runs = {
        "R6": run_r6(),
        "R9": run_r9(ema_fast=6, rsi_thresh=40, rsi_window=3, stop_atr=1.5, target_atr=0, use_macd=True, use_adx=True),
        "R10": run_r10(ema_fast=6, ema_slow=21, rsi_thresh=40, rsi_window=4, stop_atr=1.7, trail_atr=2.4, breakeven_trigger_atr=1.5, use_macd=True, use_adx=True),
    }

    rows = []
    mc_data: dict[str, np.ndarray] = {}

    for name, trades_pts in runs.items():
        m = _metrics(np.array(trades_pts), CAPITAL_INICIAL, CDI_ANUAL, bdays)
        mc = _monte_carlo_returns_pct(m["trades_r"], CAPITAL_INICIAL, MC_SIMS, SEED)
        mc_data[name] = mc
        rows.append({
            "robo": name,
            "trades": m["n_trades"],
            "win_rate_%": m["win_rate"] * 100,
            "media_trade_R$": m["media_trade_r$"],
            "desvio_trade_R$": m["desvio_trade_r$"],
            "variancia_trade_R$": m["variancia_trade_r$"],
            "sharpe": m["sharpe_trade"],
            "retorno_total_%": m["retorno_total_%"],
            "cdi_periodo_%": m["cdi_periodo_%"],
            "alpha_vs_cdi_pp": m["alpha_vs_cdi_pp"],
            "mc_media_retorno_%": mc.mean() if len(mc) else 0.0,
            "mc_desvio_retorno_%": mc.std(ddof=1) if len(mc) > 1 else 0.0,
        })

    _plot_mc_distributions(mc_data, "montecarlo_100_retornos_r6_r9_r10.png")

    df_out = pd.DataFrame(rows).sort_values("alpha_vs_cdi_pp", ascending=False).reset_index(drop=True)
    return df_out


if __name__ == "__main__":
    df_cmp = run_comparison()
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print("\n=== Comparativo R6 vs R9 vs R10 ===")
        print(df_cmp.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print("\nGráfico salvo em: montecarlo_100_retornos_r6_r9_r10.png")
