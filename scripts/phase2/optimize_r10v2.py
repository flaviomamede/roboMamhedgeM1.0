from __future__ import annotations

import argparse
import zlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboMamhedgeR11 import run_backtest_trades
from b3_costs_phase2 import default_b3_cost_model, trade_net_pnl_brl

DEFAULT_BASE_CSV = REPO_ROOT / "fase1_antigravity" / "WIN_5min.csv"
DEFAULT_TRAIN_CSV = REPO_ROOT / "fase1_antigravity" / "WIN_train.csv"
DEFAULT_TEST_CSV = REPO_ROOT / "fase1_antigravity" / "WIN_test.csv"


SPACE: dict[str, list] = {
    "ema_fast": [6, 8, 10, 12, 14],
    "ema_slow": [21, 34, 55],
    "rsi_period": [10, 14, 21],
    "rsi_thresh": [45, 50, 55, 60],
    "rsi_window": [2, 3, 4, 5],
    "stop_atr": [1.4, 1.7, 2.0, 2.3, 2.6],
    "trail_atr": [1.8, 2.0, 2.2, 2.5, 2.8],
    "breakeven_trigger_atr": [1.5, 1.8, 2.2, 2.6],
    "use_adx": [False, True],
    "adx_min": [15.0, 20.0, 25.0],
    "use_macd": [False, True],
    "use_vwap_filter": [False, True],
    "er_period": [10, 14, 20, 30],
    "er_trend_min": [0.35, 0.40, 0.45, 0.50, 0.55],
    "atr_vol_window": [20, 30, 40],
    "vol_explosion_cutoff": [1.2, 1.4, 1.6, 1.8],
    "max_bars_in_trade": [8, 12, 16, 20],
}

ANCHOR: dict = {
    "ema_fast": 10,
    "ema_slow": 34,
    "rsi_period": 14,
    "rsi_thresh": 50,
    "rsi_window": 3,
    "stop_atr": 2.0,
    "trail_atr": 2.2,
    "breakeven_trigger_atr": 2.2,
    "use_adx": False,
    "adx_min": 20.0,
    "use_macd": True,
    "use_vwap_filter": True,
    "er_period": 20,
    "er_trend_min": 0.45,
    "atr_vol_window": 30,
    "vol_explosion_cutoff": 1.6,
    "max_bars_in_trade": 12,
}


def _normalize(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def _candidate_key(params: dict, keys: list[str]) -> tuple:
    return tuple((k, _normalize(params[k])) for k in keys)


def _ensure_train_test(base_csv: Path, train_csv: Path, test_csv: Path, train_pct: float) -> tuple[Path, Path]:
    if train_csv.exists() and test_csv.exists():
        return train_csv, test_csv

    df = pd.read_csv(base_csv, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()

    cut = int(len(df) * train_pct)
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    train.to_csv(train_csv)
    test.to_csv(test_csv)
    return train_csv, test_csv


def _trades_to_reais(params: dict, csv_path: str) -> np.ndarray:
    cost_model = default_b3_cost_model()
    trades = run_backtest_trades(
        csv_path=csv_path,
        quantity=1,
        with_timestamps=False,
        enable_long=True,
        enable_short=True,
        **params,
    )
    return np.array([trade_net_pnl_brl(t, cost_model) for t in trades], dtype=float)


def _mc_mu_var(trades_r: np.ndarray, n_sims: int, seed: int) -> tuple[float, float]:
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


def _is_valid_candidate(params: dict) -> bool:
    if params["ema_fast"] >= params["ema_slow"]:
        return False
    if params["stop_atr"] <= 0 or params["trail_atr"] <= 0 or params["breakeven_trigger_atr"] <= 0:
        return False
    if params["er_trend_min"] <= 0.0 or params["er_trend_min"] >= 1.0:
        return False
    return True


def _sample_random_candidate(space: dict[str, list], rng: np.random.Generator) -> dict:
    return {k: _normalize(rng.choice(values)) for k, values in space.items()}


def optimize_r11(
    train_csv: str,
    test_csv: str,
    mc_sims: int,
    min_trades: int,
    seed: int,
    global_samples: int,
    elite_size: int,
    refine_rounds: int,
) -> dict:
    keys = list(SPACE.keys())
    rng = np.random.default_rng(seed + zlib.crc32(b"R11"))
    cache: dict[tuple, dict] = {}

    def evaluate(params: dict) -> None:
        if not _is_valid_candidate(params):
            return
        params = {k: _normalize(params[k]) for k in keys}
        key = _candidate_key(params, keys)
        if key in cache:
            return

        eval_seed = seed + int(zlib.crc32(repr(key).encode("utf-8")) % 1_000_000)
        trades_r_train = _trades_to_reais(params, train_csv)
        n_train = int(len(trades_r_train))
        if n_train < min_trades:
            cache[key] = {
                "params": params,
                "valid": False,
                "n_train": n_train,
                "score_train": -float("inf"),
            }
            return

        mu_train, var_train = _mc_mu_var(trades_r_train, n_sims=mc_sims, seed=eval_seed)
        score_train = _objective(mu_train, var_train)

        cache[key] = {
            "params": params,
            "valid": True,
            "n_train": n_train,
            "mu_train": mu_train,
            "var_train": var_train,
            "score_train": score_train,
        }

    # 1) ancora + sweep 1D
    evaluate(ANCHOR)
    for p in keys:
        for v in SPACE[p]:
            cand = dict(ANCHOR)
            cand[p] = _normalize(v)
            evaluate(cand)

    # 2) exploracao global aleatoria
    for _ in range(max(0, global_samples)):
        evaluate(_sample_random_candidate(SPACE, rng))

    valid_rows = [row for row in cache.values() if row.get("valid")]
    if not valid_rows:
        raise RuntimeError("Nenhum conjunto válido encontrado. Tente reduzir min_trades.")
    valid_rows.sort(key=lambda r: r["score_train"], reverse=True)
    elites = valid_rows[: max(1, elite_size)]

    # 3) refinamento coordenado multi-start
    for elite in elites:
        current = dict(elite["params"])
        for _ in range(max(1, refine_rounds)):
            improved = False
            cur_key = _candidate_key(current, keys)
            cur_score = cache[cur_key]["score_train"]
            for p in keys:
                best_local = dict(current)
                best_score = cur_score
                for v in SPACE[p]:
                    cand = dict(current)
                    cand[p] = _normalize(v)
                    evaluate(cand)
                    ckey = _candidate_key(cand, keys)
                    row = cache.get(ckey)
                    if row and row.get("valid") and row["score_train"] > best_score:
                        best_local = cand
                        best_score = row["score_train"]
                if best_score > cur_score:
                    current = best_local
                    cur_score = best_score
                    improved = True
            if not improved:
                break

    final_valid = [row for row in cache.values() if row.get("valid")]
    final_valid.sort(key=lambda r: r["score_train"], reverse=True)
    best = dict(final_valid[0])
    best_params = best["params"]
    eval_seed_best = seed + int(zlib.crc32(repr(_candidate_key(best_params, keys)).encode("utf-8")) % 1_000_000)
    trades_r_test = _trades_to_reais(best_params, test_csv)
    mu_test, var_test = _mc_mu_var(trades_r_test, n_sims=mc_sims, seed=eval_seed_best + 9999)
    score_test = _objective(mu_test, var_test)
    best["n_test"] = int(len(trades_r_test))
    best["mu_test"] = mu_test
    best["var_test"] = var_test
    best["score_test"] = score_test
    return {
        "best": best,
        "tested": len(cache),
        "valid": len(final_valid),
    }


def _format_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def main() -> None:
    ap = argparse.ArgumentParser(description="Otimizacao heuristica do R11 (objective: mu/variancia com MC)")
    ap.add_argument("--base_csv", default=str(DEFAULT_BASE_CSV))
    ap.add_argument("--train_csv", default=str(DEFAULT_TRAIN_CSV))
    ap.add_argument("--test_csv", default=str(DEFAULT_TEST_CSV))
    ap.add_argument("--train_pct", type=float, default=0.7)
    ap.add_argument("--mc_sims", type=int, default=1000)
    ap.add_argument("--min_trades", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--global_samples", type=int, default=180)
    ap.add_argument("--elite_size", type=int, default=6)
    ap.add_argument("--refine_rounds", type=int, default=3)
    args = ap.parse_args()

    base_csv = Path(args.base_csv)
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    train_csv, test_csv = _ensure_train_test(base_csv, train_csv, test_csv, float(args.train_pct))

    out = optimize_r11(
        train_csv=str(train_csv),
        test_csv=str(test_csv),
        mc_sims=int(args.mc_sims),
        min_trades=int(args.min_trades),
        seed=int(args.seed),
        global_samples=int(args.global_samples),
        elite_size=int(args.elite_size),
        refine_rounds=int(args.refine_rounds),
    )

    best = out["best"]
    print("=" * 110)
    print("OTIMIZACAO R11 (ex-R10v2, Long+Short) — objective: mu/variancia (MC)")
    print("=" * 110)
    print(f"Train CSV: {train_csv}")
    print(f"Test  CSV: {test_csv}")
    print(f"Combos testados: {out['tested']} | validos: {out['valid']}")
    print("-" * 110)
    print(f"BEST score train: {best['score_train']:.8f} | mu={best['mu_train']:.2f} | var={best['var_train']:.2f} | trades={best['n_train']}")
    print(f"BEST score test : {best['score_test']:.8f} | mu={best['mu_test']:.2f} | var={best['var_test']:.2f} | trades={best['n_test']}")
    print("-" * 110)
    print("BEST params:")
    for k, v in best["params"].items():
        print(f"  {k}={v}")
    print("-" * 110)
    print("Para atualizar defaults no roboMamhedgeR11.py, copie estes valores.")


if __name__ == "__main__":
    main()

