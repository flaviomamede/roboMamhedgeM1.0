from __future__ import annotations

import argparse
import zlib
from dataclasses import dataclass
from datetime import datetime
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
DEFAULT_REPORT_MD = BASE_DIR / "otimize.md"

from utils_fuso import pnl_reais  # noqa: E402
from utils_metrics_pwb import metrics_from_csv  # noqa: E402


@dataclass
class RobotSpec:
    name: str
    run_backtest: callable
    space: dict[str, list]
    anchor: dict


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


def _trades_to_reais(trades_pts) -> np.ndarray:
    if trades_pts is None:
        return np.array([])
    arr = np.asarray(trades_pts).tolist()
    return np.array([pnl_reais(t) for t in arr], dtype=float)


def _mc_mu_var(trades_r: np.ndarray, n_sims: int = 1000, seed: int = 42) -> tuple[float, float]:
    if len(trades_r) == 0:
        return 0.0, float("inf")
    if len(trades_r) == 1:
        total = float(trades_r[0])
        return total, 0.0

    rng = np.random.default_rng(seed)
    n = len(trades_r)
    idx = rng.integers(0, n, size=(n_sims, n))
    totals = trades_r[idx].sum(axis=1)
    mu = float(totals.mean())
    var = float(totals.var(ddof=1))
    return mu, var


def _objective(mu: float, var: float, eps: float = 1e-9) -> float:
    return float(mu / (var + eps))


def _normalize_value(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def _candidate_key(params: dict, keys: list[str]) -> tuple:
    return tuple((k, _normalize_value(params[k])) for k in keys)


def _is_valid_candidate(params: dict) -> bool:
    if "ema_fast" in params and "ema_slow" in params and params["ema_fast"] >= params["ema_slow"]:
        return False
    if "trail_atr" in params and "stop_atr" in params and params["trail_atr"] < params["stop_atr"]:
        return False
    if "target_atr" in params and params["target_atr"] < 0:
        return False
    return True


def _robot_spec(robot: str) -> RobotSpec:
    rb = robot.upper()

    if rb == "R0":
        from roboMamhedgeR0 import run_backtest

        return RobotSpec(
            name="R0",
            run_backtest=run_backtest,
            space={
                "ema_fast": [5, 7, 9, 11, 13, 15],
                "ema_slow": [18, 21, 26, 34, 42, 55],
                "atr_period": [7, 10, 14, 20],
                "atr_mean_period": [7, 10, 14, 20, 30, 40],
            },
            anchor={"ema_fast": 9, "ema_slow": 21, "atr_period": 10, "atr_mean_period": 10},
        )

    if rb in {"R6", "R6V2"}:
        from roboMamhedgeR6_v2 import run_backtest

        return RobotSpec(
            name="R6",
            run_backtest=run_backtest,
            space={
                "stop_atr": [1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.6],
                "target_atr": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                "rsi_thresh": [30, 35, 40, 45, 50, 55],
                "rsi_window": [2, 3, 4, 5, 7, 9],
                "use_macd_filter": [False, True],
            },
            anchor={"stop_atr": 2.0, "target_atr": 3.0, "rsi_thresh": 35, "rsi_window": 3, "use_macd_filter": False},
        )

    if rb == "R7":
        from roboMamhedgeR7 import run_backtest

        return RobotSpec(
            name="R7",
            run_backtest=run_backtest,
            space={
                "stop_atr": [1.2, 1.5, 1.8, 2.0, 2.3, 2.6],
                "target_atr": [2.0, 2.5, 3.0, 3.5, 4.0],
                "rsi_bullish": [35, 40, 45, 50, 55, 60],
                "use_macd_filter": [False, True],
            },
            anchor={"stop_atr": 2.0, "target_atr": 3.0, "rsi_bullish": 40, "use_macd_filter": False},
        )

    if rb == "R8":
        from roboMamhedgeR8 import run_backtest

        return RobotSpec(
            name="R8",
            run_backtest=run_backtest,
            space={
                "ema_fast": [5, 7, 9, 12, 15],
                "ema_slow": [21, 26, 34, 42, 55],
                "ema_trend": [34, 50, 80, 100, 120],
                "momentum_lookback": [5, 10, 15, 20, 30],
                "stop_atr": [1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.6],
                "rsi_period": [10, 14, 21],
            },
            anchor={
                "ema_fast": 9,
                "ema_slow": 34,
                "ema_trend": 50,
                "momentum_lookback": 20,
                "stop_atr": 1.5,
                "rsi_period": 14,
            },
        )

    if rb == "R9":
        from roboMamhedgeR9 import run_backtest

        return RobotSpec(
            name="R9",
            run_backtest=run_backtest,
            space={
                "ema_fast": [3, 4, 6, 8, 10, 12],
                "rsi_period": [7, 10, 14, 21],
                "rsi_thresh": [35, 40, 45, 50, 55],
                "rsi_window": [3, 5, 7, 9],
                "stop_atr": [1.0, 1.3, 1.6, 2.0, 2.4],
                "target_atr": [0.0, 1.0, 1.5, 2.0, 2.5, 3.0],
                "use_macd": [False, True],
                "use_adx": [False, True],
                "adx_min": [15, 20, 25, 30],
                "max_bars_in_trade": [24, 36, 48, 60, 72],
            },
            anchor={
                "ema_fast": 4,
                "rsi_period": 14,
                "rsi_thresh": 40,
                "rsi_window": 5,
                "stop_atr": 2.0,
                "target_atr": 0.0,
                "use_macd": True,
                "use_adx": True,
                "adx_min": 20,
                "max_bars_in_trade": 48,
            },
        )

    if rb == "R10":
        from roboMamhedgeR10 import run_backtest

        return RobotSpec(
            name="R10",
            run_backtest=run_backtest,
            space={
                "ema_fast": [4, 6, 8, 10, 12],
                "ema_slow": [13, 21, 34, 55],
                "rsi_period": [10, 14, 21],
                "rsi_thresh": [35, 40, 45, 50, 55],
                "rsi_window": [3, 4, 6, 8],
                "stop_atr": [1.0, 1.3, 1.5, 1.7, 2.0, 2.3],
                "trail_atr": [1.5, 1.8, 2.2, 2.4, 2.8, 3.2, 3.6],
                "breakeven_trigger_atr": [0.8, 1.0, 1.2, 1.5, 1.8, 2.2],
                "use_adx": [False, True],
                "adx_min": [15, 20, 25, 30],
                "use_macd": [False, True],
                "max_bars_in_trade": [12, 20, 30, 40, 60],
            },
            anchor={
                "ema_fast": 6,
                "ema_slow": 21,
                "rsi_period": 14,
                "rsi_thresh": 40,
                "rsi_window": 4,
                "stop_atr": 1.7,
                "trail_atr": 2.4,
                "breakeven_trigger_atr": 1.5,
                "use_adx": True,
                "adx_min": 20,
                "use_macd": True,
                "max_bars_in_trade": 20,
            },
        )

    raise SystemExit("Robô inválido. Use R0, R6, R7, R8, R9, R10 ou ALL.")


def _evaluate_candidate(
    spec: RobotSpec,
    params: dict,
    train_csv: str,
    n_sims: int,
    min_trades: int,
    seed: int,
) -> dict:
    if not _is_valid_candidate(params):
        return {"valid": False, "score": -float("inf"), "n_trades": 0}

    trades_pts = spec.run_backtest(train_csv, **params)
    trades_r = _trades_to_reais(trades_pts)
    if len(trades_r) < min_trades:
        return {"valid": False, "score": -float("inf"), "n_trades": int(len(trades_r))}

    mu, var = _mc_mu_var(trades_r, n_sims=n_sims, seed=seed)
    score = _objective(mu, var)
    return {
        "valid": True,
        "score": score,
        "mu": mu,
        "var": var,
        "n_trades": int(len(trades_r)),
    }


def _sample_random_candidate(space: dict[str, list], rng: np.random.Generator) -> dict:
    out = {}
    for k, values in space.items():
        out[k] = _normalize_value(rng.choice(values))
    return out


def _optimize_robot(
    robot: str,
    train_csv: str,
    test_csv: str,
    n_sims: int,
    min_trades: int,
    seed: int,
    global_samples: int,
    elite_size: int,
    refine_rounds: int,
) -> dict:
    spec = _robot_spec(robot)
    rng = np.random.default_rng(seed + zlib.crc32(spec.name.encode("utf-8")))
    keys = list(spec.space.keys())
    cache: dict[tuple, dict] = {}

    def eval_and_cache(params: dict) -> None:
        normalized = {k: _normalize_value(params[k]) for k in keys}
        key = _candidate_key(normalized, keys)
        if key in cache:
            return
        eval_seed = seed + int(zlib.crc32(repr((spec.name, key)).encode("utf-8")) % 1_000_000)
        cache[key] = {"params": normalized} | _evaluate_candidate(
            spec=spec,
            params=normalized,
            train_csv=train_csv,
            n_sims=n_sims,
            min_trades=min_trades,
            seed=eval_seed,
        )

    # 1) Âncora atual (parâmetros já otimizados) + varredura 1D em torno dela.
    eval_and_cache(spec.anchor)
    for p in keys:
        for v in spec.space[p]:
            cand = dict(spec.anchor)
            cand[p] = _normalize_value(v)
            eval_and_cache(cand)

    # 2) Exploração global aleatória (diversificação para fugir de mínimo local).
    for _ in range(max(0, global_samples)):
        eval_and_cache(_sample_random_candidate(spec.space, rng))

    valid_rows = [row for row in cache.values() if row["valid"]]
    if not valid_rows:
        raise SystemExit(f"Nenhum conjunto válido para {spec.name}. Tente reduzir min_trades.")

    valid_rows.sort(key=lambda r: r["score"], reverse=True)
    elites = valid_rows[: max(1, elite_size)]

    # 3) Refinamento local coordenado multi-start (partindo dos melhores elites).
    for elite in elites:
        current = dict(elite["params"])
        for _round in range(max(1, refine_rounds)):
            improved = False
            current_key = _candidate_key(current, keys)
            current_score = cache[current_key]["score"]
            for p in keys:
                best_local = dict(current)
                best_score = current_score
                for v in spec.space[p]:
                    cand = dict(current)
                    cand[p] = _normalize_value(v)
                    eval_and_cache(cand)
                    cand_key = _candidate_key(cand, keys)
                    cand_row = cache[cand_key]
                    if cand_row["valid"] and cand_row["score"] > best_score:
                        best_local = cand
                        best_score = cand_row["score"]
                if best_score > current_score:
                    current = best_local
                    current_score = best_score
                    improved = True
            if not improved:
                break

    final_valid = [row for row in cache.values() if row["valid"]]
    final_valid.sort(key=lambda r: r["score"], reverse=True)
    best = final_valid[0]

    # Avaliação no TEST com os melhores parâmetros do TRAIN.
    best_params = best["params"]
    test_trades_pts = spec.run_backtest(test_csv, **best_params)
    test_trades_r = _trades_to_reais(test_trades_pts)
    test_mu, test_var = _mc_mu_var(test_trades_r, n_sims=n_sims, seed=seed + 9_999)
    test_score = _objective(test_mu, test_var)

    train_trades_pts = spec.run_backtest(train_csv, **best_params)
    tested = len(cache)
    valid = len(final_valid)
    return {
        "robot": spec.name,
        "params": best_params,
        "score": float(best["score"]),
        "train_mu": float(best["mu"]),
        "train_var": float(best["var"]),
        "train_trades": int(best["n_trades"]),
        "test_mu": float(test_mu),
        "test_var": float(test_var),
        "test_score": float(test_score),
        "test_trades": int(len(test_trades_r)),
        "tested": int(tested),
        "valid": int(valid),
        "global_samples": int(global_samples),
        "elite_size": int(elite_size),
        "refine_rounds": int(refine_rounds),
        "train_metrics": metrics_from_csv(trades_pts=np.asarray(train_trades_pts), csv_path=train_csv),
        "test_metrics": metrics_from_csv(trades_pts=np.asarray(test_trades_pts), csv_path=test_csv),
    }


def _defaults_hint(params: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in params.items())


def _write_report(path: Path, results: list[dict], n_sims: int, train_pct: float, base_csv: Path) -> None:
    lines = []
    lines.append("# Otimizacao Heuristica Unificada - Fase 1 + Fase 2")
    lines.append("")
    lines.append(f"- Data/hora: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- CSV base: `{base_csv}`")
    lines.append(f"- Split temporal: {train_pct:.0%} train / {(1-train_pct):.0%} test")
    lines.append(f"- Simulacoes Monte Carlo por conjunto: {n_sims}")
    lines.append("- Funcao-objetivo: `score = mu / variancia`")
    lines.append("  - `mu`: media dos totais simulados (R$)")
    lines.append("  - `variancia`: variancia dos totais simulados (R$^2)")
    lines.append("- Busca heuristica:")
    lines.append("  - ancora atual + varredura 1D")
    lines.append("  - amostragem aleatoria global (diversificacao)")
    lines.append("  - refinamento coordenado multi-start (elites)")
    lines.append("")
    lines.append("| Robo | Combos testados | Combos validos | Score train | Score test | mu train (R$) | var train | Params otimos |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        lines.append(
            f"| {r['robot']} | {r['tested']} | {r['valid']} | {r['score']:.8f} | {r['test_score']:.8f} | {r['train_mu']:.2f} | {r['train_var']:.2f} | `{_defaults_hint(r['params'])}` |"
        )
    lines.append("")
    lines.append("## Resultado detalhado")
    lines.append("")
    for r in results:
        tm = r["train_metrics"]
        sm = r["test_metrics"]
        lines.append(f"### {r['robot']}")
        lines.append(f"- Params otimos: `{_defaults_hint(r['params'])}`")
        lines.append(
            f"- Busca usada: global_samples={r['global_samples']} elite_size={r['elite_size']} refine_rounds={r['refine_rounds']}"
        )
        lines.append(f"- Objective train: {r['score']:.8f} | mu={r['train_mu']:.2f} | var={r['train_var']:.2f}")
        lines.append(f"- Objective test: {r['test_score']:.8f} | mu={r['test_mu']:.2f} | var={r['test_var']:.2f}")
        lines.append(
            f"- TRAIN metrics: trades={tm['n']} win={tm['win_rate']*100:.1f}% epl=R$ {tm['e_pl']:.2f} total=R$ {tm['total_pl']:.2f} "
            f"payoff={tm['payoff']:.2f} riskF={tm['risk_factor']:.2f} ROI/m={tm['roi_mensal_pct']:.2f}%"
        )
        lines.append(
            f"- TEST metrics: trades={sm['n']} win={sm['win_rate']*100:.1f}% epl=R$ {sm['e_pl']:.2f} total=R$ {sm['total_pl']:.2f} "
            f"payoff={sm['payoff']:.2f} riskF={sm['risk_factor']:.2f} ROI/m={sm['roi_mensal_pct']:.2f}%"
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Otimizacao unificada por Kelly mu/variancia via Monte Carlo (heuristica: global + refinamento)"
    )
    p.add_argument("--robot", default="R7", help="R0, R6, R7, R8, R9, R10 ou ALL")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV base para split temporal")
    p.add_argument("--train_pct", type=float, default=0.7, help="Percentual para treino")
    p.add_argument("--mc_sims", type=int, default=1000, help="Numero de simulacoes Monte Carlo por conjunto")
    p.add_argument("--min_trades", type=int, default=20, help="Minimo de trades para considerar conjunto valido")
    p.add_argument("--seed", type=int, default=42, help="Seed base para Monte Carlo")
    p.add_argument("--global_samples", type=int, default=220, help="Amostras aleatorias globais por robo")
    p.add_argument("--elite_size", type=int, default=6, help="Quantidade de elites para refinamento")
    p.add_argument("--refine_rounds", type=int, default=3, help="Rodadas de refinamento coordenado por elite")
    p.add_argument("--output_md", default=str(DEFAULT_REPORT_MD), help="Arquivo markdown de relatorio")
    args = p.parse_args()

    base_csv = Path(args.csv)
    train_csv, test_csv = _ensure_train_test(base_csv, train_pct=float(args.train_pct))
    robot = args.robot.upper()

    if robot == "ALL":
        robots = ["R0", "R6", "R7", "R8", "R9", "R10"]
    else:
        robots = [robot]

    results: list[dict] = []
    for rb in robots:
        res = _optimize_robot(
            robot=rb,
            train_csv=str(train_csv),
            test_csv=str(test_csv),
            n_sims=int(args.mc_sims),
            min_trades=int(args.min_trades),
            seed=int(args.seed),
            global_samples=int(args.global_samples),
            elite_size=int(args.elite_size),
            refine_rounds=int(args.refine_rounds),
        )
        results.append(res)

    print("=" * 110)
    print("OTIMIZACAO HEURISTICA UNIFICADA (mu/variancia) - FASE 1 + FASE 2")
    print("=" * 110)
    print(f"Base: {base_csv.name} | Train: {train_csv.name} | Test: {test_csv.name}")
    print(
        f"MC sims: {args.mc_sims} | Min trades: {args.min_trades} | "
        f"Global samples: {args.global_samples} | Elites: {args.elite_size} | Rounds: {args.refine_rounds}"
    )
    print("-" * 110)
    for r in results:
        print(
            f"{r['robot']}: score_train={r['score']:.8f} score_test={r['test_score']:.8f} "
            f"| mu_train={r['train_mu']:.2f} var_train={r['train_var']:.2f} | tested={r['tested']}"
        )
        print(f"  best: {_defaults_hint(r['params'])}")

    report_path = Path(args.output_md)
    _write_report(report_path, results, n_sims=int(args.mc_sims), train_pct=float(args.train_pct), base_csv=base_csv)
    print("-" * 110)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()

