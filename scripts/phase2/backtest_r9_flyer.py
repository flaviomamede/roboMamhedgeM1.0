import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import utilities from the sibling module
# We need to make sure python path sees the current directory
import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from market_time import converter_para_brt
from b3_costs_phase2 import default_b3_cost_model, trade_costs_brl, trade_net_pnl_brl
from roboMamhedgeR9 import run_backtest_trades

# --- Configuration ---
INITIAL_CAPITAL = 10000.0
CDI_ANNUAL = 0.12 # 12% a.a.
DEFAULT_QUANTITY = 1
_COST_MODEL = default_b3_cost_model()

DEFAULT_CSV_PATH = os.path.join(REPO_ROOT, "fase1_antigravity", "WIN_5min.csv")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "reports", "phase2")
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "flyer_r9.png")

def run_detailed_backtest(csv_path=DEFAULT_CSV_PATH):
    print(f"Carregando dados de {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)
    df.sort_index(inplace=True)
    print("Simulando trades com roboMamhedgeR9.run_backtest_trades()...")
    trades_with_ts = run_backtest_trades(csv_path=csv_path, quantity=DEFAULT_QUANTITY, with_timestamps=True)

    trades_rows = []
    for idx, item in enumerate(trades_with_ts, start=1):
        t = item["trade"]
        net = trade_net_pnl_brl(t, _COST_MODEL)
        costs = trade_costs_brl(t, _COST_MODEL)
        trades_rows.append(
            {
                "trade_id": idx,
                "entry_time": item["entry_time"],
                "exit_time": item["exit_time"],
                "entry_price": float(t.entry_price_points),
                "exit_price": float(t.exit_price_points),
                "points": float(t.exit_price_points - t.entry_price_points),
                "pnl_brl": float(net),
                "costs_brl": float(costs),
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
    
    # Basic
    total_net_profit = df_trades['pnl_brl'].sum()
    final_equity = equity_curve[-1]
    roi_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    num_trades = len(df_trades)
    winning_trades = df_trades[df_trades['pnl_brl'] > 0]
    losing_trades = df_trades[df_trades['pnl_brl'] <= 0]
    
    win_rate = len(winning_trades) / num_trades * 100
    
    avg_win = winning_trades['pnl_brl'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['pnl_brl'].mean() if not losing_trades.empty else 0
    
    gross_profit = winning_trades['pnl_brl'].sum()
    gross_loss = abs(losing_trades['pnl_brl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Drawdown
    eq_series = pd.Series(equity_curve)
    rolling_max = eq_series.cummax()
    drawdown = (eq_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min() # Negative value
    
    # Sharpe (Simplificado diário)
    # Convert equity to daily returns
    # Since we have 5min data, let's look at changes in equity per trade or resample?
    # Better: Resample equity curve to Daily closes
    # But equity_curve is a list matching df len.
    # We need the DF index to resample.
    
    return {
        'Initial Capital': INITIAL_CAPITAL,
        'Final Equity': final_equity,
        'Net Profit': total_net_profit,
        'ROI (%)': roi_pct,
        'Trades': num_trades,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Max Drawdown (%)': max_drawdown,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss
    }

def generate_flyer(df, trades, equity, metrics, filename="flyer_r9.png"):
    print("Gerando gráfico flyer...")
    
    # Prepare Data
    df_trades = pd.DataFrame(trades)
    
    # Setup Figure with GridSpec
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.5])
    
    # --- Main Chart (Price & Equity) ---
    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()
    
    # Price
    # Downsample for plotting speed if needed, but 5min is okay
    ax1.plot(df.index, df['close'], color='gray', alpha=0.5, linewidth=1, label='WIN Index')
    ax1.set_ylabel('Index Points', color='gray', fontweight='bold')
    
    if not df_trades.empty:
        wins = df_trades[df_trades['pnl_brl'] > 0]
        losses = df_trades[df_trades['pnl_brl'] <= 0]
        ax1.scatter(wins['exit_time'], wins['exit_price'], color='green', marker='o', s=60, edgecolors='black', zorder=5, label='Win')
        ax1.scatter(losses['exit_time'], losses['exit_price'], color='red', marker='o', s=60, edgecolors='black', zorder=5, label='Loss')

    if not df_trades.empty:
        equity_times = [df.index[0]] + list(df_trades["exit_time"])
    else:
        equity_times = [df.index[0]]
    ax2.step(equity_times, equity, where='post', color='blue', linewidth=2.2, label='Robot Equity (R$)')
    ax2.set_ylabel('Equity (R$)', color='blue', fontweight='bold', fontsize=12)
    
    # Styling
    ax1.set_title(f"Robot R9 Performance - {df.index[0].date()} to {df.index[-1].date()}", fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # --- Drawdown Area ---
    ax3 = fig.add_subplot(gs[1], sharex=ax1)
    eq_series = pd.Series(equity)
    rolling_max = eq_series.cummax()
    drawdown = (eq_series - rolling_max) / rolling_max * 100

    ax3.fill_between(equity_times, drawdown, 0, color='red', alpha=0.3, label='Drawdown %')
    ax3.plot(equity_times, drawdown, color='red', linewidth=1)
    ax3.set_ylabel('Drawdown %')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower left')

    # --- Metrics Table/Footer ---
    ax4 = fig.add_subplot(gs[2])
    ax4.axis('off')
    
    # Prepare Text
    if metrics:
        stats_text = (
            f"Net Profit: R$ {metrics['Net Profit']:,.2f}   |   "
            f"ROI: {metrics['ROI (%)']:.2f}%   |   "
            f"Drawdown: {metrics['Max Drawdown (%)']:.2f}%   |   "
            f"Trades: {metrics['Trades']}   |   "
            f"Win Rate: {metrics['Win Rate (%)']:.1f}%   |   "
            f"Profit Factor: {metrics['Profit Factor']:.2f}"
        )
    else:
        stats_text = "No trades executed."

    # Add CDI comparison using the backtest window in BRT.
    days = (df.index[-1] - df.index[0]).days
    cdi_return = (1 + CDI_ANNUAL)**(days/365) - 1
    stats_text += f"\nBenchmark CDI ({days} days): {cdi_return*100:.2f}%"

    ax4.text(0.5, 0.5, stats_text, 
             ha='center', va='center', 
             fontsize=14, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    # Save
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Gráfico salvo em {filename}")

if __name__ == "__main__":
    current_dir = REPO_ROOT
    csv_file = os.path.join(current_dir, "fase1_antigravity", "WIN_5min.csv")
    
    if not os.path.exists(csv_file):
        print(f"Erro: {csv_file} não encontrado.")
    else:
        # Run Backtest
        df, trades, equity = run_detailed_backtest(csv_path=csv_file)
        
        # Calculate Metrics
        metrics = calculate_metrics(trades, equity)
        
        # Display Report
        print("\n--- Relatório de Performance R9 ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
        # Generate Chart
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_file = DEFAULT_OUTPUT_FILE
        generate_flyer(df, trades, equity, metrics, filename=output_file)
