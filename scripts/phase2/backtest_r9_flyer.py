import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange

# Import utilities from the sibling module
# We need to make sure python path sees the current directory
import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from market_time import converter_para_brt, dentro_horario_operacao
from b3_costs_phase2 import TradePoints, default_b3_cost_model, trade_costs_brl, trade_net_pnl_brl

# --- Configuration ---
INITIAL_CAPITAL = 10000.0
CDI_ANNUAL = 0.12 # 12% a.a.
DEFAULT_QUANTITY = 1
_COST_MODEL = default_b3_cost_model()

DEFAULT_CSV_PATH = os.path.join(REPO_ROOT, "fase1_antigravity", "WIN_5min.csv")

def run_detailed_backtest(
    csv_path=DEFAULT_CSV_PATH,
    ema_fast=4,
    rsi_period=14,
    rsi_thresh=40,
    rsi_window=5,
    stop_atr=2.0,
    target_atr=0,       # 0 = no target
    use_macd=False,
    use_adx=True,
    adx_min=20,
):
    print(f"Carregando dados de {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)
    
    # Sort just in case
    df.sort_index(inplace=True)

    print("Calculando indicadores...")
    # Indicators
    df['rsi'] = RSIIndicator(df['close'], window=rsi_period).rsi()
    df['ema'] = EMAIndicator(df['close'], window=ema_fast).ema_indicator()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    if use_macd:
        macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_hist'] = macd.macd_diff()

    if use_adx:
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

    # Peak detection
    df['rsi_peak_max'] = (df['rsi'].shift(1) > df['rsi'].shift(2)) & (df['rsi'].shift(1) > df['rsi'])
    df['rsi_bullish_window'] = (df['rsi'] > rsi_thresh).rolling(window=rsi_window).max()

    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    trades = [] # List of dicts: {entry_time, exit_time, entry_price, exit_price, type (buy/sell), result_points, result_brl}
    
    # Equity curve simulation
    # We will track equity at every bar close
    equity = [INITIAL_CAPITAL] * len(df)
    current_capital = INITIAL_CAPITAL
    
    print("Simulando trades...")
    for i in range(2, len(df)):
        ts = df.index[i]
        
        # Determine Trade Logic
        trade_executed = False
        pnl_trade = 0
        
        # 1. Force Close at End of Day
        if ts.hour >= 17 and position != 0:
            exit_price = df['close'].iloc[i]
            if position == 1:
                pts = exit_price - entry_price
            elif position == -1:
                pts = entry_price - exit_price

            trade = TradePoints(entry_price_points=float(entry_price), exit_price_points=float(exit_price), quantity=DEFAULT_QUANTITY)
            trade_res_brl = trade_net_pnl_brl(trade, _COST_MODEL)
            trades.append({
                'entry_time': entry_ts,
                'exit_time': ts,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'points': pts,
                'pnl_brl': trade_res_brl,
                'costs_brl': trade_costs_brl(trade, _COST_MODEL),
            })
            current_capital += trade_res_brl
            position = 0
            equity[i] = current_capital
            continue

        if not dentro_horario_operacao(ts):
            equity[i] = current_capital
            continue

        atr_val = df['atr'].iloc[i]
        if pd.isna(atr_val) or atr_val == 0:
            equity[i] = current_capital
            continue

        # ADX Filter
        if use_adx:
            adx_val = df['adx'].iloc[i]
            if pd.isna(adx_val) or adx_val < adx_min:
                equity[i] = current_capital
                continue

        # --- ENTRY ---
        if position == 0:
            rsi_win = df['rsi_bullish_window'].iloc[i]
            if not pd.isna(rsi_win) and rsi_win:
                ema_up = df['ema'].iloc[i] > df['ema'].iloc[i - 1]
                if ema_up:
                    macd_ok = True
                    if use_macd:
                        macd_val = df['macd_hist'].iloc[i]
                        macd_ok = pd.isna(macd_val) or macd_val > 0
                    
                    if macd_ok:
                         # For this robot, logic seems to be LONG ONLY or derived?
                         # R9 original generic code:
                         # if position <= 0... 
                         # Actually looking at R9 code: `if position <= 0:` inside entry block implies it might reverse from Short to Long?
                         # But wait, looking closer at R9 code:
                         # `if position == -1:` -> closes short then buys? 
                         # NO, R9 logic in `if position <= 0:` checks signals to BUY (Long).
                         # It does NOT seem to have Short logic implemented in the `elif` blocks or specific short signals.
                         # The provided R9 code primarily scans for `ema_up` and `rsi_bullish`. This is a LONG ONLY strategy setup in the provided snippet?
                         # Let's re-read R9 line 89: `if position <= 0:`
                         # And line 112: `elif position == 1:` which handles exit.
                         # There is no `elif position == -1:` block for managing a short position, and no trigger to go `position = -1`.
                         # So R9 is effectively Long Only as written in the provided file.
                         
                         entry_price = df['close'].iloc[i]
                         entry_ts = ts
                         stop_loss = entry_price - stop_atr * atr_val
                         take_profit = entry_price + target_atr * atr_val if target_atr > 0 else 0
                         position = 1
        
        # --- EXIT ---
        elif position == 1:
            hit_stop = df['low'].iloc[i] <= stop_loss
            hit_tp = take_profit > 0 and df['high'].iloc[i] >= take_profit
            peak = df['rsi_peak_max'].iloc[i]
            
            exit_signal = False
            exit_price_exec = 0
            
            if hit_stop:
                exit_signal = True
                exit_price_exec = stop_loss # Simulate stop execution at price
            elif hit_tp:
                exit_signal = True
                exit_price_exec = take_profit
            elif peak:
                exit_signal = True
                exit_price_exec = df['close'].iloc[i]
            
            if exit_signal:
                pts = exit_price_exec - entry_price
                trade = TradePoints(entry_price_points=float(entry_price), exit_price_points=float(exit_price_exec), quantity=DEFAULT_QUANTITY)
                trade_res_brl = trade_net_pnl_brl(trade, _COST_MODEL)
                trades.append({
                    'entry_time': entry_ts,
                    'exit_time': ts,
                    'entry_price': entry_price,
                    'exit_price': exit_price_exec,
                    'position': 1,
                    'points': pts,
                    'pnl_brl': trade_res_brl,
                    'costs_brl': trade_costs_brl(trade, _COST_MODEL),
                })
                current_capital += trade_res_brl
                position = 0
        
        equity[i] = current_capital

    return df, trades, equity

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
    
    # Buy/Sell/Win/Loss Markers
    # Since this is a "flyer" showing robot performance, let's mark the exits
    # Green Circle = Win, Red Circle = Loss
    if not df_trades.empty:
        wins = df_trades[df_trades['pnl_brl'] > 0]
        losses = df_trades[df_trades['pnl_brl'] <= 0]
        
        # Plotting at Exit Time/Price
        ax1.scatter(wins['exit_time'], wins['exit_price'], color='green', marker='o', s=80, edgecolors='black', zorder=5, label='Win')
        ax1.scatter(losses['exit_time'], losses['exit_price'], color='red', marker='o', s=80, edgecolors='black', zorder=5, label='Loss')

    # Equity Curve
    ax2.plot(df.index, equity, color='blue', linewidth=2.5, label='Robot Equity (R$)')
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
    
    ax3.fill_between(df.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown %')
    ax3.plot(df.index, drawdown, color='red', linewidth=1)
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

    # Add CDI Comparison (Approx)
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
        output_file = os.path.join(current_dir, "fase1_antigravity", "flyer_r9.png")
        generate_flyer(df, trades, equity, metrics, filename=output_file)
