"""
ROBÔ BASELINE - WIN DAY TRADE
==============================

Estratégia SIMPLES e ROBUSTA para WIN (Mini Índice Bovespa)

PRINCÍPIOS:
1. Simplicidade (menos overfitting)
2. Gestão de risco clara (Stop/Target em ATR)
3. Filtro de regime (ADX - só opera em tendência)
4. P&L correto em Reais (1 ponto = R$ 0,20)
5. Custos realistas (R$ 2,50 por round-trip)

ESTRATÉGIA:
- Entrada: Price Action + Momentum + Filtro de Tendência
  * Compra: Preço acima EMA21 + ADX > 20 + Breakout de máxima recente
  * Venda: Preço abaixo EMA21 + ADX > 20 + Breakdown de mínima recente
  
- Saída:
  * Stop Loss: 1.5 × ATR
  * Take Profit: 2.5 × ATR (R:R = 1.67)
  * Trailing Stop: Move stop para entrada quando lucro > 1.5 ATR
  
- Horário: 09:15 - 17:00 (evita abertura muito volátil)
- Máx 3 trades/dia (controle de overtrading)

EXPECTATIVA TEÓRICA:
Com 50% win rate e R:R 1.67:
E[P&L] = 0.50 × 2.5 - 0.50 × 1.5 = 0.5 ATR > 0 ✓

Com 55% win rate:
E[P&L] = 0.55 × 2.5 - 0.45 × 1.5 = 0.7 ATR
"""

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
from datetime import time

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# BOVA11 em pontos: 1 ponto = R$ 0,001
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from utils_fuso import MULT_PONTOS_REAIS, CUSTO_POR_TRADE
    MULT_WIN = MULT_PONTOS_REAIS
    CUSTO_REAIS = CUSTO_POR_TRADE * MULT_PONTOS_REAIS
except ImportError:
    MULT_WIN = 0.001  # BOVA11×1000
    CUSTO_REAIS = 2.50

# Gestão de risco
STOP_ATR = 1.5      # Stop Loss em ATR
TARGET_ATR = 2.5    # Take Profit em ATR
TRAIL_ATR = 1.5     # Ativa trailing stop quando lucro > 1.5 ATR

# Filtros
ADX_MIN = 20        # ADX mínimo (força da tendência)
EMA_PERIOD = 21     # EMA de tendência
ATR_PERIOD = 14     # ATR para volatilidade
LOOKBACK = 10       # Lookback para breakout

# Controles operacionais
HORA_INICIO = time(9, 15)   # Evita primeira hora muito volátil
HORA_FIM = time(17, 0)      # Fecha antes do after-market
MAX_TRADES_DIA = 3          # Máximo de trades por dia


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def converter_para_brt(df):
    """Converte índice para timezone de Brasília"""
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('America/Sao_Paulo')
    return df


def dentro_horario_operacao(timestamp):
    """Verifica se está dentro do horário de operação"""
    hora = timestamp.time()
    return HORA_INICIO <= hora <= HORA_FIM


def calcula_metricas(trades):
    """Calcula métricas completas de performance"""
    if len(trades) == 0:
        return None
    
    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades <= 0]
    
    # Métricas básicas
    n_trades = len(trades)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    
    # P&L
    total_pnl = trades.sum()
    avg_pnl = trades.mean()
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    # Profit Factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Drawdown
    cumulative = np.cumsum(trades)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max()
    
    # Sharpe Ratio (anualizado)
    if trades.std() > 0:
        # Assumindo ~50 trades/mês, 12 meses/ano = 600 trades/ano
        sharpe = (avg_pnl / trades.std()) * np.sqrt(600)
    else:
        sharpe = 0
    
    # Expectativa (expectancy)
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'expectancy': expectancy,
    }


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(csv_path="WIN_5min.csv", verbose=True):
    """
    Executa backtest da estratégia baseline
    
    Returns:
        trades: array com P&L em Reais de cada trade
    """
    # Carrega dados
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    df = converter_para_brt(df)
    
    # Indicadores
    df['ema'] = EMAIndicator(df['close'], window=EMA_PERIOD).ema_indicator()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=ATR_PERIOD).average_true_range()
    
    adx_ind = ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_ind.adx()
    
    # Máximas e mínimas de lookback (para breakout)
    df['high_lookback'] = df['high'].rolling(window=LOOKBACK).max().shift(1)
    df['low_lookback'] = df['low'].rolling(window=LOOKBACK).min().shift(1)
    
    # Estado
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trailing_stop_active = False
    
    trades = []
    trades_hoje = 0
    data_atual = None
    
    # Loop
    for i in range(max(EMA_PERIOD, ATR_PERIOD, LOOKBACK) + 1, len(df)):
        timestamp = df.index[i]
        
        # Reseta contador de trades no novo dia
        if data_atual is None or timestamp.date() != data_atual:
            trades_hoje = 0
            data_atual = timestamp.date()
        
        # Verifica horário
        if not dentro_horario_operacao(timestamp):
            # Fecha posição se ainda aberta fora do horário
            if position != 0:
                pnl_pontos = (df['close'].iloc[i] - entry_price) * position
                pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
                trades.append(pnl_reais)
                position = 0
                if verbose:
                    print(f"{timestamp} | FECHAMENTO HORÁRIO | P&L: R$ {pnl_reais:.2f}")
            continue
        
        # Pega valores atuais
        close = df['close'].iloc[i]
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        ema = df['ema'].iloc[i]
        atr = df['atr'].iloc[i]
        adx = df['adx'].iloc[i]
        high_lb = df['high_lookback'].iloc[i]
        low_lb = df['low_lookback'].iloc[i]
        
        # Skip se indicadores não disponíveis
        if pd.isna(ema) or pd.isna(atr) or pd.isna(adx) or pd.isna(high_lb) or pd.isna(low_lb):
            continue
        
        # =====================================================================
        # GESTÃO DE POSIÇÃO ABERTA
        # =====================================================================
        if position != 0:
            # Checa stop loss
            if position == 1 and low <= stop_loss:
                pnl_pontos = stop_loss - entry_price
                pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
                trades.append(pnl_reais)
                if verbose:
                    print(f"{timestamp} | STOP LONG | Entrada: {entry_price:.0f} | Stop: {stop_loss:.0f} | P&L: R$ {pnl_reais:.2f}")
                position = 0
                trades_hoje += 1
                continue
            
            elif position == -1 and high >= stop_loss:
                pnl_pontos = entry_price - stop_loss
                pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
                trades.append(pnl_reais)
                if verbose:
                    print(f"{timestamp} | STOP SHORT | Entrada: {entry_price:.0f} | Stop: {stop_loss:.0f} | P&L: R$ {pnl_reais:.2f}")
                position = 0
                trades_hoje += 1
                continue
            
            # Checa take profit
            if position == 1 and high >= take_profit:
                pnl_pontos = take_profit - entry_price
                pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
                trades.append(pnl_reais)
                if verbose:
                    print(f"{timestamp} | TARGET LONG | Entrada: {entry_price:.0f} | Target: {take_profit:.0f} | P&L: R$ {pnl_reais:.2f}")
                position = 0
                trades_hoje += 1
                continue
            
            elif position == -1 and low <= take_profit:
                pnl_pontos = entry_price - take_profit
                pnl_reais = pnl_pontos * MULT_WIN - CUSTO_REAIS
                trades.append(pnl_reais)
                if verbose:
                    print(f"{timestamp} | TARGET SHORT | Entrada: {entry_price:.0f} | Target: {take_profit:.0f} | P&L: R$ {pnl_reais:.2f}")
                position = 0
                trades_hoje += 1
                continue
            
            # Trailing stop (move stop para breakeven quando lucro > TRAIL_ATR)
            if not trailing_stop_active:
                if position == 1 and (close - entry_price) >= TRAIL_ATR * atr:
                    stop_loss = entry_price
                    trailing_stop_active = True
                    if verbose:
                        print(f"{timestamp} | TRAILING ATIVADO LONG | Novo stop: {stop_loss:.0f}")
                
                elif position == -1 and (entry_price - close) >= TRAIL_ATR * atr:
                    stop_loss = entry_price
                    trailing_stop_active = True
                    if verbose:
                        print(f"{timestamp} | TRAILING ATIVADO SHORT | Novo stop: {stop_loss:.0f}")
        
        # =====================================================================
        # ENTRADA EM POSIÇÃO (se flat e dentro do limite de trades)
        # =====================================================================
        if position == 0 and trades_hoje < MAX_TRADES_DIA:
            # Filtro de tendência (ADX)
            if adx < ADX_MIN:
                continue
            
            # SINAL LONG: Preço acima EMA + Breakout de máxima recente
            if close > ema and high > high_lb:
                entry_price = close
                stop_loss = entry_price - STOP_ATR * atr
                take_profit = entry_price + TARGET_ATR * atr
                position = 1
                trailing_stop_active = False
                if verbose:
                    print(f"{timestamp} | ENTRADA LONG | Preço: {entry_price:.0f} | Stop: {stop_loss:.0f} | Target: {take_profit:.0f} | ADX: {adx:.1f}")
            
            # SINAL SHORT: Preço abaixo EMA + Breakdown de mínima recente
            elif close < ema and low < low_lb:
                entry_price = close
                stop_loss = entry_price + STOP_ATR * atr
                take_profit = entry_price - TARGET_ATR * atr
                position = -1
                trailing_stop_active = False
                if verbose:
                    print(f"{timestamp} | ENTRADA SHORT | Preço: {entry_price:.0f} | Stop: {stop_loss:.0f} | Target: {take_profit:.0f} | ADX: {adx:.1f}")
    
    return np.array(trades) if trades else np.array([])


# =============================================================================
# EXECUÇÃO E RELATÓRIO
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ROBÔ BASELINE - WIN DAY TRADE")
    print("=" * 80)
    print(f"\nParâmetros:")
    print(f"  Stop Loss: {STOP_ATR} × ATR")
    print(f"  Take Profit: {TARGET_ATR} × ATR")
    print(f"  R:R Ratio: {TARGET_ATR/STOP_ATR:.2f}")
    print(f"  ADX Mínimo: {ADX_MIN}")
    print(f"  Horário: {HORA_INICIO} - {HORA_FIM}")
    print(f"  Máx trades/dia: {MAX_TRADES_DIA}")
    print(f"  Custo por trade: R$ {CUSTO_REAIS:.2f}")
    print("\n" + "=" * 80)
    
    # Executa backtest
    trades = run_backtest(verbose=False)
    
    if len(trades) == 0:
        print("\n❌ Nenhum trade executado!")
        print("   Verifique:")
        print("   1. Arquivo WIN_5min.csv existe?")
        print("   2. Dados têm volume suficiente?")
        print("   3. ADX está sempre baixo (mercado lateral)?")
    else:
        # Calcula métricas
        metricas = calcula_metricas(trades)
        
        print(f"\n📊 RESULTADOS DO BACKTEST")
        print("=" * 80)
        print(f"Total de Trades:      {metricas['n_trades']}")
        print(f"Win Rate:             {metricas['win_rate']*100:.1f}%")
        print(f"Total P&L:            R$ {metricas['total_pnl']:.2f}")
        print(f"P&L Médio/Trade:      R$ {metricas['avg_pnl']:.2f}")
        print(f"Ganho Médio:          R$ {metricas['avg_win']:.2f}")
        print(f"Perda Média:          R$ {metricas['avg_loss']:.2f}")
        print(f"Profit Factor:        {metricas['profit_factor']:.2f}")
        print(f"Max Drawdown:         R$ {metricas['max_drawdown']:.2f}")
        print(f"Sharpe Ratio:         {metricas['sharpe']:.2f}")
        print(f"Expectativa/Trade:    R$ {metricas['expectancy']:.2f}")
        
        # Análise
        print(f"\n📈 ANÁLISE")
        print("=" * 80)
        
        if metricas['win_rate'] >= 0.50:
            print("✅ Win rate >= 50%")
        else:
            print("⚠️  Win rate < 50% - ajustar estratégia ou filtros")
        
        if metricas['profit_factor'] >= 1.5:
            print("✅ Profit Factor >= 1.5 (boa relação ganho/perda)")
        elif metricas['profit_factor'] >= 1.0:
            print("⚠️  Profit Factor entre 1.0-1.5 (margens apertadas)")
        else:
            print("❌ Profit Factor < 1.0 (estratégia perde dinheiro)")
        
        if metricas['sharpe'] >= 1.0:
            print("✅ Sharpe >= 1.0 (retorno ajustado ao risco é bom)")
        else:
            print("⚠️  Sharpe < 1.0 (volatilidade alta vs retorno)")
        
        if metricas['expectancy'] > 0:
            print(f"✅ Expectativa positiva: R$ {metricas['expectancy']:.2f}/trade")
        else:
            print(f"❌ Expectativa negativa: R$ {metricas['expectancy']:.2f}/trade")
        
        # Estatística dos últimos 30 trades
        if len(trades) >= 30:
            recent = trades[-30:]
            recent_win_rate = (recent > 0).mean()
            recent_pnl = recent.sum()
            print(f"\n📊 Últimos 30 trades:")
            print(f"   Win Rate: {recent_win_rate*100:.1f}%")
            print(f"   P&L Total: R$ {recent_pnl:.2f}")
