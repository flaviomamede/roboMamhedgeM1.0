"""
SCRIPT DE DIAGNÓSTICO - Verifica problemas específicos nos seus robôs
======================================================================

Este script testa suas hipóteses e identifica o problema real.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("DIAGNÓSTICO - ANÁLISE DOS SEUS ROBÔS")
print("=" * 80)

# =============================================================================
# TESTE 1: Arquivo WIN_5min.csv existe?
# =============================================================================
print("\n🔍 TESTE 1: Verificando arquivo de dados...")
print("-" * 80)

csv_path = "WIN_5min.csv"
if not Path(csv_path).exists():
    print(f"❌ Arquivo '{csv_path}' NÃO encontrado!")
    print("   → Coloque o arquivo WIN_5min.csv no mesmo diretório")
    print("   → Ou ajuste o caminho no código")
    exit(1)
else:
    print(f"✅ Arquivo '{csv_path}' encontrado")

# =============================================================================
# TESTE 2: Estrutura dos dados está correta?
# =============================================================================
print("\n🔍 TESTE 2: Verificando estrutura dos dados...")
print("-" * 80)

try:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"✅ Arquivo carregado com sucesso")
    print(f"   Total de linhas: {len(df):,}")
    print(f"   Período: {df.index[0]} até {df.index[-1]}")
    print(f"   Dias úteis: {len(df.index.date) / 78:.0f} dias (aprox)")
    
    # Verifica colunas
    print(f"\n   Colunas disponíveis: {list(df.columns)}")
    
    # Normaliza nomes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    
    required = ['high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Colunas obrigatórias faltando: {missing}")
        exit(1)
    if 'open' not in df.columns:
        df['open'] = df['close'].shift(1).fillna(df['close'])
        print(f"⚠️  Coluna 'open' ausente — usando close.shift(1) como aproximação")
    print(f"✅ Colunas disponíveis: {list(df.columns)}")
    
    # Mostra amostra
    print(f"\n   Amostra dos dados:")
    print(df[['close', 'high', 'low']].head(3))
    
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    exit(1)

# =============================================================================
# TESTE 3: Verificar escala dos preços (unidade)
# =============================================================================
print("\n🔍 TESTE 3: Verificando escala dos preços...")
print("-" * 80)

close_mean = df['close'].mean()
close_std = df['close'].std()

print(f"   Preço médio: {close_mean:,.2f}")
print(f"   Desvio padrão: {close_std:,.2f}")

if close_mean >= 1000:
    print("✅ Preços em pontos (BOVA11×1000 ou WIN)")
    if close_mean > 100000:
        print("   → BOVA11×1000: 1 ponto = R$ 0,001 (MULT_PONTOS_REAIS)")
    else:
        print("   → Escala de milhares; MULT_PONTOS_REAIS para converter em R$")
else:
    print("⚠️  Preços em R$/cota (< 1000). Execute: python converter_csv_para_pontos.py")

# =============================================================================
# TESTE 4: Simular P&L com e sem multiplicador
# =============================================================================
print("\n🔍 TESTE 4: Simulando impacto do multiplicador WIN...")
print("-" * 80)

# Simula trade típico
entry = df['close'].iloc[100]
exit_gain = entry + 200  # Ganho de 200 pontos
exit_loss = entry - 150  # Perda de 150 pontos

print(f"\n   Exemplo de trade:")
print(f"   Entrada: {entry:.0f} pontos")
print(f"   Saída (ganho): {exit_gain:.0f} pontos (+200 pts)")
print(f"   Saída (perda): {exit_loss:.0f} pontos (-150 pts)")

print(f"\n   SEM multiplicador (ERRADO):")
print(f"   Ganho: {exit_gain - entry:.0f} = R$ {exit_gain - entry:.0f} (INFLADO 5X!)")
print(f"   Perda: {exit_loss - entry:.0f} = R$ {exit_loss - entry:.0f} (INFLADO 5X!)")

print(f"\n   COM multiplicador 0.20 (CORRETO):")
print(f"   Ganho: {exit_gain - entry:.0f} × 0.20 = R$ {(exit_gain - entry) * 0.20:.2f}")
print(f"   Perda: {exit_loss - entry:.0f} × 0.20 = R$ {(exit_loss - entry) * 0.20:.2f}")

print(f"\n   ⚠️  DIFERENÇA: 5X!")
print(f"   Se seus robôs mostram P&L de R$ 1.000,00, o real é R$ 200,00")
print(f"   Se mostram perda de R$ -5.000,00, a real é R$ -1.000,00")

# =============================================================================
# TESTE 5: Verificar custos de transação
# =============================================================================
print("\n🔍 TESTE 5: Verificando custos de transação...")
print("-" * 80)

print(f"\n   Custos realistas para WIN (mini índice):")
print(f"   • Corretagem: R$ 0,50 - R$ 2,00")
print(f"   • Emolumentos B3: ~R$ 0,30")
print(f"   • Slippage (1-2 pts): R$ 0,20 - R$ 0,40")
print(f"   • TOTAL: ~R$ 2,00 - R$ 3,00 por round-trip")

print(f"\n   ⚠️  Se você está usando valores como:")
print(f"   • CUSTO_POR_TRADE = 10 → ERRADO! (R$ 10 é muito alto)")
print(f"   • CUSTO_POR_TRADE = 50 → ABSURDO! (R$ 50 por trade)")

print(f"\n   ✅ Valor correto: CUSTO_POR_TRADE = 2.5 (R$ 2,50)")

# =============================================================================
# TESTE 6: Análise do regime de mercado
# =============================================================================
print("\n🔍 TESTE 6: Analisando regime de mercado...")
print("-" * 80)

from ta.trend import ADXIndicator

adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
adx_mean = adx.mean()

print(f"\n   ADX médio: {adx_mean:.1f}")

if adx_mean < 20:
    print(f"   ❌ Mercado LATERAL (ADX < 20)")
    print(f"   → Estratégias de TENDÊNCIA vão PERDER dinheiro!")
    print(f"   → {(adx < 20).sum() / len(adx) * 100:.1f}% do tempo está lateral")
    print(f"\n   SOLUÇÃO: Use estratégias de reversão à média (mean reversion)")
elif adx_mean < 25:
    print(f"   ⚠️  Mercado com tendência FRACA")
    print(f"   → Estratégias de tendência terão baixa performance")
else:
    print(f"   ✅ Mercado com TENDÊNCIA forte")
    print(f"   → Estratégias trend-following devem funcionar")

# Distribição ADX
print(f"\n   Distribuição ADX:")
print(f"   • ADX < 20 (lateral): {(adx < 20).sum() / len(adx) * 100:.1f}%")
print(f"   • ADX 20-25 (fraco): {((adx >= 20) & (adx < 25)).sum() / len(adx) * 100:.1f}%")
print(f"   • ADX > 25 (forte): {(adx >= 25).sum() / len(adx) * 100:.1f}%")

# =============================================================================
# TESTE 7: Volatilidade (ATR)
# =============================================================================
print("\n🔍 TESTE 7: Analisando volatilidade...")
print("-" * 80)

from ta.volatility import AverageTrueRange

atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
atr_mean = atr.mean()
atr_in_reais = atr_mean * 0.20

print(f"\n   ATR médio: {atr_mean:.0f} pontos = R$ {atr_in_reais:.2f}")

print(f"\n   Implicações para Stop/Target:")
print(f"   • Stop 2×ATR: {atr_mean*2:.0f} pts = R$ {atr_mean*2*0.20:.2f}")
print(f"   • Target 3×ATR: {atr_mean*3:.0f} pts = R$ {atr_mean*3*0.20:.2f}")

if atr_mean * 2 * 0.20 < 5:
    print(f"\n   ⚠️  ATR baixo - stops muito apertados!")
    print(f"   → Você vai levar stop frequentemente")
elif atr_mean * 2 * 0.20 > 30:
    print(f"\n   ⚠️  ATR alto - stops muito largos!")
    print(f"   → Risco grande por trade")
else:
    print(f"\n   ✅ ATR adequado para day trade")

# =============================================================================
# RESUMO DIAGNÓSTICO
# =============================================================================
print("\n" + "=" * 80)
print("RESUMO DO DIAGNÓSTICO")
print("=" * 80)

problemas = []
solucoes = []

# Verifica multiplicador
if close_mean > 100000:
    problemas.append("❌ CRÍTICO: P&L provavelmente NÃO está usando multiplicador 0.20")
    solucoes.append("   → Adicionar: pnl_reais = pnl_pontos × 0.20")

# Verifica regime
if adx_mean < 20:
    problemas.append("❌ CRÍTICO: Mercado LATERAL - estratégias de tendência vão falhar")
    solucoes.append("   → Use estratégias de reversão à média")
    solucoes.append("   → Ou adicione filtro ADX > 20")

# Verifica quantidade de dados
dias = len(df) / 78
if dias < 120:
    problemas.append(f"⚠️  Dados insuficientes: apenas {dias:.0f} dias")
    solucoes.append("   → Consiga pelo menos 6 meses de dados")

if len(problemas) > 0:
    print("\n🔴 PROBLEMAS IDENTIFICADOS:")
    for p in problemas:
        print(p)
    print("\n💡 SOLUÇÕES:")
    for s in solucoes:
        print(s)
else:
    print("\n✅ Nenhum problema crítico identificado!")
    print("   → Seus dados parecem estar corretos")
    print("   → O problema pode estar na lógica da estratégia")

print("\n" + "=" * 80)
print("PRÓXIMOS PASSOS")
print("=" * 80)
print("""
1. Se o multiplicador está errado:
   → Corrija todos os robôs para usar pnl_pontos × 0.20
   
2. Se o mercado está lateral:
   → Adicione filtro ADX > 20 nas entradas
   → Ou crie estratégia de reversão à média
   
3. Se tem poucos dados:
   → Consiga histórico maior (6+ meses)
   
4. Teste o robô baseline simples:
   → python robo_baseline.py
   → Compare com seus robôs atuais
   
5. Use walk-forward analysis:
   → Não confie em backtest único
   → Valide em períodos diferentes
""")
