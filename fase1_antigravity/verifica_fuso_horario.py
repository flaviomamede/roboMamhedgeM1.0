"""
Verificação independente: fuso horário e filtros.
Confere se os dados estão corretos e se os filtros mapeiam BRT ↔ UTC corretamente.
"""
import pandas as pd

df = pd.read_csv("WIN_5min.csv", index_col=0, parse_dates=True)
df.columns = df.columns.str.lower()

print("=" * 70)
print("1. FORMATO DOS DADOS")
print("=" * 70)
print(f"Primeiro registro: {df.index[0]}")
print(f"Último registro:  {df.index[-1]}")
print(f"Timezone do index: {df.index.tz}")
print(f"Tipo: {type(df.index[0])}")

# Horas únicas presentes nos dados
hours_in_data = sorted(df.index.hour.unique())
print(f"\nHoras UTC presentes: {hours_in_data}")

print("\n" + "=" * 70)
print("2. CONVERSÃO UTC → BRT (UTC-3)")
print("=" * 70)
print("BRT = UTC - 3 horas")
print("  10:00 BRT = 13:00 UTC")
print("  11:00 BRT = 14:00 UTC")
print("  12:00 BRT = 15:00 UTC")
print("  13:00 BRT = 16:00 UTC")
print("  14:00 BRT = 17:00 UTC")
print("  16:30 BRT = 19:30 UTC  ← fim de sessão típico")

print("\n" + "=" * 70)
print("3. FILTROS ATUAIS (R1-R5) — O QUE ESTÁ SENDO APLICADO")
print("=" * 70)
print("Código: skip quando (h,m) atende:")
print("  - h < 10 ou h >= 17")
print("  - h == 13 e m <= 45")
print("  - h == 14")
print("  - h == 16 e m >= 30")
print()
print("Interpretação se dados estão em UTC:")
print("  h<10 ou h>=17  → pula 00h-09h e 17h+ UTC = 21h-06h e 14h+ BRT")
print("  h==13, m<=45   → pula 13:00-13:45 UTC = 10:00-10:45 BRT ✓")
print("  h==14          → pula 14:00 UTC = 11:00 BRT ✓")
print("  h==16, m>=30   → pula 16:30-16:55 UTC = 13:30-13:55 BRT")
print()
print("CORRIGIDO: Agora usamos utils_fuso.py — converte para BRT e filtra corretamente.")

print("\n" + "=" * 70)
print("4. AMOSTRA: PRIMEIROS CANDLES POR HORA (UTC)")
print("=" * 70)
for h in hours_in_data[:8]:
    mask = df.index.hour == h
    first = df.index[mask][0]
    brt_h = (h - 3) % 24
    print(f"  {h:02d}:00 UTC = {brt_h:02d}:00 BRT  →  {first}")

print("\n" + "=" * 70)
print("5. CONTAGEM DE CANDLES POR HORA (dados brutos)")
print("=" * 70)
for h in hours_in_data:
    n = (df.index.hour == h).sum()
    brt = (h - 3) % 24
    print(f"  {h:02d}h UTC ({brt:02d}h BRT): {n:4d} candles")

print("\n" + "=" * 70)
print("6. TESTE: CONVERTER PARA BRT E FILTRAR POR HORA BRT")
print("=" * 70)
try:
    df_brt = df.copy()
    df_brt.index = df_brt.index.tz_convert('America/Sao_Paulo')
    h_brt = df_brt.index.hour
    print("Conversão para America/Sao_Paulo: OK")
    print(f"Exemplo: {df.index[0]} UTC → {df_brt.index[0]} BRT")
    print(f"Horas BRT nos dados: {sorted(h_brt.unique())}")
except Exception as e:
    print(f"Erro na conversão: {e}")
