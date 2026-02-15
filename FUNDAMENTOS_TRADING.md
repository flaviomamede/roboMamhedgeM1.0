# Fundamentos de Trading — WIN, Custos e Expectativa

## 1. O que é 1 trade?

### Unidades

| Conceito | WIN (Mini-Índice) |
|----------|-------------------|
| **1 contrato** | 1 unidade mínima negociável |
| **1 ponto** | R$ 0,20 de variação por contrato |
| **Tick mínimo** | 5 pontos = R$ 1,00 por contrato |
| **Exemplo** | Movimento de 50 pontos = R$ 10 por contrato |

### Nosso backtest

O P&L no código está em **pontos** (BOVA11 × 1000, alinhado ao Ibovespa). Para converter em reais:

```
P&L em R$ = P&L_pontos × MULT_PONTOS_REAIS
```

Para BOVA11 em pontos: **1 ponto = R$ 0,001** (`MULT_PONTOS_REAIS = 0.001`). O `download_win.py` converte BOVA11 para pontos automaticamente; use `converter_csv_para_pontos.py` para arquivos antigos.

---

## 2. Como a B3 cobra?

### Tarifas B3 (day trade)

| Volume diário | Taxa sobre valor financeiro |
|---------------|----------------------------|
| Até R$ 1 mi   | ~0,023% |
| R$ 1–5 mi     | ~0,0225% |
| R$ 5–10 mi    | ~0,021% |
| Acima         | Redução progressiva |

- Cobrança: **negociação** + **liquidação** (comprador e vendedor)
- **1 trade** = 2 operações (entrada + saída) = 4 cobranças (2 compra, 2 venda)

### Corretora

- Corretagem: R$ 0 a R$ 20+ por operação, conforme plano
- Day trade: muitas corretoras cobram por operação ou por contrato

### Custo total aproximado por trade (1 contrato WIN)

- B3: ~R$ 0,50 a R$ 2,00 (depende do valor)
- Corretora: R$ 0 a R$ 5
- **Total:** ~R$ 1 a R$ 7 por round-trip (entrada + saída)

No `utils_fuso.py`: `CUSTO_POR_TRADE = 2500` pontos (= R$ 2,50).

---

## 3. O que outros traders fazem?

### Win rate

- **45–50%** é comum e considerado saudável
- Muitos profissionais operam com **50–55%**
- Steve Cohen: "Meu melhor trader ganha só 63% das vezes. A maioria ganha 50–55%."

### O que importa mais que win rate

- **Expectativa matemática:** `E = p×G - (1-p)×P` (p = prob. ganho, G = ganho médio, P = perda média)
- **Risk:Reward:** 1:2 ou 1:3 (perda menor que ganho)
- **Profit factor:** ganhos totais / perdas totais > 1,5

### Exemplo

- 45% win, ganho médio R$ 100, perda média R$ 50  
- E = 0,45×100 - 0,55×50 = 45 - 27,5 = **R$ 17,50 por trade**

---

## 4. Sua tese: menos trades, mais seletividade

> "Traders não ganham mais porque arriscam demais. Fazer menos trades, só quando a probabilidade de ganho for maior, e nesses trades aumentar a quantidade. Stop para baixo menor que stop para cima."

### Alinhamento com a teoria

| Princípio | Implementação |
|-----------|----------------|
| **Menos trades** | Filtros mais rígidos (EMA200, MACD, RSI > 50) |
| **Só alta probabilidade** | Entrar apenas quando vários indicadores concordam |
| **Aumentar aposta nos bons** | Posição maior nos trades selecionados |
| **Stop assimétrico** | Stop loss < take profit (ex.: 1×ATR stop, 2,5×ATR target) |

### Meta: 45–50% acerto, 5% em 2 meses

- Capital R$ 10.000 → 5% = R$ 500 em 2 meses
- Com ~20 trades em 2 meses: E[P&L] ≈ R$ 25 por trade
- Com 1 contrato WIN: ~125 pontos por trade em média
- Ou: menos trades (ex.: 10) com E[P&L] ≈ R$ 50 por trade

---

## 5. Resumo prático

| Pergunta | Resposta |
|----------|----------|
| **1 trade = ?** | 1 round-trip (entrada + saída) em 1 contrato |
| **1 contrato WIN** | 1 ponto = R$ 0,20 |
| **Custo B3 + corretora** | ~R$ 1–7 por trade (1 contrato) |
| **Win rate alvo** | 45–50% é realista |
| **Estratégia** | Menos trades, mais seletivos, stop menor que target |

---

## 6. Problema: EMA200 com 60 dias de dados

**EMA200 em 5min** = 200 barras = 16,7 horas (não 200 dias).

Com 60 dias de dados, EMA200 é tendência de ~1 dia útil. O filtro pode não ser adequado. **R8 foi alterado para EMA50** (4h), mais estável com poucos dados.

## 7. Diagnóstico de escala

Execute `python diagnostico_escala.py` para:
- Verificar período e quantidade de dados
- Comparar R6 vs R8 (com/sem filtro de tendência)
- Estimar conversão P&L → reais
- Checar se custo está subestimado

## 8. Próximos passos no código

1. **Escala real:** Calibrar `PONTOS_POR_UNIDADE` com dados reais da corretora
2. **Custo real:** Testar CUSTO_POR_TRADE = 0.1 a 0.5 (R$ 2-10 por trade)
3. **Posição:** Parâmetro `n_contratos` para simular tamanho da aposta
4. **Filtros:** EMA50 em vez de EMA200 para dados curtos
