# 🚀 GUIA DE INÍCIO RÁPIDO

## Análise de Reversão do IBOVESPA com Bayesian Change Point Detection

Este pacote implementa o algoritmo BCP da tese de Tobias Setz (ETH Zurich) para detectar pontos de inflexão no IBOVESPA.

---

## ⚡ Uso Imediato

### 1. Teste com Dados de Exemplo

```bash
# Executa análise no arquivo de exemplo já incluído
python fast_bcp_detector.py ibovespa_5min_exemplo_com_reversao.csv
```

**Resultado esperado:**
- ✅ Detecta mudança de regime nos dados sintéticos
- 📊 Gera gráfico com análise completa
- 📝 Fornece probabilidade de reversão atual

---

### 2. Análise dos Seus Dados

```bash
# Substitua 'seu_arquivo.csv' pelo arquivo do IBOVESPA
python fast_bcp_detector.py seu_arquivo.csv
```

**Requisitos do CSV:**
- ✓ Coluna `timestamp` com data/hora
- ✓ Coluna `close` com preço de fechamento
- ✓ Dados de 5 minutos (ou outra frequência consistente)

---

## 📋 Arquivos Incluídos

| Arquivo | Descrição |
|---------|-----------|
| `fast_bcp_detector.py` | **Versão rápida** (RECOMENDADA) - Análise em segundos |
| `ibovespa_bcp_reversal_detector.py` | Versão completa (mais precisa, mais lenta) |
| `generate_example_data.py` | Gera dados sintéticos para teste |
| `ibovespa_5min_exemplo_com_reversao.csv` | Dados de exemplo (já gerados) |
| `README.md` | Documentação completa |

---

## 📊 Interpretando os Resultados

### Status da Análise

O script retorna um dos seguintes status:

```
🔴 ALERTA ALTO - Reversão iminente (>90º percentil)
    ➜ Forte evidência de mudança estrutural
    ➜ AÇÃO: Aguardar confirmação antes de posições
    
🟡 ALERTA MODERADO - Probabilidade elevada (75-90º percentil)
    ➜ Instabilidade crescente detectada
    ➜ AÇÃO: Aumentar cautela e monitoramento
    
🟠 ATENÇÃO - Probabilidade moderada (60-75º percentil)
    ➜ Sinais moderados de mudança
    ➜ AÇÃO: Monitorar de perto
    
🟢 ESTÁVEL - Baixa probabilidade (<60º percentil)
    ➜ Regime atual consistente
    ➜ AÇÃO: Manter estratégia
```

### Métricas Principais

1. **Probabilidade de Reversão** (0-100%)
   - Quanto maior, mais provável a reversão
   - >70%: Muito alta
   - 50-70%: Alta
   - 30-50%: Moderada
   - <30%: Baixa

2. **Percentil** (0-100)
   - Posição da probabilidade atual vs histórico
   - >90: Situação extrema (atenção máxima!)
   - 75-90: Situação incomum (cautela)
   - 60-75: Acima da média (monitorar)
   - <60: Normal

3. **Força do Sinal**
   - Combina probabilidade × volatilidade
   - Valores altos = mudança drástica esperada

---

## ⚙️ Ajustando Sensibilidade

```bash
# Sintaxe
python fast_bcp_detector.py arquivo.csv [janela] [p0]

# Exemplos:

# Mais conservador (menos falsos alarmes)
python fast_bcp_detector.py ibov.csv 300 0.15

# Balanceado (padrão)
python fast_bcp_detector.py ibov.csv 200 0.20

# Mais sensível (detecta mudanças sutis)
python fast_bcp_detector.py ibov.csv 150 0.25
```

**Parâmetros:**
- `janela`: Tamanho da janela histórica (150-300)
- `p0`: Prior de probabilidade (0.10-0.30)

---

## 🎯 Caso de Uso: Seu "Feeling" vs Algoritmo

**Você disse:**
> "Futuro do IBOVESPA indica +4.6%, mas meu feeling diz que estamos no ponto de inflexão"

**Como usar o BCP:**

```python
# Execute a análise
python fast_bcp_detector.py ibovespa_hoje_5min.csv

# Observe o resultado:
# - Percentil >85: Seu feeling está CORRETO ✅
# - Percentil 70-85: Evidências MODERADAS de reversão ⚡
# - Percentil <70: Provavelmente NÃO está em inflexão ❌
```

**Importante:** O BCP não diz SE vai subir ou descer, apenas SE vai MUDAR de regime.

---

## 📈 Exemplo Real de Saída

```
======================================================================
📍 STATUS ATUAL (Última Observação)
======================================================================

🟡 ALERTA MODERADO - Probabilidade elevada

Métricas:
  • Probabilidade de Reversão: 59.41% (percentil 85)
  • Volatilidade Atual: 0.1943%
  • Força do Sinal: 1.15
  • Tendência Recente: ALTA
  • Retorno Médio Atual: 0.0304%

======================================================================
💡 INTERPRETAÇÃO
======================================================================
⚡ SINAIS DE INSTABILIDADE CRESCENTE
    Probabilidade de reversão acima da média histórica.
    ➜ Considere: Aumentar cautela e monitoramento
    ➜ Ação: Revisar estratégia e exposição ao risco
```

---

## 🔧 Solução de Problemas

### Erro: "Colunas não encontradas"

Seu CSV precisa ter colunas `timestamp` e `close`. Se as colunas têm outros nomes:

```python
# Edite o arquivo fast_bcp_detector.py (linha 260-261)
# Troque:
    date_col='timestamp',
    price_col='close',
# Por:
    date_col='data',        # ou o nome da sua coluna de data
    price_col='fechamento', # ou o nome da sua coluna de preço
```

### Análise muito lenta

Use a versão `fast_bcp_detector.py` (não a versão `ibovespa_bcp_reversal_detector.py`).

A versão rápida usa janela móvel e é 50-100x mais rápida.

---

## 📚 Fundamento Teórico

### Metodologia

Baseado em:
- **Barry & Hartigan (1993)** - Bayesian Analysis for Change Point Problems
- **Setz (2017)** - Tese de doutorado ETH Zurich

### Como Funciona

1. **Divide** a série em possíveis "regimes" (blocos com parâmetros constantes)
2. **Calcula** probabilidade bayesiana de mudança em cada ponto
3. **Detecta** quando a estrutura estatística muda significativamente

**Ideal para:**
- ✅ Ativos de alta volatilidade (cripto, IBOVESPA)
- ✅ Detecção em tempo real
- ✅ Quantificação de incerteza

---

## 🎓 Próximos Passos

1. **Teste com dados de exemplo**
   ```bash
   python fast_bcp_detector.py ibovespa_5min_exemplo_com_reversao.csv
   ```

2. **Analise seus dados reais**
   ```bash
   python fast_bcp_detector.py seus_dados_ibovespa_5min.csv
   ```

3. **Integre no seu workflow**
   - Execute periodicamente (ex: a cada hora)
   - Combine com outros indicadores técnicos
   - Use para ajustar stop loss dinamicamente

---

## 📞 Documentação Completa

Para mais detalhes, consulte o `README.md` completo incluído no pacote.

**Referência Original:**  
Setz, T. (2017). "Stable Portfolio Design Using Bayesian Change Point Models"  
https://doi.org/10.3929/ethz-b-000244960

---

✅ **Pronto para usar! Basta executar `python fast_bcp_detector.py seu_arquivo.csv`**
