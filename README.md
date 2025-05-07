# Dashboard de Previsão do IPCA

Este aplicativo usa um modelo de regressão linear múltipla para prever o índice geral do IPCA com base em dados do IBGE.

## Como usar

1. Faça upload de um arquivo CSV com os dados no formato da base IPCA.
2. Veja as previsões do modelo e compare com os dados reais.
3. Visualize as métricas de desempenho do modelo.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```