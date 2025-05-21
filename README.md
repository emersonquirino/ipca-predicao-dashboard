# IPCA Dashboard com Predição

Este projeto apresenta um dashboard interativo para análise e visualização da performance de um modelo de Regressão Linear treinado para prever o IPCA.

## 📌 O que esta aplicação faz

A aplicação permite:

- Carregar um arquivo CSV com os valores reais e previstos do IPCA;
- Comparar visualmente os resultados com um gráfico de dispersão (real vs previsto);
- Calcular automaticamente os principais indicadores de desempenho do modelo;
- Avaliar a acurácia das previsões com base em uma margem de tolerância;
- Baixar o conjunto de dados completo com os erros calculados.

## 🧪 Métricas calculadas

A aplicação calcula e exibe:

- **Erro Quadrático Médio (MSE)**: penaliza mais os grandes erros;
- **Erro Absoluto Médio (MAE)**: média das diferenças absolutas;
- **Coeficiente de Determinação (R²)**: indica a proporção da variabilidade explicada;
- **Acurácia (Erro ≤ 0.1)**: percentual de previsões com erro absoluto até 0.1.

## 📈 Visualizações incluídas

- **Gráfico de Dispersão Real vs Previsto**: compara diretamente os valores reais e estimados pelo modelo.
- **Linha de tendência ideal (y = x)**: ajuda a identificar o quão próximo os pontos estão da previsão perfeita.

## 📂 Entrada esperada

O arquivo CSV deve conter ao menos duas colunas com os nomes:
- `Índice geral Real`
- `Índice geral Previsto`


## ✅ Resultados gerados

Ao processar o CSV, o app:
- Cria colunas auxiliares (`Erro`, `Erro Absoluto`, `Acerto`);
- Gera os gráficos e métricas de forma interativa;
- Permite baixar o CSV com os erros calculados.


---

Desenvolvido por Emerson Quirino.
