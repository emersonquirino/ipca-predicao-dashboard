import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Previsão do IPCA com Regressão Linear")

# Descrição introdutória
st.markdown("""
### 📈 Sobre esta aplicação

Este painel interativo permite a visualização dos resultados de um modelo de regressão linear aplicado sobre o IPCA.

Você poderá:
- Comparar valores **reais** e **previstos** do IPCA;
- Analisar **distribuição dos erros**;
- Consultar **métricas de desempenho** do modelo;
- **Baixar** o resultado completo com os cálculos de erro.
""")

# Upload do arquivo CSV com resultados prontos
uploaded_file = st.file_uploader("Escolha o arquivo CSV com os dados de previsão", type="csv")

if uploaded_file is not None:
    try:
        # Leitura do CSV
        df = pd.read_csv(uploaded_file, sep=";")

        # Renomeando colunas, se necessário
        df.columns = df.columns.str.strip()

        # Cálculo das colunas auxiliares
        df["Erro Absoluto"] = abs(df["Índice geral Real"] - df["Índice geral Previsto"])
        df["Erro"] = df["Índice geral Real"] - df["Índice geral Previsto"]
        df["Acerto"] = df["Erro Absoluto"] <= 0.1

        # Exibe os dados
        st.subheader("Resultados da Previsão")
        st.dataframe(df.drop(columns=["Período"], errors="ignore"))

        # Gráfico: Real vs Previsto (scatter)
        st.subheader("Gráfico: Comparativo Real vs Previsto")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df["Índice geral Real"], df["Índice geral Previsto"], color='purple', label="Pontos de previsão")
        ax.plot([df["Índice geral Real"].min(), df["Índice geral Real"].max()],
                [df["Índice geral Real"].min(), df["Índice geral Real"].max()],
                color='gray', linestyle='--', label="Ideal (y = x)")
        ax.set_xlabel("Índice Geral Real")
        ax.set_ylabel("Índice Geral Previsto")
        ax.set_title("Dispersão entre valores Reais e Previstos")
        ax.legend()
        st.pyplot(fig)

        # Boxplot dos erros
        st.subheader("Distribuição dos Erros de Previsão")
        fig2, ax2 = plt.subplots()
        ax2.boxplot(df["Erro"])
        ax2.set_title("Boxplot dos Erros (Real - Previsto)")
        ax2.set_ylabel("Erro")
        st.pyplot(fig2)

        # Métricas
        y_true = df["Índice geral Real"]
        y_pred = df["Índice geral Previsto"]
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        st.subheader("Métricas do Modelo")
        st.write(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        st.write(f"Erro Absoluto Médio (MAE): {mae:.4f}")
        st.write(f"Coeficiente de Determinação (R²): {r2:.4f}")

        # Métrica adicional: acurácia dentro de faixa de tolerância
        acuracia = df["Acerto"].mean() * 100
        st.write(f"Acurácia com erro ≤ 0.1: {acuracia:.2f}%")

        # Botão de download do CSV com colunas extras
        csv = df.to_csv(index=False, sep=";").encode('utf-8-sig')
        st.download_button("Baixar resultados com erros calculados", csv, file_name="resultado_completo.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
