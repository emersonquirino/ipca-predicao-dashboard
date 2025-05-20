import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Previsão do IPCA com Regressão Linear")

# Upload do arquivo CSV com resultados prontos
uploaded_file = st.file_uploader("Escolha o arquivo CSV com os dados de previsão", type="csv")

if uploaded_file is not None:
    try:
        # Leitura do CSV
        df = pd.read_csv(uploaded_file, sep=";")

        # Criação do eixo de tempo fictício baseado na ordem dos dados
        df["Período"] = [f"P{str(i+1).zfill(2)}" for i in range(len(df))]

        # Renomeando colunas, se necessário
        df.columns = df.columns.str.strip()

        # Exibe os dados
        st.subheader("Resultados da Previsão")
        st.dataframe(df)

        # Gráfico comparando valores reais e previstos
        st.subheader("Gráfico: Valor Real vs Previsto")
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df["Período"], df["Índice geral Real"], label="Real", marker='o', linestyle='-', color='blue')
        ax.plot(df["Período"], df["Índice geral Previsto"], label="Previsto", marker='x', linestyle='--', color='orange')
        ax.set_xlabel("Período")
        ax.set_ylabel("IPCA")
        ax.set_title("Comparação entre Valor Real e Previsto")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

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

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
