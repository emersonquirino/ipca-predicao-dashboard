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
        df.columns = df.columns.str.strip()  # remove espaços extras nos nomes de colunas

        # Exibe os dados (sem coluna extra)
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["Índice geral Real", "Índice geral Previsto"]])

        # Gráfico de dispersão comparando valores reais vs previstos
        st.subheader("Gráfico: Real vs Previsto (Dispersão)")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["Índice geral Real"], df["Índice geral Previsto"], color='purple', alpha=0.7)
        ax.plot([df["Índice geral Real"].min(), df["Índice geral Real"].max()],
                [df["Índice geral Real"].min(), df["Índice geral Real"].max()],
                color='gray', linestyle='--', label="Linha Ideal (y = x)")
        ax.set_xlabel("Índice Geral Real")
        ax.set_ylabel("Índice Geral Previsto")
        ax.set_title("Dispersão entre valores Reais e Previsto")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # Métricas do modelo
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
