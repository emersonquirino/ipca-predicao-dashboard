import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Previsão do IPCA com Regressão Linear")

# Upload do arquivo CSV com resultados prontos
uploaded_file = st.file_uploader("Escolha o arquivo CSV com os dados de previsão", type="csv")

if uploaded_file is not None:
    try:
        # Leitura do CSV
        df = pd.read_csv(uploaded_file, sep=";")
        df.columns = df.columns.str.strip()  # remove espaços extras

        # Exibe os dados sem colunas auxiliares
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["Índice geral Real", "Índice geral Previsto"]])

        # Cálculo das métricas
        y_true = df["Índice geral Real"]
        y_pred = df["Índice geral Previsto"]
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Treina uma regressão linear para a linha de tendência
        reg = LinearRegression()
        reg.fit(np.array(y_true).reshape(-1, 1), y_pred)

        # Geração da linha de tendência
        x_range = np.linspace(y_true.min(), y_true.max(), 100).reshape(-1, 1)
        y_trend = reg.predict(x_range)

        # Gráfico de dispersão com linha ideal e linha de tendência
        st.subheader("Gráfico: Real vs Previsto (Dispersão)")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, color='purple', alpha=0.7, label="Pontos")
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                color='gray', linestyle='--', label="Ideal (y = x)")
        ax.plot(x_range, y_trend, color='red', linestyle='-', label="Tendência (regressão)")
        ax.set_xlabel("Índice Geral Real")
        ax.set_ylabel("Índice Geral Previsto")
        ax.set_title(f"Dispersão entre Reais e Previstos (R² = {r2:.4f})")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # Métricas
        st.subheader("Métricas do Modelo")
        st.write(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        st.write(f"Erro Absoluto Médio (MAE): {mae:.4f}")
        st.write(f"Coeficiente de Determinação (R²): {r2:.4f}")

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
