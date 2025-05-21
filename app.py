import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Previsão do IPCA com Regressão Linear")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    try:
        # Leitura do CSV
        df = pd.read_csv(uploaded_file, sep=";")

        # Padroniza nomes das colunas (remove espaços extras)
        df.columns = df.columns.str.strip()

        # Exibe os resultados sem a coluna de período
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["Índice geral Real", "Índice geral Previsto"]])

        # Gráfico comparando valores reais e previstos (sem período no eixo)
        st.subheader("Gráfico: Valor Real vs Previsto")
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(df))
        ax.plot(x, df["Índice geral Real"], label="Real", marker='o', linestyle='-', color='b')
        ax.plot(x, df["Índice geral Previsto"], label="Previsto", marker='x', linestyle='--', color='orange')
        ax.set_xlabel("Observações (ordem no arquivo)", fontsize=12)
        ax.set_ylabel("IPCA", fontsize=12)
        ax.set_title("Comparação entre Valor Real e Previsto", fontsize=14)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
