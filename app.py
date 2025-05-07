import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Previsão do IPCA com Regressão Linear")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    try:
        # Leitura do CSV
        df = pd.read_csv(uploaded_file, sep=";")

        # Conversão da coluna de data
        if 'Mês_Ano' in df.columns:
            # Corrige abreviações com mês em português
            df['Mês_Ano_Numérico'] = pd.to_datetime(df['Mês_Ano'], format="%b/%y").dt.strftime('%Y%m').astype(int)
        else:
            st.error("A coluna 'Mês_Ano' não foi encontrada no arquivo.")
            st.stop()

        # Carrega o modelo treinado
        model = joblib.load("modelo_regressao_linear.pkl")

        # Separa variáveis preditoras (X) e alvo (y)
        X = df.drop(columns=["Índice geral", "Mês_Ano"])
        y = df["Índice geral"]

        # Faz as previsões
        previsoes = model.predict(X)
        df["Previsão IPCA"] = previsoes

        # Exibe os resultados
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["Mês_Ano", "Índice geral", "Previsão IPCA"]])

        # Gráfico comparando valores reais e previstos
        st.subheader("Gráfico: Valor Real vs Previsto")
        fig, ax = plt.subplots()
        ax.plot(df["Mês_Ano"], df["Índice geral"], label="Real", marker='o')
        ax.plot(df["Mês_Ano"], df["Previsão IPCA"], label="Previsto", marker='x')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
