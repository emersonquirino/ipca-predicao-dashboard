import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Previsão do IPCA com Regressão Linear")

uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")

    # Carrega o modelo treinado
    model = joblib.load("modelo_regressao_linear.pkl")

    try:
        # Define as variáveis preditoras (X) e o alvo (y)
        X = df.drop(columns=["Índice geral", "Mês_Ano"])
        y = df["Índice geral"]

        # Faz as previsões
        previsoes = model.predict(X)
        df["Previsão IPCA"] = previsoes

        # Exibe os resultados
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["Mês_Ano", "Índice geral", "Previsão IPCA"]].head())

        # Plota o gráfico
        st.subheader("Gráfico: Valor Real vs Previsto")
        fig, ax = plt.subplots()
        ax.plot(df["Mês_Ano"], df["Índice geral"], label="Real", marker='o')
        ax.plot(df["Mês_Ano"], df["Previsão IPCA"], label="Previsto", marker='x')
        ax.legend()
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"Erro: coluna ausente no arquivo - {e}")
