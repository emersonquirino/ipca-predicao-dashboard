import streamlit as st
import pandas as pd
import joblib

st.title("Previsão do IPCA com Regressão Linear")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    # Leitura do CSV
    df = pd.read_csv(uploaded_file, sep=";")

    # Carregamento do modelo treinado
    model = joblib.load("modelo_regressao_linear.pkl")

    try:
        # Definição das variáveis de entrada (X) e saída (y)
        X = df.drop(columns=["Indice Geral", "Mes_Ano"])
        y = df["Indice Geral"]

        # Geração das previsões
        previsoes = model.predict(X)

        # Inserção das previsões no dataframe
        df["Previsão IPCA"] = previsoes

        # Exibição do dataframe com resultados
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["Mes_Ano", "Indice Geral", "Previsão IPCA"]].head())

        # Gráfico
        st.subheader("Gráfico: Valor Real vs Previsto")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(df["Mes_Ano"], df["Indice Geral"], label="Real", marker='o')
        ax.plot(df["Mes_Ano"], df["Previsão IPCA"], label="Previsto", marker='x')
        ax.legend()
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"Erro: coluna ausente no arquivo - {e}")
