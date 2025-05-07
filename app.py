import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Previsão do IPCA com Regressão Linear")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    # Leitura do CSV
    df = pd.read_csv(uploaded_file, sep=";")

    # Padroniza nomes das colunas para bater com os do modelo treinado
    df.columns = [col.strip().title() for col in df.columns]

    # Carregamento do modelo treinado
    model = joblib.load("modelo_regressao_linear.pkl")

    # Definição das variáveis de entrada (X) e saída (y)
    try:
        X = df.drop(columns=["Indice Geral", "Mes_Ano"])
        y = df["Indice Geral"]

        # Geração das previsões
        previsoes = model.predict(X)

        # Inserção das previsões no dataframe
        df["Previsão IPCA"] = previsoes

        # Exibição do dataframe com resultados
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["Mes_Ano", "Indice Geral", "Previsão IPCA"]].head(10))

        # Gráfico comparativo
        st.subheader("Gráfico: Valor Real vs Previsto")
        fig, ax = plt.subplots()
        ax.plot(df["Mes_Ano"], df["Indice Geral"], label="Real", marker='o')
        ax.plot(df["Mes_Ano"], df["Previsão IPCA"], label="Previsto", marker='x')
        ax.set_xlabel("Mês/Ano")
        ax.set_ylabel("IPCA")
        ax.legend()
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"Erro ao acessar as colunas esperadas no arquivo: {e}")
