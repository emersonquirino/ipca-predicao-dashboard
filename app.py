import streamlit as st
import pandas as pd
import joblib

st.title("Previsão do IPCA com Regressão Linear")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    # Leitura do CSV
    df = pd.read_csv(uploaded_file, sep=";")

    # Normalização dos nomes das colunas
    df.columns = df.columns.str.strip().str.lower()

    # Carregamento do modelo treinado
    model = joblib.load("modelo_regressao_linear.pkl")

    # Definição das variáveis de entrada (X) e saída (y)
    try:
        X = df.drop(columns=["índice geral", "mês_ano"])
        y = df["índice geral"]

        # Geração das previsões
        previsoes = model.predict(X)

        # Inserção das previsões no dataframe
        df["previsão ipca"] = previsoes

        # Exibição do dataframe com resultados
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["mês_ano", "índice geral", "previsão ipca"]].head(10))

        # Gráfico
        st.subheader("Gráfico: Valor Real vs Previsto")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(df["mês_ano"], df["índice geral"], label="Real", marker='o')
        ax.plot(df["mês_ano"], df["previsão ipca"], label="Previsto", marker='x')
        ax.legend()
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"Erro ao acessar as colunas esperadas no arquivo: {e}")
        
print("Atualização forçada")
