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

    # Carregamento do modelo treinado
    model = joblib.load("modelo_regressao_linear.pkl")

    try:
        # Colunas utilizadas no modelo treinado
        variaveis_ipca = [
            'Alimentação e bebidas', 'Habitação', 'Artigos de residência', 'Vestuário',
            'Transportes', 'Saúde e cuidados pessoais', 'Despesas pessoais', 'Educação', 'Comunicação'
        ]

        # Variável de entrada (X) e saída (y)
        X = df[['Mês_Ano_Numérico'] + variaveis_ipca]
        y = df['Índice geral']

        # Previsão
        previsoes = model.predict(X)
        df["Previsão IPCA"] = previsoes

        # Exibição dos dados
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["Mês_Ano", "Índice geral", "Previsão IPCA"]].head())

        # Gráfico
        st.subheader("Gráfico: Valor Real vs Previsto")
        fig, ax = plt.subplots()
        ax.plot(df["Mês_Ano"], df["Índice geral"], label="Real", marker='o')
        ax.plot(df["Mês_Ano"], df["Previsão IPCA"], label="Previsto", marker='x')
        ax.legend()
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"Erro: coluna ausente no arquivo - {e}")
