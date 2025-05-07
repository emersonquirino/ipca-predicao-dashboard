import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Previsão do IPCA com Regressão Linear")

uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")

    # Normaliza os nomes das colunas para garantir compatibilidade
    df.columns = df.columns.str.strip()

    # Garante que a coluna usada no modelo exista
    if 'Mês_Ano' in df.columns:
        df = df.rename(columns={'Mês_Ano': 'Mês_Ano_Numérico'})

    try:
        model = joblib.load("modelo_regressao_linear.pkl")

        # Separa variáveis preditoras e target
        X = df.drop(columns=["Índice geral"])
        y = df["Índice geral"]

        # Previsões
        previsoes = model.predict(X)
        df["Previsão IPCA"] = previsoes

        # Exibição da tabela
        st.subheader("Resultados da Previsão")
        st.dataframe(df[['Mês_Ano_Numérico', 'Índice geral', 'Previsão IPCA']])

        # Gráfico comparativo
        st.subheader("Gráfico: Valor Real vs Previsto")
        fig, ax = plt.subplots()
        ax.plot(df["Mês_Ano_Numérico"], df["Índice geral"], label="Real", marker='o')
        ax.plot(df["Mês_Ano_Numérico"], df["Previsão IPCA"], label="Previsto", marker='x')
        ax.legend()
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"Erro: coluna ausente no arquivo - {e}")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
