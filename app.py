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

        # Mapeamento manual dos meses em português
        mapa_meses = {
            "jan": "01", "fev": "02", "mar": "03", "abr": "04", "mai": "05", "jun": "06",
            "jul": "07", "ago": "08", "set": "09", "out": "10", "nov": "11", "dez": "12"
        }

        # Converte "jan/20" em "01/2020"
        def converter_data(mes_ano):
            mes, ano = mes_ano.split("/")
            return f"{mapa_meses[mes.lower()]}/20{ano}" if len(ano) == 2 else f"{mapa_meses[mes.lower()]}/{ano}"

        df["Mês_Ano_Convertido"] = df["Mês_Ano"].apply(converter_data)
        df["Mês_Ano_Numérico"] = pd.to_datetime(df["Mês_Ano_Convertido"], format="%m/%Y").dt.strftime("%Y%m").astype(int)

        # Carrega o modelo treinado
        model = joblib.load("modelo_regressao_linear.pkl")

        # Separa variáveis preditoras (X) e alvo (y)
        X = df.drop(columns=["Índice geral", "Mês_Ano", "Mês_Ano_Convertido"])
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
