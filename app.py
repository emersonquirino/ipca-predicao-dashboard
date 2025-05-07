import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

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
            try:
                mes, ano = mes_ano.split("/")
                mes = mapa_meses.get(mes.lower())
                if mes:  # Verifica se o mês existe no mapeamento
                    return f"{mes}/20{ano}" if len(ano) == 2 else f"{mes}/{ano}"
                else:
                    raise ValueError(f"Mês '{mes}' não reconhecido")
            except Exception as e:
                raise ValueError(f"Erro ao converter a data: {e}")

        df["Mês_Ano_Convertido"] = df["Mês_Ano"].apply(converter_data)
        df["Mês_Ano_Numérico"] = pd.to_datetime(df["Mês_Ano_Convertido"], format="%m/%Y").dt.strftime("%Y%m").astype(int)

        # Carrega o modelo treinado
        model = joblib.load("modelo_regressao_linear.pkl")

        # Alinhar a ordem das colunas conforme o modelo foi treinado
        colunas_esperadas = model.feature_names_in_  # Recupera as colunas com as quais o modelo foi treinado
        X = df[colunas_esperadas]  # Assegura que as colunas de entrada estão na ordem correta

        # Separa a variável alvo
        y = df["Índice geral"]

        # Faz as previsões
        previsoes = model.predict(X)
        df["Previsão IPCA"] = previsoes

        # Exibe os resultados
        st.subheader("Resultados da Previsão")
        st.dataframe(df[["Mês_Ano", "Índice geral", "Previsão IPCA"]])

        # Gráfico comparando valores reais e previstos
        st.subheader("Gráfico: Valor Real vs Previsto")
        fig, ax = plt.subplots(figsize=(10,6))  # Aumenta o tamanho do gráfico para melhorar a visualização
        ax.plot(df["Mês_Ano"], df["Índice geral"], label="Real", marker='o', linestyle='-', color='b')
        ax.plot(df["Mês_Ano"], df["Previsão IPCA"], label="Previsto", marker='x', linestyle='--', color='orange')
        ax.set_xlabel("Mês/Ano")  # Rótulo do eixo X
        ax.set_ylabel("IPCA")  # Rótulo do eixo Y
        ax.set_title("Comparação entre Valor Real e Previsto")  # Título do gráfico
        ax.legend()
        plt.xticks(rotation=45)  # Gira as labels do eixo X para melhor leitura
        st.pyplot(fig)

        # Previsões futuras (para 12 meses)
        st.subheader("Previsões para o Futuro (próximos 12 meses)")
        # Vamos fazer previsões para os próximos 12 meses
        futuros_periodos = pd.date_range(df["Mês_Ano"].max(), periods=13, freq="M")  # Frequência mensal
        futuros_periodos = futuros_periodos[-12:].strftime("%Y%m").astype(int)  # Últimos 12 meses para previsão

        # Criando as variáveis de entrada para as previsões futuras (usando as médias das variáveis)
        previsoes_futuras = model.predict(np.tile(X.mean(), (12, 1)))  # Usando médias para as previsões

        # Visualizar as previsões futuras no gráfico
        fig, ax = plt.subplots(figsize=(10,6))  # Aumenta o tamanho do gráfico
        ax.plot(df["Mês_Ano"], df["Índice geral"], label="Real", marker='o', linestyle='-', color='b')
        ax.plot(df["Mês_Ano"], df["Previsão IPCA"], label="Previsto", marker='x', linestyle='--', color='orange')

        # Adicionando as previsões futuras
        ax.plot(futuros_periodos, previsoes_futuras, label="Previsão Futuro", marker='x', linestyle='--', color='green')
        ax.set_xlabel("Mês/Ano")  # Rótulo do eixo X
        ax.set_ylabel("IPCA")  # Rótulo do eixo Y
        ax.set_title("Comparação entre Valor Real, Previsto e Futuro")  # Título do gráfico
        ax.legend()
        plt.xticks(rotation=45)  # Gira as labels do eixo X para melhor leitura
        st.pyplot(fig)

        # Cálculo das métricas do modelo
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y, previsoes)
        mae = mean_absolute_error(y, previsoes)
        r2 = r2_score(y, previsoes)

        # Exibindo as métricas
        st.subheader("Métricas do Modelo")
        st.write(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        st.write(f"Erro Absoluto Médio (MAE): {mae:.4f}")
        st.write(f"Coeficiente de Determinação (R²): {r2:.4f}")

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
