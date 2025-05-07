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
        df["Mês_Ano_Numérico"] = pd.to_datetime(df["Mês_Ano_Convertido"], format="%m/%Y")

        # Adicionando o filtro de data
        st.subheader("Filtro de Data")
        data_inicio = st.date_input("Data de Início", df["Mês_Ano_Numérico"].min().date())
        data_fim = st.date_input("Data de Fim", df["Mês_Ano_Numérico"].max().date())

        # Filtra os dados de acordo com o intervalo de datas
        df_filtrado = df[(df["Mês_Ano_Numérico"] >= pd.to_datetime(data_inicio)) & (df["Mês_Ano_Numérico"] <= pd.to_datetime(data_fim))]

        # Carrega o modelo treinado
        model = joblib.load("modelo_regressao_linear.pkl")

        # Alinhar a ordem das colunas conforme o modelo foi treinado
        colunas_esperadas = model.feature_names_in_  # Recupera as colunas com as quais o modelo foi treinado
        X = df_filtrado[colunas_esperadas]  # Assegura que as colunas de entrada estão na ordem correta

        # Separa a variável alvo
        y = df_filtrado["Índice geral"]

        # Faz as previsões
        previsoes = model.predict(X)
        df_filtrado["Previsão IPCA"] = previsoes

        # Exibe os resultados
        st.subheader("Resultados da Previsão")
        st.dataframe(df_filtrado[["Mês_Ano", "Índice geral", "Previsão IPCA"]])

        # Gráfico comparando valores reais e previstos
        st.subheader("Gráfico: Valor Real vs Previsto")
        fig, ax = plt.subplots(figsize=(12,6))  # Aumenta o tamanho do gráfico para melhorar a visualização
        ax.plot(df_filtrado["Mês_Ano"], df_filtrado["Índice geral"], label="Real", marker='o', linestyle='-', color='b')
        ax.plot(df_filtrado["Mês_Ano"], df_filtrado["Previsão IPCA"], label="Previsto", marker='x', linestyle='--', color='orange')
        ax.set_xlabel("Mês/Ano", fontsize=12)  # Tamanho do texto no eixo X
        ax.set_ylabel("IPCA", fontsize=12)  # Tamanho do texto no eixo Y
        ax.set_title("Comparação entre Valor Real e Previsto", fontsize=14)  # Título maior
        ax.legend()

        # Ajustando as labels no eixo X
        plt.xticks(rotation=45, ha="right", fontsize=10)  # Gira as labels e ajusta o alinhamento
        plt.tight_layout()  # Ajusta o layout para não cortar labels
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
