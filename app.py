
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Previsão do IPCA com Regressão Linear Múltipla")

mes_map = {
    "jan": "01", "fev": "02", "mar": "03", "abr": "04",
    "mai": "05", "jun": "06", "jul": "07", "ago": "08",
    "set": "09", "out": "10", "nov": "11", "dez": "12"
}

arquivo = st.file_uploader("Faça upload do CSV com os dados do IPCA (formato igual ao IPCA_Base.csv)", type="csv")

if arquivo:
    df = pd.read_csv(arquivo, sep=";")
    df[['Mês', 'Ano']] = df['Mês_Ano'].str.split('/', expand=True)
    df['Mês'] = df['Mês'].map(mes_map)
    df['Ano'] = df['Ano'].apply(lambda x: f"{int(x):02d}")
    df['Mês_Ano_Numérico'] = (df['Mês'] + df['Ano']).astype(int)

    variaveis_ipca = [
        'Alimentação e bebidas', 'Habitação', 'Artigos de residência', 'Vestuário',
        'Transportes', 'Saúde e cuidados pessoais', 'Despesas pessoais', 'Educação', 'Comunicação'
    ]

    df.dropna(subset=['Mês_Ano_Numérico', 'Índice geral'] + variaveis_ipca, inplace=True)

    X = df[['Mês_Ano_Numérico'] + variaveis_ipca]
    y = df['Índice geral']

    modelo = LinearRegression()
    modelo.fit(X, y)

    y_pred = modelo.predict(X)
    df['Previsão IPCA'] = y_pred

    st.subheader("Comparativo entre Valor Real e Previsto")
    st.dataframe(df[['Mês_Ano', 'Índice geral', 'Previsão IPCA']])

    st.subheader("Gráfico: Valor Real vs Previsto")
    plt.figure(figsize=(10, 5))
    plt.plot(df['Mês_Ano'], df['Índice geral'], marker='o', label='Real')
    plt.plot(df['Mês_Ano'], df['Previsão IPCA'], marker='x', label='Previsto')
    plt.xticks(rotation=45)
    plt.xlabel("Mês/Ano")
    plt.ylabel("IPCA")
    plt.legend()
    st.pyplot(plt)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.markdown("### Métricas do Modelo")
    st.write(f"**Erro Quadrático Médio (MSE):** {mse:.4f}")
    st.write(f"**Erro Absoluto Médio (MAE):** {mae:.4f}")
    st.write(f"**Coeficiente de Determinação (R²):** {r2:.4f}")
