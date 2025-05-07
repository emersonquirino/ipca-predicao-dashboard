import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Previsão do IPCA com Regressão Linear")

uploaded_file = st.file_uploader("Escolha o arquivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")

    # Mapear meses abreviados para números
    mes_map = {
        "jan": "01", "fev": "02", "mar": "03", "abr": "04",
        "mai": "05", "jun": "06", "jul": "07", "ago": "08",
        "set": "09", "out": "10", "nov": "11", "dez": "12"
    }

    # Separar e transformar a coluna Mês_Ano
    df[['Mês', 'Ano']] = df['Mês_Ano'].str.split('/', expand=True)
    df['Mês'] = df['Mês'].map(mes_map)
    df['Ano'] = df['Ano'].apply(lambda x: f"{int(x):02d}")
    df['Mês_Ano_Numérico'] = (df['Mês'] + df['Ano']).astype(int)

    # Variáveis usadas no modelo
    variaveis_ipca = [
        'Alimentação e bebidas', 'Habitação', 'Artigos de residência', 'Vestuário',
        'Transportes', 'Saúde e cuidados pessoais', 'Despesas pessoais', 'Educação', 'Comunicação'
    ]

    try:
    st.dataframe(df[['Mês_Ano', 'Índice geral', 'Previsão IPCA']])
except KeyError as e:
    st.error(f"Erro: coluna ausente no arquivo - {e}")

