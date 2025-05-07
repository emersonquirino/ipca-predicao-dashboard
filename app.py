import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.title("Previsão do IPCA com Regressão Linear Múltipla")
st.markdown("Faça upload do CSV com os dados do IPCA (formato igual ao IPCA_Base.csv)")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    model = joblib.load("modelo_regressao_linear.pkl")
    
    X = df.drop(columns=["Indice geral", "Mes_Ano"])
    y = df["Indice geral"]
    
    previsoes = model.predict(X)
    df["Previsão IPCA"] = previsoes

    st.subheader("Comparativo entre Valor Real e Previsto")
    st.dataframe(df[["Mes_Ano", "Indice geral", "Previsão IPCA"]].head(10))

    st.subheader("Gráfico: Valor Real vs Previsto")
    fig, ax = plt.subplots()
    ax.plot(df["Mes_Ano"], df["Indice geral"], label="Real", marker='o')
    ax.plot(df["Mes_Ano"], df["Previsão IPCA"], label="Previsto", marker='x')
    ax.legend()
    plt.xticks(rotation=90)
    plt.xlabel("Mes/Ano")
    plt.ylabel("IPCA")
    st.pyplot(fig)

    st.subheader("Métricas do Modelo")
    mse = np.mean((df["Indice geral"] - df["Previsão IPCA"])**2)
    mae = np.mean(np.abs(df["Indice geral"] - df["Previsão IPCA"]))
    r2 = 1 - (np.sum((df["Indice geral"] - df["Previsão IPCA"])**2) / np.sum((df["Indice geral"] - np.mean(df["Indice geral"]))**2))
    
    st.markdown(f"**Erro Quadrático Médio (MSE):** {mse:.4f}")
    st.markdown(f"**Erro Absoluto Médio (MAE):** {mae:.4f}")
    st.markdown(f"**Coeficiente de Determinação (R²):** {r2:.4f}")
