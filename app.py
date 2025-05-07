
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

st.set_page_config(page_title="Previsão do IPCA", layout="wide")

st.title("Previsão do IPCA com Regressão Linear Múltipla")
st.caption("Faça upload do CSV com os dados do IPCA (formato igual ao IPCA_Base.csv)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    X = df.drop(columns=["Índice geral", "Mês_Ano"])
    y = df["Índice geral"]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    df_resultado = df[["Mês_Ano", "Índice geral"]].copy()
    df_resultado["Previsão IPCA"] = y_pred

    st.subheader("Comparativo entre Valor Real e Previsto")
    st.dataframe(df_resultado, use_container_width=True)

    st.subheader("Gráfico: Valor Real vs Previsto")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_resultado["Mês_Ano"], df_resultado["Índice geral"], label="Real", marker='o')
    ax.plot(df_resultado["Mês_Ano"], df_resultado["Previsão IPCA"], label="Previsto", marker='x')
    plt.xticks(rotation=90)
    plt.xlabel("Mês/Ano")
    plt.ylabel("IPCA")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Métricas do Modelo")
    st.markdown(f"- Erro Quadrático Médio (MSE): **{mean_squared_error(y, y_pred):.4f}**")
    st.markdown(f"- Erro Absoluto Médio (MAE): **{mean_absolute_error(y, y_pred):.4f}**")
    st.markdown(f"- Coeficiente de Determinação (R²): **{r2_score(y, y_pred):.4f}**")
