import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Criar o dicionário para converter os meses
mes_map = {
    "jan": "01", "fev": "02", "mar": "03", "abr": "04",
    "mai": "05", "jun": "06", "jul": "07", "ago": "08",
    "set": "09", "out": "10", "nov": "11", "dez": "12"
}

# Carregar os dados
df = pd.read_csv("IPCA_Base.csv", sep=";")


# Separar mês e ano, formatar corretamente
df[['Mês', 'Ano']] = df['Mês_Ano'].str.split('/', expand=True)
df['Mês'] = df['Mês'].map(mes_map)
df['Ano'] = df['Ano'].apply(lambda x: f"{int(x):02d}")

# Concatenar mês + ano para obter um número único
df['Mês_Ano_Numérico'] = df['Mês'] + df['Ano']
df['Mês_Ano_Numérico'] = df['Mês_Ano_Numérico'].astype(int)

# Adicionar as variações dos grupos do IPCA como variáveis independentes
variaveis_ipca = [
    'Alimentação e bebidas', 'Habitação', 'Artigos de residência', 'Vestuário',
    'Transportes', 'Saúde e cuidados pessoais', 'Despesas pessoais', 'Educação', 'Comunicação'
]

# Removendo valores nulos
df.dropna(subset=['Mês_Ano_Numérico', 'Índice geral'] + variaveis_ipca, inplace=True)

# Definir variável independente (X) e dependente (y)
X = df[['Mês_Ano_Numérico'] + variaveis_ipca]
y = df['Índice geral']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Calcular métricas de validação
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Criar tabela com resultados
resultado_df = pd.DataFrame({
    'Mês_Ano_Numérico': X_test['Mês_Ano_Numérico'].values,
    'Índice geral Real': y_test,
    'Índice geral Previsto': y_pred
})

# Salvar resultados em um arquivo CSV
resultado_df.to_csv("resultado_previsao_IPCA.csv", index=False, sep=";")
print("\nArquivo 'resultado_previsao_IPCA.csv' salvo com sucesso!")

# Exibir métricas
print("\n### Métricas do Modelo ###")
print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
print(f"Erro Absoluto Médio (MAE): {mae:.4f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Exibir tabela de previsões
print("\n### Comparação entre Índice geral real e previsto ###")
print(resultado_df.head(10))  # Exibir as 10 primeiras linhas

# Visualizar os resultados em gráfico
plt.scatter(X_test['Mês_Ano_Numérico'], y_test, label='Dados reais')
plt.plot(X_test['Mês_Ano_Numérico'], y_pred, color='red', label='Regressão Linear')
plt.xlabel("Mês_Ano Numérico")
plt.ylabel("Índice geral")
plt.legend()
plt.title("Modelo de Regressão Linear para IPCA com Grupos de Variação")
plt.show()

# Criar gráfico com apenas uma reta de regressão
plt.figure(figsize=(10, 6))
sns.regplot(x=X_test['Mês_Ano_Numérico'], y=y_test, scatter_kws={'color': 'blue', 'label': 'Dados reais'}, line_kws={'color': 'red', 'label': 'Regressão Linear'})
plt.xlabel("Mês_Ano Numérico")
plt.ylabel("Índice Geral do IPCA")
plt.legend()
plt.title("Modelo de Regressão Linear para IPCA ao longo do tempo")
plt.show()

import joblib

# Salvar o modelo treinado
joblib.dump(modelo, "modelo_regressao_linear.pkl")
print("\nModelo salvo como 'modelo_regressao_linear.pkl'")
