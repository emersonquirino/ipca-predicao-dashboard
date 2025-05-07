# IPCA Dashboard com Predição

Este projeto tem como objetivo construir um dashboard interativo para visualização e predição do IPCA (Índice Nacional de Preços ao Consumidor Amplo), utilizando técnicas de machine learning e visualização de dados com a biblioteca `streamlit`.

## 🔍 Objetivos

- Coletar e tratar dados históricos do IPCA.
- Treinar modelos de predição com base nos dados tratados.
- Construir uma aplicação web interativa para exibição de gráficos e previsões.

## 🧠 Tecnologias Utilizadas

- **Python** — linguagem principal.
- **Pandas & NumPy** — tratamento e manipulação dos dados.
- **Scikit-Learn** — treinamento de modelos de machine learning.
- **Matplotlib & Seaborn** — visualização de dados.
- **Streamlit** — criação do dashboard web.
- **Pickle** — serialização dos modelos.
- **Git** — versionamento de código.

## 📊 Funcionalidades do Dashboard

- Gráfico da série histórica do IPCA.
- Predições com base em modelos treinados.
- Interface simples e intuitiva via browser.
- Upload de novos dados para análise.

## 🚀 Como Executar Localmente

1. Clone o repositório:
    ```bash
    git clone https://github.com/emersonquirino/ipca-predicao-dashboard.git
    cd ipca-predicao-dashboard
    ```

2. Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    venv\Scripts\activate    # Windows
    source venv/bin/activate # Linux/Mac
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4. Execute o dashboard:
    ```bash
    streamlit run app.py
    ```

5. Acesse no navegador: `http://localhost:8501`

## 📁 Estrutura do Projeto

│
├── app.py # Código principal do dashboard Streamlit
├── modelo.pkl # Modelo de machine learning serializado
├── dados/ # Pasta com dados utilizados
│ └── ipca.csv
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo

## ✍️ Autor

Emerson Quirino — [LinkedIn](https://www.linkedin.com/in/emersonquirino)
Pront