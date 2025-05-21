# 📈 IPCA Dashboard com Predição

Este projeto constrói um **dashboard interativo** para análise e predição do **IPCA** (Índice Nacional de Preços ao Consumidor Amplo), utilizando **regressão linear**, métricas de avaliação e explicações visuais que facilitam a interpretação dos resultados.

---

## 🎯 Objetivos

- Visualizar e comparar os valores reais e previstos do IPCA.
- Calcular e interpretar métricas de erro do modelo de regressão.
- Tornar a análise acessível e compreensível, inclusive para quem não é técnico.
- Baixar os resultados com todos os cálculos auxiliares em CSV.

---

## 🧠 Tecnologias Utilizadas

- **Python** — linguagem principal do projeto
- **Pandas** — manipulação de dados
- **Scikit-Learn** — métricas de avaliação de regressão
- **Matplotlib** — gráficos e visualizações
- **Streamlit** — criação da aplicação web
- **Git/GitHub** — versionamento e hospedagem do código

---

## 📊 Funcionalidades do Dashboard

- ✅ Upload de um arquivo \`.csv\` com resultados de previsão do IPCA
- ✅ Cálculo de métricas de erro: MSE, MAE e R²
- ✅ Gráfico de dispersão comparando valores reais e previstos
- ✅ Interpretação automática da performance do modelo com emojis
- ✅ Acurácia com base em faixa de tolerância (erro absoluto ≤ 0.1)
- ✅ Download do arquivo \`.csv\` com colunas de erro e acurácia
- ✅ Interface clara, explicativa e acessível

---

## 🚀 Como Executar Localmente

1. Clone o repositório:
   \`\`\`bash
   git clone https://github.com/emersonquirino/ipca-predicao-dashboard.git
   cd ipca-predicao-dashboard
   \`\`\`

2. Crie e ative um ambiente virtual:
   \`\`\`bash
   python -m venv venv
   venv\\Scripts\\activate    # Windows
   source venv/bin/activate  # Linux/Mac
   \`\`\`

3. Instale as dependências:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. Execute o dashboard:
   \`\`\`bash
   streamlit run app.py
   \`\`\`

5. Acesse no navegador:  
   [http://localhost:8501](http://localhost:8501)

---

## 📁 Estrutura do Projeto

\`\`\`
.
├── app.py                      # Código principal da aplicação Streamlit
├── modelo_regressao_linear.pkl  # Modelo treinado (opcional)
├── resultado_previsao_IPCA.csv  # Base de dados com previsões
├── requirements.txt            # Lista de dependências
└── README.md                   # Este arquivo
\`\`\`

---

## ✍️ Autor

**Emerson Quirino**  
[🔗 LinkedIn](https://www.linkedin.com/in/emersonquirino)
" > README.md && git add README.md && git commit -m "Atualiza README com novas funcionalidades e explicações" && git push origin main
