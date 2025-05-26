# Projeto de Análise de Machine Learning

## Sobre o Projeto
Implementação de técnicas de machine learning para análise de dados bancários e previsão de duração de viagens, incluindo classificação supervisionada, análise de clusters e regressão.

## Estrutura do Projeto
```
ml-bankruptcy-analysis/
├── data/
│   ├── raw/              # Datasets A, B e C originais
│   └── processed/        # Dados processados e resultados
├── models/               # Modelos treinados
├── notebooks/           
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_classificacao_dataset_a.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_regressao.ipynb
│   └── 06_classificacao_dataset_b.ipynb
└── src/
    ├── data/            
    │   └── preprocessamento.py
    └── models/          
        ├── classificador.py
        └── regressor.py
```

## Análises Implementadas

### 1. Classificação (Datasets A e B)
#### Modelos
- Regressão Logística
- Árvore de Decisão 
- Naive Bayes
- Redes Neurais (MLP)

#### Métricas
- Acurácia
- Precisão
- Recall
- F1-Score
- Matriz de Confusão

### 2. Regressão (Dataset C)
#### Modelos
- Regressão Linear
- Ridge
- Lasso
- Random Forest
- MLP Regressor

#### Métricas
- R² (Coeficiente de Determinação)
- RMSE (Raiz do Erro Quadrático Médio)
- MAE (Erro Absoluto Médio)
- MAPE (Erro Percentual Absoluto Médio)

### 3. Clustering
- K-means
- Análise de Componentes Principais (PCA)
- Avaliação com Adjusted Rand Index

## Processamento de Dados

### Transformações
- Normalização/Padronização de features
- Tratamento de valores ausentes
- Redução de dimensionalidade (PCA)
- Análise exploratória
- Balanceamento de classes
- Engenharia de características

### Resultados Gerados
- `resultados_classificacao_a.csv`: Métricas do dataset A
- `resultados_classificacao_b.csv`: Métricas do dataset B
- `resultados_regressao.csv`: Métricas de regressão

## Como Executar

### 1. Configuração do Ambiente
```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/ml-bankruptcy-analysis.git

# Instalar dependências
pip install -r requirements.txt
```

### 2. Execução dos Notebooks
1. `01_analise_exploratoria.ipynb`: Análise inicial dos dados
2. `02_preprocessamento.ipynb`: Preparação dos datasets
3. `03_classificacao_dataset_a.ipynb`: Classificação do Dataset A
4. `04_clustering.ipynb`: Análise de clusters
5. `05_regressao.ipynb`: Regressão do Dataset C
6. `06_classificacao_dataset_b.ipynb`: Classificação do Dataset B

## Dependências
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Resultados
Os resultados detalhados podem ser encontrados em:
- Notebooks individuais com análises completas
- Arquivos CSV na pasta `processed/` com métricas
- Visualizações e gráficos comparativos

## Licença
Este projeto está sob a licença MIT.

---
**Nota**: Este projeto demonstra a aplicação de diferentes técnicas de machine learning em datasets variados, incluindo análise de falências bancárias e previsão de duração de viagens.