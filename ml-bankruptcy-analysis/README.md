# Projeto de Análise de Machine Learning

## Sobre o Projeto
Implementação de técnicas de machine learning para análise de dados, incluindo classificação supervisionada e análise de clusters.

## Estrutura do Projeto
```
ml-bankruptcy-analysis/
├── data/
│   ├── raw/              # Dados originais
│   └── processed/        # Dados após pré-processamento
├── models/               # Modelos treinados
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_preprocessamento.ipynb
│   └── 03_treinamento_modelos.ipynb
└── src/
    ├── data/            
    │   └── preprocessamento.py
    └── models/          
        └── classificador.py
```

## Processamento de Dados
### Transformações Implementadas
- Normalização das features numéricas
- Tratamento de valores ausentes (média/mediana)
- Redução de dimensionalidade (PCA)
- Análise exploratória inicial

### Arquivos Gerados
- `dados_preprocessados.csv`: Dataset após limpeza
- `features_pca.csv`: Dados após redução dimensional

## Modelos Implementados

### Classificação
- Regressão Logística
- KNN
- Árvore de Decisão
- Naive Bayes

### Métricas Avaliadas
- Acurácia
- Precisão
- Recall
- F1-Score
- Matriz de Confusão

## Como Usar

### 1. Configuração do Ambiente
```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/ml-bankruptcy-analysis.git

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar Análises
Abrir os notebooks na ordem:
1. `01_analise_exploratoria.ipynb`
2. `02_preprocessamento.ipynb`
3. `03_treinamento_modelos.ipynb`

## Dependências Principais
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn


## Licença
Este projeto está sob a licença MIT.