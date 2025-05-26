import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from .preprocessamento import ProcessadorDados

__all__ = ['ProcessadorDados']

# Definindo o caminho correto para o arquivo
caminho_arquivo = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'data',
    'raw',
    'data_set_grupo_a.csv'
)

try:
    # 2. Carregamento dos Dados
    data = pd.read_csv(caminho_arquivo)

    # 3. Análise Exploratória de Dados (EDA)
    print("Primeiras 5 linhas do dataset:")
    print(data.head())
    print("\nInformações do dataset:")
    print(data.info())
    print("\nEstatísticas descritivas:")
    print(data.describe())

    # Visualização de dados
    sns.pairplot(data)
    plt.show()

    # Verificar valores ausentes
    print(data.isnull().sum())

    # 4. Pré-processamento dos Dados
    # Exemplo: Preenchendo valores ausentes
    data.fillna(data.mean(), inplace=True)

    # Codificação de variáveis categóricas, se necessário
    data = pd.get_dummies(data, drop_first=True)

    # Separar características e rótulos
    X = data.drop('target_column', axis=1)  # Substitua 'target_column' pelo nome da coluna alvo
    y_classification = data['target_column']  # Para classificação
    y_regression = data['target_column']  # Para regressão (se aplicável)

    # 5. Divisão dos Dados em Conjuntos de Treinamento e Teste
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

    # 6. Modelagem
    # Classificação
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train_class)

    # Regressão
    regressor = RandomForestRegressor()
    regressor.fit(X_train_reg, y_train_reg)

    # 7. Avaliação dos Modelos
    # Avaliação da Classificação
    y_pred_class = classifier.predict(X_test)
    print(classification_report(y_test_class, y_pred_class))
    print(confusion_matrix(y_test_class, y_pred_class))

    # Avaliação da Regressão
    y_pred_reg = regressor.predict(X_test_reg)
    print("Mean Squared Error:", mean_squared_error(y_test_reg, y_pred_reg))
    print("R^2 Score:", r2_score(y_test_reg, y_pred_reg))

except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em {caminho_arquivo}")
    print("Verifique se o arquivo existe no diretório correto.")
