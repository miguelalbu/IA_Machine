# 1. Importação de Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# 2. Carregamento dos Dados
data = pd.read_csv('data_set_groupo_a.csv')

# 3. Análise Exploratória de Dados (EDA)
print(data.head())
print(data.info())
print(data.describe())

# Visualização de dados
sns.pairplot(data)
plt.show()

# 4. Pré-processamento dos Dados
# Verificar valores ausentes
print(data.isnull().sum())

# Preencher ou remover valores ausentes conforme necessário
data.fillna(data.mean(), inplace=True)

# Codificação de variáveis categóricas, se houver
data = pd.get_dummies(data, drop_first=True)

# 5. Divisão dos Dados em Conjuntos de Treinamento e Teste
# Para classificação
X_class = data.drop('target_class', axis=1)  # Substitua 'target_class' pelo nome da coluna alvo
y_class = data['target_class']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Para regressão
X_reg = data.drop('target_reg', axis=1)  # Substitua 'target_reg' pelo nome da coluna alvo
y_reg = data['target_reg']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 6. Modelagem
# Classificação
classifier = RandomForestClassifier()
classifier.fit(X_train_class, y_train_class)

# Regressão
regressor = RandomForestRegressor()
regressor.fit(X_train_reg, y_train_reg)

# 7. Avaliação dos Modelos
# Avaliação da Classificação
y_pred_class = classifier.predict(X_test_class)
print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))

# Avaliação da Regressão
y_pred_reg = regressor.predict(X_test_reg)
print("Mean Squared Error:", mean_squared_error(y_test_reg, y_pred_reg))
print("R^2 Score:", r2_score(y_test_reg, y_pred_reg))

# 8. Conclusão
# Aqui você pode adicionar suas observações sobre o desempenho dos modelos e possíveis melhorias.