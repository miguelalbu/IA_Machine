from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class BankruptcyClassifier:
    def __init__(self, model, param_grid):
        """
        Inicializa o classificador com um modelo e seus hiperparâmetros
        
        Args:
            model: Modelo sklearn
            param_grid: Dicionário com parâmetros para GridSearch
        """
        self.model = model
        self.param_grid = param_grid
        self.grid_search = None
        
    def fit(self, X, y):
        """Treina o modelo usando GridSearchCV"""
        self.grid_search = GridSearchCV(
            self.model,
            self.param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        self.grid_search.fit(X, y)
        return self
    
    def predict(self, X):
        """Realiza predições"""
        return self.grid_search.predict(X)
    
    def predict_proba(self, X):
        """Retorna probabilidades das predições"""
        return self.grid_search.predict_proba(X)
    
    def get_best_params(self):
        """Retorna melhores parâmetros encontrados"""
        return self.grid_search.best_params_

# Configurações dos modelos
MODEL_CONFIGS = {
    'Regressão Logística': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'max_iter': [1000]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    },
    'Árvore de Decisão': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {}
    },
    'Rede Neural': {
        'model': MLPClassifier(),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'max_iter': [1000]
        }
    }
}