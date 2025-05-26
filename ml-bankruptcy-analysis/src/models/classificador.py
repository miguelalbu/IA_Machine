from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, auc)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class BankruptcyClassifier:
    def __init__(self, modelo, params_grid=None):
        """
        Inicializa o classificador
        modelo: algoritmo de classificação (ex: LogisticRegression, DecisionTree)
        params_grid: dicionário com parâmetros para otimização
        """
        self.modelo = modelo
        self.params_grid = params_grid
        self.melhor_modelo = None

    def treinar(self, X_treino, y_treino):
        """
        Treina o modelo com os dados fornecidos
        """
        if self.params_grid:
            grid_search = GridSearchCV(self.modelo, self.params_grid, cv=5)
            grid_search.fit(X_treino, y_treino)
            self.melhor_modelo = grid_search.best_estimator_
        else:
            self.modelo.fit(X_treino, y_treino)
            self.melhor_modelo = self.modelo

    def validacao_cruzada(self, X, y, cv=5):
        """
        Realiza validação cruzada e retorna métricas
        """
        metricas = {
            'acuracia': cross_val_score(self.modelo, X, y, cv=cv, 
                                      scoring='accuracy'),
            'precisao': cross_val_score(self.modelo, X, y, cv=cv, 
                                      scoring='precision'),
            'recall': cross_val_score(self.modelo, X, y, cv=cv, 
                                    scoring='recall'),
            'f1': cross_val_score(self.modelo, X, y, cv=cv, 
                                scoring='f1')
        }
        return metricas

    def avaliar(self, X_teste, y_teste):
        """
        Avalia o modelo com dados de teste
        """
        y_pred = self.melhor_modelo.predict(X_teste)
        
        return {
            'acuracia': accuracy_score(y_teste, y_pred),
            'precisao': precision_score(y_teste, y_pred),
            'recall': recall_score(y_teste, y_pred),
            'f1': f1_score(y_teste, y_pred),
            'matriz_confusao': confusion_matrix(y_teste, y_pred)
        }

    def prever_probabilidades(self, X):
        """
        Retorna probabilidades de predição
        """
        return self.melhor_modelo.predict_proba(X)

    def curva_roc(self, X_teste, y_teste):
        """
        Calcula pontos para curva ROC
        """
        y_prob = self.prever_probabilidades(X_teste)[:, 1]
        fpr, tpr, _ = roc_curve(y_teste, y_prob)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }

# Configurações dos modelos
MODEL_CONFIGS = {
    'LogisticRegression': {
        'modelo': LogisticRegression(),
        'params_grid': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    },
    'KNeighborsClassifier': {
        'modelo': KNeighborsClassifier(),
        'params_grid': {
            'n_neighbors': [3, 5, 11, 21],
            'weights': ['uniform', 'distance']
        }
    },
    'DecisionTreeClassifier': {
        'modelo': DecisionTreeClassifier(),
        'params_grid': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'GaussianNB': {
        'modelo': GaussianNB(),
        'params_grid': {}
    },
    'MLPClassifier': {
        'modelo': MLPClassifier(),
        'params_grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'learning_rate': ['constant', 'adaptive']
        }
    }
}