import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

class ProcessadorDados:
    def __init__(self):
        self.escalonador = None
        self.imputador = None
        self.pca = None

    def carregar_dados(self, caminho_arquivo):
        """
        Carrega os dados do arquivo CSV
        """
        return pd.read_csv(caminho_arquivo)

    def tratar_valores_ausentes(self, dados, estrategia='media'):
        """
        Trata valores ausentes usando a estratégia especificada
        Estratégias: 'media', 'mediana', 'mais_frequente'
        """
        self.imputador = SimpleImputer(strategy=estrategia)
        return pd.DataFrame(
            self.imputador.fit_transform(dados),
            columns=dados.columns,
            index=dados.index
        )

    def escalonar_features(self, dados, metodo='padronizacao'):
        """
        Normaliza ou padroniza os dados
        metodo: 'padronizacao' ou 'normalizacao'
        """
        if metodo == 'padronizacao':
            self.escalonador = StandardScaler()
        else:
            self.escalonador = MinMaxScaler()
            
        return pd.DataFrame(
            self.escalonador.fit_transform(dados),
            columns=dados.columns,
            index=dados.index
        )

    def reduzir_dimensoes(self, dados, n_componentes=2):
        """
        Aplica PCA para redução de dimensionalidade
        """
        self.pca = PCA(n_components=n_componentes)
        dados_reduzidos = self.pca.fit_transform(dados)
        return pd.DataFrame(
            dados_reduzidos,
            columns=[f'CP{i+1}' for i in range(n_componentes)],
            index=dados.index
        )

    def obter_importancia_features(self):
        """
        Retorna a importância das features após PCA
        """
        if self.pca is not None:
            return pd.DataFrame(
                self.pca.components_,
                columns=self.pca.feature_names_in_,
                index=[f'CP{i+1}' for i in range(self.pca.n_components_)]
            )
        return None

    def pipeline_preprocessamento(self, dados, metodo_escala='padronizacao', 
                                estrategia_imputacao='media'):
        """
        Pipeline completo de pré-processamento
        """
        dados_limpos = self.tratar_valores_ausentes(dados, 
                                                   estrategia=estrategia_imputacao)
        dados_escalonados = self.escalonar_features(dados_limpos, 
                                                   metodo=metodo_escala)
        return dados_escalonados