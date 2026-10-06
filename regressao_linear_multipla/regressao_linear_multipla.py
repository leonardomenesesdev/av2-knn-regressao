import os
import sys
import time

import numpy as np

# common.py fica na raiz do projeto (pasta acima de regressao_linear_multipla/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (kfold_indices, load_dataset, mae, r2_adjusted, r2_score,
                    rmse, zscore_apply, zscore_fit)

caminho_dataset_treino = "data/house_16H.arff"
K_FOLDS = 5 # número de folds para validação cruzada
SEED = 42 # semente para reprodutibilidade

# ---------------------------------------------------------------
# Etapa 1 - Regressão Linear Múltipla (sua parte)
# ---------------------------------------------------------------
class MRegression:
    def __init__(self, metodo="solve"):
        self.metodo = metodo
        self.coeficientes = None
        self.intercepto = None

    # Método para treinar o modelo
    def fit(self, X, y):
        # Adiciona uma coluna de 1s para o intercepto
        X = np.c_[np.ones(X.shape[0]), X]
        # Calcula os coeficientes usando a equação normal
        self.coeficientes = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercepto = self.coeficientes[0]
        self.coeficientes = self.coeficientes[1:]

    def predict(self, X):
        return X @ self.coeficientes + self.intercepto

# ---------------------------------------------------------------
# Experimento: CV com médias, desvios e tempos
# ---------------------------------------------------------------
def rodar_experimento(X, y, metodo="solve"):
    p = X.shape[1]
    res = {"R2": [], "R2_aj": [], "MAE": [], "RMSE": [],
           "t_treino": [], "t_teste": []}
 
    for treino, teste in kfold_indices(len(y), K_FOLDS, SEED):
        mu, sigma = zscore_fit(X[treino])
        X_tr = zscore_apply(X[treino], mu, sigma)
        X_te = zscore_apply(X[teste], mu, sigma)
 
        modelo = MRegression(metodo)
 
        t0 = time.perf_counter()
        modelo.fit(X_tr, y[treino])
        res["t_treino"].append(time.perf_counter() - t0)
 
        t0 = time.perf_counter()
        y_pred = modelo.predict(X_te)
        res["t_teste"].append(time.perf_counter() - t0)
 
        res["R2"].append(r2_score(y[teste], y_pred))
        res["R2_aj"].append(r2_adjusted(y[teste], y_pred, p))
        res["MAE"].append(mae(y[teste], y_pred))
        res["RMSE"].append(rmse(y[teste], y_pred))
 
    return res
 
 
def imprimir_tabela(res, nome):
    print(f"\n{nome}")
    for chave, valores in res.items():
        v = np.array(valores)
        print(f"  {chave:9s}: {v.mean():.4f} ± {v.std():.4f}")


if __name__ == "__main__":
    # Ajusta o caminho relativo caso execute da raiz ou da subpasta
    caminho = caminho_dataset_treino
    if not os.path.exists(caminho):
        caminho = os.path.join(os.path.dirname(__file__), "..", "data", "house_16H.arff")

    print(f"Carregando: {caminho}...")
    X, y, nomes, alvo = load_dataset(caminho)
    print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} atributos.")

    res = rodar_experimento(X, y)
    imprimir_tabela(res, "Regressão Linear Múltipla")