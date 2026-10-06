from importlib import machinery
from importlib import machinery
import csv
import numpy as np
import time

caminho_dataset_treino = "data/house_16H.arff"
K_FOLDS = 5 # número de folds para validação cruzada
SEED = 42 # semente para reprodutibilidade

# ---------------------------------------------------------------
# Etapa 0 - base comum (combinar a interface com o seu amigo)
# ---------------------------------------------------------------
def ler_arff(arquivo):
    """Parser manual de ARFF numérico. Retorna nomes das colunas, X e y (última coluna)."""
    nomes, linhas, em_dados = [], [], False
    with open(arquivo, encoding="utf-8") as f:
        for linha in f:
            s = linha.strip()
            if not s or s.startswith("%"):
                continue
            baixo = s.lower()
            if baixo.startswith("@attribute"):
                nomes.append(s.split()[1])
            elif baixo.startswith("@data"):
                em_dados = True
            elif em_dados:
                linhas.append([float(v) for v in s.split(",")])
    dados = np.array(linhas)
    return nomes, dados[:, :-1], dados[:, -1]

# Retorna os índices dos folds para treino e teste
def kfold_indices(n, k_folds, seed=None):
    # Cria semente para reprodutibilidade
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    # Embaralha os índices
    indices = np.arange(n)
    rng.shuffle(indices)
    # Define o tamanho de cada fold
    fold_size = n // k_folds
    # Cria os folds
    for i in range(k_folds):
        inicio = i * fold_size
        fim = inicio + fold_size if i < k_folds - 1 else n
        indice_teste = indices[inicio:fim]
        indice_treino = np.concatenate([indices[:inicio], indices[fim:]])
        yield indice_treino, indice_teste

# Normalização z-score (padronização)
def zscore_fit(X):
    medias = np.mean(X, axis=0)
    desvios = np.std(X, axis=0)
    desvios[desvios == 0] = 1.0 # Evita divisão por zero
    return medias, desvios

# Transforma os dados usando a normalização z-score
def zscore_transform(X, medias, desvios):
    return (X - medias) / desvios

# Calcula o R² (coeficiente de determinação)
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# Calcula o R² ajustado
def r2_ajustado(y_true, y_pred, n_features):
    # R² Ajustado = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
    # R² = Coeficiente de Determinação
    # n = número de observações
    # p = número de variáveis independentes
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = n_features
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Calcula o MAE (Mean Absolute Error)
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Calcula o RMSE (Root Mean Squared Error)
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

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
        X_tr = zscore_transform(X[treino], mu, sigma)
        X_te = zscore_transform(X[teste], mu, sigma)
 
        modelo = MRegression(metodo)
 
        t0 = time.perf_counter()
        modelo.fit(X_tr, y[treino])
        res["t_treino"].append(time.perf_counter() - t0)
 
        t0 = time.perf_counter()
        y_pred = modelo.predict(X_te)
        res["t_teste"].append(time.perf_counter() - t0)
 
        res["R2"].append(r2_score(y[teste], y_pred))
        res["R2_aj"].append(r2_ajustado(y[teste], y_pred, p))
        res["MAE"].append(mae(y[teste], y_pred))
        res["RMSE"].append(rmse(y[teste], y_pred))
 
    return res
 
 
def imprimir_tabela(res, nome):
    print(f"\n{nome}")
    for chave, valores in res.items():
        v = np.array(valores)
        print(f"  {chave:9s}: {v.mean():.4f} ± {v.std():.4f}")


if __name__ == "__main__":
    import os
    # Ajusta o caminho relativo caso execute da raiz ou da subpasta
    caminho = caminho_dataset_treino
    if not os.path.exists(caminho):
        caminho = os.path.join(os.path.dirname(__file__), "..", "data", "house_16H.arff")

    print(f"Carregando: {caminho}...")
    nomes, X, y = ler_arff(caminho)
    print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} atributos.")

    res = rodar_experimento(X, y)
    imprimir_tabela(res, "Regressão Linear Múltipla")