import csv
import time

import numpy as np

# converte texto em float ou, a depender do seu formato, vira nan
def _para_float(txt):
    txt = txt.strip().strip("'\"")
    if txt == "" or txt == "?":
        return np.nan
    return float(txt)


def _ler_arff(path):
    nomes, linhas, em_dados = [], [], False
    with open(path, encoding="utf-8", errors="ignore") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("%"):      # comentário / vazio
                continue
            baixo = linha.lower()
            if em_dados:
                linhas.append([_para_float(c) for c in linha.split(",")])
            elif baixo.startswith("@attribute"):
                partes = linha.split(None, 2)           # @attribute nome tipo
                nomes.append(partes[1].strip("'\""))
            elif baixo.startswith("@data"):
                em_dados = True
            # @relation e o resto são ignorados
    return nomes, linhas


def _ler_csv(path):
    """Lê CSV com cabeçalho usando o módulo csv."""
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        leitor = csv.reader(f)
        nomes = [c.strip() for c in next(leitor)]
        linhas = [[_para_float(c) for c in r] for r in leitor if r]
    return nomes, linhas

def load_dataset(path, target=None):
    if path.lower().endswith(".arff"):
        nomes, linhas = _ler_arff(path)
    else:
        nomes, linhas = _ler_csv(path)
    dados = np.array(linhas, dtype=np.float64)
    j = nomes.index(target) if target else len(nomes) - 1
    y = dados[:, j].copy()
    X = np.delete(dados, j, axis=1)
    feature_names = [n for i, n in enumerate(nomes) if i != j]
    return X, y, feature_names, nomes[j]

def kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    partes = np.array_split(perm, k)                  
    folds = []
    for i in range(k):
        treino = np.concatenate([partes[j] for j in range(k) if j != i])
        folds.append((treino, partes[i]))
    return folds


def zscore_fit(X_train):
    """Média e desvio do TREINO (std=0 vira 1 para não dividir por zero)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def zscore_apply(X, mean, std):
    return (X - mean) / std

# MÉTRICAS

def r2_score(y, y_pred):
    sq_res = np.sum((y - y_pred) ** 2)
    sq_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - sq_res / sq_tot)


def r2_adjusted(y, y_pred, n_features):
    n = len(y)
    r2 = r2_score(y, y_pred)
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))


def mae(y, y_pred):
    return float(np.mean(np.abs(y - y_pred)))


def rmse(y, y_pred):
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))


#TEMPO

class Timer:
    def __enter__(self):
        self._ini = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._ini


# ----------------------------------------------------------------- resumo
def summarize(lista_de_dicts):
    """Média e desvio padrão de cada métrica entre os folds.

    Usa ddof=1 (desvio amostral), pois os folds são uma amostra de execuções.
    Retorna {metrica: (media, desvio)}.
    """
    resumo = {}
    for chave in lista_de_dicts[0]:
        v = np.array([d[chave] for d in lista_de_dicts], dtype=float)
        resumo[chave] = (float(v.mean()), float(v.std(ddof=1)))
    return resumo
