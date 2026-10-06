import numpy as np

#glossário:
# k: número de vizinhos
# x: amostra de teste
# y: rótulo da amostra (preço, nesse caso)
# A (usada em _distancias): lote de amostras de teste
# B (usada em _distancias): treino guardado no fit
class KNNRegressor:
    def __init__(self, k=5, metric="euclidean", batch_size=500):
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric deve ser 'euclidean' ou 'manhattan'")
        if k < 1:
            raise ValueError("k deve ser >= 1")
        self.k = k
        self.metric = metric
        self.batch_size = batch_size

    # ajusta o modelo, armazenando os dados de treino e pré-calculando a soma dos quadrados de cada linha (para a distância euclidiana)
    def fit(self, X, y):
        if self.k > len(X):
            raise ValueError("k maior que a qtd de amostras de treino")
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.float64)
        self.sq_ = (self.X_ ** 2).sum(axis=1) #soma dos quadrados de cada linha (distância euclidiana)   
        return self

    def _distancias(self, A):
        B = self.X_
        if self.metric == "euclidean":
            # ||a-b||² = ||a||² + ||b||² - 2 a·b 
            d2 = (A ** 2).sum(axis=1)[:, None] + self.sq_[None, :] - 2.0 * A @ B.T
            return np.sqrt(np.maximum(0.0, d2)) #maximum evita negativo por erro numérico
        #manhattan: soma |a_j - b_j| coluna a coluna (memória: lote x n_treino)
        d = np.zeros((A.shape[0], B.shape[0])) # inicializa a matriz de distâncias zerada
        for j in range(A.shape[1]):
            d += np.abs(A[:, j][:, None] - B[:, j][None, :])
        return d

    def predict(self, X):
        """Prevê com o k do objeto."""
        return self.predict_multi(X, [self.k])[self.k]

    def predict_multi(self, X, ks):
        """Prevê para vários k de uma vez (reaproveita o cálculo das distâncias).

        Retorna {k: predições}. Processa em lotes para economizar memória.
        """
        X = np.asarray(X, dtype=np.float64)
        kmax = max(ks)
        if kmax > len(self.X_):
            raise ValueError("k maior que o número de amostras de treino")
        saidas = {k: np.empty(len(X)) for k in ks}
        for ini in range(0, len(X), self.batch_size):
            fim = ini + self.batch_size
            D = self._distancias(X[ini:fim])
            # argpartition separa os kmax menores sem ordenar tudo
            if kmax < D.shape[1]:
                idx = np.argpartition(D, kmax - 1, axis=1)[:, :kmax]
            else:
                idx = np.tile(np.arange(D.shape[1]), (D.shape[0], 1))
            # ordena só esses kmax, para pegar os k menores de cada k
            ordem = np.argsort(np.take_along_axis(D, idx, axis=1), axis=1)
            idx = np.take_along_axis(idx, ordem, axis=1)
            alvos = self.y_[idx]                        # (lote, kmax)
            for k in ks:
                saidas[k][ini:fim] = alvos[:, :k].mean(axis=1)
        return saidas
