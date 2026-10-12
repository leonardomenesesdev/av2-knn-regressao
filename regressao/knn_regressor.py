import numpy as np

#glossário:
# k: número de vizinhos
# x: amostra de teste
# y: rótulo da amostra (preço, nesse caso)
# A (usada em _distancias): lote de amostras de teste
# B (usada em _distancias): treino guardado no fit
# batch_size: tamanho do lote de amostras - importante pra nâo estourar a memória
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

    # prevê com o k do modelo, retornando um array de previsões
    def predict(self, X):
        return self.predict_multi(X, [self.k])[self.k]

    # prevê com vários k ao mesmo tempo, retornando um dicionário {k: y_pred}
    def predict_multi(self, X, ks):
        X = np.asarray(X, dtype=np.float64) #garante que X será um array numpy
        kmax = max(ks) # retorna o maior k (maior número de vizinhos), permitindo calcular as distâncias uma vez só
        if kmax > len(self.X_):
            raise ValueError("k maior que o número de amostras de treino")
        saidas = {k: np.empty(len(X)) for k in ks} # cria um dicionário vazio para armazenar as previsões de cada k
        for ini in range(0, len(X), self.batch_size):
            fim = ini + self.batch_size
            D = self._distancias(X[ini:fim]) #matriz de distâncias (lote, treino)
            if kmax < D.shape[1]: #D.shape[1] - > número de amostras de treino
                #se kmax for menor que o n° de amostras de treino, pega só os kmax mais próximos 
                idx = np.argpartition(D, kmax - 1, axis=1)[:, :kmax] 
            else:
                idx = np.tile(np.arange(D.shape[1]), (D.shape[0], 1)) # pega todas as amostras de treino (D.shape[1]) para cada amostra do lote (D.shape[0])
            # ordena só esses kmax, para pegar os k menores de cada k
            ordem = np.argsort(np.take_along_axis(D, idx, axis=1), axis=1)
            idx = np.take_along_axis(idx, ordem, axis=1) #indexa os kmax vizinhos mais próximos
            alvos = self.y_[idx] # pega os rótulos correspondentes aos kmax vizinhos mais próximos
            for k in ks:
                saidas[k][ini:fim] = alvos[:, :k].mean(axis=1)
        return saidas
