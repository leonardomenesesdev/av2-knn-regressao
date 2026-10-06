"""Testes sem pandas/sklearn. Rode de dentro de av2/: python testes/test_knn.py"""
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)                          # common.py
sys.path.insert(0, os.path.join(RAIZ, "regressao"))  # knn_regressor.py
import numpy as np

from common import (kfold_indices, mae, r2_adjusted, r2_score, rmse,
                    zscore_apply, zscore_fit)
from knn_regressor import KNNRegressor

rng = np.random.default_rng(0)

# 1) kNN vetorizado == versão ingênua com laços
X = rng.normal(size=(300, 6))
y = rng.normal(size=300)
Xt = rng.normal(size=(40, 6))
for metric in ("euclidean", "manhattan"):
    p = KNNRegressor(k=7, metric=metric, batch_size=16).fit(X, y).predict(Xt)
    ref = []
    for a in Xt:
        if metric == "euclidean":
            d = [np.sqrt(sum((a - b) ** 2)) for b in X]
        else:
            d = [sum(abs(a - b)) for b in X]
        ref.append(np.mean(y[np.argsort(d)[:7]]))
    assert np.allclose(p, ref), metric
print("OK kNN == ingênuo (euclidiana e manhattan)")

# 2) métricas contra conta à mão
yt = np.array([3., -0.5, 2., 7.])
yp = np.array([2.5, 0., 2., 8.])
assert np.isclose(r2_score(yt, yp), 0.9486081370449679)
assert np.isclose(mae(yt, yp), 0.5) and np.isclose(rmse(yt, yp), np.sqrt(0.375))
r2 = r2_score(yt, yp)
assert np.isclose(r2_adjusted(yt, yp, 1), 1 - (1 - r2) * 3 / 2)
print("OK métricas")

# 3) folds disjuntos e completos
folds = kfold_indices(1003, 5, 42)
for tr, te in folds:
    assert not set(tr) & set(te)
    assert len(tr) + len(te) == 1003
todos = np.concatenate([te for _, te in folds])
assert sorted(todos) == list(range(1003))
print("OK folds")

# 4) zscore protege std=0
Z = np.c_[rng.normal(size=10), np.ones(10)]
mu, sd = zscore_fit(Z)
assert np.isfinite(zscore_apply(Z, mu, sd)).all()
print("OK zscore")

# 5) tempo no tamanho real (~22784 x 16)
N = 22784
X = rng.normal(size=(N, 16))
y = X @ rng.normal(size=16) + rng.normal(size=N)
tr, te = kfold_indices(N, 5, 42)[0]
for metric in ("euclidean", "manhattan"):
    m = KNNRegressor(k=5, metric=metric).fit(X[tr], y[tr])
    t = time.perf_counter()
    m.predict(X[te])
    print(f"{metric}: predict de 1 fold = {time.perf_counter() - t:.2f}s")
