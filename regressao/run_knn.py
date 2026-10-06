"""Experimento kNN regressor: escolha de k + avaliação final com 5-fold."""
import argparse
import csv
import os
import sys

import numpy as np

# common.py fica na raiz do projeto (pasta acima de regressao/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (Timer, kfold_indices, load_dataset, mae, r2_adjusted,
                    r2_score, rmse, summarize, zscore_apply, zscore_fit)
from knn_regressor import KNNRegressor

KS = [1, 3, 5, 7, 11, 15, 21]
METRICAS = ["euclidean", "manhattan"]
NOMES = {"euclidean": "kNN (Euclidiana)", "manhattan": "kNN (Manhattan)"}


def escolher_k(X, y, folds, seed):
    """Escolhe k por validação INTERNA: dentro do treino de cada fold separa
    20% para validar; o conjunto de teste do fold não é usado.
    Retorna o R² médio de validação por distância e por k."""
    r2 = {m: {k: [] for k in KS} for m in METRICAS}
    for f, (tr, _) in enumerate(folds):
        rng = np.random.default_rng(seed + f)
        tr = rng.permutation(tr)
        corte = int(0.8 * len(tr))
        sub, val = tr[:corte], tr[corte:]
        mean, std = zscore_fit(X[sub])
        Xs, Xv = zscore_apply(X[sub], mean, std), zscore_apply(X[val], mean, std)
        for m in METRICAS:
            preds = KNNRegressor(metric=m).fit(Xs, y[sub]).predict_multi(Xv, KS)
            for k in KS:
                r2[m][k].append(r2_score(y[val], preds[k]))
    return {m: {k: float(np.mean(v)) for k, v in r2[m].items()} for m in METRICAS}


def avaliar(X, y, folds, metric, k):
    """Avaliação final: normaliza com o treino, mede tempos e métricas por fold."""
    linhas = []
    for tr, te in folds:
        mean, std = zscore_fit(X[tr])
        Xtr, Xte = zscore_apply(X[tr], mean, std), zscore_apply(X[te], mean, std)
        modelo = KNNRegressor(k=k, metric=metric)
        with Timer() as t_treino:
            modelo.fit(Xtr, y[tr])
        with Timer() as t_teste:
            pred = modelo.predict(Xte)
        linhas.append({
            "r2": r2_score(y[te], pred),
            "r2_ajustado": r2_adjusted(y[te], pred, X.shape[1]),
            "mae": mae(y[te], pred),
            "rmse": rmse(y[te], pred),
            "t_treino": t_treino.elapsed,
            "t_teste": t_teste.elapsed,
        })
    return summarize(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="arquivo .arff ou .csv")
    ap.add_argument("--target", default=None, help="nome do alvo (padrão: última coluna)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=None, help="força um k (pula a escolha)")
    ap.add_argument("--out", default="resultados")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X, y, feats, alvo = load_dataset(a.data, a.target)
    faltantes = int(np.isnan(X).sum() + np.isnan(y).sum())
    print(f"Instâncias: {X.shape[0]} | Atributos: {X.shape[1]} | Alvo: {alvo}")
    print(f"Valores faltantes: {faltantes}")
    if faltantes:
        raise SystemExit("Há valores faltantes: trate-os antes de rodar.")
    print(f"Semente: {a.seed} | {a.folds}-fold")

    folds = kfold_indices(len(y), a.folds, a.seed)
    if a.k is None:
        curvas = escolher_k(X, y, folds, a.seed)
        escolhidos = {m: max(curvas[m], key=curvas[m].get) for m in METRICAS}
        for m in METRICAS:
            print(f"R² de validação ({m}): " +
                  ", ".join(f"k={k}:{v:.4f}" for k, v in curvas[m].items()))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for m in METRICAS:
            plt.plot(KS, [curvas[m][k] for k in KS], marker="o", label=NOMES[m])
        plt.xlabel("k")
        plt.ylabel("R² (validação interna)")
        plt.title("Escolha de k")
        plt.grid(alpha=.3)
        plt.legend()
        plt.savefig(os.path.join(a.out, "knn_k_vs_r2.png"), dpi=150)
    else:
        escolhidos = {m: a.k for m in METRICAS}
    print("k escolhido:", escolhidos)

    cab = ["Regressor", "k", "R²", "R² ajustado", "MAE", "RMSE",
           "Tempo Treino (s)", "Tempo Teste (s)"]
    chaves = ["r2", "r2_ajustado", "mae", "rmse", "t_treino", "t_teste"]
    linhas = []
    for m in METRICAS:
        res = avaliar(X, y, folds, m, escolhidos[m])
        linhas.append([NOMES[m], str(escolhidos[m])] +
                      [f"{res[c][0]:.4f} ± {res[c][1]:.4f}" for c in chaves])

    tabela = "| " + " | ".join(cab) + " |\n|" + "---|" * len(cab) + "\n"
    tabela += "\n".join("| " + " | ".join(l) + " |" for l in linhas)
    print("\n" + tabela)
    with open(os.path.join(a.out, "knn_resultados.txt"), "w", encoding="utf-8") as f:
        f.write(f"semente={a.seed} folds={a.folds}\n\n{tabela}\n")
    with open(os.path.join(a.out, "knn_resultados.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cab)
        w.writerows(linhas)


if __name__ == "__main__":
    main()
