# Regressão com kNN (house_16H, OpenML 574)

Estrutura: `common.py` (compartilhado) na raiz; `regressao/` com o kNN (e a regressão linear do Anderson).

Só NumPy + biblioteca padrão (matplotlib apenas para o gráfico). Sem pandas e sem scikit-learn.

## Como rodar
```
python testes/test_knn.py                      # testes (dados sintéticos)
python regressao/run_knn.py --data data/<arquivo>.arff   # experimento (ou .csv)
```
Opções: `--target price`, `--seed 42`, `--folds 5`, `--k N` (pula a escolha de k), `--out resultados`.
No terminal do Windows, se os acentos saírem quebrados: `set PYTHONUTF8=1`.

## Protocolo (usar igual ao Anderson)
- 5-fold manual, `kfold_indices(n, 5, seed=42)` de `common.py`.
- Normalização z-score calculada só com o treino de cada fold.
- Escolha de k: validação interna (20% do treino de cada fold), sem usar o teste.
  Limitação: o k sai da média dos 5 folds, então o teste de um fold pode ter entrado no treino interno de outro.
- Desvio padrão com `ddof=1`.

## Resultados
Execução real em `data/house_16H.arff` (22.784 instâncias, 16 atributos, alvo `price`), semente 42, 5-fold.

| Regressor | k | R² | R² ajustado | MAE | RMSE | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|---|---|
| kNN (Euclidiana) | 7 | 0.5178 ± 0.0201 | 0.5161 ± 0.0202 | 18389.44 ± 445.82 | 36667.06 ± 819.03 | 0.0012 ± 0.0004 | 1.4188 ± 0.0885 |
| kNN (Manhattan) | 11 | 0.5514 ± 0.0223 | 0.5499 ± 0.0224 | 17250.74 ± 478.69 | 35362.33 ± 936.63 | 0.0010 ± 0.0002 | 9.3279 ± 0.3895 |
| Regressão Linear Múltipla | — | 0.2567 ± 0.0156 | 0.2541 ± 0.0157 | 25455.32 ± 388.17 | 45533.87 ± 788.74 | 0.0036 ± 0.0018 | 0.0001 ± 0.0000 |

kNN (Manhattan e Euclidiana) teve R² e MAE bem melhores que a Regressão Linear Múltipla — esperado, já que o house_16H foi desenhado de propósito para ter baixa correlação linear entre atributos e alvo ("16H" = 16 atributos, dificuldade Alta). Em troca, a Regressão Linear é muitíssimo mais rápida no teste (praticamente instantânea) contra ~1.4s (Euclidiana) e ~9.3s (Manhattan) por fold do kNN. Tabela completa do kNN em `resultados/knn_resultados.csv`/`.txt`, curva de escolha de k em `resultados/knn_k_vs_r2.png`.

`regressao_linear_multipla.py` agora importa `kfold_indices`, `zscore_fit`/`zscore_apply`, `r2_score`, `r2_adjusted`, `mae` e `rmse` direto de `common.py`, então os 3 regressores usam exatamente os mesmos folds.
