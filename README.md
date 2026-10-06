# Regressão com kNN (house_16H, OpenML 574)

Só NumPy + biblioteca padrão (matplotlib apenas para o gráfico). Sem pandas e sem scikit-learn.

## Como rodar
```
python testes/test_knn.py                      # testes (dados sintéticos)
python run_knn.py --data data/<arquivo>.arff   # experimento (ou .csv)
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
Preencher com a execução no dataset real (k escolhido, semente, tabela em `resultados/`).
