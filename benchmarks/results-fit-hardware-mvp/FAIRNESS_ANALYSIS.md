# Análise da comparação Bankai vs. LightGBM

## Resumo

No workload binário profundo medido, o Bankai levou **0,274 s** no fit com `importance_type="permutation"`; o LightGBM RF levou **2,335 s**. A razão observada foi de aproximadamente **8,5×**. Esse resultado é plausível para esta carga e implementação, mas não deve ser tratado como uma razão geral de desempenho entre as bibliotecas.

## Protocolo medido

- 20.000 linhas de treino, 4.000 linhas de validação e 32 features sintéticas; dataset criado com `random_state=1729`.
- 40 árvores/iterações, seed de modelo 42, limite de profundidade 20, até 511 folhas, mínimo de 5 amostras por folha e 63 bins.
- Bankai: `max_features=None`, `max_samples=0.8`, bootstrap e `importance_type="permutation"`.
- LightGBM RF: `boosting_type="rf"`, `bagging_fraction=0.8`, `bagging_freq=1`, `feature_fraction=1.0` e `max_bin=63`.
- `n_jobs=-1` nas duas bibliotecas. A máquina reportou 8 CPUs lógicas.
- Um warmup e cinco fits medidos; a ordem dos modelos alternou. O cronômetro cobre `fit`, não a predição nem as métricas de validação.
- A permutation importance foi calculada pelo Bankai durante o fit. Não foi calculada pelo LightGBM.

O benchmark mede o caminho público de fit de cada biblioteca, incluindo a preparação interna dos dados. Não mede apenas o kernel de crescimento das árvores.

## Resultados

Medianas das cinco execuções, no mesmo conjunto de validação:

| Modelo | Fit (s) | Acurácia | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bankai, permutation importance | **0,274** | 0,942 | 0,957 | 0,927 | 0,942 |
| LightGBM RF | 2,335 | 0,923 | 0,926 | 0,922 | 0,924 |
| LightGBM GBDT | 2,338 | **0,961** | **0,965** | **0,958** | **0,961** |
| LightGBM DART | 2,412 | 0,945 | 0,949 | 0,943 | 0,946 |
| LightGBM GOSS | 2,266 | 0,955 | 0,957 | 0,955 | 0,956 |

Os boosters LightGBM foram comparados com 40 iterações, `learning_rate=0.1`, limites de bins/folhas/profundidade iguais e todas as features. GBDT e DART usaram bagging de 80%; GOSS usou `top_rate=0.2` e `other_rate=0.1`, sem bagging. São configurações fixas para comparação, não resultados de tuning.

## Custo da permutation importance no Bankai

Foi feita uma ablação com cinco fits por configuração, mantendo dataset, seed, limites e `n_jobs=-1`:

| Importance no Bankai | Fit mediano | Folhas totais nas 40 árvores |
| --- | ---: | ---: |
| `gain` | 0,241 s | 20.440 |
| `permutation` | 0,263 s | 20.440 |

Neste caso, a permutation importance acrescentou aproximadamente **0,023 s (9,4%)** ao fit. Ela é calculada nativamente sobre as observações OOB e os percursos das árvores; não é uma chamada externa repetida cinco vezes por feature. Esse custo está incluído nos 0,274 s apresentados para Bankai.

## O que a comparação sustenta

Os modelos receberam os mesmos dados e limites nominais, e os tempos foram medidos em condições intercaladas. Isso sustenta a afirmação estreita de que, neste workload, o fit do Bankai com permutation importance foi mais rápido que as configurações medidas do LightGBM.

Há razões de implementação que podem contribuir para o resultado: o Bankai treina árvores independentes em paralelo e usa histogramas compactos; o LightGBM organiza as rodadas do boosting de forma sequencial, pois cada árvore depende das predições/gradientes atualizados. Essas diferenças são explicações plausíveis, não uma atribuição causal medida por profiler. O benchmark não coletou ciclos, utilização por thread, bandwidth de memória ou contadores de cache.

## Limitações de justiça e interpretação

1. **O trabalho não é idêntico.** Bankai treina uma floresta aleatória com bootstrap e critério Gini. `gbdt`, `dart` e `goss` treinam boosters com gradientes. Mesmo número de árvores/iterações, bins e limites de folhas não torna os algoritmos equivalentes.
2. **Importance assimétrica.** O fit do Bankai inclui permutation importance. O fit de LightGBM não inclui uma operação de importance. Isso atende ao cenário solicitado, mas o cronômetro compara fit do Bankai com importance contra fit de LightGBM sem importance.
3. **`n_jobs=-1` não garante escala igual.** Ambos receberam esse valor, mas as bibliotecas distribuem o trabalho de formas diferentes. No benchmark anterior, LightGBM RF com quatro threads levou cerca de 1,186 s; na medição atual com `n_jobs=-1`, a mediana foi 2,335 s. Portanto, usar todas as threads não foi mais rápido para LightGBM nesta carga.
4. **Métricas são de um único dataset/seed.** As cinco repetições medem sobretudo variação de tempo. Elas não são cinco amostras independentes para estimar a variação estatística de acurácia.
5. **Configuração não foi ajustada por modelo.** GBDT, DART e GOSS usaram 40 iterações e `learning_rate=0.1`. Mais iterações ou outro learning rate podem mudar tempo e métricas. O GBDT já atingiu F1 0,961, acima do F1 0,942 do Bankai, neste conjunto.
6. **Limites não provam trabalho idêntico.** `num_leaves`/`max_leaf_nodes`, `min_child_samples`/`min_samples_leaf` e `max_bin`/`max_bins` são limites com semânticas de implementação distintas. O runner desta rodada não grava contagem de folhas efetivas para cada booster LightGBM.

## Conclusão

O resultado de 8,5× é real para esta execução controlada e ainda inclui o custo de permutation importance do Bankai. É uma evidência de desempenho promissora no caso sintético de 20 mil linhas e 32 features. Não demonstra superioridade geral nem uma comparação de custo por árvore matematicamente equivalente. Para essa conclusão mais ampla, seria necessário medir vários datasets e seeds, registrar o tamanho efetivo das árvores e comparar também modelos ajustados para uma qualidade-alvo comum.

Dados brutos e runner: [deep-binary-permutation.json](deep-binary-permutation.json) e [run_deep_binary_fit_benchmark.py](../../run_deep_binary_fit_benchmark.py).
