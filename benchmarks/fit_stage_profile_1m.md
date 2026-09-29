# Perfil do `fit` em 1 milhão de linhas

## Resultado

A espera longa antes de as threads entrarem no processamento é explicada principalmente pelo trabalho de preparação em Python e pela cópia integral da matriz para o Rust. O despacho Python→Rust em si ocorre uma vez e não é o gargalo: antes de a extensão começar, este perfil gastou **301,2 s**; dentro da chamada nativa, a cópia da matriz consumiu mais **62,5 s**.

| Etapa | 100 mil × 500 | 1 milhão × 500 |
|---|---:|---:|
| Validação sklearn / conversão para `float64` | 0,060 s | 20,698 s |
| `_prepare_training_data` | 4,830 s | 280,504 s |
| `matrix_to_owned` (`Vec<f64>`) | 0,086 s | 62,481 s |
| Construção dos cortes dos bins | 0,202 s | 2,153 s |
| Aplicação dos cortes / matriz discretizada | 0,083 s | 3,358 s |
| Preparação total de `DenseInput` | 0,324 s | 22,552 s |
| Construção da floresta | 0,015 s | 1,310 s |
| Chamada nativa total | 0,425 s | 86,402 s |
| `fit` total | **5,322 s** | **387,852 s** |
| Pico RSS observado pelo amostrador | 1,147 GiB | 3,782 GiB |

As etapas internas de `DenseInput` estão contidas no total dessa linha: cortes + aplicação somam 5,511 s no caso de 1M; o restante inclui validações e setup. Por isso as linhas da tabela não devem ser somadas entre si.

## Como foi medido

- Máquina: macOS 26.6.2, Apple Silicon ARM64, 8 CPUs lógicas e 8 GiB de RAM; Python 3.11.11.
- Entrada sintética reproduzível com seed 1729: `float32`, 500 features, 90 relevantes, sem matriz esparsa; 2 GB de dados em 1M linhas.
- `n_jobs=-1`, `max_bins=63`, `binning_strategy='sampled_select'`, amostra de 200 mil linhas para os cortes, `max_samples=0.8`.
- Uma árvore rasa (`n_estimators=1`, `max_depth=1`) e `importance_type='gain'` foram usados para isolar preparação, cópia e histogramização. A geração do dataset ficou fora do tempo de `fit`.
- Os tempos são de uma execução por tamanho, em build Rust `--release`. O RSS veio de amostragem a cada 50 ms. Uma amostragem do processo no macOS registrou footprint físico máximo de 8,8 GiB; houve pressão de memória, então os tempos do caso de 1M são representativos deste host sob pressão, não uma estimativa limpa para outra máquina.

O perfil não é o benchmark integral do notebook: ele usa uma árvore rasa para isolar o atraso inicial. O notebook está configurado com 40 árvores, profundidade 20 e importância por permutation. Além disso, a variável `BANKAI_BINNING_STRATEGY='sampled_select'` existe no notebook, mas o argumento `binning_strategy=BANKAI_BINNING_STRATEGY` está comentado na construção do Bankai; naquela célula o default atual (`exact_sort`) é usado. Portanto, os tempos de histogramização acima usam `sampled_select` e não devem ser tratados como o tempo de histogramização da configuração efetiva do notebook. A validação, ordenação de linhas e cópia da matriz são anteriores e não dependem dessa escolha de binning.

## Onde o tempo é gasto

1. [`estimator.py`](../python/bankai_random_forest/estimator.py#L196) valida a entrada pedindo `float64`; isso converte a entrada `float32` de 2 GB para uma matriz de 4 GB.
2. [`estimator.py`](../python/bankai_random_forest/estimator.py#L750) chama `_prepare_training_data` em todo `fit`. No caminho denso, cria as chaves de `np.lexsort` com todas as features ([linha 776](../python/bankai_random_forest/estimator.py#L776)), calcula a ordem ([linha 777](../python/bankai_random_forest/estimator.py#L777)) e materializa `X[order]` ([linha 781](../python/bankai_random_forest/estimator.py#L781)). O timer atual cobre a função inteira; não separa o custo da ordenação do custo da cópia/reordenação. Uma amostragem da pilha durante a execução observou cópia NumPy (`memmove`) nessa etapa.
3. [`lib.rs`](../src/lib.rs#L436) transforma a matriz validada em um `Vec<f64>` com `view.iter().copied().collect()` ([linha 439](../src/lib.rs#L439)). Esse passe serial levou 62,5 s no caso de 1M.
4. Depois disso, [`dense.rs`](../src/dense.rs#L917) cria os cortes e aplica os bins; no perfil `sampled_select`, os tempos foram 2,2 s e 3,4 s. Essas rotinas usam as threads disponíveis para esse pré-processamento. Logo, a ponte Python/Rust não custa minutos por chamadas repetidas: os minutos decorrem da ordenação/reordenação antes da chamada e da movimentação integral dos dados.

## Plano priorizado para mitigar

### P0 — Separar a ordenação da cópia e verificar se a ordenação é necessária

Instrumentar `np.lexsort` e `X[order]` como etapas distintas na configuração real do notebook (`n_estimators=40`, 500 features e binning explicitamente escolhido). Em seguida, avaliar remover a ordenação canônica do caminho comum de `fit`: `_training_row_order` só é solicitado quando `_track_training_row_order` está ativo, enquanto a ordenação e a cópia ocorrem também no caminho normal. Preservar o comportamento exigido pelo caminho multioutput. Antes de adotar a mudança, comparar métricas, previsões, importância permutation, pesos/classes e comportamento com `random_state` fixo; a ordem das linhas pode alterar quais amostras bootstrap são selecionadas para a mesma seed.

### P1 — Evitar a expansão `float32 → float64 → Vec<f64>`

Adicionar ingestão densa `float32` no limite Rust/PyO3 e histogramizar a partir do buffer de origem, produzindo diretamente os bins compactos `u8` e mantendo os cortes em `f64` quando necessário. Isso pode eliminar a matriz Python de 4 GB e o `Vec<f64>` de 4 GB no caso de 1M. Validar NaN/inf, thresholds, previsões e métricas nos caminhos `exact_sort`, `sampled_select` e entrada `float64`; medir RSS e tempo de cópia novamente.

### P2 — Reavaliar localidade e paralelismo após reduzir as cópias

Com a preparação reduzida, medir separadamente cálculo de cortes e aplicação dos bins na configuração efetiva do notebook. Manter a comparação com a mesma semente, `n_jobs=-1`, importância permutation e número/profundidade de árvores; executar várias repetições e comparar mediana e dispersão. Só então priorizar otimizações SIMD ou mudanças de escalonamento das threads: no perfil isolado, os dois passes de histograma somaram 5,5 s, muito abaixo dos 343,0 s gastos na preparação Python e na cópia serial para Rust.

## Conclusão

O atraso observado antes do uso alto de CPU **não é causado principalmente pelo custo de atravessar a ponte Python→Rust**. O maior alvo é `_prepare_training_data` (280,5 s), seguido pelo cast da entrada para `float64` e pela cópia serial da matriz em `matrix_to_owned` (62,5 s). A primeira otimização deve atacar ordenação/reordenação e depois eliminar cópias de matriz; acelerar o loop de histograma antes disso não remove a espera de vários minutos.
