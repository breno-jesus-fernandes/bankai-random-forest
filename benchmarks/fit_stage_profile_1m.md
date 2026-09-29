# Perfil do `fit` em 1 milhão de linhas

## Perfil original (antes das otimizações)

No baseline, a espera longa antes de as threads entrarem no processamento era explicada principalmente pelo trabalho de preparação em Python e pela cópia integral da matriz para o Rust. O despacho Python→Rust em si ocorre uma vez e não é o gargalo: antes de a extensão começar, o perfil gastou **301,2 s**; dentro da chamada nativa, a cópia da matriz consumiu mais **62,5 s**.

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

## Otimizações implementadas e nova medição

O trabalho priorizado já foi aplicado em duas etapas. Os tempos abaixo usam o
mesmo gerador, seed e configuração rasa nas três medições, em build Rust
`--release`. A geração de `X` e `y` continua fora do cronômetro.

| Tamanho | Baseline | P0: ordenação/cópia no Rust | P1: preserva `float32` | Ganho baseline → P1 |
|---|---:|---:|---:|---:|
| 100 mil × 500 | 5,322 s | 0,557 s | 0,517 s | 10,3× |
| 1 milhão × 500 | 387,852 s | 248,845 s | **41,571 s** (mediana de 3) | **9,3×** |

No caso de 1M, as três medições P1 foram 41,571 s, 34,535 s e 54,615 s. Duas
execuções repetidas produziram as mesmas previsões nas primeiras 10 mil linhas.
O ganho de P1 contra a medição P0 única foi 6,0× (83,3% menos tempo). Os
resultados são uma referência de diagnóstico, não um intervalo estatístico: o
baseline e P0 tiveram uma execução cada, e este host tem 8 GiB de RAM e sofre
pressão de memória variável.

O perfil de 1M continua usando uma árvore (`n_estimators=1`, `max_depth=1`) e
`importance_type='gain'` para isolar custo de preparação e transferência. Não é
uma medição do notebook completo nem uma comparação direta com LightGBM. O pico
RSS amostrado foi 3,782 GiB no baseline e aproximadamente 2,0 GiB nas primeiras
execuções P0/P1; em um processo que repetiu P1 duas vezes, o amostrador observou
2,592 GiB. O RSS do macOS é sensível a compressão e paginação, portanto esses
números são indicativos.

### P0 — Ordenação canônica e cópia densa

O caminho denso single-output deixa de construir as chaves de `np.lexsort` e
`X[order]` em Python. O Rust calcula uma ordem lexicográfica estável refinando
grupos empatados feature a feature, com NaNs depois dos valores finitos, e
preenche o `Vec<f64>` final diretamente nessa ordem. O vetor é contíguo e evita
a matriz NumPy intermediária reordenada. O índice usado também mantém labels,
pesos e predições OOB alinhados.

A ordenação canônica não foi removida: uma tentativa de ignorá-la alterou o
comportamento de bootstrap com seed fixa e falhou na equivalência entre peso
inteiro e repetição de linhas e no check comum do sklearn. Multioutput e sparse
continuam no caminho Python anterior; o ganho P0 se aplica ao caminho denso
single-output.

### P1 — Preservação de `float32` até a fronteira nativa

`validate_data` agora mantém matrizes `float32` e `float64` no formato recebido;
outros tipos numéricos continuam sendo convertidos para `float64`. O extrator
PyO3 aceita os dois formatos e converte valores `float32` para `f64` enquanto
preenche o buffer nativo, já na ordem final. Como todo valor `float32` é
representável exatamente em `f64`, isso preserva os valores e evita criar antes
uma matriz Python `float64` de 4 GB para o caso de 1M × 500. O núcleo de treino
continua armazenando os valores em `f64`.

Foi acrescentado um teste comparando `float32` com os mesmos valores em
`float64`, incluindo NaNs, histogramas, OOB, previsões e importâncias. O teste
de linha invariável cobre também ties, zeros com sinal e NaNs.

### Controles de histogramas incluídos na branch

O estimator expõe `binning_strategy` (`exact_sort`, `sampled_sort`,
`exact_select`, `sampled_select`) e `bin_sample_size` (200 mil por default). As
estratégias sampled selecionam um subconjunto determinístico por feature; as
estratégias select usam multi-select de ranks em vez de ordenar todas as
observações. Histogramas e pré-processamento dividem o orçamento de threads
com a construção das árvores em matrizes grandes. Essas opções mudam o custo e,
dependendo de empates/amostragem, podem mudar os cortes; compare qualidade junto
com velocidade ao escolher uma estratégia.

## Diagnóstico do código antes das mudanças

1. No baseline, `validate_data(dtype=np.float64)` expandia a matriz `float32` de 2 GB para 4 GB. O caminho atual preserva `float32`/`float64` em [`estimator.py`](../python/bankai_random_forest/estimator.py#L193).
2. O baseline sempre construía chaves com `np.lexsort` e materializava `X[order]`. O caminho denso single-output agora adia essa operação em [`_prepare_training_data`](../python/bankai_random_forest/estimator.py#L781); o código Python de ordenação continua como fallback para sparse/multioutput.
3. [`lib.rs`](../src/lib.rs#L463) determina a ordem canônica e [`matrix_to_owned`](../src/lib.rs#L573) converte/copia as linhas direto para o buffer contíguo nativo. No baseline, esse passe serial levou 62,5 s no caso de 1M.
4. [`dense.rs`](../src/dense.rs#L917) cria os cortes e aplica os bins. No perfil baseline `sampled_select`, esses passes somaram 5,5 s em 1M, muito abaixo do tempo de preparação/cópia. A ponte Python→Rust acontece uma vez; não explica sozinha a espera de minutos.

## Próxima etapa do plano

### P2 — Evitar o buffer denso de `f64` e medir o fit integral

A fronteira nativa ainda materializa um `Vec<f64>` completo antes de produzir os
bins compactos `u8`. A próxima etapa é alimentar a discretização diretamente do
buffer `float32` de origem, preservando os cortes em `f64` e a ordem de linhas.
Depois, medir separadamente construção/aplicação de bins e o notebook integral
(40 árvores, profundidade 20, importância permutation, `n_jobs=-1`), em várias
repetições e comparando qualidade e pico de memória. O perfil atual mostra que
reduzir a preparação tem impacto alto; não mede a velocidade relativa ao
LightGBM nem comprova ainda ganho no treino completo.

## Conclusão

O atraso inicial vinha da preparação Python e da transferência da matriz, não de
chamadas repetidas pela ponte Python→Rust. Ordenar e copiar a matriz densa no
Rust e preservar `float32` até essa fronteira reduziram o perfil raso de 1M de
387,9 s para uma mediana de 41,6 s neste host. O próximo gargalo concreto é a
materialização do buffer nativo `f64`; eliminá-la e medir o fit integral são os
próximos passos antes de concluir sobre desempenho frente ao LightGBM.
