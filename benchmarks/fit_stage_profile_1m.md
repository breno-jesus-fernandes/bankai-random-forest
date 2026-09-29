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

O perfil não é o benchmark integral do notebook: ele usa uma árvore rasa para isolar o atraso inicial. O notebook está configurado com 40 árvores, profundidade 20 e importância por permutation no Bankai. O argumento `binning_strategy=BANKAI_BINNING_STRATEGY` agora está ativo. Em 100k linhas, `bin_sample_size=200_000` cobre todas as linhas; em 1M, amostra 200 mil. A geração do dataset fica fora do tempo de fit.

## Otimizações implementadas e nova medição

O trabalho priorizado já foi aplicado em três etapas. Os tempos abaixo usam o
mesmo gerador, seed e configuração rasa nas quatro versões, em build Rust
`--release`. A geração de `X` e `y` continua fora do cronômetro.

| Tamanho | Baseline | P0: ordenação/cópia no Rust | P1: preserva `float32` | P2: bins direto da matriz | Ganho baseline → P2 |
|---|---:|---:|---:|---:|---:|
| 100 mil × 500 | 5,322 s | 0,557 s | 0,517 s | 0,509 s | 10,4× |
| 1 milhão × 500 | 387,852 s | 248,845 s | 41,571 s (mediana de 3) | **3,306 s** (mediana de 3) | **117,3×** |

No caso raso de 1M, P1 mediu 41,571 s, 34,535 s e 54,615 s; P2 mediu 3,678 s,
3,306 s e 3,216 s. As previsões foram iguais entre repetições com a mesma seed.
P2 ficou 12,6× abaixo da mediana P1. O baseline e P0 tiveram uma execução cada;
portanto, os ganhos são referências de diagnóstico, não intervalos estatísticos.
Este host tem 8 GiB de RAM e sofre pressão de memória variável.

O caso de 100k raso no quadro usa `float32`, igual ao notebook. Também medi
`float64` em 100k × 500: o caminho anterior teve mediana de 0,453 s
(0,439–0,481 s) e o accessor P2, 0,492 s (0,492–0,518 s), cerca de 8,6% mais
lento. O ganho de evitar a cópia depende do tamanho e do dtype; o caminho direto
não é mais rápido para toda carga pequena.

O perfil raso de 1M usa uma árvore (`n_estimators=1`, `max_depth=1`) e
`importance_type='gain'` para isolar a preparação. O RSS amostrado foi 3,782 GiB
no baseline, aproximadamente 2,0 GiB nas primeiras execuções P0/P1 e 1,983 GiB
na primeira execução P2; os processos com várias repetições chegaram a 2,6 GiB.
O RSS do macOS é sensível a compressão e paginação, então não há evidência
conclusiva de queda de pico de RSS entre P1 e P2, apesar da alocação `Vec<f64>`
de 4 GB ter sido removida no caminho P2.

Também foi medido o fit completo do notebook, com 40 árvores, profundidade 20,
511 folhas, `n_jobs=-1`, Bankai permutation importance e LightGBM gain sem
permutation importance:

| Dados | Bankai P1 | Bankai P2 | LightGBM 4.7 RF | Repetições |
|---|---:|---:|---:|---:|
| 100k × 500 | 7,278 s | 7,284 s | 12,822 s | 3 por configuração |
| 1M × 500 | — | **49,703 s** | 89,578 s | 3 por configuração |

No conjunto de 1M, P2 variou de 43,298 a 50,356 s e LightGBM de 88,118 a
93,356 s. As medianas indicam 1,80× para Bankai frente ao LightGBM, mesmo com
permutation importance apenas no Bankai; a medição não isola apenas construção
de árvores e as configurações RF são aproximadas, não matematicamente idênticas.
Em 100k, P1 e P2 ficaram empatados no fit integral: a preparação já é pequena
frente ao treino e à importance permutation.

O maior RSS amostrado foi 1,982 GiB para o primeiro fit completo P2 de 1M e
3,794 GiB para o primeiro LightGBM; um processo posterior que alternou execuções
observou 3,145 GiB de pico total. A medição do RSS do macOS variou entre
processos e não sustenta uma comparação precisa de memória.

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

### P2 — Discretização direto da matriz de origem

Quando a entrada é densa `float32` ou `float64` e histogramas estão ativos, o
Rust mantém uma view emprestada da matriz NumPy e acessa as linhas na ordem
canônica. O cálculo de cortes e a aplicação dos bins escrevem diretamente no
buffer compacto `u8`; esse caminho elimina a alocação intermediária completa de
`Vec<f64>` (4 GB em 1M × 500). Entrada sparse e treino sem histogramas mantêm o
caminho anterior. Um teste Rust compara os cortes e todos os bins do accessor
com o caminho de matriz própria, incluindo NaNs, seleção amostrada e
pré-processamento paralelo; testes Python cobrem dtypes e matriz `float32` com
strides.

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
3. [`lib.rs`](../src/lib.rs#L564) determina a ordem canônica. O caminho baseline/fallback em [`matrix_to_owned`](../src/lib.rs#L674) converte e copia as linhas para um buffer contíguo; no baseline, esse passe serial levou 62,5 s no caso de 1M. O caminho P2 histogramizado acessa a view diretamente e evita essa cópia.
4. [`dense.rs`](../src/dense.rs#L1036) cria os cortes e aplica os bins. No perfil baseline `sampled_select`, esses passes somaram 5,5 s em 1M, muito abaixo do tempo de preparação/cópia. A ponte Python→Rust acontece uma vez; não explica sozinha a espera de minutos.

## Próxima etapa do plano

### P3 — Melhorar o ponto de troca e reduzir passes

O caminho direto cobre agora entrada densa `float32` e `float64` com histograma.
O próximo alvo é medir um ponto de troca por dtype/tamanho — o accessor é
ligeiramente mais lento em 100k `float64` — e combinar validação de NaNs/inf com
a aplicação dos bins para remover um passe. Também foi observado que o vetor de
permutation importance muda entre execuções multithread com seed fixa, enquanto
previsões e soma global da importance permanecem estáveis; vale separar essa
variância de agregação do trabalho de desempenho antes de depender de
comparações por feature.

## Conclusão

O atraso inicial vinha da preparação Python e da transferência da matriz, não de
chamadas repetidas pela ponte Python→Rust. A ordenação no Rust, a preservação de
`float32` e a discretização direta reduziram o perfil raso de 1M de 387,9 s para
3,3 s de mediana neste host. No fit integral de 1M, três execuções deram 49,7 s
de mediana para Bankai P2 e 89,6 s para LightGBM RF; o Bankai calcula permutation
importance e o LightGBM não. Esse resultado consolida o patamar neste host, mas
não remove a diferença de trabalho nem substitui uma comparação em máquinas de
produção.
