# Crescimento por maior ganho com limite de folhas

Implementado na branch `feature/fit-hardware-mvp`, a partir de `548e3a5`.

O crescimento com `max_leaf_nodes` finito agora escolhe a folha cujo split produz o maior ganho global de impureza. A mudança recuperou acurácia e manteve tempos competitivos nos workloads locais. O custo em relação ao MVP anterior foi de **4,7% a 30,5%** na mediana de fit, dependendo do cenário. Trata-se de uma correção que muda a árvore e o trabalho realizado; o custo não pode ser atribuído apenas à fila de prioridade.

## Implementação

- `crates/xrf/src/forest/tree.rs`: uma `BinaryHeap` armazena candidatos de folhas, ordenados pelo score global retornado pelo input. Empates entre nós usam o índice da arena, de forma determinística. O score do `DenseInput` já é ponderado por peso do nó / peso total de treinamento.
- Os candidatos guardam máscara, histograma, profundidade restante, bounds monotônicos e melhor split. Ao expandir o melhor candidato, seus dois filhos são avaliados e entram na fila caso sejam divisíveis. Nós não expandidos permanecem folhas válidas.
- A arena com filhos `usize` continua sendo usada. O histograma do pai é consumido para derivar os filhos por subtração; folhas terminais liberam seu cache. Ao atingir o limite, as máscaras da fila são devolvidas ao pool e os caches descartados. Os últimos dois filhos não têm histogramas construídos quando não há orçamento para mais splits.
- O caminho sem limite de folhas mantém o crescimento recursivo existente. Poda, valores ausentes, pesos, exportação SHAP e limites monotônicos continuam usando as mesmas representações.
- O contrato de `RfInput::new_split_with_bounds` documenta que scores precisam ser comparáveis entre nós. Implementações Rust externas que devolvem apenas ganho local precisam normalizá-lo para utilizar o crescimento com limite de folhas.

## Testes

`cargo test --workspace`: **21 aprovados**.

`.venv/bin/python -m pytest -q`: **155 aprovados**, com dois warnings já conhecidos de `log(0)`.

O novo arquivo `tests/test_best_first_growth.py` inclui oito casos: seleção do maior ganho global mesmo quando o ganho local é menor e o ramo seria visitado por último; modos exato/histograma; pesos; respeito a profundidade/folhas/mínimo de amostras; reproducibilidade entre uma e três threads; monotonicidade e poda. A fixture de ganho global falharia com o crescimento anterior em profundidade.

## Tempo de fit

Segundos, mediana de cinco repetições após um warmup por cenário e implementação. Mesmos dados e parâmetros do benchmark anterior, com seed 42. Os fits das duas bibliotecas alternam a ordem a cada repetição. Sem testes ou compilação concorrentes.

| Cenário | Threads | MVP anterior (DFS) | Maior ganho | Custo adicional | LightGBM RF |
| --- | ---: | ---: | ---: | ---: | ---: |
| binary | 1 | 0.1895 | 0.2473 | 30.5% | 0.2039 |
| binary | 4 | 0.0960 | 0.1137 | 18.5% | 0.2074 |
| wide_multiclass | 1 | 0.6770 | 0.8424 | 24.4% | 1.1378 |
| wide_multiclass | 4 | 0.2826 | 0.3312 | 17.2% | 1.0660 |
| deep_binary | 1 | 0.7345 | 0.7689 | 4.7% | 0.5437 |
| deep_binary | 4 | 0.2430 | 0.2698 | 11.0% | 1.1857 |

Com quatro threads, o binário de 63 folhas levou aproximadamente **0,114 s**, versus **0,207 s** do LightGBM (1,82× de speedup local). No binário de 511 folhas foram **0,270 s** versus **1,186 s**. O LightGBM não escalou bem nesses casos pequenos; não generalizar essa diferença para datasets maiores. Com uma thread, ele foi mais rápido nos dois binários.

## Acurácia e tamanho dos modelos

Acurácia na validação com 4.000 exemplos, seed 42; igual nas execuções de uma e quatro threads.

| Cenário | MVP anterior | Maior ganho | LightGBM RF | Árvores Bankai / LightGBM | Folhas totais Bankai / LightGBM |
| --- | ---: | ---: | ---: | ---: | ---: |
| binary | 73.625% | 89.975% | 86.925% | 40 / 40 | 2520 / 2520 |
| wide_multiclass | 56.500% | 74.725% | 79.225% | 40 / 160 | 2520 / 10080 |
| deep_binary | 90.700% | 94.200% | 92.300% | 40 / 40 | 20440 / 20440 |

Nos dois cenários binários, o Bankai ficou mais acurado que o LightGBM neste dataset. No multiclasse, ainda ficou abaixo. O LightGBM constrói uma árvore por classe em cada rodada: são 160 árvores contra 40 do Bankai nesse cenário. As bibliotecas diferem em critério, amostragem e valores das folhas; não há equivalência de algoritmo/trabalho.

### Verificação em três seeds

Mantendo 63 folhas e seeds 42, 43 e 44, a melhora também se repetiu:

| Cenário | Média DFS anterior | Média maior ganho | Faixa maior ganho |
| --- | ---: | ---: | ---: |
| binary | 74.02% | 90.19% | 89.98%–90.58% |
| wide_multiclass | 57.17% | 75.00% | 74.72%–75.20% |

Todas as 240 árvores dessa verificação tiveram exatamente 63 folhas e subdividiram o segundo filho da raiz. O problema de esgotamento do orçamento em apenas um lado não apareceu nesses casos. Isso não significa que todo irmão precise sempre ser subdividido: na política implementada a decisão depende do ganho.

## Limitações e reprodução

- A fronteira de candidatos retém histogramas de folhas pendentes. Seu tamanho pode crescer com o orçamento de folhas; o consumo de memória pode superar o caminho DFS. Não foram medidos pico de RSS ou contadores de cache nesta rodada.
- As diferenças de sorteios e estruturas entre políticas são esperadas. O teste de paralelismo confirma reproducibilidade dentro da nova política, não identidade com DFS.
- Os resultados são locais: macOS ARM64 com oito CPUs lógicas, Python 3.11.11, Rust 1.97.1 e LightGBM 4.7.0. Os JSONs guardam amostras individuais, versões e hashes da fonte/extensão. O commit registrado é o pai porque a medição ocorreu antes do commit da implementação.
- A comparação mede `fit` completo, incluindo conversão/discretização; não apenas o kernel de crescimento. Não houve tuning para igualar acurácia nem medição em outras máquinas. O sorteio de features com reposição não foi alterado nesta correção.

Build e benchmark na raiz do checkout:

```sh
VIRTUAL_ENV="$PWD/.venv" RUSTFLAGS='-C target-cpu=native' .venv/bin/maturin develop --release --skip-install
.venv/bin/python benchmarks/run_fit_hardware_mvp.py --label best-first --output benchmarks/results-fit-hardware-mvp/best-first.json
.venv/bin/python benchmarks/investigate_fit_accuracy.py --only-63 --output benchmarks/results-fit-hardware-mvp/best-first-accuracy.json
```

A versão anterior foi recompilada a partir de `548e3a5` para gerar `dfs-control.json`, com as mesmas flags e runner. Depois foi restaurada a implementação por maior ganho e recompilada a extensão. O hash da fonte restaurada coincidiu com o da medição nova.

Arquivos: [controle DFS](dfs-control.json), [maior ganho](best-first.json), [três seeds](best-first-accuracy.json). O [relatório original](README.md) e a [investigação](ACCURACY.md) documentam o histórico anterior à correção.
