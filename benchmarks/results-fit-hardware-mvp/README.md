# MVP de construção de árvores: arena, histogramas e localidade

Branch: `feature/fit-hardware-mvp`. Baseline: `964710b153ae6ac55db8bcaf5a00f86181342253`.

O MVP reduziu a mediana do tempo de `fit` entre **9,6% e 16,3%** nos seis cenários medidos (speedup de aproximadamente **1,11× a 1,19×**). As previsões e quantidades de folhas do Bankai coincidiram com a baseline em todas as 30 medições do Bankai. Os ganhos são do conjunto de alterações; não houve ablação para atribuir percentuais a cada técnica.

## Implementação

- **Arena por árvore:** `crates/xrf/src/forest/tree.rs` guarda `Node<I>` em um `Vec`, com filhos `usize`. A inserção em pós-ordem preserva a sequência de construção e o consumo do RNG. A reserva considera amostras e limite de folhas, com teto inicial de 8.191 nós e crescimento geométrico posterior. Predição, poda, percurso, importância por permutação e exportação SHAP usam os índices.
- **Subtração in-place:** `RfInput::split_cache_children` recebe a propriedade do cache do pai. O filho menor continua sendo contado diretamente; o buffer do pai é subtraído e transferido ao maior. Os ancestrais deixam de reter cópias dos histogramas durante a descida.
- **Histogramas contíguos:** `src/dense.rs` usa um vetor de pesos, um de contagens e um de offsets por cache, com views por feature. Pela estrutura do código, as novas alocações de histogramas por split passam de `4F + 2` para `3`, com `F` features; essa contagem não inclui máscaras, estatísticas dos alvos e demais temporários. O cache raiz passa de `2F + 1` para `3`. São contagens analíticas, não medições de um profiler de alocação.
- **Kernel de contagem:** preserva os bins densos `u8` por linha, acessa uma fatia da linha e offsets contíguos, e especializa o cálculo de endereços para duas e quatro classes. As demais quantidades de classes usam o caminho genérico.
- **Valores ausentes:** dois buffers candidatos são reutilizados entre os bins, eliminando os clones de vetores dentro desse loop.

## Medições de tempo

Tempo de parede em segundos, mediana de cinco fits medidos após um warmup por implementação e cenário. `Redução` significa `1 − MVP / baseline`; não é aumento percentual de throughput. A coluna LightGBM usa as medições intercaladas com o MVP.

| Cenário | Threads | Bankai baseline | Bankai MVP | Redução | LightGBM RF | MVP / LightGBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| binary | 1 | 0.2150 | 0.1889 | 12.1% | 0.1766 | 1.07× |
| binary | 4 | 0.1098 | 0.0993 | 9.6% | 0.2312 | 0.43× |
| wide_multiclass | 1 | 0.7681 | 0.6767 | 11.9% | 1.1535 | 0.59× |
| wide_multiclass | 4 | 0.3443 | 0.3017 | 12.4% | 1.0915 | 0.28× |
| deep_binary | 1 | 0.8432 | 0.7395 | 12.3% | 0.5496 | 1.35× |
| deep_binary | 4 | 0.3029 | 0.2535 | 16.3% | 1.2730 | 0.20× |

No binário com 63 folhas e uma thread, a razão de tempo caiu de **1,22× para 1,07×** usando a mesma referência LightGBM da rodada MVP. No binário com 511 folhas, caiu de **1,53× para 1,35×**. Assim, o MVP se aproximou do LightGBM em tempo de processamento nesses casos, mas ainda ficou atrás com uma thread.

Com quatro threads, o Bankai foi mais rápido neste ambiente. O LightGBM apresentou overhead/variabilidade e não escalou bem nesses workloads pequenos; isso é um resultado local, sem diagnóstico causal de escalabilidade. Não extrapolar as razões para datasets maiores ou outras máquinas.

## Qualidade e trabalho efetivo

Acurácia em 4.000 amostras de validação. Os valores são iguais com uma e quatro threads e entre os builds de cada implementação.

| Cenário | Acurácia Bankai | Acurácia LightGBM | Árvores Bankai / LightGBM | Folhas totais Bankai / LightGBM |
| --- | ---: | ---: | ---: | ---: |
| binary | 73.62% | 86.92% | 40 / 40 | 2520 / 2520 |
| wide_multiclass | 56.50% | 79.22% | 40 / 160 | 2520 / 10080 |
| deep_binary | 90.70% | 92.30% | 40 / 40 | 20440 / 20440 |

**Não há equivalência de qualidade.** O LightGBM foi mais acurado; com apenas 63 folhas, a diferença foi grande. O caso `deep_binary` tem qualidade mais próxima (90,70% versus 92,30%). Não houve ajuste de hiperparâmetros para igualar acurácia. LightGBM multiclasse construiu quatro vezes mais árvores; seu tempo não representa o mesmo número de árvores do Bankai nesse cenário.

Bankai usa Gini, bootstrap com reposição e seu crescimento atual; LightGBM RF usa estatísticas de gradiente e bagging. Igualar limites de folhas, profundidade e bins não iguala o trabalho nem os algoritmos.

## Protocolo e ambiente

- macOS 26.6.2, ARM64, 8 CPUs lógicas; Rust 1.97.1, Python 3.11.11, NumPy 2.4.6, LightGBM 4.7.0.
- Ambos os builds Bankai em release, `opt-level=3`, LTO fat, uma codegen unit e `RUSTFLAGS='-C target-cpu=native'`.
- `binary`: 20.000 linhas × 32 features, duas classes, 63 folhas, profundidade 12, mínimo de 20 amostras por folha.
- `wide_multiclass`: 12.000 linhas × 128 features, quatro classes; mesmos limites acima.
- `deep_binary`: 20.000 linhas × 32 features, duas classes, 511 folhas, profundidade 20, mínimo de cinco amostras por folha.
- Todos: `n_estimators=40`, 63 bins, todas as features elegíveis, fração amostral nominal de 80%, seed do dataset 1729 e do modelo 42. Cinco repetições da mesma seed medem ruído de execução, não variabilidade estatística entre datasets/seeds.
- Os fits de Bankai e LightGBM alternam a ordem a cada repetição. Importação, geração dos dados, predição, métrica e exportação de árvores ficam fora do cronômetro. Conversão de entrada, discretização e demais etapas internas de `fit` ficam dentro dele. Não foi isolado apenas o kernel de construção.
- Sem importância por permutação ou OOB no benchmark. A geração de estatísticas normais do modelo permanece incluída.
- A baseline definitiva foi recompilada e executada sem testes ou compilação concorrentes; substitui a medição exploratória inicial. O MVP também foi medido sem essas tarefas concorrentes. O build MVP foi restaurado ao final.
- Os JSONs registram todas as repetições, hashes das previsões, fonte e extensão carregada. O campo `commit` aponta para o commit base porque as alterações MVP ainda não estavam commitadas durante a medição; `source_sha256` e `extension_sha256` distinguem os builds.

### Dispersão observada

Intervalos mínimo–máximo em segundos. Os controles LightGBM dos dois builds mostram a variabilidade do ambiente; não foram descartadas amostras.

| Cenário | Threads | Bankai baseline | Bankai MVP | LightGBM com baseline | LightGBM com MVP |
| --- | ---: | ---: | ---: | ---: | ---: |
| binary | 1 | 0.2145–0.2153 | 0.1886–0.1906 | 0.1779–0.2341 | 0.1755–0.1803 |
| binary | 4 | 0.1090–0.1104 | 0.0976–0.1067 | 0.2175–0.2455 | 0.2158–0.3032 |
| wide_multiclass | 1 | 0.7660–0.8026 | 0.6750–0.7336 | 1.1568–1.2062 | 1.1480–1.3798 |
| wide_multiclass | 4 | 0.3345–0.3662 | 0.2914–0.3276 | 1.0766–1.1346 | 1.0620–1.2012 |
| deep_binary | 1 | 0.8337–0.9527 | 0.7364–0.7401 | 0.5377–0.5694 | 0.5456–0.5523 |
| deep_binary | 4 | 0.2939–0.3047 | 0.2516–0.2551 | 1.1401–1.3434 | 1.2485–1.4472 |

## SIMD e limites do MVP

O assembly ARM64 de `split_cache_children`, com a subtração inline, contém `fsub.2d`, `fmaxnm.2d` e `sub.2d`, com loads/stores de registradores vetoriais. Isso confirma vetorização SIMD da **subtração** neste build; o trecho extraído está em [subtraction-arm64.s](subtraction-arm64.s). Não foi demonstrado que a baseline carecia de SIMD, nem que a contagem com escritas indiretas foi vetorizada. A melhoria de tempo não pode ser atribuída exclusivamente a SIMD.

Não foram medidos contadores de cache L1/L2, ciclos de CPU, largura de banda ou pico de memória. O layout de features permanece por linha; transposição/tiling, acumuladores replicados, pool completo de histogramas por worker e discretização compacta do CSR ficam para outra iteração. O MVP continua criando o buffer do filho menor e temporários por feature.

A arena altera a API Rust pública de `Tree` e o contrato de `RfInput`; consumidores Rust externos precisarão adaptar-se. A API Python foi preservada nos testes. A poda deixa nós inacessíveis na arena, e a reserva pode reter capacidade excedente até a destruição da árvore; não houve compactação. Construção e predição continuam recursivas.

## Validação

- `cargo test --workspace`: **21 testes aprovados**, incluindo CLI, equivalência de histogramas, bootstrap, reutilização dos buffers e índices da arena.
- **135 testes Python aprovados** em fit, núcleo nativo, caminhos SHAP, compatibilidade SHAP, serialização e auditoria de compatibilidade sklearn. Os dois warnings foram de `log(0)` no teste de probabilidades.
- Comparação dos 30 resultados Bankai baseline/MVP: hashes das previsões e número de folhas idênticos. Os 30 controles LightGBM também coincidiram nesses campos.
- `git diff --check` sem erros.

## Reprodução

Na raiz do checkout MVP, com as dependências do projeto e de benchmark instaladas:

```sh
VIRTUAL_ENV="$PWD/.venv" RUSTFLAGS='-C target-cpu=native' .venv/bin/maturin develop --release --skip-install
.venv/bin/python benchmarks/run_fit_hardware_mvp.py --label mvp --output benchmarks/results-fit-hardware-mvp/mvp.json
```

Para a baseline, usar checkout separado do commit informado, copiar o mesmo runner para ele e compilar com as mesmas flags. Executar em ambiente equivalente com `--label baseline` e saída separada. `--skip-install` evita uma incompatibilidade observada entre o maturin instalado e o comando `uv pip install --group`; as dependências já estavam disponíveis.

Para gerar assembly:

```sh
RUSTFLAGS='-C target-cpu=native' cargo rustc --release --lib -- --emit=asm
```

Dados completos: [baseline.json](baseline.json), [mvp.json](mvp.json). Runner: [run_fit_hardware_mvp.py](../run_fit_hardware_mvp.py).
