# Extra Trees: campanha rastreável

As medições antigas dos notebooks são históricas. O notebook local anterior
executava warmups, mas aceitava CSVs existentes sem identificar os fontes e o
binário. Elas não demonstram o efeito do build atual.

Por orienta??o do usu?rio, futuros benchmarks usam `bootstrap=False` e `max_samples=None` em Bankai e sklearn. LightGBM e XGBoost mant?m seu subsampling pr?prio. Campanhas anteriores conservam seus par?metros originais.

O runner `benchmarks/run_extra_trees_campaign.py` usa por padrão 500 mil linhas,
500 features, 100 árvores e três repetições. Fraude vem primeiro. Bankai Extra
Trees e sklearn Extra Trees recebem os mesmos dados, split e seed nominal;
Bankai Random Forest é o controle. A ordem é invertida nas repetições ímpares.
Os offsets dos cenários são estáveis quando só um subconjunto é executado.

```powershell
.venv/Scripts/python.exe benchmarks/run_extra_trees_campaign.py
.venv/Scripts/python.exe benchmarks/run_extra_trees_campaign.py --all-scenarios --all-models
```

Cada execução cria uma pasta exclusiva `benchmarks/campaign-extra-trees-*`.
`manifest.json` identifica os fontes, `_core.pyd`, versões, hardware e configuração.
`timing-*.json` salva o fit antes dos diagnósticos; `fit-*.json` é o checkpoint
completo. Um timing sem métricas completas é reexecutado na retomada. `raw.csv`
contém as observações; `medians.csv`, `paired.csv` e `paired-medians.csv` apresentam
medianas e diferenças pareadas. Os JSONs conservam parâmetros efetivos, ranking,
features informativas perdidas, número de nós e profundidade. As contagens de
positivos nas folhas usam o conjunto de teste sem pesos; não são o cover do bag.

Para retomar, passe `--resume <pasta>` e os mesmos argumentos originais. Mudanças
nos fontes, binário, versões, hardware ou configuração recusam a retomada.
Não copie CSVs antigos para uma campanha nova.

## Semântica implementada

- `extra_trees=True` exige `max_bins=None` tanto na entrada densa quanto na esparsa. Inteiros s?o rejeitados com `ValueError` no Python e na fronteira nativa.

- Bootstrap Extra Trees usa os pesos efetivos como probabilidades de sorteio.
  Cada linha sorteada aparece uma vez no bag; sua multiplicidade vira o peso.
  `min_samples_leaf` e `min_samples_split` contam linhas distintas. Pesos de
  `balanced_subsample` são calculados sobre as multiplicidades do bag.
- Pesos inteiros não são expandidos em linhas duplicadas no Python para Extra
  Trees. Random Forest conserva a implementação anterior.
- Importância por gain é normalizada por árvore, somada e renormalizada. Árvores
  sem splits contribuem zero. O método nativo `gain_importances()` conserva a
  soma global para diagnóstico de ablação sem alterar as árvores.
- Features são visitadas sem reposição. Constantes descobertas são herdadas
  pelos filhos; constantes não consomem o orçamento de features. Se nenhum
  candidato válido existir, a busca pode ultrapassar `max_features`.
- O amostrador reinicia seu estado por árvore. `extra_seed` é misturado antes
  da conversão para o uniforme, evitando descartar seeds pequenas nos bits baixos.
- Entrada densa float32 permanece em armazenamento float32 durante o treino.
  A conversão inicial ainda passa temporariamente por float64; essa alocação
  transitória deve ser considerada em futuras otimizações de preparação.
- A busca binária densa reutiliza um buffer contíguo por worker; cada feature é
  lida da matriz uma vez, e a segunda passagem lê o buffer. A partição acontece
  apenas para o candidato vencedor. O layout nativo permanece por linha e a
  ordenação canônica é preservada.

A referência auditada é scikit-learn 1.9.1, em especial
[`_forest.py`](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/ensemble/_forest.py),
[`_bootstrap.py`](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/ensemble/_bootstrap.py)
e [`_splitter.pyx`](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.1/sklearn/tree/_splitter.pyx).
Não se exige igualdade das árvores entre bibliotecas: RNG, tratamento de NaN
(Bankai avalia as duas direções) e probabilidades preditas (Bankai vota pela
classe da folha) continuam diferentes. A detecção de constantes do Bankai usa
igualdade exata; sklearn usa tolerância float32. Essas diferenças permanecem
na interpretação de qualidade.

## Instrumentação

```powershell
.venv/Scripts/python.exe benchmarks/run_extra_trees_campaign.py --instrumented --repeats 1 --warmups 0
```

`BANKAI_PROFILE_FIT=1` habilita `fit_profile_` no estimador. O perfil separa
preparação/validação Python, ordenação canônica, preparação nativa, construção
das árvores, bootstrap, busca, partição e importância. Tempos internos de
workers são somas de tempo decorrido por etapa e podem superar o tempo de
parede; não devem ser somados entre si para reconstruir o tempo total. Os
contadores de leituras de amostras, rejeições e buffer são atualmente específicos
da busca binária densa. Não constituem um perfil completo das alocações do fit.
Campanhas instrumentadas têm configuração distinta; o runner rejeita perfil
habilitado acidentalmente numa campanha de timing.

## Baseline e critérios

`benchmarks/baseline-extra-trees` preserva o binário anterior, os fontes da
fronteira nativa e do splitter e o wrapper Python. O subdiretório `python` pode
ser usado via `PYTHONPATH` em um processo separado, sem substituir o build atual.

Fraude tem metas provisórias: mediana de tempo Bankai ≤ sklearn, diferença
pareada mediana de AP ≥ −0,01 e no máximo uma feature informativa adicional
perdida nas top 24. Três repetições são triagem; o aceite definitivo continua
exigindo dez repetições. A comparação adicional de bootstrap mantém separados
os resultados de `bootstrap=False` e os parâmetros originais do benchmark.

A otimiza??o final impede busca de splits em n?s com menos de `2 * min_samples_leaf` linhas distintas. Foi motivada pelo perfil da primeira campanha. A compara??o hist?rica de bootstrap foi encerrada; os novos benchmarks usam exclusivamente bootstrap desligado.

Atualiza??o da API: `extra_trees=True` agora exige `max_bins=None`; inteiros s?o rejeitados com `ValueError` no Python e no n?cleo nativo. A equival?ncia com `max_bins=63` descrita acima corresponde ao comportamento anterior. Novas campanhas passam `None` explicitamente para Extra Trees.
