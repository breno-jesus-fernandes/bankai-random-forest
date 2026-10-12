# Roadmap de otimização do Extra Trees

## Objetivo

Fazer o modo `extra_trees=True` do Bankai competir com `ExtraTreesClassifier` do scikit-learn em velocidade e qualidade de ranking para seleção de features, mantendo a API atual. A implementação pode ser reestruturada se as medições justificarem.

## Diagnóstico atual

O notebook `benchmarks/banking_feature_importance_benchmark.ipynb` compara o wheel Bankai `0.1.0a12` com scikit-learn `1.9.1` em Linux x86_64. No benchmark completo, cada fit usa 400 mil linhas de treino, 500 features, 100 árvores, bootstrap com `max_samples=0.8`, e 10 repetições por cenário. O Bankai usa histogramas com 63 bins.

As medianas registradas foram:

| Cenário | Bankai Extra Trees | Extra Trees sklearn |
| --- | ---: | ---: |
| Inadimplência | 4,61 s | 2,64 s |
| Churn | 4,22 s | 2,78 s |
| Aceitação de oferta | 3,81 s | 2,94 s |
| Fraude em transações | 4,79 s | 2,66 s |

O código do Bankai já evita ordenar os valores no caminho Extra Trees sem histogramas. Porém, com `max_bins=63`, o treino passa por histogramas: prepara bins para a matriz, constrói histogramas por nó e avalia um bin aleatório por feature amostrada. A construção e atualização do cache considera todas as features, embora cada split avalie apenas um subconjunto. No modo histograma, a amostragem de features também não usa o caminho sem reposição disponível no modo exato.

O splitter aleatório do scikit-learn escolhe features sem reposição, encontra mínimo e máximo para cada feature visitada, sorteia um limiar e avalia a partição em código Cython. Ele também mantém informação sobre features constantes para reduzir trabalho nos descendentes. O uso de código nativo compilado e a partição direta das amostras evitam a preparação global de histogramas do caminho atual do Bankai.

O notebook dá suporte à hipótese de que o caminho atual é mais caro, mas não é um perfil de CPU. A contribuição relativa de preparação de bins, construção dos caches, partição das máscaras, cálculo de importância e paralelismo ainda precisa ser medida separadamente.

## Plano de implementação

1. **Comparar os caminhos de referência.** Clonar/inspecionar a versão `1.9.1` do scikit-learn, especialmente `_splitter.pyx`, o particionador denso e a execução paralela das árvores. Reproduzir os tempos do notebook no mesmo ambiente e conferir os parâmetros efetivos de cada estimador.
2. **Adicionar instrumentação do fit.** Separar tempo de validação/conversão de entrada, preparação dos bins, construção das árvores, cálculo de importância e demais etapas relevantes. Medir contagens de amostras e features visitadas, bytes temporários e alocações por etapa.
3. **Implementar um caminho especializado para Extra Trees.** Para cada nó, amostrar features sem reposição; calcular mínimo e máximo das features visitadas; sortear um limiar por feature; acumular contagens e pesos das classes para avaliar o candidato; selecionar o melhor candidato válido; particionar as amostras uma vez para criar os filhos. Evitar construir histogramas globais de todas as features nesse caminho.
4. **Reduzir custo de dados e alocação.** Medir layouts de memória e acessos em matrizes float32 contíguas; reutilizar buffers de partição e estatísticas; evitar cópias e caches de features não amostradas. Usar os perfis para decidir se uma representação de dados diferente ou paralelismo adicional traz ganho real.
5. **Preservar os demais modos.** Manter o caminho atual para Random Forest e outros usos de histogramas. Fazer o modo Extra Trees especializado continuar sob `extra_trees=True`, sem exigir mudança na API pública.

## Semântica e compatibilidade

- Preservar bootstrap e `max_samples`, pesos de amostras/classes, critérios, limites de profundidade, `min_samples_leaf`, importância por ganho e seeds reproduzíveis.
- Preservar o tratamento de valores ausentes, features constantes e splits inviáveis por limites de amostras/peso.
- Comparar distribuição e qualidade dos resultados, sem exigir igualdade de árvores ou rankings entre bibliotecas: a geração de números aleatórios e alguns detalhes internos diferem.
- Manter os modos não Extra Trees cobertos por regressão funcional e de desempenho.

## Validação e critérios de aceitação

- Reexecutar o benchmark completo do notebook em x86_64 com os mesmos datasets, seeds, parâmetros, número de árvores e jobs; medir pelo menos 10 repetições pareadas por cenário.
- Comparar tempo mediano de fit e métricas de ranking de features: average precision sobre features informativas, recall@24 e precision@24. Reportar intervalos de confiança pareados para avaliar estabilidade dos ganhos.
- Critério de aceite: tempo mediano do Bankai igual ou menor que o do Extra Trees sklearn e qualidade do ranking não inferior de forma estatisticamente significativa nos cenários do benchmark.
- Confirmar o comportamento em dados com features constantes, classes desbalanceadas, valores ausentes, limites de folha grandes e matrizes float32 contíguas.
- Verificar que as rotas Random Forest e histograma não Extra Trees mantêm resultados funcionais e que não sofrem regressão material de runtime.

## Referência de implementação

- Notebook e resultados: `benchmarks/banking_feature_importance_benchmark.ipynb`.
- Bankai: `src/dense.rs` (`best_split_exact`, `best_random_split_exact`, `best_split_histogram`, `split_cache` e `split_cache_children`) e `crates/xrf/src/forest/tree.rs` (crescimento e partição dos nós).
- scikit-learn 1.9.1: [`_splitter.pyx`, splitter aleatório](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/tree/_splitter.pyx#L461-L650); [documentação de ExtraTreesClassifier](https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).
