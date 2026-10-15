# Extra Trees exato: layout por coluna (2026-10-14)

O caminho denso exato agora guarda a matriz compartilhada por coluna durante o fit.
A conversão existente para float32 e a transposição são feitas juntas, em blocos
para aproveitar cache, distribuídas pelos jobs. Todos os acessos exatos respeitam
o layout. Não há bins, mudança no RNG, critérios ou hiperparâmetros.
A matriz de previsão continua sendo construída normalmente. Dados sparse e o
algoritmo RF não recebem a transposição.

## Benchmark curto

```sh
.venv/bin/maturin develop --release --skip-install
.venv/bin/python benchmarks/run_extra_trees_quick.py --output /tmp/exact.json
.venv/bin/python benchmarks/run_extra_trees_quick.py --no-bootstrap --output /tmp/exact-no-bootstrap.json
```

O script importa o build local em `python/`, reutiliza apenas as definições do
gerador do notebook Boruta e mede um fit com features sombra. Padrão: 100 mil
linhas geradas, 80 mil para treino, 500 features originais + 500 sombras, 64 árvores,
6 jobs, cenário credit_default, três seeds. Mesmos hiperparâmetros nominais para
Bankai e sklearn: profundidade 8, folha mínima 500, sqrt, class_weight balanced.
Bootstrap usa max_samples=0.8. A geração, previsão e métricas ficam fora do timer;
a preparação interna e a transposição fazem parte do fit medido. Alterna a ordem
dos modelos entre repetições. Não há warmup separado; são medições curtas locais,
sujeitas a cache, temperatura e ruído. Cada execução leva dezenas de segundos.

Resultados em `benchmarks/results-extra-trees-quick/measurements.csv`; configuração,
plataforma, hashes dos binários e verificações de equivalência no manifest ao lado.
A baseline foi recompilada do HEAD indicado no manifest com as mesmas opções release.

| Bootstrap | Versão | Fit mediano (s) | AP das features | Recall@24 |
| --- | --- | ---: | ---: | ---: |
| Sim | Bankai anterior | 2.519 | 0.7013 | 0.6667 |
| Sim | Bankai por coluna | 1.656 | 0.7013 | 0.6667 |
| Sim | Bankai por coluna + conversão junta | 1.512 | 0.7013 | 0.6667 |
| Sim | sklearn na execução final | 1.817 | 0.7757 | 0.7083 |
| Não | Bankai por coluna + conversão junta | 2.329 | 0.7538 | 0.7500 |
| Não | sklearn | 4.968 | 0.7597 | 0.7083 |

Com bootstrap, o fit mediano do Bankai caiu 40.0% frente ao anterior. Na execução
final, ficou 16.8% abaixo do sklearn; venceu nas três seeds. O sklearn variou entre
as execuções (medianas 1.58–1.82 s), então essa margem precisa de mais repetições
para ser considerada estável. Sem bootstrap, o Bankai ficou 53.1% abaixo do sklearn.
Não foi medida uma baseline anterior sem bootstrap nesta investigação.

Nas três comparações antes/depois com bootstrap, as probabilidades do Bankai
foram idênticas, a diferença máxima de importância foi 6.94e-18, e as métricas de
qualidade se mantiveram. Isso preserva a qualidade anterior; não elimina a diferença
para o sklearn com bootstrap. Sem bootstrap, o AP do ranking foi próximo, mas o
sklearn teve ROC AUC e AP preditivo maiores (métricas completas no CSV).
As bibliotecas têm RNG e semânticas de bootstrap ponderado diferentes; parâmetros
iguais não significam árvores ou amostras idênticas.

Este teste mede o trabalho inicial de uma rodada do Boruta. Não valida convergência,
recall/FDR finais, outros cenários nem escala de 500 mil linhas. O notebook não foi
alterado nesta investigação. O caminho de entrada ainda materializa float64 antes
de convertê-lo para float32; remover esse intermediário pode reduzir preparação e
pico de memória em trabalho futuro. Para float64, a transposição exige outra matriz
temporária do mesmo tamanho; em float32, foi fundida à conversão já existente.

Validação: 256 testes Python e 56 testes Rust passaram, incluindo teste de layout
retangular, NaN, float32/float64, compartilhamento e proteção contra retransposição.
