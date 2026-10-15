# Extra Trees exato: hipóteses de aceleração, 2026-10-15

Foram mantidas as otimizações que preservaram divisões, predições e qualidade:

- Entrada float32 diretamente em matriz por coluna; elimina o intermediário float64.
- Buffer de feature com o dtype original, mantendo contas de limiar/impureza em float64.
- Especialização dos loops conforme a presença de NaN, decidida fora do loop.
- Pesos contíguos no nó binário exato, reutilizados entre features.

Entradas float64 continuam armazenadas em float64. O backend binned não recebeu
mudanças no algoritmo. Não foram mantidos flags experimentais no código de produção.

## Comparação final alternada

Dois passes AB/BA, com três seeds por passe e um warmup por modelo antes de medir a
primeira seed: seis fits medidos por variante e configuração. A e B são extensões
nativas em pacotes temporários separados, sem sobrescrever o binário ativo.
Usam os mesmos dados, wrappers Python, hiperparâmetros e ordem de modelos. Registramos
tempo de parede, CPU do processo (inclui workers), hash das divisões com ordem das
árvores normalizada, predições e importâncias. O timer inclui preparação interna.

**A baseline já contém a otimização por coluna da etapa anterior desta investigação.**
Não é uma medição direta contra o wheel PyPI alpha14. Os builds de performance são
anteriores ao bump de metadados para alpha15; os hashes individuais estão nos manifests.
O wheel final alpha15 foi validado separadamente pela suíte completa de testes.

| Configuração | Exato anterior (s) | Exato novo (s) | Binned novo (s) |
| --- | ---: | ---: | ---: |
| Credit, com bootstrap, 100k × 500, 64 árvores | 1.1754 | 0.7544 | 0.8755 |
| Credit, sem bootstrap, 100k × 500, 64 árvores | 1.7508 | 1.3352 | 1.3548 |
| Offer, float64, 50k × 100, 64 árvores | 0.2322 | 0.1996 | 0.1201 |
| Churn, árvores profundas, 10k × 50, 16 árvores | 0.01680 | 0.01614 | — |

Valores são medianas de fit; dimensões são antes de separar 80% de treino e de
acrescentar igual número de features sombra. Credit: depth=8, min_leaf=500.
Offer: depth=6, min_leaf=300. Churn profundo: sem limite de profundidade, min_leaf=1.
Todos usam 6 jobs, sqrt, pesos balanceados e seeds 20261011–20261013. Bootstrap,
quando ligado, usa max_samples=0.8. Binned: 63 bins, sampled_select, amostra de 20k.

A redução entre medianas do exato foi 35.8% com bootstrap e 23.7% sem bootstrap.
As medianas das razões pareadas foram 1.542× e 1.261×; de CPU, 1.179× e 1.153×.
Essas medidas não são intercambiáveis. O controle binned permaneceu próximo:
0.8780→0.8755 s e 1.4011→1.3548 s. As razões pareadas do binned variaram um pouco
em torno de 1, inclusive 0.988× de tempo de parede no caso com bootstrap e 0.990×
de CPU sem bootstrap, compatível com ruído destas medições curtas.

Nos 24 pares de fits exatos dessas quatro configurações, as divisões das árvores
foram idênticas, a diferença máxima nas predições foi zero e as métricas de ranking
permaneceram iguais. Diferenças de importâncias ficaram até 5.56e-17 no exato e
1.12e-16 no controle binned, associadas à agregação paralela. Não houve evidência de
regressão de tempo nos cenários exatos medidos. O teste profundo dura poucos ms e
serve apenas como triagem, não para estimar uma margem de performance confiável.

## Hipóteses descartadas e contribuição incremental

`screening.csv` contém as triagens iniciais com três seeds, sem warmup. Elas não
substituem a comparação alternada acima, pois a carga do hardware variou.

| Hipótese | Resultado da triagem |
| --- | --- |
| Buffer tipado + caminho sem NaN | Fit mediano 1.712→1.491 s; predições e métricas preservadas. As duas mudanças foram testadas juntas. |
| Entrada float32 direta, além das anteriores | 0.933 s na triagem; preservou qualidade. |
| Bins com refinamento no bin atravessado pelo limiar exato | 1.281 s: preservou as divisões do teste, mas não mostrou ganho sobre a entrada direta; descartado. |
| Sorteio na escala original com aproximação pela borda inferior de bin | AP do ranking 0.644 contra 0.701 no exato e fit 2.101 s; descartado. |
| Cache de pesos por nó, além da entrada direta | Comparação AB/BA em `cache-abba`: 0.765→0.704 s; razão pareada 1.068×, CPU 1.041×, divisões idênticas. Mantido. |

O protótipo de refinamento preserva os valores originais, constrói também códigos
de bins e mantém uma lista de amostras por bin no nó. Soma estatísticas dos bins
inteiros e reavalia valores originais somente no bin atravessado. O custo de
pré-processamento, estatísticas e listas não compensou nesta implementação.
Isso rejeita este protótipo, não demonstra que toda implementação de refinamento
será lenta. Seu patch e os das etapas anteriores foram preservados para revisão.
O patch experimental deriva da etapa `direct`, não da versão final com cache.
Os patches usam contexto zero (`git apply --unidiff-zero`) para preservar somente
as mudanças de cada hipótese.

## Validação e limites

O wheel 0.1.0a15 foi compilado com `maturin build --release --locked`, instalado no
venv sem alterar dependências e testado: **269 testes Python, 57 unitários Rust e
5 testes CLI passaram**. Novos testes cobrem float32 C/F/strided contra float64,
NaN, pesos zero, multiclass, bootstrap, valores especiais e diferenças float64
menores que a precisão de float32. Os manifests e lockfiles concordam em alpha15.

São medições locais no macOS ARM64; não demonstram ausência de regressão em todo
hardware/dataset nem a convergência final do Boruta. Os dados são sintéticos e
raros; não se deve extrapolar estes resultados para a qualidade em produção.
O gerador AST do notebook de trabalho foi conferido contra master e é idêntico;
as alterações não relacionadas do notebook ficaram fora da mudança.

## Reprodução

```sh
.venv/bin/maturin develop --release --skip-install --locked
.venv/bin/python benchmarks/run_extra_trees_quick.py --include-binned --warmup --output /tmp/current.json
```

Para comparar dois builds preservados (cada comando leva dezenas de segundos):

```sh
.venv/bin/python benchmarks/run_extra_trees_ablation.py \
  --baseline-core /caminho/baseline/_core.abi3.so \
  --candidate-core python/bankai_random_forest/_core.abi3.so \
  --include-binned --output-dir /tmp/exact-abba
```

Use `--no-bootstrap`, `--dtype float64`, ou `--min-leaf 1 --max-depth 0` para os
outros cenários. Arquivos `measurements.csv`, `paired.json`, `summary.json` e
`manifest.json` em cada subpasta documentam os resultados. O runner também guarda
os vetores completos nos JSONs da pasta de saída; eles não foram incluídos no repo
para evitar duplicar muitos MB de dados sintéticos.
